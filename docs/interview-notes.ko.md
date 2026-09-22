# 구현을 설명하기 위한 질문 노트

[프로젝트](../README.ko.md) · [선행연구](related-work/README.ko.md)

코드 읽기와 설명 연습용입니다. 사용자가 직접 집필하거나 구술 검증을 완료했다는 기록은 아닙니다. 기존 Case008 소스와 새 Case009 실행 근거를 구분합니다.

## 1. 왜 meta device를 사용했나?

먼저 원래 full-width 가중치를 할당하지 않고 모델 형상을 만듭니다. [load_dense](../tools/modelpack/artifact.py)는 meta skeleton에서 변경한 층을 작은 MLP로 교체한 뒤 실제 tensor를 할당합니다. meta에는 데이터가 없으므로 rotary 같은 파생 buffer도 복원해야 합니다. 대안인 원본 로드 후 교체는 초기 메모리와 원본 checkpoint를 요구합니다. [저장·로딩 테스트](../cases/008-build-reconstruct-reload/tests/test_core.py)와 [CPU roundtrip](../tools/modelpack_demo/roundtrip.py)을 읽고, “어떤 tensor는 저장되고 어떤 buffer는 재생성되는가?”를 설명해 봅니다.

## 2. strict=True와 assign=True는 각각 무엇을 하나?

strict는 key 누락·추가를 거절하고 assign은 state tensor를 실제 parameter/buffer로 할당합니다. meta tensor에 일반 copy_를 시도하는 것과 다릅니다. 이후 tied weights를 다시 묶습니다. optimizer를 먼저 만들었다면 parameter reference가 달라질 수 있어 별도 주의가 필요하지만 이 도구는 추론 loader입니다. [실제 할당 순서](../tools/modelpack/artifact.py)와 누락 tensor 실패 테스트를 확인하고 “형상 검사와 strict key 검사는 서로 무엇을 잡는가?”를 설명합니다.

## 3. 왜 .partial → 검사 → rename인가?

[save_dense](../tools/modelpack/artifact.py)는 쓰는 중인 artifact를 final 이름으로 노출하지 않고 검사 후 같은 filesystem에서 rename합니다. 이는 디렉터리 이름의 atomic visibility를 이용하며 fsync를 통한 crash durability까지 보장하는 구현은 아닙니다. 다른 filesystem으로 이동하거나 rename 후 전원이 꺼지는 경우는 별도 문제입니다. CPU roundtrip의 손상 artifact 거절을 보고 “독자가 어느 시점부터 final 경로를 읽을 수 있는가?”를 답합니다.

## 4. λ=η tr(G)/m인 이유는?

[fit_ridge](../tools/modelpack/numerics.py)의 G는 활성값 Gram입니다. 평균 diagonal로 λ를 정규화해 activation scale에 대응하는 dimensionless η를 사용합니다. 목적은 ||W H−Y||² + λ||W−W₀||²이며 W₀는 원래 작은 down projection입니다. 최적 η의 이론 보장이 아닙니다. 원래 DEV 선택 규칙과 solver 테스트를 읽고 “η와 λ를 바꾸면 어느 가중치에 가까워지는가?”를 설명합니다.

## 5. 왜 Cholesky와 solve인가?

유효 Gram과 양의 λ는 SPD 구조를 이용할 수 있게 합니다. 명시적 역행렬 대신 두 번의 solve로 계산합니다. normal equations는 conditioning을 악화시킬 수 있고 QR/SVD는 다른 비용의 대안입니다. 작은 normal-equation residual은 계산 문제를 풀었다는 증거이며 held-out 일반화 증거는 아닙니다. [solver](../tools/modelpack/numerics.py)·[테스트](../cases/008-build-reconstruct-reload/tests/test_core.py)를 보고 “residual과 예측 오차는 왜 다른가?”를 설명합니다.

## 6. 252개와 144개는 왜 다른가?

Qwen4B는 36층입니다. 저장 전 층마다 q/k/v/o/gate/up/down의 7개 Linear를 세면 36×7=252입니다. vLLM은 qkv와 gate/up을 묶어 qkv/o/gate_up/down 4개로 실행하므로 36×4=144입니다. 이는 동적 GPU 호출 수가 아닙니다. [변환 대상 검사](../tools/modelpack/quantized.py)와 [runtime_state](../tools/modelpack/q_runtime.py)가 각 경계를 기록하며, 새 DEV trace에서는 실제 Marlin kernel도 확인합니다. lm_head는 GPTQ 변환에서 제외됩니다. “저장 tensor의 이름을 fused runtime module에 어떻게 대응할 것인가?”를 설명합니다.

## 7. W4A16이 decode에 유리할 수 있는 이유는?

Linear FLOPs≈2MKN이고, weight-heavy byte 수는 대략 bytes_per_weight×KN+scale bytes입니다. 작은 M에서는 가중치 이동 감소가 도움이 될 수 있지만 M이 커지면 재사용과 산술 집약도, dequant 비용이 달라집니다. attention/KV와 host 비용도 남습니다. [MARLIN](related-work/README.md)과 [요청 runner](../cases/009-q-serving-quality/scripts/serving.py)를 읽고 “client concurrency가 늘 때 M이 반드시 같은 값으로 늘어나는가?”를 설명합니다.

## 8. kernel speedup과 모델 speedup은 왜 다른가?

나머지 비용이 같다는 가정 아래 S_total=1/((1−f)+f/S_kernel)입니다. f가 작거나 다른 비용이 늘면 커널 향상이 전체 요청에 적게 반영됩니다. 이 식은 설명 모델이며 scheduling·cache·graph 변화까지 실측 대신 계산하지 않습니다. Case004 complete operator, Case007 prefill, Case009 serving은 서로 다른 경계입니다. “무엇을 타이머 안에 넣었는가?”부터 답합니다.

## 9. 압축 후 정답률이 높아지면?

기준선 보존과 정답 품질은 다른 목표입니다. 손실·개선·다른 오답 전환을 보고 문항 설계와 표본 불확실성을 따로 검사합니다. 기존 code 생성기에는 index→gold, count→gold, 정답 최솟값 단서가 있었습니다. 단서의 존재가 모델의 이용을 증명하지는 않습니다. [새 data-v2 검사](../cases/009-q-serving-quality/scripts/data_v2.py)와 [paired core](../packages/diova-compare/src/diova_compare/core.py)를 읽고 “원인이 관측된 것인가, 해석 후보인가?”를 설명합니다.

## 10. 실제 Marlin과 graph replay의 근거는?

configuration flag와 module 이름만으로 충분하지 않습니다. 새 DEV는 C=1/16의 짧은 profiler 구간에서 cudaGraphLaunch와 Marlin CUDA kernel 이름을 남깁니다. [요약](../cases/009-q-serving-quality/provenance/dev_profile_summary.json)은 이 진단의 근거입니다. 본 timing은 profiler를 끄고 측정합니다. trace가 보여준 것은 관측한 DEV 구간이며 모든 요청의 모든 연산을 개별 증명한 것은 아닙니다. “capture와 replay의 증거는 각각 어느 로그인가?”를 설명합니다.

## 11. concurrency와 active batch는 어떻게 구분했나?

client semaphore는 동시에 진행 중인 HTTP 요청의 상한입니다. scheduler의 running sequence 수와 GEMM token-row 수는 별도 관측입니다. [serving.py](../cases/009-q-serving-quality/scripts/serving.py)는 client concurrency를 조건으로 기록하고 vLLM metrics의 preemption/queue 기록을 보존합니다. streaming event 수로 token 수를 대신하지 않고 API usage 및 반환 token IDs를 대조합니다. “동시성32에서 active32라고 말하려면 무엇이 더 필요한가?”를 답합니다.

## 12. 직접 구현과 upstream은 어디서 나뉘나?

프로젝트는 artifact 계약·검사, 고정 구조 solver 연결, loader, 측정 및 pairing, 결과 UI를 구현합니다. 모델·GPTQ 방법·serialization 라이브러리·추론 runtime·커널은 upstream을 사용합니다. [기여 표](../README.ko.md)와 [선행연구](related-work/README.ko.md), 원래 NOTICE를 따라 함수와 artifact 단위로 대응합니다. “이 저장소가 없어도 upstream이 제공하는 기능과 여기에서 추가한 검사는 무엇인가?”를 설명합니다.
