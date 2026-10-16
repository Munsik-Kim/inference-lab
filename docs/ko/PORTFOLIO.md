# 도구, 구현, 기술 협업

[English](../en/PORTFOLIO.md) | 한국어 · [홈](../../README.ko.md)

[처음 보는 분](START_HERE.md) · [쉬운 용어집](GLOSSARY.md)

[모델별 실전 안내서](LLM_GUIDE.md)에서 Qwen3-4B의 검증된 실행 경로와 품질·실패·완화 기록을 연결합니다.

## 30초 소개

<!-- claims: c008-tracks c008-artifact-implementation c007-implementation -->
모델을 작게 저장해도 원래 맞히던 질문을 틀리거나 요청한 작업을 끝내지 못할 수 있습니다. 저는 DIOVA에서 변경한 모델을 저장·재실행하는 도구, 같은 질문의 답변·실행 비용을 비교하는 도구, 실제 도구 작업을 확인하는 도구를 만들었습니다. 공개 모델과 RTX 5080 16GB의 실행 기록이 근거입니다.

로컬 AI 실행·추론 최적화·변경 후 품질 검사와 관련된 구현입니다. [기술 협업 범위](#기술-협업-범위)에서 논의할 수 있는 작업을 확인할 수 있습니다.

[대표작과 직접 실행하기](https://munsik-kim.github.io/inference-lab/ko/index.html#featured-stories) · [내 결과 비교 예제](../../packages/diova-compare/examples/compare/README.ko.md)

## 대표작 A · 모델을 작게 저장하고 실제 요청까지 실행하기

Case008 Q의 **기존 압축 방법(GPTQ)으로 변환 → 변환해 저장한 모델 파일(artifact) 검사 → 새로 시작한 프로그램에서 재실행**을 Case009의 **실제 요청 시간·메모리·답변 품질 비교**로 연결했습니다. [변환 코드](../../tools/modelpack/quantized.py) · [저장본 검사](../../tools/modelpack/qartifact.py) · [실제 요청 실행기](../../cases/009-q-serving-quality/scripts/serving.py).

Case008 Q의 가중치 파일은 **8.045 GB → 2.652 GB, 67.0% 감소**했습니다. 같은 실험의 PyTorch GPU 할당 최고값은 **10.4868 → 5.3971 GiB, 약 48.5% 감소**입니다. eager 실행·요청 1개·캐시 1 GiB 조건으로 초기화와 실행을 포함합니다. [파일·할당 메모리 원자료](https://munsik-kim.github.io/inference-lab/ko/qwen3-4b.html#storage).

후속 Case009의 GPU 전체 메모리 최고 표본값은 **14.08–14.51 → 9.08–9.52 GiB**로, 같은 라운드끼리 **4.98–5.06 GiB·34.4–35.8% 감소**했습니다. graph 실행·같은 캐시 4 GiB·3라운드의 1초 간격 표본이며, 시작·준비 실행·바탕화면 사용량을 포함합니다. [VRAM 수치·계산·조건](https://munsik-kim.github.io/inference-lab/ko/qwen3-4b.html#device-memory). 파일 크기와 두 메모리 지표는 서로 다른 경계의 측정입니다.

Case008 R은 별도 Qwen3-0.6B 한 블록의 구조 축소·복원입니다. [Case008 Q 구현](https://munsik-kim.github.io/inference-lab/ko/case008.html#track-q) → [Case009 실제 요청 비교](https://munsik-kim.github.io/inference-lab/ko/case009.html#serving).

**기술적 판단 — 현재 설계 해석:** 파일이 작아도 불러오기에 실패하거나 의도한 계산을 사용하지 못할 수 있어, 새 프로그램에서 실행하는 것까지 검사합니다. 16비트 가중치(BF16)와 4비트 가중치(W4)는 같은 실행 방식 안에서 비교합니다. GPU 실행 순서를 미리 준비하는 graph와 매번 연산을 호출하는 eager의 차이를 가중치 변경 효과와 구분하기 위해서입니다. [저장 경계 테스트](../../cases/008-build-reconstruct-reload/tests/test_boundaries.py)와 [요청 프로토콜](../../cases/009-q-serving-quality/configs/serving_protocol.json)이 근거입니다. 다른 장치의 이득과 Case009 품질 실행의 비정상 종료 문제는 해결된 것으로 보지 않습니다. 이는 현재 코드·기록의 해석이며 당시 의도를 새로 단정하지 않습니다.

## 대표작 B · 같은 질문의 답변을 비교하는 diova-compare

모델 없이 쓰는 독립 CPU 비교 도구를 만들었습니다. 두 파일의 질문과 정답 기준이 같은지 검사하고, 원래 맞혔다가 틀린 질문·새로 맞힌 질문·다른 오답으로 바뀐 질문을 나눠 보여줍니다. 원래 프로그램의 종료 코드가 제공되면 점수와 함께 보존합니다. [작은 합성 예제로 실행](../../packages/diova-compare/examples/compare/README.ko.md) · [입력 형식·명령어 사용법](../../packages/diova-compare/README.ko.md) · [집계 코드](../../packages/diova-compare/src/diova_compare/core.py) · [테스트](../../packages/diova-compare/tests/test_compare.py).

**기술적 판단 — 현재 설계 해석:** 예제의 두 파일은 모두 2/4 정답이지만, 원래 맞힌 답 하나를 잃고 다른 답 하나를 새로 맞힙니다. 질문을 하나씩 짝지어 보는 비교(paired comparison)로 평균 정답률이 가린 변화를 찾습니다. 기본 비교에는 모델의 원시 점수(logits)가 필요 없습니다. 자유롭게 생성한 답의 정오와 입력·결과 파일 식별자는 사용자가 제공합니다. 파일 검사가 자동 채점이나 원자료의 진위를 보증하지는 않습니다. 새 업무에는 그 업무에 맞는 채점·질문 대응 규칙이 필요합니다.

## 대표작 C · 답변이 아니라 실제 도구 작업 완료 확인하기

AI가 성공했다는 최종 메시지를 냈어도 시험용 파일을 고치거나 요구한 검사를 실행하지 않은 경우가 있었습니다. 사용 가능한 도구 이름만 허용하는 제한과 호출 오류 안내를 만들고, 실제 실행 기록과 파일 변경을 확인했습니다. [도구 이름 제한 코드](../../research/agent_tool_contract_repair_20261014/guard.py) · [실제 작업 기록 분석](../../research/agent_tool_contract_repair_20261014/analyze.py) · [기능 테스트](../../research/agent_tool_contract_repair_20261014/test_guard.py) · [전체 보고서](../../research/agent_tool_contract_repair_20261014/records/REPORT_KO.md).

**설정 조정에 쓰던 기존 문제(DEV) 8개 재사용 · 작은 표본:** D16은 6/8 → 8/8, D4는 0/8 → 2/8이며 D4의 6개 실패가 남았습니다. D16은 가중치를 4비트 정수(INT4)로 저장하고 계산 입력은 BF16으로 유지합니다. D4는 같은 가중치에 입력까지 INT4로 바꿉니다. 두 경로 모두 압축 값을 BF16으로 복원해 품질을 보는 참조 계산(reference)입니다. 새 문제의 성능과 압축 상태로 직접 계산하는 가속 효과는 검증하지 않았습니다.

**기술적 판단 — 현재 설계 해석:** 최종 JSON 메시지만 채점하면 실제로 하지 않은 작업을 놓칩니다. 도구 이름 제한·오류 안내·둘의 결합을 따로 시험하고 실행 기록과 파일 변경을 검사했습니다. 이름 제한(ENUM)과 결합(BOTH)이 동률이어서 오류 안내의 추가 이득은 확인하지 못했습니다. 새 문제와 남은 D4 실패를 더 확인해야 합니다.

## 연구·수치 엔지니어링 기록

[Case014](https://munsik-kim.github.io/inference-lab/ko/case014.html#overview)는 변환한 행렬의 INT4 저장 바이트를 유지하면서, 압축 오차를 줄이도록 가중치를 조정한 별도 연구입니다. PG19 책에서는 텍스트 예측 점수(PPL; 낮을수록 좋음)가 개선됐지만 ArXiv 논문에서는 같은 이득을 확인하지 못했습니다. 전체 모델 파일 크기·과제 정답률·속도·GPU 메모리(VRAM)가 개선됐다는 결과는 아닙니다.

Case006·007·008 R·010–012를 포함한 [전체 사례](CASEBOOK.md)와 아래 코드 투어에서 다른 구현·부정 결과도 계속 확인할 수 있습니다.

## 구조 복원과 평가 도구의 기여

![고정 코드 입력 하나를 비교하는 실제 Case007 로컬 화면](../../presentation/screenshots/case007-comparison.png)

*로컬 Edge 화면: 첫 고정 CODE 입력의 25% 삭제 비교입니다. [Case006 출력 계산 화면](../../presentation/screenshots/case006-readout.png)은 별도로 표시한 같은 입력 진단을 보여줍니다. 탐색을 안내하는 선택 화면이며 전체 성능 집계가 아닙니다.*

Qwen3-0.6B의 한 MLP를 줄인 저장본의 구조 복원은 [코드 읽는 순서](#code-tour)에서, 다른 사례는 [전체 사례](CASEBOOK.md)에서 확인합니다.

<a id="implementation-map"></a>
## 구현 역량과 실제 근거

| 프로젝트 구현 | 기존 연구·라이브러리에서 제공 | 코드·테스트 |
|---|---|---|
| 변환해 저장한 모델 파일 검사·불완전한 저장 방지·구조가 맞지 않는 재로딩 거절 | PyTorch · Transformers · safetensors | [코드](../../tools/modelpack/artifact.py) · [테스트](../../cases/008-build-reconstruct-reload/tests/test_core.py) |
| Q 모델의 변환·저장·복사·새로 시작한 추론 프로그램에서 실행 검사 | GPTQ · LLM Compressor · compressed-tensors · vLLM/Marlin | [코드](../../tools/modelpack/quantized.py) · [테스트](../../cases/008-build-reconstruct-reload/tests/test_boundaries.py) |
| 작은 계산 블록의 출력 오차를 줄이는 가중치 적합(ridge 회귀)·보정 저장 | NumPy · CPU FP64 · ridge 회귀 · Cholesky · 채널 재구성 선행연구 | [코드](../../tools/modelpack/numerics.py) · [테스트](../../cases/008-build-reconstruct-reload/tests/test_core.py) |
| 같은 질문을 짝지어 측정하고 개별 변화를 보여주는 비교 보고서 | vLLM benchmark · lm-evaluation-harness · 비교 지표 선행연구 | [코드](../../packages/diova-compare/src/diova_compare/core.py) · [테스트](../../packages/diova-compare/tests/test_compare.py) |
| 모델의 변화하는 기억을 적은 비트로 저장·실패 상태 보존·새 프로그램에서 이어 실행 | NumPy · PyTorch · ComplexKDA recurrence | [구현](../../cases/010-ckda-finite-precision-memory-horizon/versions/v2/source/online_v2.py) · [합성 재시작 테스트](../../cases/010-ckda-finite-precision-memory-horizon/versions/v2/tests/test_online_v2.py) |
| 같은 기억에서 답을 읽는 마지막 층만 교체·변경한 두 배열 저장·재적용 | ComplexKDA · PyTorch L-BFGS · NumPy | [모델 연결 코드](../../cases/011-frozen-state-readout-adaptation/source/adapter.py) · [변경 가중치 저장·재적용 검사](../../cases/011-frozen-state-readout-adaptation/tests/test_fitting_patch.py) |
| 같은 초기 잡음에서 생성 단계·내부 반복·시간·요청한 개수·색·좌우 조건 충족 여부 비교 | Looped-DiT · FLAN-T5 · PyTorch · Pillow | [모델 연결 코드](../../cases/012-looped-dit-inference-budget/source/adapter.py) · [CPU 검사](../../cases/012-looped-dit-inference-budget/tests/test_analysis.py) |
| 배율을 유지하는 4비트 정수(INT4) 변환·저장된 품질 결과 재검산 | DualQuant 공개 수식 · PyTorch · NumPy | [변환](../../cases/013-dualquant-reference-quality/source/paper_int4.py) · [CPU 검산](../../cases/013-dualquant-reference-quality/analysis/audit.py) |
| 앞선 층 압축으로 달라진 중간 계산값을 다음 층에서 수집해 보정·INT4 저장/재로딩·모델 전체 문장 품질 비교 | Guo compensation · PyTorch · Transformers · NumPy | [코드](../../cases/014-quantization-compensation/followups/p2_core/source/run_p2.py) · [테스트](../../cases/014-quantization-compensation/followups/p2_core/tests/test_p2.py) |
| API 이름 변경·관측 이력·상태 회귀시험 | Qwen · PyTorch · Transformers · Python stdlib | [runner](../../cases/015-tool-interface-sensitivity/followups/order-api-regression-v1/src/order_regression/runner.py) · [tests](../../cases/015-tool-interface-sensitivity/followups/order-api-regression-v1/tests/test_order_contracts.py) |
| 같은 입력의 GPU 실행 시간·메모리 측정 | vLLM · CUDA · NVML · PyTorch SDPA · SageAttention | [실행기](../../cases/002-bf16-fp8-document-extraction/scripts/run.py) · [시간 측정 경계](../../cases/004-low-precision-attention-break-even/src/timing.py) |
| 공개 기록 재계산·화면 수치 대조 | Python · NumPy · JavaScript · unittest · GitHub Actions · Pages | [재계산](../../tools/showcase/replay_selection.py) · [표시 검사](../../tests/showcase/test_showcase.py) |

- **실행 경로 진단:** RTX 5080의 GPU 구조(SM120)에 맞는 연산을 선택하도록 하는 기존 라이브러리 수정을 전후 비교 실행기로 확인했습니다. [Case001 실행기](../../cases/001-sm120-int8-fallback/run.py).
- **비용 측정:** 입력 정보를 조합하는 attention 연산에서 값 범위 조정(smoothing)·적은 비트로 변환(양자화)·형식 변환을 포함한 호출 시간을 측정했습니다. [Case004 어댑터](../../cases/004-low-precision-attention-break-even/src/backends.py).
- **수치 분석:** EFQ 방법의 숫자 대응과 비교에 쓰는 배율 선택을 나눠 확인하고, 미리 정한 속도·오차 조건을 만족하는 설정이 없을 때 중단한 판정을 기록했습니다. [Case003 분석·영어](../../cases/003-efq-softmax-numerical-audit/ANALYSIS.md) · [Case005 분석·영어](../../cases/005-attention-precision-pareto/ANALYSIS.md).

<a id="code-tour"></a>
## 구현 코드 읽기

**저장본 제작과 재로딩부터 읽습니다.** [`save_dense()`](https://github.com/Munsik-Kim/inference-lab/blob/9bc8b8fced8dfd5147f9bfdc67564eda04670810/tools/modelpack/artifact.py#L32)는 숫자 배열(tensor)과 모델 구조 목록을 저장합니다. [`load_dense()`](https://github.com/Munsik-Kim/inference-lab/blob/9bc8b8fced8dfd5147f9bfdc67564eda04670810/tools/modelpack/artifact.py#L109)는 빈 모델 구조(meta)에서 층별 크기를 맞추고 저장한 가중치를 빠짐없이 대입합니다. 함께 쓰는 가중치를 연결하고, 위치 정보 계산에 필요한 보조 배열(rotary buffer)도 복원합니다. [`fit_ridge()`](https://github.com/Munsik-Kim/inference-lab/blob/9bc8b8fced8dfd5147f9bfdc67564eda04670810/tools/modelpack/numerics.py#L13)는 64비트 계산(FP64)으로 작은 블록의 출력 오차를 줄이는 가중치를 CPU에서 구합니다. [원래 저장 테스트](../../cases/008-build-reconstruct-reload/tests/test_core.py)는 누락된 숫자 배열과 맞지 않는 구조를 거절합니다. [독립 CPU 예제](../../tools/modelpack_demo/roundtrip.py)는 이 로더를 다른 프로세스에서 실행합니다.

<!-- claims: c006-implementation -->
1. **모델의 attention 호출을 제한합니다.** [`ScopedAttention`](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/006-attention-decision-stability/src/intervention.py#L30)부터 읽습니다. `route`는 질문을 처음 읽는 단계(prefill)의 13번 층에서만, 참조할 위치와 읽을 위치 수가 같은 정사각형 attention 계산을 교체합니다. `__exit__`는 앞서 등록돼 있던 함수를 복구합니다. [예외 복구 테스트](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/006-attention-decision-stability/tests/test_core.py#L193)에서 이 동작을 확인할 수 있습니다.
2. **행렬 크기를 줄입니다.** [`sliced`](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/007-interaction-aware-mlp-pruning/src/surgery.py#L6)는 남긴 그룹을 gate/up 행과 down 열에 대응시킵니다. `masked`가 비교 기준을 만들고 `replace_mlp`는 `finally`에서 원래 모듈을 복구합니다. [배열 크기·편향·계산 일치 테스트](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/007-interaction-aware-mlp-pruning/tests/test_core.py#L62)가 이 연결을 검사합니다.
3. **불완전한 근거를 거절합니다.** [`assess`](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/006-attention-decision-stability/src/validity.py#L10)는 전체 출력과 실행 경로 검사 결과를 요구합니다. [같은 질문을 짝짓는 원래 테스트](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/006-attention-decision-stability/tests/test_core.py#L249)는 누락·중복·서로 다른 입력의 결합을 거절합니다. 새 [화면 검사기](../../tools/showcase/check.py)는 표시용 데이터를 보존된 원자료와 대조합니다.

## 60–90초 시연 대본

*CPU 합성 예제를 실제로 실행하며 보여줄 동선입니다. 이 문서는 녹화 영상이 아닙니다.*

- **0–15초:** “모델 변경 전후의 정답률이 같아도 원래 맞히던 질문을 틀릴 수 있습니다.”
- **15–40초:** [예제](../../packages/diova-compare/examples/compare/README.ko.md)의 두 파일을 validate하고 compare합니다. 모델을 실행하는 시연이 아니라 저장된 합성 답변을 CPU에서 비교한다고 밝힙니다.
- **40–65초:** 생성된 report.md와 pairs.csv에서 sum-01의 정답 손실과 sum-02의 회복을 엽니다.
- **65–90초:** [집계 코드](../../packages/diova-compare/src/diova_compare/core.py)와 [전환·거절·실패 보존 테스트](../../packages/diova-compare/tests/test_compare.py)를 확인합니다.

과거 GPU 결과 조회는 [Case008 Q](https://munsik-kim.github.io/inference-lab/ko/case008.html#track-q) → [Case009](https://munsik-kim.github.io/inference-lab/ko/case009.html#quality)로 별도 보여줍니다. 작은 무작위 모델의 [CPU 저장·재로딩 예제](GETTING_STARTED.md#tiny-model-demo)도 별개 구현 경로입니다.

<a id="근거로-연결되는-소개-문장-세-개"></a>
## 도구의 성능 평가

<!-- claims: c007-transfer c007-quality c006-native c006-readout -->
Case008 R은 Qwen 0.6B의 한 층 MLP에서 짧은 합성 평가 입력 192개의 삭제로 생긴 국소 제곱 출력 오차를 **94.1–95.4%** 줄였습니다. 정답 점수 변화는 확률 지표와 보정 구조에 따라 달랐습니다. 측정한 요청 시간비의 불확실성 구간에는 1이 포함돼 전체 요청이 빨라졌다고 확인할 수 없었습니다. [Q 품질·속도 평가](https://munsik-kim.github.io/inference-lab/ko/case008.html#q-evaluation)와 [R 평가](https://munsik-kim.github.io/inference-lab/ko/case008.html#r-evaluation)에서 비교할 수 있습니다.

MLP 25% 삭제에서 그룹 간 영향을 함께 보는 선택법(PAIRWISE)은 출력 오차가 소폭 낮고 원래 모델의 답변을 더 많이 유지했습니다. 그러나 정답에 준 확률로 계산하는 손실(NLL; 낮을수록 좋음)은 그룹을 따로 보는 선택법(INDEPENDENT)이 더 좋았습니다. 50%의 선택 모듈은 같았으며 원래 판정은 **뚜렷한 이득을 확인하지 못한 채 완료(`COMPLETED_NO_CLEAR_TRANSFER`)**입니다. Case006에서는 표준 192개 중 A_PUBLIC은 5개, V4는 8개의 선택이 달라졌습니다. 같은 입력의 사후 출력 계산 진단은 동률과 마지막 어휘 점수 계산을 살펴봅니다. 구현상 선택과 서로 다른 평가 목표를 이 예제들에서 확인할 수 있습니다. [사례 안내](CASEBOOK.md) · [로컬 화면 열기](GETTING_STARTED.md#local-showcase).

<!-- claims: c004-cost c005-stop -->
Attention 전체 호출 비용을 측정하고, 관측된 정밀도·비용 절충과 **설정 개발 검사에서 중단했다는 판정(`STOP_DEV_SCREEN`)**을 함께 기록했습니다. [Case004 비용·영어](../../cases/004-low-precision-attention-break-even/README.md) · [Case005 판정·영어](../../cases/005-attention-precision-pareto/README.md).

<a id="contribution-and-reuse"></a>
## 프로젝트 기여와 재사용

<!-- claims: attribution -->
프로젝트의 작업은 비교 실행, 범위를 제한한 어댑터, 행렬 축소, 유효성 검사, 수치·답변 분석과 결과 탐색기입니다. Qwen과 Red Hat AI가 체크포인트를, Transformers·PyTorch·vLLM이 모델과 실행 구성요소를, SageAttention이 저정밀 GPU 연산(커널)을 제공합니다. Case001 guard의 저자는 hclsys이며 EFQ-Softmax는 Han과 공동저자의 작업입니다. HOPE는 여러 전문 계산 블록(expert) 중 일부를 사용하는 모델(MoE)에서 블록을 제거할 때 상호작용을 고려하는 선행연구입니다. 여기서는 모든 채널을 계산하는 MLP(dense MLP)를 그룹으로 나누고, 그 출력 기여에서 그룹 선택 점수를 직접 구했습니다.

GPTQ는 기존 양자화 방법이며 LLM Compressor·compressed-tensors는 변환·압축 저장을, safetensors는 가중치 형식을 제공합니다. Case008에서는 저장본 검사, 층별 구조 로더, 고정 구조 ridge 적합을 연결했습니다. [Case008 출처·영어](../../cases/008-build-reconstruct-reload/NOTICE.md).

OpenAI Codex가 구현·로컬 실행·테스트·분석·문서 작성을 지원했습니다. [Case001](../../cases/001-sm120-int8-fallback/NOTICE.md), [Case003](../../cases/003-efq-softmax-numerical-audit/NOTICE.md), [Case006](../../cases/006-attention-decision-stability/NOTICE.md), [Case007 출처·영어](../../cases/007-interaction-aware-mlp-pruning/NOTICE.md)에 재사용한 작업을 기록했습니다. 프로젝트 코드는 [Apache-2.0](../../LICENSE)이며 모델·기존 라이브러리의 이용 조건은 별도입니다.

## 기술 협업 범위

현재 공개 구현이 보여주는 역량을 요청 단위로 정리하면 다음과 같습니다.

- 로컬 모델의 양자화·실행 설정 비교와 병목 진단.
- 모델 변경 전후의 답변·도구 실행 회귀 평가.
- 측정·기록·검산 과정을 재사용 가능한 도구로 정리.


어떤 모델·업무인지, 무엇이 실패하거나 비교가 필요한지, 어떤 결과가 도움이 될지부터 설명하면 됩니다. 구체적인 측정 범위와 지표는 그다음 함께 정할 수 있습니다. [저장소 이슈](https://github.com/Munsik-Kim/inference-lab/issues)는 공개 기술 질문 경로이며, 인증값이나 비공개 고객 자료는 공개 보고에 넣지 않습니다.

근거는 특정 모델·장치 하나에서 수행한 합성 실험과 Case009의 별도 공식 과제 결과입니다. 실제 서비스 배포 적합성은 **미평가(`NOT_ASSESSED`)**입니다. CPU 검산은 저장된 개별 수치를 다시 계산합니다. 제외된 전체 배열을 복구하거나 GPU 모델 실행을 독립적으로 반복하는 절차는 [재현 안내](GETTING_STARTED.md)에서 따로 설명합니다.


## Case010 — state 저장과 실패 보존

모델이 입력마다 갱신하는 기억(recurrent state)을 적은 비트로 묶어 저장하고, 저장 오차의 보정값을 다음 계산에 전달하는 residual transport를 연결했습니다. 각 행의 갱신을 함께 완료하고 실패 상태도 저장해 재시작 때 실패가 사라지지 않게 했습니다. [v2 구현](../../cases/010-ckda-finite-precision-memory-horizon/versions/v2/source/online_v2.py), [새 프로세스 합성 테스트](../../cases/010-ckda-finite-precision-memory-horizon/versions/v2/tests/test_online_v2.py), [저장 예산과 판독 길이 비교](../../cases/010-ckda-finite-precision-memory-horizon/REPORT.ko.md) 순서로 확인할 수 있습니다. [CPU 검산 안내](GETTING_STARTED.md#case010-cpu). ComplexKDA의 상태 갱신·학습 모델과 이 프로젝트의 저장·복원 코드(codec)·평가 도구는 [출처 안내](../../cases/010-ckda-finite-precision-memory-horizon/NOTICE.md)에 구분했습니다.

## Case011 — 같은 상태에서 판독층만 교체하기

세 가지 답변 판독층(readout)이 같은 저장 상태를 읽게 했습니다. 변경한 두 숫자 배열의 크기와 원본 모델의 파일 지문을 검사하고 새 프로그램에서 다시 적용합니다. [모델 상태를 읽는 연결 코드](../../cases/011-frozen-state-readout-adaptation/source/adapter.py) → [변경 가중치 파일을 읽어 적용하는 코드](../../cases/011-frozen-state-readout-adaptation/source/head_patch.py) → [저장·재적용 규칙 테스트](../../cases/011-frozen-state-readout-adaptation/tests/test_fitting_patch.py) → [독립 검산기](../../cases/011-frozen-state-readout-adaptation/analysis/audit.py) 순서로 읽을 수 있습니다.

모델의 변화하는 기억에 추가한 저장량은 0 B입니다. 가중치를 맞추는 데 쓰지 않은 위치의 정답 확률 손실(교차 엔트로피, CE; 낮을수록 좋음)은 개선됐습니다. 그러나 INT8 저장에서는 세 모델 저장본 모두 첫 오답 전까지 연속해서 맞힌 평균 길이가 줄었습니다. [쉬운 화면](https://munsik-kim.github.io/inference-lab/ko/case011.html) · [상세 결과·가중치 적합 계산의 종료 상태](../../cases/011-frozen-state-readout-adaptation/REPORT.ko.md) · [CPU 실행](GETTING_STARTED.md#case011-cpu). PyTorch/NumPy와 ComplexKDA의 역할은 [기여·출처](../../cases/011-frozen-state-readout-adaptation/NOTICE.md)에 있습니다.


## Case 013 — INT4 스케일 보존과 품질 검산

[4비트 변환·모델 연결 코드](../../cases/013-dualquant-reference-quality/source/paper_int4.py) · [4비트 값을 묶어 저장·복원하는 코드](../../cases/013-dualquant-reference-quality/source/reference_int4.py) · [저장한 개별 수치 검산](../../cases/013-dualquant-reference-quality/analysis/audit.py)

<!-- case014-entry -->
## Case014 — 자체 경로 보정과 전층 품질

앞선 층을 압축하면서 달라진 중간 계산값을 다음 층에서 수집하고, 64비트 계산(FP64)으로 오차 보정값을 맞춘 뒤 INT4 저장·독립 재로딩을 구현했습니다. 오차 원인을 세 항으로 나누고 같은 문서 구간의 전후 결과를 검산합니다. PG19의 텍스트 품질 이득과 ArXiv에서 확인된 한계를 함께 남겼습니다.

[구현 설명](../../cases/014-quantization-compensation/PORTFOLIO.md) · [코드](../../cases/014-quantization-compensation/followups/p2_core/source/run_p2.py) · [테스트](../../cases/014-quantization-compensation/followups/p2_core/tests/test_p2.py) · [모델 없는 CPU 검산](../../cases/014-quantization-compensation/REPRODUCE.md)

<!-- case015-entry -->
<a id="case015-cpu"></a>
## API 이름 변경 회귀시험

도구 이름만 바뀌어도 AI가 같은 일을 끝내는지 확인합니다. 예제 작업은 **구매한 개수와 개당 가격을 읽고→총금액을 계산하고→저장한 뒤→맞게 기록됐는지 확인**하는 일입니다. 기능과 입력 형식은 유지하고 `read_order`를 `readOrder`처럼 이름 표기만 바꿉니다. 16개 문제에서 BF16은 **15/16→16/16**, RTN은 **16/16→16/16**, D16은 **12/16→12/16**이었습니다. [작업·용어 안내](../../cases/015-tool-interface-sensitivity/README.ko.md)에서 주문(구매 내역)·단가(개당 가격)와 결과표 읽는 법을 설명합니다. 기존 시험 P1과 새 응용시험 A1의 각 96회는 별도로 집계합니다.