# 도구, 구현, 기술 협업

[English](../en/PORTFOLIO.md) | 한국어 · [홈](../../README.ko.md)

[처음 보는 분](START_HERE.md) · [쉬운 용어집](GLOSSARY.md)

[모델별 실전 안내서](LLM_GUIDE.md)에서 Qwen3-4B의 검증된 실행 경로와 품질·실패·완화 기록을 연결합니다.

## 30초 소개

<!-- claims: c008-tracks c008-artifact-implementation c007-implementation -->
DIOVA는 공개 모델을 제한된 GPU 환경에 맞게 변경하고 실행하며, 효율과 답변·실제 작업 결과를 함께 검증하는 추론 엔지니어링 프로젝트입니다. 모델을 작게 저장하는 데서 끝내지 않고, 다른 프로세스에서도 실행되는지와 어떤 답을 잃거나 회복했는지 확인하는 도구를 만듭니다.

[대표작과 직접 실행하기](https://munsik-kim.github.io/inference-lab/ko/index.html#featured-stories) · [내 결과 비교 예제](../../packages/diova-compare/examples/compare/README.ko.md)

## 대표작 A · 모델을 작게 저장하고 실제 요청까지 실행하기

Case008 Q의 **GPTQ 변환 → 저장본 검사 → 새 프로세스 재로딩**을 Case009의 **실제 요청 비용·품질 비교**로 연결합니다. [변환 코드](../../tools/modelpack/quantized.py) · [저장본 검사](../../tools/modelpack/qartifact.py) · [실제 요청 실행기](../../cases/009-q-serving-quality/scripts/serving.py).

가중치 파일 **8.045 GB → 2.652 GB, 약 67.0% 감소**는 Case008 Q의 Qwen3-4B 파일 크기입니다. Case009의 속도·메모리는 별도 실행 조건으로 읽습니다. [Case008 Q 원페이지](https://munsik-kim.github.io/inference-lab/ko/case008.html#track-q) → [Case009 원페이지](https://munsik-kim.github.io/inference-lab/ko/case009.html#serving). 0.6B의 pruning 복원인 Case008 R은 이 4B 양자화 흐름과 별개입니다.

**기술적 판단 — 현재 설계 해석:** 파일 크기만 확인하는 대안으로는 로딩 누락·실행 경로 문제를 알 수 없어 새 프로세스 실행까지 검사합니다. graph와 eager는 같은 모드 안에서 BF16/W4를 비교해 실행 준비 비용과 가중치 변경을 혼동하지 않게 합니다. [저장 경계 테스트](../../cases/008-build-reconstruct-reload/tests/test_boundaries.py)와 [요청 프로토콜](../../cases/009-q-serving-quality/configs/serving_protocol.json)이 근거입니다. 다른 장치의 이득과 Case009 품질 실행의 비정상 종료 문제는 해결된 것으로 보지 않습니다. 이는 현재 코드·기록의 해석이며 당시 의도를 새로 단정하지 않습니다.

## 대표작 B · 같은 질문의 답변을 비교하는 diova-compare

서로 다른 입력·정답 정의를 잘못 묶지 않도록 검사하고, 정답 손실·새 정답·다른 오답을 나눕니다. 원 실행 종료 코드가 있으면 수치와 함께 보존합니다. [작은 합성 예제로 실행](../../packages/diova-compare/examples/compare/README.ko.md) · [schema·CLI](../../packages/diova-compare/README.ko.md) · [집계 코드](../../packages/diova-compare/src/diova_compare/core.py) · [테스트](../../packages/diova-compare/tests/test_compare.py).

**기술적 판단 — 현재 설계 해석:** 평균 정답률만 보는 대안 대신 입력 쌍의 손실·회복을 따로 봅니다. 네 쌍의 예제는 양쪽 2/4여도 손실과 회복이 각각 하나임을 보여줍니다. 기본 비교에는 logits가 필요 없습니다. 자유 생성 정오와 입력·저장본 식별자는 사용자가 제공하므로 자동 채점이나 데이터 인증 범위로 넓히지 않습니다. 새 업무의 다음 기준은 채점 규칙과 입력 대응이 타당한지입니다.

## 대표작 C · 답변이 아니라 실제 도구 작업 완료 확인하기

성공했다는 최종 JSON 뒤에 실제 수정·검증이 없던 문제를 확인하고, 등록 이름 제한과 오류 피드백을 시험했습니다. [guard](../../research/agent_tool_contract_repair_20261014/guard.py) · [이벤트 분석](../../research/agent_tool_contract_repair_20261014/analyze.py) · [기능 테스트](../../research/agent_tool_contract_repair_20261014/test_guard.py) · [전체 보고서](../../research/agent_tool_contract_repair_20261014/records/REPORT_KO.md).

**기존 DEV 8과제 재사용 · 작은 표본 · BF16 복원 reference 계산:** D16은 6/8 → 8/8, D4는 0/8 → 2/8이며 D4의 6개 실패가 남았습니다. 새 입력 일반화와 실제 packed 저비트 가속은 검증하지 않았습니다.

**기술적 판단 — 현재 설계 해석:** 최종 JSON만 채점하는 대안은 실제 작업 누락을 놓칩니다. 이름 제한·피드백·결합 조건을 나누고 실행 이벤트와 변경 결과를 검사했습니다. 이름 제한과 결합 조건이 동률이어서 피드백의 추가 이득은 확인하지 못했습니다. 다음 판단에는 새 과제와 남은 D4 실패의 확인이 필요합니다.

## 연구·수치 엔지니어링 기록

[Case014](https://munsik-kim.github.io/inference-lab/ko/case014.html#overview)는 같은 INT4 projection bytes에서 품질 보정을 시험하고 데이터별 효과와 한계를 확인한 별도 연구입니다. PG19 구간의 RTN 대비 PPL 감소와 ArXiv에서 같은 이득이 확인되지 않은 결과를 나누어 읽습니다. 전체 checkpoint 크기·정답률·속도·VRAM 개선으로 바꾸지 않습니다.

Case006·007·008 R·010–012를 포함한 [전체 사례](CASEBOOK.md)와 아래 코드 투어에서 다른 구현·부정 결과도 계속 확인할 수 있습니다.

## 구조 복원과 평가 도구의 기여

![고정 코드 입력 하나를 비교하는 실제 Case007 로컬 화면](../../presentation/screenshots/case007-comparison.png)

*로컬 Edge 화면: 첫 고정 CODE 입력의 25% 삭제 비교입니다. [Case006 출력 계산 화면](../../presentation/screenshots/case006-readout.png)은 별도로 표시한 같은 입력 진단을 보여줍니다. 탐색을 안내하는 선택 화면이며 전체 성능 집계가 아닙니다.*

R은 작은 down projection의 ridge 적합과 구조별 저장·재로딩을 연결합니다. 층별 metadata → meta skeleton → strict 할당 → 공유 가중치·buffer 복원 → 원본 없이 새 프로세스 실행 순서입니다. 지원 범위는 Qwen3의 한 층 MLP입니다. 국소 SSE와 정답 점수는 아래 성능 평가 및 원래 보고서에 남습니다.

[기여 표](../../README.ko.md) · [선행연구와 구현 대응](../related-work/README.ko.md) · [CPU paired 비교 CLI](../../packages/diova-compare/README.ko.md) · [코드 설명 노트](../interview-notes.ko.md)

Case009는 기존 Qwen 4B 저장본의 같은 조건 graph/eager 요청 측정과 공식 품질 과제 기록을 추가합니다. [전체 곡선과 품질 보고서](../../cases/009-q-serving-quality/REPORT.ko.md)는 프로세스 종료 실패도 계산값과 함께 남깁니다. 새 CLI는 이 스칼라와 과거 기록을 같은 버전 계약으로 읽습니다.

## 구현 역량과 실제 근거

- **실행 경로 진단:** SM120에서 upstream의 커널 선택 guard를 수정 전후 실행기로 확인했습니다. [Case001 실행기](../../cases/001-sm120-int8-fallback/run.py).
- **비용 측정:** smoothing, 양자화, 변환을 포함한 호출 시간을 attention 어댑터에서 측정했습니다. [Case004 어댑터](../../cases/004-low-precision-attention-break-even/src/backends.py).
- **수치 분석:** EFQ 매핑과 scale 선택을 나눠 비교하고, 고정된 정밀도 screen의 중단 판정을 기록했습니다. [Case003 분석·영어](../../cases/003-efq-softmax-numerical-audit/ANALYSIS.md) · [Case005 분석·영어](../../cases/005-attention-precision-pareto/ANALYSIS.md).
- **모델 구조 변경:** 대응하는 행·열을 선택해 작은 dense MLP를 만들고, 마스킹한 계산과 비교했습니다. [Case007 구조 변경 코드](../../cases/007-interaction-aware-mlp-pruning/src/surgery.py).
- **입력별 평가:** 별도로 계산한 정답(gold)을 기준으로 전환 유형, 점수 변화, 정확한 동률을 볼 수 있습니다. [Case006 분석·영어](../../cases/006-attention-decision-stability/ANALYSIS.md) · [출력 계산 스칼라 검사기](../../cases/006-attention-decision-stability/supplemental/readout-ties-v1/scripts/verify_scalar.py).
- **분석 도구 전달:** source hash로 화면의 기록을 원자료와 연결하고, 작은 선택기 예제로 보정 목적함수를 실행합니다. [화면 빌더](../../tools/showcase/build.py) · [CPU 예제](../../tools/showcase/replay_selection.py).

<a id="code-tour"></a>
## 구현 코드 읽기

**저장본 제작과 재로딩부터 읽습니다.** [`save_dense()`](https://github.com/Munsik-Kim/inference-lab/blob/9bc8b8fced8dfd5147f9bfdc67564eda04670810/tools/modelpack/artifact.py#L32)는 tensor·구조 목록을 저장합니다. [`load_dense()`](https://github.com/Munsik-Kim/inference-lab/blob/9bc8b8fced8dfd5147f9bfdc67564eda04670810/tools/modelpack/artifact.py#L109)는 meta에서 층별 폭을 바꾸고 가중치를 엄격하게 할당한 뒤 공유 가중치와 rotary buffer를 복원합니다. [`fit_ridge()`](https://github.com/Munsik-Kim/inference-lab/blob/9bc8b8fced8dfd5147f9bfdc67564eda04670810/tools/modelpack/numerics.py#L13)는 CPU FP64 보정 문제를 풉니다. [원래 저장 테스트](../../cases/008-build-reconstruct-reload/tests/test_core.py)는 누락 tensor와 잘못된 구조를 거절합니다. [독립 CPU 예제](../../tools/modelpack_demo/roundtrip.py)는 이 로더를 다른 프로세스에서 실행합니다.

<!-- claims: c006-implementation -->
1. **모델의 attention 호출을 제한합니다.** [`ScopedAttention`](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/006-attention-decision-stability/src/intervention.py#L30)부터 읽습니다. `route`는 13번 층의 정사각형 prefill 개입을 허용하고 `__exit__`는 이전 등록 함수를 복구합니다. [예외 복구 테스트](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/006-attention-decision-stability/tests/test_core.py#L193)에서 이 동작을 확인할 수 있습니다.
2. **행렬 크기를 줄입니다.** [`sliced`](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/007-interaction-aware-mlp-pruning/src/surgery.py#L6)는 남긴 그룹을 gate/up 행과 down 열에 대응시킵니다. `masked`가 비교 기준을 만들고 `replace_mlp`는 `finally`에서 원래 모듈을 복구합니다. [형상·bias·계산 대응 테스트](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/007-interaction-aware-mlp-pruning/tests/test_core.py#L62)가 이 연결을 검사합니다.
3. **불완전한 근거를 거절합니다.** [`assess`](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/006-attention-decision-stability/src/validity.py#L10)는 전체 출력과 실행 경로 검사 결과를 요구합니다. [원래 pairing 테스트](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/006-attention-decision-stability/tests/test_core.py#L249)는 누락·중복·서로 다른 입력의 결합을 거절합니다. 새 [화면 검사기](../../tools/showcase/check.py)는 표시용 데이터를 보존된 원자료와 대조합니다.

## 60–90초 시연 대본

*CPU 합성 예제를 실제로 실행하며 보여줄 동선입니다. 이 문서는 녹화 영상이 아닙니다.*

- **0–15초:** “모델 변경 전후의 정답률이 같아도 원래 맞히던 질문을 틀릴 수 있습니다.”
- **15–40초:** [예제](../../packages/diova-compare/examples/compare/README.ko.md)의 두 파일을 validate하고 compare합니다. 모델을 실행하는 시연이 아니라 저장된 합성 답변을 CPU에서 비교한다고 밝힙니다.
- **40–65초:** 생성된 report.md와 pairs.csv에서 sum-01의 정답 손실과 sum-02의 회복을 엽니다.
- **65–90초:** [집계 코드](../../packages/diova-compare/src/diova_compare/core.py)와 [전환·거절·실패 보존 테스트](../../packages/diova-compare/tests/test_compare.py)를 확인합니다.

과거 GPU 결과 조회는 [Case008 Q](https://munsik-kim.github.io/inference-lab/ko/case008.html#track-q) → [Case009](https://munsik-kim.github.io/inference-lab/ko/case009.html#quality)로 별도 보여줍니다. 작은 무작위 모델의 [CPU 저장·재로딩 예제](GETTING_STARTED.md#tiny-model-demo)도 별개 구현 경로입니다.

## 근거로 연결되는 소개 문장 세 개

<!-- claims: c004-cost c005-stop -->
- 그룹 선택과 행렬 축소 도구를 만들고, 마스킹 계산 및 미관측 입력과 비교했습니다. [Case007 방법·영어](../../cases/007-interaction-aware-mlp-pruning/METHODS.md).
- Qwen 한 층 개입과 답변 확률·정답 손실·정답 획득·동률의 쌍별 분석을 구현했습니다. [Case006 방법·영어](../../cases/006-attention-decision-stability/METHODS.md).
- Attention 전체 호출 비용을 측정하고, 관측된 정밀도·비용 절충과 **STOP_DEV_SCREEN** 판정을 함께 기록했습니다. [Case004 비용·영어](../../cases/004-low-precision-attention-break-even/README.md) · [Case005 판정·영어](../../cases/005-attention-precision-pareto/README.md).

## 도구의 성능 평가

<!-- claims: c007-transfer c007-quality c006-native c006-readout -->
Case008 R은 Qwen 0.6B의 한 층 MLP에서 짧은 합성 평가 입력 192개의 삭제로 생긴 국소 제곱 출력 오차를 **94.1–95.4%** 줄였습니다. 정답 점수 변화는 확률 지표와 보정 구조에 따라 달랐고, 측정한 요청 시간비 구간은 1을 포함했습니다. [Q 품질·속도 평가](https://munsik-kim.github.io/inference-lab/ko/case008.html#q-evaluation)와 [R 평가](https://munsik-kim.github.io/inference-lab/ko/case008.html#r-evaluation)에서 비교할 수 있습니다.

MLP 25% 삭제에서 PAIRWISE의 국소 오차는 소폭 낮았고 기준선 선택을 더 많이 보존했지만, 정답 NLL은 INDEPENDENT가 더 좋았습니다. 50%의 선택 모듈은 같았으며 고정 판정은 **COMPLETED_NO_CLEAR_TRANSFER**입니다. Case006에서는 표준 192개 중 A_PUBLIC은 5개, V4는 8개의 선택이 달라졌습니다. 같은 입력의 사후 출력 계산 진단은 동률과 마지막 어휘 점수 계산을 살펴봅니다. 구현상 선택과 서로 다른 평가 목표를 이 예제들에서 확인할 수 있습니다. [사례 안내](CASEBOOK.md) · [로컬 화면 열기](GETTING_STARTED.md#local-showcase).

<a id="contribution-and-reuse"></a>
## 프로젝트 기여와 재사용

<!-- claims: attribution -->
프로젝트의 작업은 비교 실행, 범위를 제한한 어댑터, 행렬 축소, 유효성 검사, 수치·답변 분석과 결과 탐색기입니다. Qwen과 Red Hat AI가 체크포인트를, Transformers·PyTorch·vLLM이 모델과 실행 구성요소를, SageAttention이 저정밀 커널을 제공합니다. Case001 guard의 저자는 hclsys이며 EFQ-Softmax는 Han과 공동저자의 작업입니다. HOPE는 별도 MoE expert-pruning 맥락에서 상호작용을 고려한 삭제의 동기를 제공합니다. 여기의 dense 그룹 목적함수는 선택한 MLP 분해에서 직접 구했습니다.

GPTQ는 기존 양자화 방법이며 LLM Compressor·compressed-tensors는 변환·압축 저장을, safetensors는 가중치 형식을 제공합니다. Case008에서는 저장본 검사, 층별 구조 로더, 고정 구조 ridge 적합을 연결했습니다. [Case008 출처·영어](../../cases/008-build-reconstruct-reload/NOTICE.md).

OpenAI Codex가 구현·로컬 실행·테스트·분석·문서 작성을 지원했습니다. [Case001](../../cases/001-sm120-int8-fallback/NOTICE.md), [Case003](../../cases/003-efq-softmax-numerical-audit/NOTICE.md), [Case006](../../cases/006-attention-decision-stability/NOTICE.md), [Case007 출처·영어](../../cases/007-interaction-aware-mlp-pruning/NOTICE.md)에 재사용한 작업을 기록했습니다. 프로젝트 코드는 [Apache-2.0](../../LICENSE)이며 모델·upstream 이용 조건은 별도입니다.

## 기술 협업 범위

현재 공개 구현이 보여주는 역량을 요청 단위로 정리하면 다음과 같습니다.

- 로컬 모델의 양자화·실행 설정 비교와 병목 진단.
- 모델 변경 전후의 답변·도구 실행 회귀 평가.
- 측정·기록·검산 과정을 재사용 가능한 도구로 정리.


**모델 변경 전후 평가**, **로컬 추론 문제 진단**, **재현 가능한 분석 도구와 결과 화면**을 주제로 코드를 검토할 수 있습니다. 비교할 모델 경계, 공개 입력 ID, 평가 지표에서 논의를 시작할 수 있습니다. [저장소 이슈](https://github.com/Munsik-Kim/inference-lab/issues)는 공개 기술 질문 경로이며, 인증값이나 비공개 고객 자료는 공개 보고에 넣지 않습니다.

근거는 특정 모델·장치 하나에서 수행한 합성 실험과 Case009의 별도 공식 과제 결과입니다. 배포 판단은 **NOT_ASSESSED**입니다. CPU 감사는 남긴 스칼라를 검산하며, 제외된 전체 벡터와 별도 GPU 재현의 확인 범위는 [재현 안내](GETTING_STARTED.md)에서 설명합니다.


## Case010 — state 저장과 실패 보존

반복 state의 packing, residual transport, 행 단위 갱신과 실패 상태 직렬화를 연결했습니다. [v2 구현](../../cases/010-ckda-finite-precision-memory-horizon/versions/v2/source/online_v2.py), [새 프로세스 합성 테스트](../../cases/010-ckda-finite-precision-memory-horizon/versions/v2/tests/test_online_v2.py), [저장 예산과 판독 길이 비교](../../cases/010-ckda-finite-precision-memory-horizon/REPORT.ko.md) 순서로 확인할 수 있습니다. [CPU 검산 안내](GETTING_STARTED.md#case010-cpu). ComplexKDA의 전이·학습 모델과 프로젝트의 codec·평가 도구는 [출처 안내](../../cases/010-ckda-finite-precision-memory-horizon/NOTICE.md)에 구분했습니다.

## Case011 — 같은 상태에서 판독층만 교체하기

같은 cache를 세 판독기가 읽게 하고, 마지막 두 tensor의 저장·shape·원본 hash를 검사해 새 프로세스에서 다시 적용합니다. [특징 경계](../../cases/011-frozen-state-readout-adaptation/source/adapter.py) → [patch loader](../../cases/011-frozen-state-readout-adaptation/source/head_patch.py) → [계약 테스트](../../cases/011-frozen-state-readout-adaptation/tests/test_fitting_patch.py) → [독립 검산기](../../cases/011-frozen-state-readout-adaptation/analysis/audit.py) 순서로 읽을 수 있습니다.

추가 recurrent-state 저장은 0 B입니다. fitting 범위 밖의 정답 CE는 개선됐지만 INT8의 평균 연속 정답 길이는 세 checkpoint 모두 감소했습니다. [쉬운 화면](https://munsik-kim.github.io/inference-lab/ko/case011.html) · [정확한 결과와 solver 상태](../../cases/011-frozen-state-readout-adaptation/REPORT.ko.md) · [CPU 실행](GETTING_STARTED.md#case011-cpu). PyTorch/NumPy와 ComplexKDA의 역할은 [기여·출처](../../cases/011-frozen-state-readout-adaptation/NOTICE.md)에 있습니다.


## Case 013 — INT4 스케일 보존과 품질 검산

[INT4 변환·adapter](../../cases/013-dualquant-reference-quality/source/paper_int4.py) · [Packed 저장](../../cases/013-dualquant-reference-quality/source/reference_int4.py) · [저장 scalar 검산](../../cases/013-dualquant-reference-quality/analysis/audit.py)

<!-- case014-entry -->
## Case014 — 자체 경로 보정과 전층 품질

Capture·FP64 full-input 적합·INT4 codec/독립 재로딩·세 항 분해·문서 paired auditor를 구현했습니다. PG19 품질 이득과 ArXiv 전이 한계를 함께 제공합니다.

[Implementation](../../cases/014-quantization-compensation/PORTFOLIO.md) · [Code](../../cases/014-quantization-compensation/followups/p2_core/source/run_p2.py) · [Tests](../../cases/014-quantization-compensation/followups/p2_core/tests/test_p2.py) · [CPU reproduction](../../cases/014-quantization-compensation/REPRODUCE.md)

<!-- case015-entry -->
<a id="case015-cpu"></a>
## API 이름 변경 회귀시험

[Case015](CASEBOOK.md#case-015)은 같은 주문 업무의 이름 변경을 실행·관측 이력·최종 상태로 비교합니다. BF16 15/16→16/16, RTN16/16→16/16, D1612/16→12/16. [Report](../../cases/015-tool-interface-sensitivity/REPORT.ko.md) · [CPU commands](../../cases/015-tool-interface-sensitivity/REPRODUCE.ko.md) · [Code](../../cases/015-tool-interface-sensitivity/followups/order-api-regression-v1/src/order_regression/runner.py)
