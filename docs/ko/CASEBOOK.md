# 로컬 추론을 조사하는 일곱 질문

[English](../en/CASEBOOK.md) | 한국어 · [홈](../../README.ko.md)

실행 진단, 비용 측정, 수치 검산, 모델 변경 평가에 필요한 도구를 사례별로 구현했습니다. 행렬 축소는 [Case007](#case-007), 답변 비교는 [Case006](#case-006)부터 읽고 앞선 조사로 이어갈 수 있습니다. 각 사례는 자체 모델·입력·실험 조건을 갖습니다.

<a id="case-001"></a>
## 001 — 커널 선택 조건을 고치면 실행할 수 있는가?

<!-- claims: c001-runtime -->
사용 가능한 대체 구현이 있어도 먼저 선택한 구현이 실패하면 모델을 실행할 수 없습니다. RTX 5080 / SM120에서 원래 vLLM 0.29.0의 W8A8 선택기는 INT8을 지원하지 않는 CUTLASS 경로를 허용했습니다.

hclsys의 PR에서 Python runtime guard 부분만 공식 wheel에 적용하고, 같은 RedHatAI Qwen2.5-0.5B W8A8 체크포인트를 실행했습니다. 수정 전 초기화는 exit 1로 실패했고, 수정 후에는 exit 0으로 “The capital of France is Paris.”를 생성했습니다. 관찰한 INT8 모듈 96개는 `TritonInt8ScaledMMLinearKernel`을 선택했고 가중치는 CUDA에 있었습니다. 이 수는 로드된 모듈 수이며 GPU 커널을 동적으로 96회 호출했다는 뜻이 아닙니다.

프로젝트는 수정 전후 실행 도구, 환경 잠금 자료와 실행 증거를 정리했습니다. guard와 Triton 구현 자체는 upstream의 작업입니다. 재실행은 같은 PC의 기존 환경과 체크포인트를 사용했으며 다른 사람의 신규 설치가 아닙니다. 짧은 생성 하나로 해당 호환성 경로를 확인한 것이고, 처리량·일반 과제 품질·PR 전체 빌드를 검증한 것은 아닙니다.

[결과와 명령 — 기술 원문(영어)](../../cases/001-sm120-int8-fallback/README.md) · [원실행 JSON 근거](../../cases/001-sm120-int8-fallback/evidence/original.json) · [기록된 upstream 검증 댓글 — 영어](https://github.com/vllm-project/vllm/pull/54316#issuecomment-5614091211)

<a id="case-002"></a>
## 002 — 공식 FP8 체크포인트로 바꾸면 무엇이 달라지는가?

<!-- claims: c002-quality c002-cost -->
배포 파일이 작아도 자원 사용과 과제 성능을 함께 봐야 실제 선택을 판단할 수 있습니다. 공식 BF16·FP8 Qwen3-4B-Instruct-2507 배포본을 한국어 합성 문서 추출 과제에서 비교했습니다.

추론 전에 구조화된 사실과 확정 변경으로 정답을 계산했습니다. 담당자·작업·날짜·금액 네 필드가 모두 맞아야 문서 정답으로 셌습니다. 첫 품질 평가에서 BF16은 31/100, FP8은 32/100이었습니다. FP8−BF16 차이는 +1 퍼센트포인트(%p), 95% 쌍별 구간은 −2~+5%p였습니다. 개선이나 동등성을 확정하는 결과는 아닙니다.

같은 1 GiB BF16 KV cache 예산에서 장치 전체의 관측 최대 메모리는 약 11.62 GiB와 8.21 GiB였습니다. FP8 요청은 대체로 먼저 끝났지만 출력 길이가 달라 순수 decode 속도 비교는 아닙니다. 장치 전체 수치에는 배경 할당도 포함됩니다.

사실 기반 데이터, 엄격한 채점, streaming client와 쌍별 측정·분석을 구현했습니다. 세 라운드는 같은 100개 문서를 반복 측정했으므로 독립 품질 표본 300개가 아닙니다. JSON 형식은 맞아도 두 모델의 추출 정답률은 낮았습니다. 비교 범위는 해당 공식 체크포인트와 실행 구성입니다.

[결과 — 기술 원문(영어)](../../cases/002-bf16-fp8-document-extraction/README.md) · [집계 JSON 근거](../../cases/002-bf16-fp8-document-extraction/results/aggregate.json) · [방법과 재현 — 영어](../../cases/002-bf16-fp8-document-extraction/METHODS.md)

<a id="case-003"></a>
## 003 — 수치 비교의 기준이 달라지면 결론은 어떻게 달라지는가?

<!-- claims: c003-reference -->
같은 scale을 쓰는 기준 구현과 비교하면 오차 차이가 작아 보여도, 다른 scale 규칙을 쓰는 기준 구현이 더 낮은 오차를 낼 수 있습니다. 실제 Qwen3-0.6B attention trace에서 코드 매핑의 효과와 비교 기준의 scale 선택을 나누어 검사했습니다.

논문 식을 독립적으로 구현하고 Q/K/V·mask·FP32 계산 조건을 공통으로 유지했습니다. 세 길이에서 EFQ-Mean의 상대 출력 오차 중앙값은 11.49~14.49%였습니다. 개발 자료가 선택한 설정은 공개 EFQ-MMLU와 같았고 같은 EFQ scale의 nearest rounding과 집계 수준에서 비슷했습니다. 하지만 headroom scale의 nearest는 모든 길이에서 중앙값과 p95가 더 낮았습니다. 길이 512·2048·4096에서 보정 EFQ의 중앙값 오차는 이 headroom 기준의 각각 1.133×·1.179×·1.148×였습니다.

식 구현·검사, trace 수집 검증, 증거의 독립 재계산이 프로젝트의 작업입니다. EFQ 방법은 Han과 공동저자의 기여입니다. 길이별 192개 head 단위는 16문서×3층×4heads에서 나온 의존 관측입니다. 원래 screen의 분모는 같은 scale의 nearest이며 더 강한 headroom 기준이 아닙니다. FP32 수치 모사로 packed-FP4 하드웨어 속도나 downstream 모델 품질을 측정하지 않았습니다.

[결과 — 기술 원문(영어)](../../cases/003-efq-softmax-numerical-audit/README.md) · [scale·구현 정의 — 영어](../../cases/003-efq-softmax-numerical-audit/METHODS.md) · [독립 분해 계산 JSON](../../cases/003-efq-softmax-numerical-audit/results/comparison_decomposition.json)

<a id="case-004"></a>
## 004 — attention 연산 전체를 호출해도 빠른가?

<!-- claims: c004-cost -->
커널만 재면 저정밀 입력을 준비하는 비용이 빠질 수 있습니다. GPU에 있는 BF16 Q/K/V에서 시작해 BF16 출력을 얻는 호출 전체를 비교했습니다. smoothing·양자화·변환·wrapper가 포함됩니다.

공식 SageAttention INT8-QK/FP8-PV와 개발 자료에서 선택하고 fused 실행을 확인한 BF16 SDPA 경로를 비교했습니다. 실제 Qwen3-0.6B layer 13 입력에서 길이 512·2048·4096의 전체 연산 속도비는 0.373×·1.465×·2.100×였습니다. 1보다 작으면 느리다는 뜻입니다. 세 길이 모두 고정 국소 오차 기준인 중앙값 ≤1%, p95 ≤3%를 넘었습니다. 긴 호출이 빨라져도 비용과 오차를 함께 보는 선택 조건은 통과하지 못했습니다.

어댑터, 실행 경로 검사, 쌍별 전체 비용 측정과 표본 query 오차 분석을 구현했습니다. 호출은 모든 query를 처리하지만 오차는 head별 32개 위치와 그 위치의 모든 causal-valid key에서 계산합니다. 16개 문서의 head와 길이는 의존 관측입니다. 별도 합성 H8/H8 조건은 Qwen H16/H8 조건과 다릅니다. 수치는 모델 전체 시간이 아닌 연산 시간입니다. 커널은 SageAttention의 작업이며 오차 screen은 이 프로젝트의 공학적 기준으로 과제 정답률의 보증이 아닙니다.

[설계와 결과 — 기술 원문(영어)](../../cases/004-low-precision-attention-break-even/README.md) · [비용·오차 해석 — 영어](../../cases/004-low-precision-attention-break-even/ANALYSIS.md) · [기록된 선택표 JSON](../../cases/004-low-precision-attention-break-even/results/selection_table.json)

<a id="case-005"></a>
## 005 — 설정을 바꾸면 오차와 비용의 절충이 나아지는가?

<!-- claims: c005-stop c005-tradeoff c005-validity -->
Case004 다음에는 제한된 정밀도 설정만 바꾸어 국소 오차를 낮추면서 전체 호출 속도를 유지할 수 있는지 물었습니다. 개발 비교는 합성 문서 8개, Qwen3-0.6B layer 13, causal L4096으로 한정했습니다.

변경 설정 세 개가 실제 입력 비교를 마쳤고 V3는 작은 유효성 검사에서 제외됐습니다. 중앙값 오차 ≤1%, p95 ≤3%, 속도비 ≥1.50×를 모두 충족한 변경 설정은 없었습니다. 결과는 **STOP_DEV_SCREEN**이며 최종 후보와 새로운 확인 실험이 없습니다.

V4는 A_PUBLIC의 중앙값 오차 2.294%를 1.958%로 낮췄지만, 전체 호출 속도비도 2.110×에서 1.338×로 낮아졌습니다. 관측된 충실도·비용 절충점이지만 원래 합격 영역 밖입니다. V1/V2는 속도를 유지했으나 집계 오차 감소가 작았습니다. 누산과 V 범위 정책이 함께 바뀌어 각각의 원인을 분리한 비교는 아닙니다. V3의 smoothing 경로는 버전을 고정한 구현의 작은 fixture에서 중심화된 범위가 0이 되어 NaN/Inf가 포함된 유효하지 않은 출력을 냈습니다.

실제 dispatch 감사, 명시적 어댑터와 측정, 중단·실패 이력 보존이 프로젝트의 작업입니다. 뒤의 임계값 민감도와 쌍별 단위 분석은 사후 해석이며 새 통과 결과가 아닙니다. 국소 오차는 과제 정답률이 아니고, 이 유한 후보 목록으로 저정밀 attention 전체를 부정할 수 없습니다.

[개발 결과 — 기술 원문(영어)](../../cases/005-attention-precision-pareto/README.md) · [판정·측정 JSON](../../cases/005-attention-precision-pareto/results/dev_summary.json) · [별도 사후 검토 — 영어](../../cases/005-attention-precision-pareto/POSTHOC_THRESHOLD_REVIEW.md)

<a id="case-006"></a>
## 006 — 점수와 개별 답변 선택은 언제 달라지는가?

<!-- claims: c006-native c006-readout c006-limits c006-timing c006-implementation -->
전체 정답률이 비슷해도 정답 획득·손실·다른 오답으로의 변경은 가려질 수 있습니다. 후보를 BF16뿐 아니라 별도로 계산한 정답(gold)과 함께 비교했습니다. 회귀는 기준선이 맞힌 답을 후보가 틀린 경우입니다. B는 원래 BF16 실행인 비교 기준선입니다. A_PUBLIC과 V4는 같은 모델의 attention 연산 한 곳에 적용한 두 저정밀 설정의 이름이며, 모델 가중치는 BF16으로 유지합니다. A_PUBLIC은 INT8 QK / FP8 PV, V4는 INT8 QK / FP16 PV를 사용합니다.

**한 일과 원래 출력 계산(native)의 결과.** BF16 Qwen3-0.6B의 0부터 센 layer 13에서 정사각 causal 프롬프트 prefill attention만 교체했습니다. 다른 층과 decode는 BF16입니다. 동일 토큰 입력, 전체 출력 유효성 검사, 고정된 한 토큰 선택지 인터페이스로 비교 조건을 통제했습니다. 표준 192개에서 B/A_PUBLIC/V4의 정답 수는 104/106/108이었습니다. A_PUBLIC은 192개 중 5개가 바뀌어 정답 획득 2개·다른 오답 3개였고, V4는 192개 중 8개가 바뀌어 정답 획득 4개·다른 오답 4개였습니다. 이 집합에서는 정답 손실이 없었지만 BF16 점수로 별도 선정한 스트레스 46개에서는 A_PUBLIC 2개, V4 1개가 정답을 잃었습니다. 두 설정 모두 과제별 가중치를 같게 둔 정답 선택지의 음의 로그우도(NLL) 쌍별 변화 평균에 대해 95% 구간을 구했으며, 그 구간은 0을 포함했습니다. 변화는 후보−BF16이며, 양수는 정답에 부여한 확률이 낮아지는 방향입니다. 이 결과만으로 품질의 동등성을 입증하지는 않습니다.

Readout은 모델 내부 상태를 어휘 점수로 바꾸는 마지막 출력 계산입니다. 원래 출력 계산(native)과 별도 진단용 출력 계산(shadow)을 구분합니다. native 표준셋의 모든 선택 변경에는 B 또는 후보의 정확한 최고점 동률이 있었습니다. 같은 hidden state와 BF16에 표현된 가중치를 FP32 출력 head로 계산한 사후 진단에서는 후보마다 192개 중 3개가 바뀌었고 새로운 변경도 생겼습니다. 각 readout 안에서 후보를 B와 비교했고 같은 192 + 46 시나리오를 재사용했습니다. 동률 처리는 native 동작의 일부이며, 계산 민감도 검사가 본실험을 대체하거나 더 좋은 모델을 입증하지 않습니다.

모델 내부 개입의 통제, 쌍별 채점, 동률 분석과 오프라인 항목 탐색을 연결했습니다. BF16 코드 정답은 15/64, 엄격한 구조화 생성 유효율은 모든 설정에서 0/24였습니다. 모델 prefill 전체 속도비는 1.0041×·1.0014×로 Case004의 연산 속도비와 다릅니다. **COMPLETED_CONTROLLED_STUDY**는 근거 수집 완료를, **NOT_ASSESSED**는 배포 판단을 하지 않았음을 뜻합니다. 동등성이나 일반적인 품질 보존을 주장하지 않습니다.

[본실험과 소스 안내 — 기술 원문(영어)](../../cases/006-attention-decision-stability/README.md) · [별도 readout 진단 — 기존 한국어 설명](../../cases/006-attention-decision-stability/supplemental/readout-ties-v1/README.ko.md) · [저장된 탐색기 열기](GETTING_STARTED.md#offline-explorers)

<a id="case-007"></a>
## 007 — 함께 지울 그룹을 고려하면 더 나은 작은 MLP를 고를 수 있는가?

<!-- claims: c007-transfer c007-quality c007-random -->
그룹을 함께 지울 때의 오차는 개별 중요도를 더한 값과 다를 수 있습니다. Case007은 Qwen3-0.6B layer 13의 SwiGLU 순방향 연산 블록을 연속 채널 그룹 16개로 나눴습니다. INDEPENDENT는 개별 중요도를, PAIRWISE는 부호가 있는 그룹 간 교차항까지 사용합니다. 보정 입력 96개로 선택하고 개발 입력 48개로 검사한 뒤, gate/up 행과 down 열을 실제로 잘라 미관측 입력 192개에서 평가했습니다. 삭제 예산은 4/16(25%)과 8/16(50%)이며, 512토큰 입력마다 고정 위치 32개의 국소 오차를 계산했습니다.

**관측한 결과.** 25%에서 평균 국소 상대오차는 29.8291%에서 29.6288%로 줄었습니다. 쌍별 차이는 −0.200277퍼센트포인트, 95% 구간은 [−0.284779, −0.114472]이며, 141/192개에서 오차가 낮았습니다. 50%에서는 삭제 집합·가중치·출력이 같았습니다. 고정된 두 예산 규칙에 따른 종합 판정은 **COMPLETED_NO_CLEAR_TRANSFER**입니다. 두 선택법은 예산별로 고정한 무작위 집합 20개 각각보다 held-out 평균 국소 오차가 낮았습니다. 이는 국소 reference의 기술통계이며 전체 최적성이나 무작위 삭제 모델의 품질을 입증하지 않습니다.

25%에서 PAIRWISE는 삭제하지 않은 BF16 기준선 B의 선택을 더 보존했습니다. 선택 변경은 12/192개로 INDEPENDENT의 48/192개보다 적고, 전체 어휘 KL(B ∥ 후보)도 낮았습니다. 그러나 별도로 계산한 정답의 확률 손실인 정답 선택지 NLL은 INDEPENDENT가 더 좋았습니다. 별도 사후 비교의 PAIRWISE−INDEPENDENT NLL 차이는 +0.069860 nats [0.025925, 0.111997]이며 양수는 더 나쁜 방향입니다. 기준선 충실도와 정답 품질은 서로 다른 목표입니다.

그룹 기여 계산, 전체 조합 선택, 실제 행렬 축소와 미관측 입력 검증을 연결했습니다. 국소 MLP 속도비는 1.176–1.412×였지만 모델 전체 prefill 속도비 구간은 모두 1을 포함했습니다. 모델·층 하나, 영어 합성 과제, 재학습 없는 범위이며 배포 적합성은 **NOT_ASSESSED**입니다. 부호 있는 교차항 항등식은 새 알고리즘이나 HOPE의 MoE 연구를 직접 재현한 것이 아닙니다.

[검토된 한국어 결과와 출처](../../cases/007-interaction-aware-mlp-pruning/README.ko.md) · [방법과 동결 범위 — 기술 원문(영어)](../../cases/007-interaction-aware-mlp-pruning/METHODS.md) · [패키지와 탐색기](GETTING_STARTED.md#case007-explorer)

## 여기서 사용하는 용어

- **추론(inference):** 이미 학습된 모델을 실행해 점수나 출력을 얻는 과정입니다.
- **양자화·BF16/FP8/INT8:** 제한된 정밀도의 수 표현을 사용합니다. 가중치와 attention의 각 피연산자가 같은 형식일 필요는 없습니다.
- **Attention:** query와 key로 구한 점수를 이용해 value 벡터를 결합하는 연산입니다.
- **Prefill / decode:** 관측한 프롬프트를 처리하는 단계 / cache를 사용해 이후 토큰을 생성하는 단계입니다.
- **전체 연산 비용:** 필요한 준비와 출력 변환까지 포함한 측정 대상 어댑터의 호출 비용입니다.
- **국소 출력 오차:** 한 연산 경계의 수치 차이이며 과제 정답률 손실과 다릅니다.
- **Gold:** BF16 모델 예측이 아닌, 별도로 확정한 정답입니다.
- **쌍별 비교:** 같은 입력을 설정이나 반복 간에 대응시킨 뒤 차이를 계산합니다.
- **최고점 동률:** 기록된 계산에서 허용 선택지의 최대 점수가 정확히 같은 경우입니다.
- **사후 진단:** 결과를 본 뒤 동기가 생긴 분석으로, 원래 고정 평가와 구분합니다.
- **Readout:** 최종 hidden state를 어휘 점수로 바꾸는 출력 projection입니다.
