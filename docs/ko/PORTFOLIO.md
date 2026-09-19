# 도구, 구현, 기술 협업

[English](../en/PORTFOLIO.md) | 한국어 · [홈](../../README.ko.md)

## 30초 소개

<!-- claims: c007-transfer c007-quality c006-native c006-readout -->
Inference Lab은 모델 변경을 직접 확인할 수 있는 측정 결과와 연결합니다. 압축 도구는 채널 그룹을 고르고 Qwen의 MLP 행렬을 줄인 뒤, 선택에 쓰지 않은 입력에서 재구성 오차·답변·실행 시간을 비교합니다. Attention 연구는 한 층만 바꾸는 개입으로 답변 점수와 선택을 추적합니다. 한·영 결과 화면에서 개별 질문과 비교값을 함께 보고, CPU 예제로 공개 보정 통계에서 원래 선택기를 다시 실행할 수 있습니다.

MLP 25% 삭제에서 PAIRWISE의 국소 오차는 소폭 낮았고 기준선 선택을 더 많이 보존했지만, 정답 NLL은 INDEPENDENT가 더 좋았습니다. 50%의 선택 모듈은 같았으며 고정 판정은 **COMPLETED_NO_CLEAR_TRANSFER**입니다. Case006에서는 표준 192개 중 A_PUBLIC은 5개, V4는 8개의 선택이 달라졌습니다. 같은 입력의 사후 출력 계산 진단은 동률과 마지막 어휘 점수 계산을 살펴봅니다. 구현상 선택과 서로 다른 평가 목표를 이 예제들에서 확인할 수 있습니다. [사례 안내](CASEBOOK.md) · [로컬 화면 열기](GETTING_STARTED.md#local-showcase).

![고정 코드 입력 하나를 비교하는 실제 Case007 로컬 화면](../../presentation/screenshots/case007-comparison.png)

*로컬 Edge 화면: 첫 고정 CODE 입력의 25% 삭제 비교입니다. [Case006 출력 계산 화면](../../presentation/screenshots/case006-readout.png)은 별도로 표시한 같은 입력 진단을 보여줍니다. 탐색을 안내하는 선택 화면이며 전체 성능 집계가 아닙니다.*

## 구현 역량과 실제 근거

- **실행 경로 진단:** SM120에서 upstream의 커널 선택 guard를 수정 전후 실행기로 확인했습니다. [Case001 실행기](../../cases/001-sm120-int8-fallback/run.py).
- **비용 측정:** smoothing, 양자화, 변환을 포함한 호출 시간을 attention 어댑터에서 측정했습니다. [Case004 어댑터](../../cases/004-low-precision-attention-break-even/src/backends.py).
- **수치 분석:** EFQ 매핑과 scale 선택을 나눠 비교하고, 고정된 정밀도 screen의 중단 판정을 기록했습니다. [Case003 분석·영어](../../cases/003-efq-softmax-numerical-audit/ANALYSIS.md) · [Case005 분석·영어](../../cases/005-attention-precision-pareto/ANALYSIS.md).
- **모델 구조 변경:** 대응하는 행·열을 선택해 작은 dense MLP를 만들고, 마스킹한 계산과 비교했습니다. [Case007 구조 변경 코드](../../cases/007-interaction-aware-mlp-pruning/src/surgery.py).
- **입력별 평가:** 별도로 계산한 정답(gold)을 기준으로 전환 유형, 점수 변화, 정확한 동률을 볼 수 있습니다. [Case006 분석·영어](../../cases/006-attention-decision-stability/ANALYSIS.md) · [출력 계산 스칼라 검사기](../../cases/006-attention-decision-stability/supplemental/readout-ties-v1/scripts/verify_scalar.py).
- **분석 도구 전달:** source hash로 화면의 기록을 원자료와 연결하고, 작은 선택기 예제로 보정 목적함수를 실행합니다. [화면 빌더](../../tools/showcase/build.py) · [CPU 예제](../../tools/showcase/replay_selection.py).

<a id="code-tour"></a>
## 세 군데 코드 읽기

<!-- claims: c006-implementation -->
1. **모델의 attention 호출을 제한합니다.** [`ScopedAttention`](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/006-attention-decision-stability/src/intervention.py#L30)부터 읽습니다. `route`는 13번 층의 정사각형 prefill 개입을 허용하고 `__exit__`는 이전 등록 함수를 복구합니다. [예외 복구 테스트](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/006-attention-decision-stability/tests/test_core.py#L193)에서 이 동작을 확인할 수 있습니다.
2. **행렬 크기를 줄입니다.** [`sliced`](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/007-interaction-aware-mlp-pruning/src/surgery.py#L6)는 남긴 그룹을 gate/up 행과 down 열에 대응시킵니다. `masked`가 비교 기준을 만들고 `replace_mlp`는 `finally`에서 원래 모듈을 복구합니다. [형상·bias·계산 대응 테스트](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/007-interaction-aware-mlp-pruning/tests/test_core.py#L62)가 이 연결을 검사합니다.
3. **불완전한 근거를 거절합니다.** [`assess`](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/006-attention-decision-stability/src/validity.py#L10)는 전체 출력과 실행 경로 검사 결과를 요구합니다. [원래 pairing 테스트](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/006-attention-decision-stability/tests/test_core.py#L249)는 누락·중복·서로 다른 입력의 결합을 거절합니다. 새 [화면 검사기](../../tools/showcase/check.py)는 표시용 데이터를 보존된 원자료와 대조합니다.

## 60–90초 시연 대본

*실제 화면을 열어 설명하기 위한 대본입니다. 녹화 영상은 포함하지 않습니다.*

“Case007 로컬 화면을 엽니다. B는 삭제 전 기준선이고 옆의 두 열은 같은 수의 그룹을 제거한 결과입니다. 25% 예산에서 고정 CODE 안내 사례를 고릅니다. 이 입력에서 국소 재구성, 기준선 선택 보존, 정답에 부여한 확률을 따로 볼 수 있습니다. 50%로 바꾸면 선택 그룹과 모듈 출력이 같습니다. 시간은 이 입력의 값으로 붙이지 않고, 별도 고정 입력 6개의 측정으로 보여줍니다.

“이제 Case006에서 표준 입력 하나를 고릅니다. 정답과 최고점 집합, 네 선택지 확률을 비교합니다. 회귀는 기준선이 맞힌 답의 손실이며, 선택 변경에는 새 정답이나 다른 오답도 있습니다. FP32 진단으로 전환하면 같은 시나리오를 그 출력 계산의 기준선과 비교합니다. 마지막으로 원자료 링크를 열고 CPU 선택기 명령을 실행합니다. 모델을 실행하지 않고 보정 행렬 Q에서 그룹 조합을 열거해 저장된 선택과 대조합니다.”

## 근거로 연결되는 소개 문장 세 개

<!-- claims: c004-cost c005-stop -->
- 그룹 선택과 행렬 축소 도구를 만들고, 마스킹 계산 및 미관측 입력과 비교했습니다. [Case007 방법·영어](../../cases/007-interaction-aware-mlp-pruning/METHODS.md).
- Qwen 한 층 개입과 답변 확률·정답 손실·정답 획득·동률의 쌍별 분석을 구현했습니다. [Case006 방법·영어](../../cases/006-attention-decision-stability/METHODS.md).
- Attention 전체 호출 비용을 측정하고, 관측된 정밀도·비용 절충과 **STOP_DEV_SCREEN** 판정을 함께 기록했습니다. [Case004 비용·영어](../../cases/004-low-precision-attention-break-even/README.md) · [Case005 판정·영어](../../cases/005-attention-precision-pareto/README.md).

<a id="contribution-and-reuse"></a>
## 프로젝트 기여와 재사용

<!-- claims: attribution -->
프로젝트의 작업은 비교 실행, 범위를 제한한 어댑터, 행렬 축소, 유효성 검사, 수치·답변 분석과 결과 탐색기입니다. Qwen과 Red Hat AI가 체크포인트를, Transformers·PyTorch·vLLM이 모델과 실행 구성요소를, SageAttention이 저정밀 커널을 제공합니다. Case001 guard의 저자는 hclsys이며 EFQ-Softmax는 Han과 공동저자의 작업입니다. HOPE는 별도 MoE expert-pruning 맥락에서 상호작용을 고려한 삭제의 동기를 제공합니다. 여기의 dense 그룹 목적함수는 선택한 MLP 분해에서 직접 구했습니다.

OpenAI Codex가 구현·로컬 실행·테스트·분석·문서 작성을 지원했습니다. [Case001](../../cases/001-sm120-int8-fallback/NOTICE.md), [Case003](../../cases/003-efq-softmax-numerical-audit/NOTICE.md), [Case006](../../cases/006-attention-decision-stability/NOTICE.md), [Case007 출처·영어](../../cases/007-interaction-aware-mlp-pruning/NOTICE.md)에 재사용한 작업을 기록했습니다. 프로젝트 코드는 [Apache-2.0](../../LICENSE)이며 모델·upstream 이용 조건은 별도입니다.

## 기술 협업 범위

**모델 변경 전후 평가**, **로컬 추론 문제 진단**, **재현 가능한 분석 도구와 결과 화면**을 주제로 코드를 검토할 수 있습니다. 비교할 모델 경계, 공개 입력 ID, 평가 지표에서 논의를 시작할 수 있습니다. [저장소 이슈](https://github.com/Munsik-Kim/inference-lab/issues)는 공개 기술 질문 경로이며, 인증값이나 비공개 고객 자료는 공개 보고에 넣지 않습니다.

근거는 특정 모델·장치 하나·합성 과제의 결과입니다. 배포 판단은 **NOT_ASSESSED**입니다. CPU 감사는 남긴 스칼라를 검산하며, 제외된 전체 벡터와 별도 GPU 재현의 확인 범위는 [재현 안내](GETTING_STARTED.md)에서 설명합니다.
