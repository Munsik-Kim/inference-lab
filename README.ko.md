# Inference Lab

[English](README.md) | 한국어

Inference Lab은 모델 압축, 수치 진단, 답변 비교 도구를 RTX 5080의 측정 결과와 연결한 추론 엔지니어링 프로젝트입니다. Qwen의 MLP를 실제로 줄이는 코드, 입력별 모델 출력을 비교하는 분석, CPU 선택기 재계산, 브라우저 결과 탐색기를 제공합니다.

[결과 보기](docs/ko/GETTING_STARTED.md#local-showcase) · [구현 살펴보기](docs/ko/PORTFOLIO.md#code-tour) · [선택기 재계산](docs/ko/GETTING_STARTED.md#cpu-selector)

## Case 007: 채널을 선택하고 작은 MLP를 만듭니다

<!-- claims: c007-transfer c007-quality -->
압축 도구는 Qwen3-0.6B 한 층 MLP의 채널을 16그룹으로 나눠 측정하고, 삭제할 그룹을 고른 뒤 gate/up 행과 down 열을 잘라냅니다. INDEPENDENT는 그룹을 개별 평가하고 PAIRWISE는 그룹 사이의 부호 있는 상호작용도 반영합니다. 같은 삭제 예산과 보정 입력을 사용하고, 선택에 쓰지 않은 입력 192개로 결과를 확인했습니다.

25% 삭제에서 PAIRWISE의 평균 국소 재구성 오차는 29.8291%에서 29.6288%로 소폭 낮아졌습니다. 50%에서는 두 방법이 같은 축소 모듈을 만들었습니다. 고정된 두 예산 판정은 **COMPLETED_NO_CLEAR_TRANSFER**입니다. 25%의 PAIRWISE가 기준선 선택을 더 많이 보존했지만, 별도로 계산한 정답(gold) 확률의 음의 로그(NLL)는 INDEPENDENT가 더 좋았습니다. NLL은 낮을수록 좋습니다.

축소한 **한 층 MLP**는 국소 호출에서 1.176–1.412배 빨랐고, 모델 전체 prefill 속도비 구간은 모두 1을 포함했습니다. [비교 화면](docs/ko/GETTING_STARTED.md#local-showcase)에서 입력별 삭제 그룹, 국소 오차, 정답 전환을 볼 수 있습니다. [실험과 측정값·영어](cases/007-interaction-aware-mlp-pruning/README.md) · [행렬 축소 코드](cases/007-interaction-aware-mlp-pruning/src/surgery.py).

## Case 006: 한 층을 바꾼 뒤 점수와 선택을 따라갑니다

<!-- claims: c006-native c006-readout c006-limits -->
범위를 제한한 어댑터가 13번 층의 프롬프트 prefill attention만 바꿉니다. B는 원래 BF16 기준선이며, A_PUBLIC과 V4는 같은 BF16 가중치 모델에 적용한 두 저정밀 attention 설정입니다. 탐색기에서 정답 확률, 선택 변경, 기준선이 맞힌 답을 후보가 잃은 경우인 회귀를 비교합니다.

표준 입력 192개 중 A_PUBLIC은 **192개 중 5개**, V4는 **192개 중 8개**의 선택이 바뀌었습니다. 이 집합의 회귀는 0회였고, 별도로 선정한 스트레스 입력 46개에서는 각각 2회·1회였습니다. 과제를 같은 비중으로 둔 평균 쌍별 NLL 변화의 95% 구간은 두 설정 모두 0을 포함했습니다. 변화는 후보−B이며 양수가 나빠지는 방향입니다. 이 결과로 동등성을 입증한 것은 아닙니다.

사후 출력 계산(readout) 진단은 같은 내부 상태와 BF16 가중치의 표현값에서 마지막 어휘 점수를 다시 계산했습니다. 원래 표준셋의 모든 선택 변경에는 B 또는 후보의 최고점 동률이 끼어 있었습니다. 별도 FP32 출력 계산에서는 각 후보 **192개 중 3개**가 바뀌었으며 새 변경도 생겼습니다. 화면은 같은 입력을 재사용한 두 계산을 따로 보여줍니다. B의 코드 정답은 **15/64개**, 엄격한 구조화 생성 성공은 각 설정 **0/24개**였습니다. [원실험·영어](cases/006-attention-decision-stability/README.md) · [출력 계산 진단·영어](cases/006-attention-decision-stability/supplemental/readout-ties-v1/README.md).

## 일곱 가지 연결된 질문

| 사례 | 도구와 관측 |
|---|---|
| [001 — 모델을 실행할 수 있는가?](docs/ko/CASEBOOK.md#case-001) | 전후 비교 실행기로 upstream의 SM120 선택 guard를 확인했습니다. |
| [002 — FP8에서 무엇이 달라지는가?](docs/ko/CASEBOOK.md#case-002) | 문서 추출을 쌍으로 비교해 자원 감소와 낮은 정답률을 측정했습니다. |
| [003 — 어떤 수치 기준과 비교하는가?](docs/ko/CASEBOOK.md#case-003) | 방정식 검산으로 코드 매핑과 scale 선택을 나눠 봤습니다. |
| [004 — 호출 전체 비용은 얼마인가?](docs/ko/CASEBOOK.md#case-004) | 입력 준비와 변환을 포함한 attention 시간을 측정했습니다. |
| [005 — 정밀도와 비용은 어떻게 맞바뀌는가?](docs/ko/CASEBOOK.md#case-005) | 제한된 후보 비교에서 절충점과 STOP_DEV_SCREEN을 기록했습니다. |
| [006 — 개별 답변은 어떻게 바뀌는가?](docs/ko/CASEBOOK.md#case-006) | 점수와 동률을 함께 보며 정답률 뒤의 선택 변경을 조사했습니다. |
| [007 — 어느 MLP 그룹을 제거할 것인가?](docs/ko/CASEBOOK.md#case-007) | 선택, 실제 행렬 축소, 미관측 입력 검증을 연결했습니다. |

사례들은 관련된 엔지니어링 질문을 서로 다른 조건에서 조사했으며, 하나의 연속 성능 곡선은 아닙니다.

## 도구 사용과 구현 기여

[시작 안내](docs/ko/GETTING_STARTED.md)는 한·영 로컬 화면, 보존된 영어 탐색기, 작은 CPU 선택기 예제와 스칼라 검산을 연결합니다. 결과를 보는 데 GPU는 필요 없습니다. [포트폴리오](docs/ko/PORTFOLIO.md)에는 코드 읽기 순서와 60–90초 시연 대본이 있습니다.

Qwen, Transformers, PyTorch, vLLM, SageAttention 위에 비교 실행기, 모델 구조 변경, 유효성 검사, 분석과 결과 탐색기를 구현했습니다. [기여·출처](docs/ko/PORTFOLIO.md#contribution-and-reuse)에서 기존 방법과 커널의 저자를 확인할 수 있습니다. OpenAI Codex가 구현·실행·분석·문서 작성을 지원했습니다. 프로젝트 코드는 [Apache-2.0](LICENSE)이며 upstream 이용 조건은 별도입니다.

모델 변경 전후 평가, 로컬 추론 문제 진단, 재현 가능한 분석 도구를 주제로 기술 협업을 논의할 수 있습니다. [공개 기술 질문](https://github.com/Munsik-Kim/inference-lab/issues)에는 사례와 공개 입력 ID를 연결할 수 있습니다. 결과는 측정한 장치와 합성 과제의 범위이며 배포 판단은 **NOT_ASSESSED**입니다.
