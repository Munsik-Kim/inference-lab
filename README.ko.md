# DIOVA

[English](README.md) | 한국어

**D**eep-learning **I**nference **O**ptimization, **V**alidation & **A**nalysis

DIOVA는 딥러닝 모델을 압축하고 실행 성능과 답변 변화를 비교하는 프로젝트입니다. PyTorch로 구현한 모델 수정 도구, RTX 5080 측정, 입력별 결과 탐색기를 함께 제공합니다.

**Python · PyTorch · Transformers · vLLM · NumPy**

[구현 기능](#capabilities) · [기술스택](#tech-stack) · [프로젝트 보기](#projects)

<a id="capabilities"></a>
## 구현 기능

### 모델 연산 교체와 구조 압축

모델의 특정 연산을 교체하고, 채널을 선택해 실제 행렬 크기를 줄입니다. 변경한 계산의 대응과 원래 모듈로의 복원을 테스트합니다.

Python · PyTorch · Transformers. [Attention 교체 코드](cases/006-attention-decision-stability/src/intervention.py) · [행렬 축소와 테스트](cases/007-interaction-aware-mlp-pruning/tests/test_core.py)

### GPU 추론 성능과 답변 비교

같은 입력에서 지연시간·GPU 메모리·정답 점수를 비교합니다. 부분 연산과 모델 전체의 실행 비용을 나누어 측정합니다.

vLLM · CUDA 실행 환경 · NVML · NumPy. [요청·메모리 측정 코드](cases/002-bf16-fp8-document-extraction/scripts/run.py) · [시간 측정 경계](cases/004-low-precision-attention-break-even/src/timing.py)

### 재계산 도구와 결과 탐색기

공개 기록을 재계산하는 Python 도구와 입력별 결과를 보는 웹 화면을 만듭니다. 자동 검사로 표시 데이터와 원자료의 연결을 확인합니다.

Python · JavaScript · GitHub Actions · GitHub Pages. [CPU 선택기 재계산](tools/showcase/replay_selection.py) · [표시 계층 검사와 테스트](tests/showcase/test_showcase.py)

<a id="tech-stack"></a>
## 기술스택별 작업

| 작업 | 기술과 구현 |
|---|---|
| 모델 구현·수정 | **Python / PyTorch / Transformers** — 모델 객체 검사, 연산 교체, 행렬 축소, shape·dtype 검사를 구현했습니다. |
| GPU 추론 통합 | **vLLM / PyTorch SDPA / SageAttention / CUDA 실행 환경 / NVML** — 공개 backend 연결, 로컬 서버 실행, 실행 경로·자원 사용 기록을 구현했습니다. |
| 수치 분석·그룹 선택 | **NumPy / Matplotlib / 조합 탐색 / 쌍별 통계** — 기여 행렬과 제거 그룹을 구하고, 오차·점수·불확실성을 분석했습니다. |
| 재현·자동 검사 | **Git / unittest / SHA256 / GitHub Actions** — 파일 보존과 입력 짝짓기 검사, 스칼라 재계산, 문서·표시 계층 CI를 구현했습니다. |
| 결과 화면 | **HTML / CSS / JavaScript / GitHub Pages** — 필터·입력별 직접 링크·소스 탐색을 갖춘 한·영 정적 화면을 만들었습니다. |

[구현 코드·관련 테스트 읽기](docs/ko/PORTFOLIO.md#code-tour). CUDA는 GPU 실행·측정에, 공개 커널은 연산 교체에 사용했습니다.

<a id="projects"></a>
## 대표 프로젝트

### Case 007 — 채널 선택부터 실제 모델 구조 축소까지

<!-- claims: c007-transfer c007-quality -->
MLP는 모델 안에서 정보를 변환하는 계산 블록입니다. 채널 그룹을 선택하고 gate/up/down 행렬을 잘라 작은 모듈을 만드는 도구를 구현했습니다. 사용 기술은 **PyTorch · NumPy · 구조화 가지치기(pruning)**입니다.

한 층 MLP 그룹의 25% 삭제에서는 상호작용 선택의 국소 오차가 미관측 입력 192개에서 소폭 낮았고, 50%에서는 두 방법이 같은 구조를 골랐습니다. 축소한 **한 층 MLP**는 1.176–1.412배 빨랐으며, 질문을 처음 처리하는 prefill의 전체 모델 가속은 확인되지 않았습니다. 기준 모델을 가깝게 보존하는 것과 정답에 더 좋은 확률을 주는 것은 다른 결과였습니다.

[결과 탐색](https://munsik-kim.github.io/inference-lab/ko/case007.html) · [행렬 축소 코드](cases/007-interaction-aware-mlp-pruning/src/surgery.py) · [상세 연구](docs/ko/CASEBOOK.md#case-007)

### Case 006 — 모델의 계산을 바꾸고 답변 변화를 추적하는 도구

<!-- claims: c006-native c006-readout c006-limits -->
한 층 attention을 교체하고 같은 질문의 정답 확률·선택·최고점 동률을 비교하는 도구를 구현했습니다. 사용 기술은 **PyTorch · Transformers · SageAttention**입니다.

두 설정은 표준 192개 중 5개, 192개 중 8개의 선택을 바꿨습니다. 이 집합에서는 기준선의 정답을 잃지 않았지만 별도로 선정한 스트레스 집합에서는 정답 손실이 있었습니다. 같은 입력의 사후 진단은 마지막 어휘 점수 계산의 정밀도를 살폈습니다. 한 모델의 한 층을 합성 질문으로 비교한 기록입니다.

[결과 탐색](https://munsik-kim.github.io/inference-lab/ko/case006.html) · [Attention 교체 코드](cases/006-attention-decision-stability/src/intervention.py) · [상세 연구·추가 진단](docs/ko/CASEBOOK.md#case-006)

### Case 008 — 바꾼 모델을 저장하고 다시 실행하기

<!-- claims: c008-tracks -->
GPTQ 변환부터 새 vLLM 실행까지, 고정된 작은 MLP의 보정부터 층별 구조 로딩까지 두 경로를 연결했습니다. Q 가중치 파일은 약 67.0% 작아졌고 R의 국소 제곱오차는 짧은 합성 입력에서 94.1–95.4% 감소했습니다. 요청 가속은 확정하지 못했고 정답 점수 변화는 혼재했습니다.

[구조화 보고서](cases/008-build-reconstruct-reload/REPORT.ko.md) · [코드·재현(영어)](cases/008-build-reconstruct-reload/REPRODUCTION.md) · [측정 자료 ZIP](downloads/case008_build_reconstruct_reload_reviewed_publication_v2.zip)

## 여덟 질문과 구현물

| 쉬운 질문 | 이 사례에서 볼 수 있는 것 |
|---|---|
| [001 — GPU에서 모델이 실행되지 않는다면?](docs/ko/CASEBOOK.md#case-001) | 기존 실행 경로 수정의 전후 비교 도구. 기술명: SM120, vLLM, W8A8. |
| [002 — 적은 비트로 저장하면 메모리와 시간이 줄까?](docs/ko/CASEBOOK.md#case-002) | 공식 모델의 문서 추출 비교. 자원 사용은 줄었지만 과제 정답률은 낮았습니다. 기술명: BF16, FP8. |
| [003 — 간단해진 계산은 원래 값과 얼마나 비슷할까?](docs/ko/CASEBOOK.md#case-003) | 비교 기준의 scale을 나누어 검사한 수치 오차. 기술명: EFQ-Softmax, FP32 모의 계산. |
| [004 — 핵심 연산이 빠르면 호출 전체도 빨라질까?](docs/ko/CASEBOOK.md#case-004) | 준비 비용을 포함한 attention 시간. 긴 입력 일부는 빨랐지만 출력 오차는 고정 기준을 넘었습니다. 기술명: kernel, 전체 호출 비용. |
| [005 — 속도와 오차를 함께 만족하는 설정이 있을까?](docs/ko/CASEBOOK.md#case-005) | 측정한 절충 관계. 바꿔 본 설정 중 두 조건을 만족한 것이 없어 새 확인 단계로 넘어가지 않았습니다. 기술명: 정밀도, 국소 오차. |
| [006 — 어느 질문의 답이 바뀌는가?](docs/ko/CASEBOOK.md#case-006) | 입력별 답변 점수·선택·최고점 동률 진단. 기술명: NLL, top-score tie. |
| [007 — 작은 계산 블록에 어느 그룹을 남길까?](docs/ko/CASEBOOK.md#case-007) | 그룹 선택, 실제 행렬 축소, 선택에 쓰지 않은 입력의 평가. 기술명: MLP, pruning. |
| [008 — 바꾼 모델을 저장하고 다시 실행할 수 있을까?](docs/ko/CASEBOOK.md#case-008) | GPTQ 저장·실행 통합과 고정 MLP ridge 복구. Q 파일과 R 국소 오차가 감소했고 정답 점수 변화는 혼재. |

## 더 깊게 보기

5분 [처음 보기](docs/ko/START_HERE.md)에서 시작하거나 [용어집](docs/ko/GLOSSARY.md)에서 궁금한 말을 찾아보세요. [사례 안내](docs/ko/CASEBOOK.md)는 방법·소스·정확한 결과로, [이용 안내](docs/ko/GETTING_STARTED.md)는 ZIP 다운로드·로컬 HTML·CPU 검사로 이어집니다. 탐색기는 저장된 측정값을 보여주므로 GPU가 필요 없습니다. GitHub의 HTML 미리보기는 작동 화면이 아닌 소스 코드입니다. 기존 탐색기 UI와 상세 기술 원문은 영어입니다.

## 기여와 출처

DIOVA는 **Deep-learning Inference Optimization, Validation & Analysis**의 약자입니다.

공개 모델과 실행 라이브러리에 비교 실행기, 행렬 축소 도구, 수치 검사와 근거 탐색기를 연결했습니다. [포트폴리오의 기여·출처](docs/ko/PORTFOLIO.md#contribution-and-reuse)에서 Qwen, Transformers, PyTorch, vLLM, SageAttention 및 기존 방법·수정의 저자를 확인할 수 있습니다. OpenAI Codex가 구현·실행·분석·작성을 지원했습니다. 프로젝트 자료는 [Apache-2.0](LICENSE)을 따르며 사례별 고지에 upstream 조건을 남겼습니다. 이 작업은 모델 변경 평가와 추론 문제 진단에 연결되며, 배포 적합성은 평가하지 않았습니다.
