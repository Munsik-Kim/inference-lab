# 도구, 구현, 기술 협업

[English](../en/PORTFOLIO.md) | 한국어 · [홈](../../README.ko.md)

[처음 보는 분](START_HERE.md) · [쉬운 용어집](GLOSSARY.md)

## 30초 소개

<!-- claims: c008-tracks c008-artifact-implementation c007-implementation -->
DIOVA는 학습된 모델을 더 가볍게 만드는 코드와, 변경한 모델을 다시 실행하고 비교하는 도구를 개발합니다. 실제 구현과 측정 결과를 프로젝트별로 확인할 수 있습니다.

Case008은 GPTQ 저장본 제작, 압축 가중치 검사, 층별 구조 로더와 고정 MLP의 출력 복구를 연결합니다. Qwen 4B 경로는 **8.045 GB에서 2.652 GB로 줄인 가중치 파일**을 만듭니다. Qwen 0.6B 경로는 이미 작은 MLP에 보정값을 저장하고, 원래 checkpoint 없이 그 구조를 다시 불러옵니다. [두 구현 과정 보기](https://munsik-kim.github.io/inference-lab/ko/case008.html) · [작은 CPU 저장·재로딩 예제 실행](GETTING_STARTED.md#tiny-model-demo).

Case007은 채널 그룹 선택을 실제 gate/up/down 행렬 축소로 연결합니다. Case006에서는 질문별 정답 확률, 정답 손실·개선과 동률을 살펴볼 수 있습니다. 결과 화면에서 실제 입력 하나를 고르고, 연결된 구현 코드를 읽은 뒤 공개 선택 계산을 재실행할 수 있습니다. [코드 투어](#code-tour) · [CPU 선택기](GETTING_STARTED.md#cpu-selector).

![고정 코드 입력 하나를 비교하는 실제 Case007 로컬 화면](../../presentation/screenshots/case007-comparison.png)

*로컬 Edge 화면: 첫 고정 CODE 입력의 25% 삭제 비교입니다. [Case006 출력 계산 화면](../../presentation/screenshots/case006-readout.png)은 별도로 표시한 같은 입력 진단을 보여줍니다. 탐색을 안내하는 선택 화면이며 전체 성능 집계가 아닙니다.*

## 구조 복원과 평가 도구의 기여

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

*실제 화면을 열어 설명하기 위한 대본입니다. 녹화 영상은 포함하지 않습니다.*

“Case008에서 Q를 엽니다. BF16 원본을 GPTQ로 변환하고 새 vLLM 프로세스에서 실행한 순서를 봅니다. 파일 바이트와 allocator 메모리는 서로 다른 자원입니다. 정답 수는 같아도 확률 점수의 방향은 달랐습니다.

“R로 이동해 I25의 보정 전후를 비교합니다. 채널 수는 고정된 채 국소 오차가 줄었습니다. 정답 수는 별도 결과로 확인합니다. CPU 예제 링크에서 작은 무작위 모델 명령을 실행하고 summary.json을 엽니다. 저장 프로세스가 종료된 뒤 다른 프로세스가 복사본을 읽었습니다. 구조, 전체 출력, 공유 가중치가 모두 같아야 합니다. 이어서 로더 소스와 실패 테스트를 봅니다. Case007과 Case006에서는 앞선 그룹 선택과 입력별 답변 비교를 볼 수 있습니다.”

## 근거로 연결되는 소개 문장 세 개

<!-- claims: c004-cost c005-stop -->
- 그룹 선택과 행렬 축소 도구를 만들고, 마스킹 계산 및 미관측 입력과 비교했습니다. [Case007 방법·영어](../../cases/007-interaction-aware-mlp-pruning/METHODS.md).
- Qwen 한 층 개입과 답변 확률·정답 손실·정답 획득·동률의 쌍별 분석을 구현했습니다. [Case006 방법·영어](../../cases/006-attention-decision-stability/METHODS.md).
- Attention 전체 호출 비용을 측정하고, 관측된 정밀도·비용 절충과 **STOP_DEV_SCREEN** 판정을 함께 기록했습니다. [Case004 비용·영어](../../cases/004-low-precision-attention-break-even/README.md) · [Case005 판정·영어](../../cases/005-attention-precision-pareto/README.md).

## 도구의 성능 평가

<!-- claims: c007-transfer c007-quality c006-native c006-readout -->
Case008은 Qwen 0.6B의 한 층 MLP에서 짧은 합성 평가 입력 192개의 삭제로 생긴 국소 제곱 출력 오차를 **94.1–95.4%** 줄였습니다. 정답 점수 변화는 확률 지표와 보정 구조에 따라 달랐고, 측정한 요청 시간비 구간은 1을 포함했습니다. [Q 품질·속도 평가](https://munsik-kim.github.io/inference-lab/ko/case008.html#q-evaluation)와 [R 평가](https://munsik-kim.github.io/inference-lab/ko/case008.html#r-evaluation)에서 비교할 수 있습니다.

MLP 25% 삭제에서 PAIRWISE의 국소 오차는 소폭 낮았고 기준선 선택을 더 많이 보존했지만, 정답 NLL은 INDEPENDENT가 더 좋았습니다. 50%의 선택 모듈은 같았으며 고정 판정은 **COMPLETED_NO_CLEAR_TRANSFER**입니다. Case006에서는 표준 192개 중 A_PUBLIC은 5개, V4는 8개의 선택이 달라졌습니다. 같은 입력의 사후 출력 계산 진단은 동률과 마지막 어휘 점수 계산을 살펴봅니다. 구현상 선택과 서로 다른 평가 목표를 이 예제들에서 확인할 수 있습니다. [사례 안내](CASEBOOK.md) · [로컬 화면 열기](GETTING_STARTED.md#local-showcase).

<a id="contribution-and-reuse"></a>
## 프로젝트 기여와 재사용

<!-- claims: attribution -->
프로젝트의 작업은 비교 실행, 범위를 제한한 어댑터, 행렬 축소, 유효성 검사, 수치·답변 분석과 결과 탐색기입니다. Qwen과 Red Hat AI가 체크포인트를, Transformers·PyTorch·vLLM이 모델과 실행 구성요소를, SageAttention이 저정밀 커널을 제공합니다. Case001 guard의 저자는 hclsys이며 EFQ-Softmax는 Han과 공동저자의 작업입니다. HOPE는 별도 MoE expert-pruning 맥락에서 상호작용을 고려한 삭제의 동기를 제공합니다. 여기의 dense 그룹 목적함수는 선택한 MLP 분해에서 직접 구했습니다.

GPTQ는 기존 양자화 방법이며 LLM Compressor·compressed-tensors는 변환·압축 저장을, safetensors는 가중치 형식을 제공합니다. Case008에서는 저장본 검사, 층별 구조 로더, 고정 구조 ridge 적합을 연결했습니다. [Case008 출처·영어](../../cases/008-build-reconstruct-reload/NOTICE.md).

OpenAI Codex가 구현·로컬 실행·테스트·분석·문서 작성을 지원했습니다. [Case001](../../cases/001-sm120-int8-fallback/NOTICE.md), [Case003](../../cases/003-efq-softmax-numerical-audit/NOTICE.md), [Case006](../../cases/006-attention-decision-stability/NOTICE.md), [Case007 출처·영어](../../cases/007-interaction-aware-mlp-pruning/NOTICE.md)에 재사용한 작업을 기록했습니다. 프로젝트 코드는 [Apache-2.0](../../LICENSE)이며 모델·upstream 이용 조건은 별도입니다.

## 기술 협업 범위

**모델 변경 전후 평가**, **로컬 추론 문제 진단**, **재현 가능한 분석 도구와 결과 화면**을 주제로 코드를 검토할 수 있습니다. 비교할 모델 경계, 공개 입력 ID, 평가 지표에서 논의를 시작할 수 있습니다. [저장소 이슈](https://github.com/Munsik-Kim/inference-lab/issues)는 공개 기술 질문 경로이며, 인증값이나 비공개 고객 자료는 공개 보고에 넣지 않습니다.

근거는 특정 모델·장치 하나에서 수행한 합성 실험과 Case009의 별도 공식 과제 결과입니다. 배포 판단은 **NOT_ASSESSED**입니다. CPU 감사는 남긴 스칼라를 검산하며, 제외된 전체 벡터와 별도 GPU 재현의 확인 범위는 [재현 안내](GETTING_STARTED.md)에서 설명합니다.
