# DIOVA — 모델 압축·실행·결과 검사

공개 AI 모델을 제한된 그래픽카드에서 실행하고, 변경 전후의 답변과 실제 작업 결과를 검사하는 도구를 만들었습니다. DIOVA의 공개 구현과 측정 기록은 **로컬 모델 실행·최적화, 변경 전후 품질 검사, 도구 자동화 평가**와 관련됩니다.

**Munsik Kim | DIOVA** · [English](README.md) | 한국어

[포트폴리오·모델별 안내 (Pages)](https://munsik-kim.github.io/inference-lab/ko/index.html) · [직접 사용: 내 결과 비교 예제](packages/diova-compare/examples/compare/README.ko.md)

<a id="모델을-작게-저장하고-실행하고-달라진-답변까지-확인합니다"></a>

## 대표작 A · B · C

<!-- claims: c008-tracks c008-artifact-implementation -->
| 문제 | 직접 구현 | 근거·사용 경로 |
|---|---|---|
| A. 모델을 작게 저장하고 실제 요청까지 실행하기 | 기존 압축 방법(GPTQ)을 연결하고, 저장 파일 검사·새로 시작한 프로그램에서 재실행·동일 조건의 요청 비교 구현 | [Case008 Q](https://munsik-kim.github.io/inference-lab/ko/case008.html#track-q) → [Case009](https://munsik-kim.github.io/inference-lab/ko/case009.html#serving) |
| B. 바꾼 모델의 답변을 같은 질문 기준으로 비교하기 | 독립 CPU 도구 diova-compare 제작: 질문·정답 기준이 같은지 검사하고, 잃은 정답·새 정답·그 밖의 변경과 원 실행 실패 표시 | [CPU 실행 예제](packages/diova-compare/examples/compare/README.ko.md) · [코드·테스트](packages/diova-compare/README.ko.md#구현) |
| C. 도구 작업의 실제 완료와 실행 비용 확인 | 저장된 값·수정 후 검증 관측·최종 답을 함께 채점하고, 호출 정책과 KV 구성에 따른 완주·실패·비용을 재생 검사 | [Case017 결과와 실제 사건](cases/017-agent-contract-cost-bench/REPORT.ko.md) · [모델 없이 CPU 재생](cases/017-agent-contract-cost-bench/REPRODUCE.ko.md) · 앞선 연구: [Case013 후속 구현](research/agent_tool_contract_repair_20261014/README.ko.md) |

## 확인된 저장 공간·실행 메모리

Qwen3-4B-Instruct-2507 · RTX 5080 16GB · 원본 BF16 → 가중치를 4비트로 압축한 GPTQ W4A16. 아래 행은 측정 대상과 실행 조건이 다릅니다.

| 기존 실험·측정 대상 | 원본 → 압축 모델 | 의미·조건 |
|---|---|---|
| [Case008 Q · 가중치 파일](https://munsik-kim.github.io/inference-lab/ko/case008.html#q-evaluation) | **8.045 GB → 2.652 GB** | 파일 크기 **67.0% 감소**, 십진 GB |
| [Case008 Q · PyTorch GPU 할당 최고값](https://munsik-kim.github.io/inference-lab/ko/qwen3-4b.html#storage) | **10.4868 → 5.3971 GiB** | 약 48.5% 감소. 연산을 매번 호출(eager)·요청 1개·캐시 1 GiB, 초기화·실행 포함 |
| [Case009 · GPU 전체 메모리 최고 표본값](https://munsik-kim.github.io/inference-lab/ko/qwen3-4b.html#device-memory) | **14.08–14.51 → 9.08–9.52 GiB** | 같은 라운드끼리 **4.98–5.06 GiB·34.4–35.8% 감소**. GPU 작업 순서를 미리 준비(graph)·캐시 4 GiB·3라운드·1초 표본 |

Case009는 시작·준비 실행·바탕화면 사용량을 포함한 표본입니다. 정답 손실과 평가 프로그램의 비정상 종료도 [모델별 안내](docs/ko/LLM_GUIDE.md)에서 함께 읽습니다. Case008 R은 별도 Qwen3-0.6B의 한 계산 블록을 줄이고 복원한 연구입니다.

<!-- case016-entry -->
## Case016 — 같은 KV 메모리에서 수용량과 요청 비용 검증

RTX 5080에서 실제 NVFP4 KV 저장·읽기 검사와 고정 pool 비교 도구를 구현했습니다. 같은 약4GiB 캐시에서 token 수용량은 3.56배였고, 16K 입력의 동시 요청2/4에서 처리량 비율(F4/B16)은 1.21/1.35배였습니다. 모든 조건이 빨라진 것은 아니며 원 agent 기준선 실패도 보존했습니다.

[Report](cases/016-kv-agent-bench/REPORT.ko.md) · [CPU](cases/016-kv-agent-bench/REPRODUCE.ko.md) · [Code](cases/016-kv-agent-bench/followups/nvfp4-technical-r1/src/nvfp4_technical/runner.py)

<!-- case017-entry -->
## Case017 — 도구 호출 규칙과 업무 완주 비용

상품 합계 기록·창고 이동·문서 정보 변경을 시험하고, 저장된 값·수정 후 받은 검증 결과·최종 답을 함께 검사하는 평가기를 구현했습니다.

BF16 가중치를 고정한 신규 합성 24과제에서, NVFP4 KV의 문법 제약은 자유 출력보다 완주 7/24→15/24, 상태 정확 7/24→22/24를 기록했습니다. 실패를 포함한 실행 비용과 남은 오류도 공개합니다.

DEV에서 BF16 KV·자유 출력은 5/8로 확대 기준 6/8에 미달했지만, 로컬 gate의 기준 누락으로 P2가 진행됐습니다. 이 이탈을 공개한 탐색적 측정입니다.

[결과·실제 사건](cases/017-agent-contract-cost-bench/REPORT.ko.md) · [CPU 재생](cases/017-agent-contract-cost-bench/REPRODUCE.ko.md) · [확대 기준 이탈 경위](cases/017-agent-contract-cost-bench/PROTOCOL_DEVIATION.ko.md)

<!-- case018-entry -->
## Case018 — 재고 복구와 완료 보고를 따로 검사했습니다

재고 복구 환경과 검사 도구를 만들어 복구된 상태와 마지막 보고를 따로 확인했습니다. 같은 24개 합성 과제에서 두 정책 모두 상태·후속 검사 통과는 BF16 KV 24/24, NVFP4 KV 19/24였고, 검증·완료 보고까지 충족한 수는 각각 0/24, 5/24였습니다. 양쪽 가중치는 BF16으로 고정했습니다.

[결과·실제 사건](cases/018-agent-state-recovery/REPORT.ko.md) · [모델 없이 검산하기](cases/018-agent-state-recovery/REPRODUCE.ko.md)

<!-- case020-entry -->
## Case020 — 임베딩을 작게 보관하면 실행도 빨라질까?

같은 OPTIC 압축 표현을 전체 미리 계산·요청 복원·중복 제거 복원으로 비교했습니다. 합성 표의 PyTorch allocated peak는 약304→81MiB로 줄었지만, P4의30개 process block에서 중복 제거의 호출 속도 이득은0/30이었습니다. 실제1024/4096회 재사용 세션은 전체를 미리 계산한 방식이 더 빨랐습니다.

[쉬운 설명](cases/020-optic-on-demand-decode/README.ko.md) · [보고서](cases/020-optic-on-demand-decode/REPORT.ko.md) · [모델 없이 CPU 검산](cases/020-optic-on-demand-decode/REPRODUCE.ko.md)

## 직접 사용·재현

- 내 모델의 저장된 답 비교: [두 파일 준비 → 검사 → 비교 → 보고서](packages/diova-compare/examples/compare/README.ko.md). Python 3.10+와 CPU로 실행하며, 자유 서술 답의 정오는 사용자가 제공합니다.
- 로컬 모델 실행: [모델별 안내](docs/ko/LLM_GUIDE.md). 현재 상세 모델은 Qwen3-4B이며 모든 16GB GPU나 업무로 범위를 넓히지 않습니다.
- 기존 기록 검산: [재현 안내](docs/ko/GETTING_STARTED.md). GPU 재실행과 저장 기록의 CPU 재계산을 구분합니다.

<a id="capabilities"></a>
<a id="tech-stack"></a>
## 구현·사용 기술의 근거

[직접 구현과 기존 기술·코드·테스트 표](docs/ko/PORTFOLIO.md#implementation-map)에서 기여를 확인하고, [코드 읽는 순서](docs/ko/PORTFOLIO.md#code-tour)로 이어갈 수 있습니다. [기술 협업 범위](docs/ko/PORTFOLIO.md#기술-협업-범위)는 현재 공개 구현이 보여주는 작업을 설명합니다.

<a id="case-006--모델-변경을-질문별로-추적하는-평가-도구"></a>
<a id="case-007--채널-선택부터-실제-행렬-축소까지"></a>
<a id="case-008--모델을-변환하고-복구하고-다시-실행하기"></a>
<a id="case-010--저정밀-state의-저장재시작과-기억-수명"></a>
<a id="case-011--같은-상태-다른-판독기"></a>
<a id="case-012--이미지-생성-시간-어디에-계산을-더-쓸까"></a>
<a id="case-013--가중치-오차가-줄면-실제-답변도-좋아질까"></a>
<a id="case015--같은-도구-다른-이름"></a>
<a id="gpu-추론-성능과-답변-비교"></a>
<a id="q--gptq-변환저장새-runtime-실행"></a>
<a id="r--구조가-달라진-모델의-저장재로딩과-가중치-보정"></a>
<a id="구현-기능"></a>
<a id="구현-상세와-연구-기록"></a>
<a id="기술스택별-작업"></a>
<a id="모델을-압축하고-저장하고-다시-실행합니다"></a>
<a id="새-측정--긴-decode와-동시-요청"></a>
<a id="연구수치-엔지니어링--case014"></a>
<a id="재계산-도구와-결과-탐색기"></a>
<a id="직접-구현한-부분과-재사용한-기술"></a>
<a id="후속-구현--도구-이름-오류와-실제-작업-성공"></a>
<a id="projects"></a>
## 질문과 구현물

<!-- claims: c007-implementation c006-implementation c006-readout c010-state-bytes -->
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
| [009 — 긴 decode와 동시 요청에서 비용은 어떻게 바뀔까?](cases/009-q-serving-quality/README.ko.md) | 같은 조건의 graph/eager 요청 비용과 공식 품질 계산. 비정상 프로세스 종료 기록도 함께 표시. |
| [010 — 저비트 상태를 실패 후에도 다시 실행할 수 있는가?](docs/ko/CASEBOOK.md#case-010) | 입력 흐름 하나의 저장 상태 12,305 → 3,137 bytes, 74.5% 감소. 상태 저장량이며 전체 RAM·VRAM 감소율과 별개입니다. |
| [011 — 같은 상태에서 판독기만 바꾸면?](docs/ko/CASEBOOK.md#case-011) | 마지막 층 두 tensor 보정·재적용, 정답 점수와 첫 오류 길이의 비교. |
| [012 — 이미지 생성 시간을 어디에 더 쓸까?](docs/ko/CASEBOOK.md#case-012) | 같은 초기 잡음에서 생성 단계·내부 반복·시간을 비교하는 도구. 전체 192장의 개수·색·좌우 요청 조건 충족 여부와 조건별 획득·손실 평가. |
| [013 — 가중치 오차가 줄면 실제 답도 좋아질까?](docs/ko/CASEBOOK.md#case-013) | INT4 스케일 보존·저장·재로딩과 동일 입력 품질 기록. Packed 전체 모델 수치 기준 미통과. |
| [014 — 같은 4비트 저장량에서 품질을 개선할 수 있는가?](docs/ko/CASEBOOK.md#case-014) | PG19 텍스트 예측 품질 개선과 ArXiv 한계. BF16으로 복원한 품질 평가이며 VRAM 절감은 측정하지 않았습니다. |
| [015 — 도구 이름만 바뀌어도 AI가 같은 일을 끝낼까?](https://munsik-kim.github.io/inference-lab/ko/case015.html#overview) | 구매한 개수와 개당 가격을 조회해 총금액을 계산·저장·확인하는 예제로 비교합니다. 16개 문제에서 BF16 15/16→16/16, RTN 16/16→16/16, D16 12/16→12/16. [작업 설명·용어·CPU 예제](cases/015-tool-interface-sensitivity/README.ko.md). |
| [016 — 실제 NVFP4 KV 저장과 동시 요청 비용은?](cases/016-kv-agent-bench/README.ko.md) | BF16 가중치·같은4GiB pool에서 KV 저장·읽기, token 수용량, 36개 block의 처리량·대기·수치 sanity 비교. |
| [017 — 도구 호출 규칙과 업무 완주 비용](cases/017-agent-contract-cost-bench/README.ko.md) | 엄격한 JSON·실제 상태·최신 검증 관측·완주 비용. 확대 기준 누락을 공개한 탐색적 측정. |

## 더 깊게 보기

[처음 보는 분](docs/ko/START_HERE.md) · [용어집](docs/ko/GLOSSARY.md) · [전체 사례와 원자료](docs/ko/CASEBOOK.md) · [선행연구와 구현 대응](docs/related-work/README.ko.md)

## 기여와 출처

DIOVA는 **Deep-learning Inference Optimization, Validation & Analysis**의 약자입니다. 공개 모델·라이브러리에 비교 실행기, 모델 변경 도구와 검사·결과 화면을 연결했습니다. 원저자와 재사용 범위는 [기여·출처](docs/ko/PORTFOLIO.md#contribution-and-reuse)에 있습니다. OpenAI Codex가 구현·실행·분석·작성을 지원했습니다. 프로젝트 자료는 [Apache-2.0](LICENSE)을 따르며 사례별 고지에 upstream 조건을 남겼습니다. 실제 서비스 배포 적합성은 미평가입니다.
