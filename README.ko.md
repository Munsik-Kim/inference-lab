# DIOVA · 모델 실행과 변경 후 검증

[English](README.md) | 한국어

**Munsik Kim** · 모델을 실행하고, 바뀐 답변과 실패 원인을 확인하는 도구를 만듭니다. 작은 모델의 저장·재시작, 같은 질문의 결과 비교, 실제 작업 완료 여부를 코드와 검산 기록으로 연결합니다.

[대표작과 결과 보기](https://munsik-kim.github.io/inference-lab/ko/index.html#featured-stories) · [모델 없이 작은 예제 실행](https://munsik-kim.github.io/inference-lab/ko/compare.html#run)

## 목적에 맞게 시작하기

| 확인하고 싶은 것 | 다음 단계 |
|---|---|
| 기술 역량과 직접 만든 부분 | [설계 판단·코드·테스트](docs/ko/PORTFOLIO.md) |
| 내 두 답변 파일의 변화 | [diova-compare 입력 예시와 실행](packages/diova-compare/examples/compare/README.ko.md) |
| 연구의 결과와 적용 범위 | [전체 사례와 원자료](docs/ko/CASEBOOK.md), [공개 연구 지도](https://munsik-kim.github.io/inference-lab/ko/index.html#research) |

## 대표작 A · B · C

<!-- claims: c008-tracks c008-artifact-implementation -->
| 작업 | 직접 구현한 것과 확인할 근거 |
|---|---|
| A. 작은 모델을 다시 실행 | GPTQ 변환 연결·파일 검사·새 프로세스 재로딩: [Case008 Q](https://munsik-kim.github.io/inference-lab/ko/case008.html#track-q) → [Case009 요청 비용·품질](https://munsik-kim.github.io/inference-lab/ko/case009.html#serving). 파일 크기, GPU 메모리, 정답 변화는 다른 측정입니다. |
| B. 평균 뒤에 숨은 답변 변화 | [diova-compare](https://munsik-kim.github.io/inference-lab/ko/compare.html)의 입력 짝 검사와 질문별 정답 손실·회복 보고서. 같은 2/4 정답도 잃은 답 1개와 새 정답 1개가 다릅니다. 작은 합성 예제의 결과입니다. |
| C. 완료 선언과 실제 상태 | [Case017 보고서](cases/017-agent-contract-cost-bench/REPORT.ko.md) · [CPU 재생](cases/017-agent-contract-cost-bench/REPRODUCE.ko.md) · [앞선 구현](research/agent_tool_contract_repair_20261014/README.ko.md). [Case018](cases/018-agent-state-recovery/README.ko.md)은 상태·후속 절차와 정확한 최종 보고를 분리합니다. Case017은 기준 누락을 공개한 탐색적 결과이며 추가 비용도 남겼습니다. |

최근 **Case020**은 작은 상주량이 빠른 호출을 뜻하는지, **Case021**은 같은 압축 표현의 실행 차이가 어디서 생기는지 조사했습니다. Case021의 전체 평균 KL 감소 여부는 **INCONCLUSIVE**, 즉 이번 자료로 확정하지 못했습니다. [최근 진단 읽기](https://munsik-kim.github.io/inference-lab/ko/index.html#latest-work).

## 바로 사용할 도구

- **diova-compare:** 이미 실행·채점한 두 JSONL을 CPU로 비교합니다. `validate → compare` 뒤 `report.md`부터 읽고 `pairs.csv`에서 질문별 변화를 확인하세요. 모델 실행이나 자유 답변의 정답 생성은 하지 않습니다.
- **[inference-engineering-kit](https://github.com/Munsik-Kim/inference-engineering-kit):** 지원 과제와 외부 정답 계약을 갖춘 설치형 평가 도구입니다. 기본 CPU 경로와 별도 모델/HTTP 실행을 구분하며 비교 기능에 diova-compare를 재사용합니다. [공개 v0.2.1](https://github.com/Munsik-Kim/inference-engineering-kit/releases/tag/v0.2.1)과 최신 source의 범위는 해당 안내에서 확인하세요.

## 전체 사례

21개 사례는 서로 다른 모델·질문·조건의 기록입니다. 번호는 성능 순위가 아닙니다. [사이트에서 검색·주제별 찾기](https://munsik-kim.github.io/inference-lab/ko/index.html#cases).

<a id="capabilities"></a>
<a id="case-006--모델-변경을-질문별로-추적하는-평가-도구"></a>
<a id="case-007--채널-선택부터-실제-행렬-축소까지"></a>
<a id="case-008--모델을-변환하고-복구하고-다시-실행하기"></a>
<a id="case-010--저정밀-state의-저장재시작과-기억-수명"></a>
<a id="case-011--같은-상태-다른-판독기"></a>
<a id="case-012--이미지-생성-시간-어디에-계산을-더-쓸까"></a>
<a id="case-013--가중치-오차가-줄면-실제-답변도-좋아질까"></a>
<a id="case015--같은-도구-다른-이름"></a>
<a id="case016--같은-kv-메모리에서-수용량과-요청-비용-검증"></a>
<a id="case017--도구-호출-규칙과-업무-완주-비용"></a>
<a id="case018--재고-복구와-완료-보고를-따로-검사했습니다"></a>
<a id="case020--임베딩을-작게-보관하면-실행도-빨라질까"></a>
<a id="case021--같은-int4인데-왜-계산-결과가-달라질까"></a>
<a id="diova--모델-압축실행결과-검사"></a>
<a id="gpu-추론-성능과-답변-비교"></a>
<a id="projects"></a>
<a id="q--gptq-변환저장새-runtime-실행"></a>
<a id="r--구조가-달라진-모델의-저장재로딩과-가중치-보정"></a>
<a id="tech-stack"></a>
<a id="구현-기능"></a>
<a id="구현-상세와-연구-기록"></a>
<a id="구현사용-기술의-근거"></a>
<a id="기술스택별-작업"></a>
<a id="기여와-출처"></a>
<a id="더-깊게-보기"></a>
<a id="모델을-압축하고-저장하고-다시-실행합니다"></a>
<a id="모델을-작게-저장하고-실행하고-달라진-답변까지-확인합니다"></a>
<a id="새-측정--긴-decode와-동시-요청"></a>
<a id="연구수치-엔지니어링--case014"></a>
<a id="재계산-도구와-결과-탐색기"></a>
<a id="직접-구현한-부분과-재사용한-기술"></a>
<a id="직접-사용재현"></a>
<a id="질문과-구현물"></a>
<a id="확인된-저장-공간실행-메모리"></a>
<a id="후속-구현--도구-이름-오류와-실제-작업-성공"></a>
<!-- claims: c007-implementation c006-implementation c006-readout c010-state-bytes -->
| 쉬운 질문 | 관측과 구현 범위 |
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
| [018 — 상태 복구와 최종 보고는 함께 맞을까?](docs/ko/CASEBOOK.md#case-018) | 상태·후속 절차 적합 Q와 정확 보고까지 요구하는 W를 분리합니다. |
| [019 — 손실은 그대로인데 학습은 진행될까?](docs/ko/CASEBOOK.md#case-019) | 방향 정렬·크기·손실의 관계를 기존 이론의 부분 재현과 자체 대조로 검사합니다. |
| [020 — 작게 보관하면 실행도 빨라질까?](docs/ko/CASEBOOK.md#case-020) | 합성 임베딩의 상주량·호출·실제 반복 사용 비용을 구분합니다. |
| [021 — 같은 INT4인데 왜 값이 달라질까?](docs/ko/CASEBOOK.md#case-021) | 단일항은 같고 결합에서 차이. 중간 정밀도 증가의 전체 효과는 INCONCLUSIVE입니다. |

## 안내 문서와 출처

[처음 보는 분](docs/ko/START_HERE.md)은 목적별 길잡이, [PORTFOLIO](docs/ko/PORTFOLIO.md)는 기여·판단·코드 근거, [CASEBOOK](docs/ko/CASEBOOK.md)은 전체 색인입니다. [CPU 검산과 GPU 재실행](docs/ko/GETTING_STARTED.md), [모델별 실행 안내](docs/ko/LLM_GUIDE.md), [용어집](docs/ko/GLOSSARY.md)은 필요할 때 이어서 읽으세요.

DIOVA는 Deep-learning Inference Optimization, Validation & Analysis입니다. 공개 모델·논문 방법·런타임을 재사용하며, 직접 작성한 연결·평가·검산 코드와 구분합니다. OpenAI Codex가 구현·분석·문서화를 지원했습니다. [기여·출처](docs/ko/PORTFOLIO.md#contribution-and-reuse) · [Apache-2.0](LICENSE) · [선행연구](docs/related-work/README.ko.md). 사례별 upstream 조건과 실패·미실행 기록을 유지합니다.
