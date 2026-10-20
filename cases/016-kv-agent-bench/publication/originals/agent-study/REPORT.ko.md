# Case016 — 작은 KV 캐시로 더 많은 업무를 정확히 끝낼 수 있을까?

실제 압축 KV를 사용하는 vLLM과 장문 업무 실행·감사 도구를 구현했습니다. 고정한 DEV 96개 조건에서 B16은 0/48, NVFP4는 6/48의 검증된 업무 성공을 기록했습니다. 같은 12개 과제를 메뉴·응답 한도 4개 조건으로 반복한 결과입니다. 기준선 확대 gate는 NOT_MET입니다. 이번 범위 밖의 P2/P3는 미실행입니다.

[한눈에 보기](#overview) · [실제 업무](#business) · [KV와 기준선](#cache) · [실험](#experiment) · [성공과 비용](#results) · [실패 사건](#events) · [직접 실행](#reproduce) · [상세 출처](#sources)

<a id="overview"></a>

## 한눈에 보기

이번 질문은 작은 KV가 장문 업무 완주에 어떤 영향을 주는지입니다. 품질이 좋아야 실험 완료인 것은 아닙니다. 모든 96개 조건은 관측 완료이며, 원기록과 실패를 유지합니다. 후속48개 EVAL은 생성·CPU 정답경로 검사만 했고 모델로 열지 않았습니다.

<a id="business"></a>

## 실제 업무

네 종류는 구매·견적, 최신 버전, 지연 도착, 수정·취소입니다. 여러 자료에서 규칙에 맞는 자료를 고르고, 해당 자료 확인 → 금액 계산 → 기록 수정 → 최신 상태 검증 → 최종 JSON 순서로 진행합니다. 정답 경로는 도구4회와 생성5회로 예산 안에 있습니다.

정수 최소단위는 소수점 계산을 피하기 위한 금액 표현입니다. `quantity`는 개수, `unit_price_minor`는 한 개 가격입니다. 할인은 정수비율이고 금액은 `floor((quantity*price*(100-discount)+50)/100)`입니다. UTC cutoff·버전·취소 우선 규칙은 초기 자료에 공개됩니다.

선택 규칙, 수량·할인, 후보 자료를8192token 입력의3600/5500/7350 위치에 둡니다. 도구8/32쌍은 길이와 근거 위치가 같습니다. 무관한 합성 padding으로 위치를 통제한 실험이며 자연 문서 모집단의 대표성은 측정하지 않았습니다. 초기 사실은 처음부터 사용 가능하고 명시한 자료 확인 외에 숨은 재조회 의무는 없습니다.

<a id="cache"></a>

## KV와 기준선

| KV | storage | actual pool B | capacity tokens | layers | max context |
|---|---|---|---|---|---|
| B16 | BF16 | 4,293,918,720 | 29,120 | 36 | 28672 |
| F8 | FP8 E4M3 | 4,293,918,720 | 58,240 | 36 | 28672 |
| F4 | NVFP4 packed + FP8 block scales | 4,294,508,544 | 103,552 | 36 | 28672 |

목표 pool은 모두4,294,967,296B입니다. 실제 page 정렬 후의 저장량을 표에 기록했습니다. 같은 storage의 view를 중복 합산하지 않습니다. B16은147,456B/token, FP8은73,728B/token, NVFP4는 packed36,864+block-scale4,608=41,472B/token입니다. 저장토큰 용량은 처리량이 아닙니다. F8은 global scale1.0이 요청 전후 동일한 F8_DEFAULT_SCALE이며 P0에만 사용했습니다.

NVFP4는 실제 uint8 packed data와 FP8 E4M3 block scale을 사용했습니다. GPU trace에 NVFP4 writer, FP4 cache를 읽는 FA2 prefill과 decode kernel이 있습니다. query/output이 BF16인 것은 BF16 cache fallback이 아닙니다. Kernel 이름은 `results/p0_diagnostics_r2/F4_kernel_profile.json`에 있습니다.

<a id="experiment"></a>

## 실험

P0는 KV별8개 SMOKE를 기술적으로 실행했습니다. 작은 NVFP4 검사에서 같은 압축 cache를 두 decoder로 읽은 차이는0이었고 BF16 원본 대비 상대 norm 오차는 약0.095였습니다. 별도 공식 test body는4layout PASS, 전체 upstream pytest suite는 NOT_RUN입니다. 실행 전 Case016 CPU 경계검사 34종을 실행했습니다. 실행 후 자료의 지연 도착·취소·ID/순서 성질을 확인하는 CPU 검토 3종도 별도로 통과했습니다.

| KV | SMOKE observed | verified workflow |
|---|---|---|
| B16 | 8 | 1 |
| F8 | 8 | 1 |
| F4 | 8 | 0 |

P1는 DEV12×KV2×메뉴2×cap2=96입니다. BF16 weights, TP1, FlashInfer, eager, async scheduling OFF, prefix caching OFF, chunked prefill2048, max-num-seqs4, context28672, clientC1을 공통 적용했습니다. generation 총8192, 요청8, 도구시도6, 요청180초, episode900초, temperature0, EOS준수입니다. 모델 결과를 본 뒤 과제·도구·예산을 변경하지 않았습니다. 환경·계측 오류와 수정은 amendment01~08로 구분합니다.

![All eight P1 cells](figures/p1_success.svg)

매 턴 전체 대화를 다시 보내며 prefix cache가 꺼져 있어 prefill이 반복됩니다. 각 HTTP 요청 내부에서만 autoregressive KV를 사용하고 종료 후 반납합니다. 한 episode 전체가 KV를 영구 점유하는 모형이 아닙니다.

<a id="results"></a>

## 성공과 비용

| KV | menu | cap | verified/12 | state/12 | final/12 | false done |
|---|---|---|---|---|---|---|
| B16 | 8 | 256 | 0/12 | 2/12 | 0/12 | 4 |
| B16 | 8 | 1024 | 0/12 | 2/12 | 0/12 | 4 |
| B16 | 32 | 256 | 0/12 | 4/12 | 0/12 | 2 |
| B16 | 32 | 1024 | 0/12 | 4/12 | 0/12 | 2 |
| F4 | 8 | 256 | 0/12 | 2/12 | 0/12 | 3 |
| F4 | 8 | 1024 | 0/12 | 2/12 | 0/12 | 3 |
| F4 | 32 | 256 | 3/12 | 5/12 | 5/12 | 1 |
| F4 | 32 | 1024 | 3/12 | 5/12 | 5/12 | 1 |

| comparison | both success | loss | recovery | both fail |
|---|---|---|---|---|
| B16/menu8: cap256 to1024 | 0 | 0 | 0 | 12 |
| B16/menu32: cap256 to1024 | 0 | 0 | 0 | 12 |
| F4/menu8: cap256 to1024 | 0 | 0 | 0 | 12 |
| F4/menu32: cap256 to1024 | 3 | 0 | 0 | 9 |
| menu8/cap256: B16 toF4 | 0 | 0 | 0 | 12 |
| menu8/cap1024: B16 toF4 | 0 | 0 | 0 | 12 |
| menu32/cap256: B16 toF4 | 0 | 0 | 3 | 9 |
| menu32/cap1024: B16 toF4 | 0 | 0 | 3 | 9 |

Baseline usable: **NOT_MET**; B16/menu8/cap1024 = **0/12**, families {'A': 0, 'B': 0, 'C': 0, 'D': 0}. 등록된9/12 및 각family2/3 기준은 후속 실행범위 관리 제안입니다. 기준선의 낮은 완주율 때문에 이 DEV의 paired 변화만으로 실용 우위나 동등성·비열등성을 판단하지 않습니다.

| KV | model requests | tool attempts | events | output tokens | worker attempt s | peak allocated GiB | peak reserved GiB |
|---|---|---|---|---|---|---|---|
| B16 | 266 | 284 | 284 | 13164 | 513.292 | 12.601 | 12.688 |
| F4 | 212 | 260 | 251 | 11916 | 568.285 | 12.601 | 12.688 |

P1 worker clock: **1084.509s**. 로드·실패·I/O·계측을 포함하며 GPU busy나 serving speedup이 아닙니다. C1 혼합 DEV block의 makespan/correct-job 장부는 원 summary의 attempts에 있습니다. 성공0의 seconds/tokens-per-correct는 null입니다. 취소 probe의 수신 token은 기록됐지만 서버가 중단 전 만든 전체 token 수는 정확히 관측하지 못했습니다.

<a id="events"></a>

## 실패 사건

| terminal reason | episodes |
|---|---|
| TOOL_CALL_LIMIT | 62 |
| FINAL | 30 |
| PARSE_STOP | 4 |

| first unmet obligation | episodes |
|---|---|
| validation | 20 |
| write | 6 |
| source | 48 |
| calculation | 16 |
| none | 6 |

도구시도는 예산 내에서 호스트가 처리한 slot 수입니다. 생성됐지만 한도 때문에 실행에 진입하지 못한 slot은 원응답에 남습니다. 사건 수에는 registry·인자 거절도 포함하며, 실제 callable 성공 수는 diagnostics에서 따로 확인합니다.

도구결과 valid:true는 공개 schema·불변식 통과이며 정답 금액이나 올바른 버전 선택을 보장하지 않습니다. B16 DEV-B00/menu8/cap1024는 필요한 버전3 대신2를 선택해348142를 저장했고 valid:true를 받았습니다. 독립 정답은253559이며 실패 판정은 유지됩니다. 이 사례는 전체 raw로 연결되며 다른 조건을 대표하는 유일한 사례로 선택한 것이 아닙니다.

같은 response 안의 모든 call은 같은 생성 전 관측 목록을 가집니다. write→validate는 순차 실행을 허용하지만, 조회결과·계산결과가 필요한 다음 행동은 그 결과를 받은 뒤 생성되어야 합니다. first-unmet는 공개 DAG의 정해진 순서이고 모델 내부의 유일한 원인 진단이 아닙니다. 자유 rollout의 후속 대화가 다르면 동일 관측이라고 부르지 않습니다.

[전체96개 raw](results/p1_r1/episodes/) · [가족별·파싱·길이 진단](results/p1_r1/diagnostics.json) · [paired CSV](results/p1_r1/paired.csv)

<a id="reproduce"></a>

## 직접 실행

[실행 명령과 CPU 감사 범위](REPRODUCE.ko.md). 원 SSE를 재조립하고 새 fixture에서 상태·관측·endpoint를 다시 계산합니다. scalar audit는 모델 추론 자체를 재현하지 않습니다. 입력과 원 모델을 재다운로드하거나 quantizer를 실행하지 않습니다.

<a id="sources"></a>

## 상세 출처

Model revision: `cdbee75f17c01a7cc42f958dc650907174af0554`. P1 execution source SHA256: `ab8463e9bfde41564e44a863bc1159dfecf26cada54a7fdceea5ebec127e53ee`. Git base: `0c86afa4728cbf901f6c3df64802810db6581a00`.

| scope | status |
|---|---|
| vLLM/NVFP4 storage + 36 layers | observed |
| P1 DEV96 | completed |
| EVAL48 /16K /C2,C4 /P2,P3 | NOT_RUN |
| weight W4 /D16 /training | NOT_RUN |
| Git stage/commit/push/PR/Pages | NOT_RUN |
| full model logits independently checked finite | NOT_RUN; small cache decode finite checked |

P2/P3를 시작하려면 기준선이 왜 실패했는지 확인하고 별도 승인·새 버전·새 동결 계획이 필요합니다. 이번 결과를 고치기 위한 즉석 prompt/도구/한도 sweep은 하지 않았습니다.

[vLLM0.32 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0), [NVFP4 SM8x/SM12x support](https://github.com/vllm-project/vllm/pull/46963), [SM12x XQA decode](https://github.com/vllm-project/vllm/pull/60452), [Qwen function calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html). [Source audit](provenance/source_audit.json), [reuse map](REUSE_MAP.md), [NOTICE](NOTICE.md).

설치한 공식 engine/kernel/model과 자체 구현한 fixture·client·실행·채점·감사를 구분합니다. Codex 보조와 별도 scalar 계산 검토를 사용했으며 사람의 독립 심사나 외부 GPU 재현으로 표시하지 않습니다.
