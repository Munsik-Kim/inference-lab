# Case017 — 도구 호출 규칙과 업무 완주 비용

AI가 상품 합계를 계산해 저장하고, 창고 사이의 재고를 옮기며, 문서 정보를 바꾸는 과정을 시험했습니다. 출력 형식을 제한하거나 호출 오류 뒤 한 번 다시 시도하는 규칙이, 저장된 값·검증 결과·최종 답까지 맞게 완료하는 비율을 어떻게 바꾸는지 비교했습니다.

[요약](#overview) · [배경](#background) · [질문](#questions) · [원리·실행 계약](#contract) · [방법](#method) · [실험](#experiments) · [결과](#results) · [분석](#analysis) · [결론](#conclusion) · [재현·출처](#reproduce)

## <a id="overview"></a>요약

모델 가중치는 같은 Qwen3-4B BF16으로 고정했습니다. 모델이 앞선 내용을 처리하며 저장하는 중간 계산값을 KV 캐시라고 합니다. BF16 KV(K0)는 16비트 기준선, NVFP4 KV(K1)는 4비트 값과 배율(scale)을 함께 저장하는 구성입니다.

자유 출력(C0)은 형식을 프롬프트로 안내하고 첫 호출 규칙 위반에서 종료합니다. 문법 제약(G)은 매 응답에 공개 JSON schema를 적용합니다. 중립 재시도(N)와 오류 설명 재시도(E)는 첫 위반 뒤 각각 중립 안내 또는 공개 오류 코드·위치·기대 타입으로 한 번 다시 시도합니다. schema는 정답값이나 다음 행동을 알려주지 않습니다.

| KV 저장 | C0 | G | N | E |
| --- | --- | --- | --- | --- |
| BF16 KV (K0) | 16/24 | 16/24 | 16/24 | 16/24 |
| NVFP4 KV (K1) | 7/24 | 15/24 | 9/24 | 13/24 |

같은 BF16 가중치를 사용하는 NVFP4 KV 구성에서, 매 응답에 JSON 문법을 적용하자 신규 합성 24과제의 완주 수가 7/24에서 15/24로 늘었습니다. 자유 출력에서 실패한 8개가 회복됐고 손실은 0개였습니다. 실제 저장된 값이 맞은 과제는 7/24에서 22/24로 늘었지만, 수정 후 최신 검증 결과를 받지 않은 경우는 업무 완주로 세지 않았습니다. 완주 차이는 33.33pp이며, 탐색적 95% 구간은 [16.67, 50.00]pp입니다. pp는 완주율의 퍼센트포인트 차이입니다.

**확대 기준 이탈: 설정 점검용 DEV에서 BF16 KV+C0의 완주는 5/8로 원안의 6/8 기준에 미달했습니다. 당시 로컬 계획과 실행 코드가 이 성공 조건을 누락하고 완료·동결·봉인만 검사해 P2를 진행했습니다. 별도 면제 승인 기록은 확인하지 못했습니다. 이 경위를 공개하며, P2는 해당 기준을 통과한 확인시험이 아닌 신규 합성 과제의 탐색적 측정으로 제시합니다.** [PROTOCOL_DEVIATION.ko.md](PROTOCOL_DEVIATION.ko.md)

완주 수가 늘면서 전체 실행 비용도 늘었습니다. 실패를 포함한 24개 실행의 비용을 완주 수로 나누면 입력 토큰은 약 3.62%, 출력 토큰은 12.36%, 한 과제 실행 시간(episode wall)은 11.27% 증가했습니다.

## <a id="background"></a>배경

Case015는 출력 형식과 도구 이름의 실패를, Case016은 native KV 저장·수용량과 서빙 비용을 다뤘습니다. 이번에는 새 합성 작업에서 엄격한 실행 규칙과 KV 저장 구성의 완주·비용을 함께 측정했습니다. 앞선 선택 실패의 복구율이나 서빙 성능은 이번 분모에 옮기지 않습니다.

## <a id="questions"></a>질문

Q1: 출력 규칙을 바꾸면 실제 파일·레코드와 마지막 답까지 맞게 마치는 비율이 달라지는가? Q2: BF16 가중치를 유지한 BF16 KV와 native NVFP4 KV에서 결과와 토큰·호출·시간 비용은 어떻게 다른가? 성공 증가를 전제하지 않습니다.

## <a id="contract"></a>원리·실행 계약

모델 가중치는 같은 Qwen3-4B BF16으로 고정했습니다. 모델이 앞선 내용을 처리하며 저장하는 중간 계산값을 KV 캐시라고 합니다. BF16 KV(K0)는 16비트 기준선, NVFP4 KV(K1)는 4비트 값과 배율(scale)을 함께 저장하는 구성입니다.

자유 출력(C0)은 형식을 프롬프트로 안내하고 첫 호출 규칙 위반에서 종료합니다. 문법 제약(G)은 매 응답에 공개 JSON schema를 적용합니다. 중립 재시도(N)와 오류 설명 재시도(E)는 첫 위반 뒤 각각 중립 안내 또는 공개 오류 코드·위치·기대 타입으로 한 번 다시 시도합니다. schema는 정답값이나 다음 행동을 알려주지 않습니다.

모든 정책은 같은 chat endpoint·template·단일 JSON envelope·엄격 parser를 씁니다. whitespace 외 여분 문장·여러 object·중복 key·틀린 type·미등록 이름을 거절합니다. schema-valid 잘못된 값은 실제 fixture에 전달하며 hidden gold로 고치지 않습니다. validation 도구는 공개 불변식만 검사합니다.

G의 `uniqueItems`는 backend 미지원으로 사전 등록부터 host 검사 범위였습니다. N/E는 첫 계약 실패만 한 번 재시도하며 business 오류는 트리거가 아닙니다. 여섯 번째 행동이 final이면 허용하고, non-final이면 실행 후 TOOL_CALL_LIMIT로 끝나는 원 정책을 유지했습니다.

| KV | Bytes/token | Actual pool bytes | Token slots |
| --- | --- | --- | --- |
| K0 | 147456 | 4293918720 | 29120 |
| K1 | 41472 | 4294508544 | 103552 |

36층의 native packed storage/write/read 증거는 P0 원 로그에 보존했습니다. 슬롯 비율 약3.556배는 pool 수용량이며, 전체 VRAM 감소·단일100K 문맥·serving speedup을 뜻하지 않습니다.

## <a id="method"></a>방법

A: 여러 상품의 개수·한 개 가격·할인으로 정수 금액을 계산해 저장. B: 두 창고의 재고·revision을 읽고 원자적으로 이동. C: 문서 revision을 읽고 상태·태그·담당자 변경. DEV8(A3/B3/C2), 신규 HOLDOUT24(각8), P3용 COST6을 분리했습니다. HOLDOUT은 DEV와 다른 template·상태·ID·값을 쓰지만 실제 회사 모집단은 아닙니다.

V는 원 `VERIFIED_WORKFLOW_SUCCESS`, S는 `STATE_CORRECT`, F는 `FINAL_ANSWER_CORRECT`입니다. 수정 후 현재 상태의 성공 validation을 실제로 받은 뒤 final을 생성해야 합니다. 보호 필드·다른 record·관측 의존성·엄격한 최종 답도 검사합니다.

Fresh*는 보완 단계의 **사후 파생 지표**입니다. event의 대상·SUCCESS·실제 bool true, 최종 상태 digest/version, final 생성 전 available_result_event_ids를 대조했습니다. final이 없으면 마지막 실제 generation의 관측을 사용합니다. Fresh 자체는 요청의 정답을 보장하지 않습니다.

## <a id="experiments"></a>실험

Qwen/Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. BF16 weights, RTX 5080 16GB, vLLM0.32.0/FlashInfer/xgrammar, TP1, 공통 GRAPH, episode C1, prefix cache OFF. 실제 초기 길이는 DEV1726–1784/HOLDOUT1734–1798 tokens입니다.

512 output tokens/request, 8 model attempts, 6 parsed tool/final attempts, 4096 output tokens/episode, 1 retry, request180s/episode900s, client context24576/server28672. TF32=False, BF16 reduced precision reduction=True는 양쪽 공통입니다. 첫 입력 IDs는 대응하며 자유 rollout 뒤에는 이력이 갈라집니다.

P1/P2 COMPLETE는 실행 완료만 뜻합니다. HOLDOUT 생성·봉인·P2 freeze는 첫 기록 응답보다 앞섰고 공통35개 파일 hash는 동일합니다. 로컬 자기기록이며, 확대 기준 준수와는 별개입니다.

## <a id="results"></a>결과

### P1 DEV

| KV | C0 | G | N | E |
| --- | --- | --- | --- | --- |
| K0 | 5/8 | 5/8 | 5/8 | 5/8 |
| K1 | 2/8 | 5/8 | 2/8 | 4/8 |

### P2 HOLDOUT

| KV | Policy | V | S | F | Fresh* |
| --- | --- | --- | --- | --- | --- |
| K0 | C0 | 16 | 21 | 21 | 16 |
| K0 | G | 16 | 23 | 23 | 16 |
| K0 | N | 16 | 21 | 21 | 16 |
| K0 | E | 16 | 21 | 21 | 16 |
| K1 | C0 | 7 | 7 | 7 | 7 |
| K1 | G | 15 | 22 | 22 | 16 |
| K1 | N | 9 | 12 | 12 | 9 |
| K1 | E | 13 | 16 | 16 | 13 |

V/S/F/Fresh*: 셀당24개, 서로 다른 성공 정의입니다. Fresh는 사후 지표입니다.

| KV | Policy | A /8 | B /8 | C /8 |
| --- | --- | --- | --- | --- |
| K0 | C0 | 8 | 0 | 8 |
| K0 | G | 8 | 0 | 8 |
| K0 | N | 8 | 0 | 8 |
| K0 | E | 8 | 0 | 8 |
| K1 | C0 | 4 | 0 | 3 |
| K1 | G | 8 | 0 | 7 |
| K1 | N | 5 | 1 | 3 |
| K1 | E | 7 | 0 | 6 |

| 비교 | 회복 | 손실 | 차이 pp | 95% pp |
| --- | --- | --- | --- | --- |
| K0:G-C0 | 0 | 0 | 0.00 | [0.00, 0.00] |
| K0:N-C0 | 0 | 0 | 0.00 | [0.00, 0.00] |
| K0:E-C0 | 0 | 0 | 0.00 | [0.00, 0.00] |
| K0:E-N | 0 | 0 | 0.00 | [0.00, 0.00] |
| K1:G-C0 | 8 | 0 | 33.33 | [16.67, 50.00] |
| K1:N-C0 | 2 | 0 | 8.33 | [0.00, 20.83] |
| K1:E-C0 | 6 | 0 | 25.00 | [8.33, 41.67] |
| K1:E-N | 5 | 1 | 16.67 | [0.00, 33.33] |
| C0:K1-K0 | 0 | 9 | -37.50 | [-54.17, -20.83] |
| G:K1-K0 | 0 | 1 | -4.17 | [-12.50, 0.00] |
| N:K1-K0 | 1 | 8 | -29.17 | [-45.83, -12.50] |
| E:K1-K0 | 0 | 3 | -12.50 | [-25.00, 0.00] |

| KV | Policy | Requests | Input tokens | Output tokens | Episode wall (s) | Output tokens/verified job |
| --- | --- | --- | --- | --- | --- | --- |
| K0 | C0 | 102 | 201511 | 4996 | 77.864714401 | 312.250 |
| K0 | G | 110 | 217801 | 5399 | 84.562621448 | 337.438 |
| K0 | N | 104 | 205193 | 5139 | 80.317695542 | 321.188 |
| K0 | E | 108 | 213191 | 5546 | 85.310678150 | 346.625 |
| K1 | C0 | 51 | 98437 | 2470 | 39.188918025 | 352.857 |
| K1 | G | 110 | 218579 | 5947 | 93.437362778 | 396.467 |
| K1 | N | 88 | 171277 | 4355 | 68.265962043 | 483.889 |
| K1 | E | 102 | 200653 | 5001 | 79.399216725 | 384.692 |

완주 수가 늘면서 전체 실행 비용도 늘었습니다. 실패를 포함한 24개 실행의 비용을 완주 수로 나누면 입력 토큰은 약 3.62%, 출력 토큰은 12.36%, 한 과제 실행 시간(episode wall)은 11.27% 증가했습니다.

입력과 출력 토큰은 서로 같은 계산 비용이 아닙니다. 위 비율은 성공한 episode만의 평균 latency가 아니라 실패까지 포함한 전체 비용/완주 수입니다. P3 처리량·과금·전력 비용으로 해석하지 않습니다.

P0/P1/P2의 요청은22/233/775(총1030), 출력은678/11054/38853(총50585)입니다. P0는12개 probe와2개 smoke의10요청입니다. 모두 실제 응답을 받았으며 인프라 재시도는0회입니다. Worker wall은164.583118317/216.100527464/654.464649490초, 합계1035.148295271초입니다. 서버 로드·컴파일·종료·I/O를 포함하며 episode wall 합계와 다릅니다.

## <a id="analysis"></a>분석

G의 창고 실패16회는 고유7과제×KV2의 최신 검증 누락14회와 B05×KV2의 SKU 접두사 누락·이동 미실행2회입니다. 전부 같은 오류가 아닙니다. `Inspect validation after the intended update.`와 `An older validation is stale if a subsequent action changed the state.`는 실제 system prompt에 있었습니다([원 소스](src/agent_contract_cost/prompts.py)).

A00_K1_C0는 여분 `}`로 첫 조회부터 거절됐고 G는 계산·저장·검증·final까지 완료했습니다. B01_K0_G는 올바르게 이동했지만 이동 전 검증만 관측했습니다. C00_K1_G는 review 요청에 published를 저장해 Fresh1/S0/F0/V0입니다. B03_K1_N은 중립 재시도 뒤 이동→검증→final로 유일한 NVFP4 N 창고 완주를 기록했습니다. 같은 task의 G/E는 이동 전에 검증했습니다. N/E의 최초 입력·실패 raw는 같았지만 이후 관측 이력은 달랐습니다. 한 사례로 오류 설명만의 인과효과를 결론내리지 않습니다.

원 INCORRECT_FINAL은 최종 숫자 오류뿐 아니라 최종 생성 때 관측 계약 미충족도 포함합니다. 보조 `terminal_budget_or_infrastructure` 라벨은 실제 예산 소진의 증거로 쓰지 않습니다. 별도 종료 독립 감사의 점수 변경은0/256입니다.

구간은 원 정의대로 family별 task의8조건을 함께 유지한 paired bootstrap10000회/seed6102013입니다. 합성 메뉴의 탐색적 pointwise95% 구간이며, 일반 업무 모집단·비열등성·동등성 검증이 아닙니다. NVFP4 E−N의 구간은0을 포함합니다.

## <a id="conclusion"></a>결론

이 조건에서 문법 제약은 NVFP4의 실제 완주를 늘렸지만 상태값 의미와 수정 후 검증 계획을 보장하지 못했습니다. BF16의 완주 수는 같아도 상태 정확 수는21→23으로 늘었습니다. 추가 토큰·episode 시간 비용과 확대 기준 누락을 함께 공개합니다.

범위: 한 모델·한 장비·짧은 C1 합성 작업의 serving 구성 비교입니다. 다른 모델·실제 업무·전체 품질 동등성·전체 VRAM 감소·속도 우위·P3 동시 처리량은 미평가입니다. P3/새 원인 실험은 NOT_RUN입니다.

## <a id="reproduce"></a>재현·출처

[REPORT.ko.md](REPORT.ko.md) · [REPRODUCE.ko.md](REPRODUCE.ko.md) · [실행 코드](src/agent_contract_cost/runner.py) · [전체 ledger](publication/summary/episode_ledger.csv)

공개 층은 256개 전체 원응답·SSE·tokens·events를 손실 없이 압축 보관하고, 새 디렉터리에 원 상대경로·바이트를 복원해 frozen verifier를 실행합니다. 원 scorer/parser/protocol의 hash를 새 문서에 맞춰 바꾸지 않습니다. 공개 doc과 source/build identity는 실행 source SHA와 구분합니다.

Codex 지원으로 구현·문서화했습니다. 별도 CPU 계산은 AI 보조 감사이며 사람의 독립 심사·외부 GPU 재현·저자 승인이 아닙니다. ChatGPT의 보완 CSV 검토는256행 스칼라·paired 산술 범위이고, 전체 fixture 재생은 로컬 CPU receipt의 범위입니다.

[NOTICE](NOTICE.md) · [REUSE_MAP](REUSE_MAP.md) · [Frozen protocol](specs/protocol.json) · [Execution freeze](provenance/execution_freeze_p2.json) · [Preservation map](publication/provenance/original_mapping.json)
