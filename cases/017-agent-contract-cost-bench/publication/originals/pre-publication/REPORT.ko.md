# Case017 — 도구 호출 규칙을 지키면 AI가 일을 더 정확히 끝낼까?

가상 기록을 읽고 계산하거나 수정한 뒤, 실제 저장된 상태와 마지막 답이 모두 맞는지 시험했습니다. 상품별 개수와 한 개 가격으로 합계를 구하는 작업, 창고 사이에 물건을 옮기는 작업, 문서의 상태·태그·담당자를 바꾸는 작업입니다. 기본 모델 가중치는 그대로 두고, 출력 규칙 네 가지와 대화 기억(KV cache)의 저장 정밀도 두 가지를 비교했습니다.

[한눈에 보기](#overview) · [배경과 질문](#question) · [구현 계약](#implementation) · [방법·데이터](#method) · [결과](#results) · [분석](#analysis) · [결론](#conclusion) · [재현·출처](#reproduce)

## <a id="overview"></a>한눈에 보기

| KV | C0 FREE | G GRAMMAR | N NEUTRAL | E ERROR |
| --- | --- | --- | --- | --- |
| K0 | 16/24 | 16/24 | 16/24 | 16/24 |
| K1 | 7/24 | 15/24 | 9/24 | 13/24 |

표는 새 HOLDOUT 24개 과제에서의 실제 업무 완주 수입니다. 파일 값만 맞거나 JSON만 유효한 것은 성공으로 세지 않습니다.

NVFP4에서 G는 C0보다 완주 8개를 회복했고 손실은 0개였습니다. 차이는 33.33 percentage points이며 탐색 구간은 [16.67, 50.00]pp입니다. BF16에서는 네 정책의 완주수가 같았습니다. NVFP4의 E-N은 회복 5개·손실 1개이며 구간은0을 포함합니다.

P0: PASS; total actual model requests: **1030**; output tokens: **50585**; cumulative worker wall: **1035.148s**.

## <a id="question"></a>배경과 질문

Case015의 선택된 실패 복구와 Case016의 KV 저장 검증은 동기입니다. 이번 결과의 분모에 옮기지 않았습니다. Q1은 출력 규칙이 실제 완주와 비용을 바꾸는지, Q2는 같은 BF16 가중치에서 native NVFP4 KV로 바꿀 때 무엇이 달라지는지입니다.

## <a id="implementation"></a>기술 원리와 구현 계약

직접 구현한 것은 엄격 parser, 정책·예산 실행기, 세 가지 안전 fixture, 관측 이력, 독립 endpoint 감사, 비교 도구입니다. vLLM·FlashInfer·xgrammar의 엔진과 커널은 upstream 구현입니다. KV cache는 한 요청 안에서 앞선 토큰의 key/value를 저장하며, 매 턴 전체 history를 재전송하고 prefix caching은 끕니다.

| Policy | Contract |
| --- | --- |
| C0 | Free generation; first contract rejection terminates |
| G | Public branch-specific JSON schema on every response; no retry |
| N | Free generation; one neutral retry after first contract rejection |
| E | Free generation; one retry with public error code/path/type |

모두 동일 endpoint·template·단일 JSON envelope·parser를 사용합니다. G에는 정답 ID·금액·행동 순서를 넣지 않습니다. `uniqueItems`는 설치 backend가 지원하지 않아 모든 정책의 host 검사로만 적용합니다. 여섯 번째 action이 final이면 허용하며, 다른 도구이면 실행 후 종료합니다. final도 예산을 쓰므로 일곱 번째 생성은 없습니다.

| KV | Actual storage | Bytes/token | Actual pool bytes | Token slots |
| --- | --- | --- | --- | --- |
| K0 | BF16 | 147456 | 4293918720 | 29120 |
| K1 | packed uint8 + FP8 block scales | 41472 | 4294508544 | 103552 |

이 비율은 같은 약 4GiB KV pool의 token 슬롯 비율입니다. 전체 VRAM 감소나 단일 100K context 지원이 아닙니다.

## <a id="method"></a>방법·실험

Qwen/Qwen3-4B-Instruct-2507; revision `cdbee75f17c01a7cc42f958dc650907174af0554`; BF16 weights; vLLM 0.32.0; FlashInfer; xgrammar; TP1; GRAPH; episode concurrency1.

Actual numerical settings: TF32=False; BF16 reduced-precision reduction=True, identical across KV configurations.

DEV8(A3/B3/C2)×8조건=64개, 이후 동결된 생성기로 다른 template·상태의 HOLDOUT24(A/B/C 각8)×8조건=192개를 생성·봉인했습니다. COST6은 CPU 정답 경로만 준비했고 P3는 미실행입니다. 초기 입력은 약 2K 목표이며 실제 길이는 data/*_model_inputs.json에 있습니다. 같은 task의 최초 token IDs는 KV·정책 사이에 일치해야 하며 이후 자유 이력은 달라질 수 있습니다.

Limits: 512 tokens/request; 8 model attempts; 6 parsed tool/final attempts; 4096 output tokens; one retry; request180s/episode900s; client context24576; server context28672.

비용표의 출력 token은 반환된 원 token IDs의 개수이며 usage와 대조했습니다. EOS 등 제어 token도 원 기록에 보존합니다. episode wall에는 로컬 tokenization·도구·통신·실패가 포함됩니다. worker wall에는 서버 로드·컴파일·종료도 포함하며, warmup forward를 별도 API 생성 요청으로 세지 않습니다.

각 generation 직전의 관측 사건 ID를 고정합니다. 정답 경로는 read 결과를 받은 뒤 계산·수정하고, 최신 validation을 받은 뒤 final을 생성해야 합니다. validation은 공개 불변식만 검사하며 hidden gold를 알려주지 않습니다. 최종 정답은 별도 CPU oracle가 판정합니다.

## <a id="results"></a>실험 결과

### P1 — COMPLETE

| KV | C0 FREE | G GRAMMAR | N NEUTRAL | E ERROR |
| --- | --- | --- | --- | --- |
| K0 | 5/8 | 5/8 | 5/8 | 5/8 |
| K1 | 2/8 | 5/8 | 2/8 | 4/8 |

| KV | Policy | Requests | Output tokens | Retries | Episode wall sum (s) | Tokens / verified job |
| --- | --- | --- | --- | --- | --- | --- |
| K0 | C0 | 26 | 1184 | 0/0 | 18.752 | 236.8 |
| K0 | G | 36 | 1648 | 0/0 | 26.561 | 329.6 |
| K0 | N | 29 | 1387 | 3/3 | 21.630 | 277.4 |
| K0 | E | 35 | 1850 | 3/3 | 28.408 | 370.0 |
| K1 | C0 | 15 | 710 | 0/0 | 11.230 | 355.0 |
| K1 | G | 36 | 1732 | 0/0 | 27.951 | 346.4 |
| K1 | N | 23 | 1042 | 6/6 | 16.538 | 521.0 |
| K1 | E | 33 | 1501 | 6/6 | 24.464 | 375.2 |

![P1 verified success](figures/p1_success.svg)

| KV | Policy | A: totals | B: transfer | C: metadata |
| --- | --- | --- | --- | --- |
| K0 | C0 | 3/3 | 0/3 | 2/2 |
| K0 | G | 3/3 | 0/3 | 2/2 |
| K0 | N | 3/3 | 0/3 | 2/2 |
| K0 | E | 3/3 | 0/3 | 2/2 |
| K1 | C0 | 1/3 | 0/3 | 1/2 |
| K1 | G | 3/3 | 0/3 | 2/2 |
| K1 | N | 1/3 | 0/3 | 1/2 |
| K1 | E | 3/3 | 0/3 | 1/2 |

| Comparison | Pairs | Both success | Loss | Recovery | Both fail | Difference (pp) | 95% interval (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K0:G-C0 | 8 | 5 | 0 | 0 | 3 | 0.00 | [0.00, 0.00] |
| K0:N-C0 | 8 | 5 | 0 | 0 | 3 | 0.00 | [0.00, 0.00] |
| K0:E-C0 | 8 | 5 | 0 | 0 | 3 | 0.00 | [0.00, 0.00] |
| K0:E-N | 8 | 5 | 0 | 0 | 3 | 0.00 | [0.00, 0.00] |
| K1:G-C0 | 8 | 2 | 0 | 3 | 3 | 37.50 | [12.50, 62.50] |
| K1:N-C0 | 8 | 2 | 0 | 0 | 6 | 0.00 | [0.00, 0.00] |
| K1:E-C0 | 8 | 2 | 0 | 2 | 4 | 25.00 | [0.00, 37.50] |
| K1:E-N | 8 | 2 | 0 | 2 | 4 | 25.00 | [0.00, 37.50] |
| C0:K1-K0 | 8 | 2 | 3 | 0 | 3 | -37.50 | [-62.50, -12.50] |
| G:K1-K0 | 8 | 5 | 0 | 0 | 3 | 0.00 | [0.00, 0.00] |
| N:K1-K0 | 8 | 2 | 3 | 0 | 3 | -37.50 | [-62.50, -12.50] |
| E:K1-K0 | 8 | 4 | 1 | 0 | 3 | -12.50 | [-25.00, 0.00] |

### P2 — COMPLETE

| KV | C0 FREE | G GRAMMAR | N NEUTRAL | E ERROR |
| --- | --- | --- | --- | --- |
| K0 | 16/24 | 16/24 | 16/24 | 16/24 |
| K1 | 7/24 | 15/24 | 9/24 | 13/24 |

| KV | Policy | Requests | Output tokens | Retries | Episode wall sum (s) | Tokens / verified job |
| --- | --- | --- | --- | --- | --- | --- |
| K0 | C0 | 102 | 4996 | 0/0 | 77.865 | 312.2 |
| K0 | G | 110 | 5399 | 0/0 | 84.563 | 337.4 |
| K0 | N | 104 | 5139 | 2/2 | 80.318 | 321.2 |
| K0 | E | 108 | 5546 | 2/2 | 85.311 | 346.6 |
| K1 | C0 | 51 | 2470 | 0/0 | 39.189 | 352.9 |
| K1 | G | 110 | 5947 | 0/0 | 93.437 | 396.5 |
| K1 | N | 88 | 4355 | 16/16 | 68.266 | 483.9 |
| K1 | E | 102 | 5001 | 16/16 | 79.399 | 384.7 |

![P2 verified success](figures/p2_success.svg)

| KV | Policy | A: totals | B: transfer | C: metadata |
| --- | --- | --- | --- | --- |
| K0 | C0 | 8/8 | 0/8 | 8/8 |
| K0 | G | 8/8 | 0/8 | 8/8 |
| K0 | N | 8/8 | 0/8 | 8/8 |
| K0 | E | 8/8 | 0/8 | 8/8 |
| K1 | C0 | 4/8 | 0/8 | 3/8 |
| K1 | G | 8/8 | 0/8 | 7/8 |
| K1 | N | 5/8 | 1/8 | 3/8 |
| K1 | E | 7/8 | 0/8 | 6/8 |

| Comparison | Pairs | Both success | Loss | Recovery | Both fail | Difference (pp) | 95% interval (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K0:G-C0 | 24 | 16 | 0 | 0 | 8 | 0.00 | [0.00, 0.00] |
| K0:N-C0 | 24 | 16 | 0 | 0 | 8 | 0.00 | [0.00, 0.00] |
| K0:E-C0 | 24 | 16 | 0 | 0 | 8 | 0.00 | [0.00, 0.00] |
| K0:E-N | 24 | 16 | 0 | 0 | 8 | 0.00 | [0.00, 0.00] |
| K1:G-C0 | 24 | 7 | 0 | 8 | 9 | 33.33 | [16.67, 50.00] |
| K1:N-C0 | 24 | 7 | 0 | 2 | 15 | 8.33 | [0.00, 20.83] |
| K1:E-C0 | 24 | 7 | 0 | 6 | 11 | 25.00 | [8.33, 41.67] |
| K1:E-N | 24 | 8 | 1 | 5 | 10 | 16.67 | [0.00, 33.33] |
| C0:K1-K0 | 24 | 7 | 9 | 0 | 8 | -37.50 | [-54.17, -20.83] |
| G:K1-K0 | 24 | 15 | 1 | 0 | 8 | -4.17 | [-12.50, 0.00] |
| N:K1-K0 | 24 | 8 | 8 | 1 | 7 | -29.17 | [-45.83, -12.50] |
| E:K1-K0 | 24 | 13 | 3 | 0 | 8 | -12.50 | [-25.00, 0.00] |

| Stage | Terminal reason | Episodes |
| --- | --- | --- |
| P1 | CONTRACT_FAILURE | 22 |
| P1 | INCORRECT_FINAL | 4 |
| P1 | INCORRECT_STATE | 5 |
| P1 | VERIFIED_SUCCESS | 33 |
| P2 | CONTRACT_FAILURE | 40 |
| P2 | INCORRECT_FINAL | 35 |
| P2 | INCORRECT_STATE | 9 |
| P2 | VERIFIED_SUCCESS | 108 |

## <a id="analysis"></a>분석과 불확실성

업무 실패·계약 거절·실행 미관측을 구분했습니다. 위 구간은 family별 task를 함께 재표집한 10,000회 탐색적 pointwise 구간(seed6102013)입니다. 같은 task의 8조건을 같이 유지합니다. 합성 24과제는 독립 회사나 일반 업무 모집단이 아닙니다. 폭0인 구간도 동등성 인증이 아닙니다. E와 N은 독립 자유 실행 정책이므로 차이를 오류 설명 정보만의 인과효과로 해석하지 않습니다. KV 변경에는 커널 경로 차이가 포함될 수 있습니다.

- [DEV-A00_K1_C0](results/p1_r1/cells/DEV-A00_K1_C0.json): `CONTRACT_FAILURE`; first unmet `read`.
- [DEV-A00_K1_N](results/p1_r1/cells/DEV-A00_K1_N.json): `CONTRACT_FAILURE`; first unmet `read`.
- [DEV-A01_K1_C0](results/p1_r1/cells/DEV-A01_K1_C0.json): `CONTRACT_FAILURE`; first unmet `read`.
- [DEV-A01_K1_N](results/p1_r1/cells/DEV-A01_K1_N.json): `CONTRACT_FAILURE`; first unmet `read`.
- [DEV-B00_K0_C0](results/p1_r1/cells/DEV-B00_K0_C0.json): `CONTRACT_FAILURE`; first unmet `source_read`.
- [DEV-B00_K0_E](results/p1_r1/cells/DEV-B00_K0_E.json): `INCORRECT_STATE`; first unmet `source_read`.
- [DEV-B00_K0_G](results/p1_r1/cells/DEV-B00_K0_G.json): `INCORRECT_STATE`; first unmet `source_read`.
- [DEV-B00_K0_N](results/p1_r1/cells/DEV-B00_K0_N.json): `CONTRACT_FAILURE`; first unmet `source_read`.

대표 원행: [HOLDOUT-A00 / NVFP4 C0](results/p2_r1/cells/HOLDOUT-A00_K1_C0.json)는 첫 read JSON 뒤에 `}`를 하나 더 생성해 실행되지 않았습니다. [같은 과제의 G](results/p2_r1/cells/HOLDOUT-A00_K1_G.json)는 read→계산→저장→검증→final을 실제 완료했습니다. 이는 이번 조건에서 형식 병목이 완화된 사례입니다.

[HOLDOUT-B01 / BF16 G](results/p2_r1/cells/HOLDOUT-B01_K0_G.json)는 이동 전 검증을 받은 뒤 재고를 옮기고, 새 상태 검증 없이 final을 냈습니다. 파일과 최종 숫자는 맞지만 관측 계약을 충족하지 못했습니다. [HOLDOUT-C00 / NVFP4 G](results/p2_r1/cells/HOLDOUT-C00_K1_G.json)는 요청된 `review` 대신 schema상 허용되는 `published`를 저장했습니다. G는 올바른 상태값을 알려주지 않으므로 이 의미 오류는 그대로 실패입니다.

## <a id="conclusion"></a>결론과 범위

정확한 유지·손실·회복 수는 위 paired 표에 있습니다. 문법에 맞는 출력은 실제 상태·인자·검증 관측까지 맞다는 보장이 아닙니다. 비용은 실패를 포함한 episode wall 합계와 생성 token입니다. 이 측정으로 serving speedup이나 동시 처리량을 주장하지 않습니다.

| Scope | Status |
| --- | --- |
| P0 native cache / grammar | Measured; see P0 gates and kernel/finite receipts |
| P1 DEV / P2 HOLDOUT | Separate measured denominators and stored-trace audits |
| P3 COST / concurrent serving | NOT_RUN |
| Weight INT4 / training / other models | NOT_RUN |
| Git publication / Pages | NOT_RUN; separate approval required |

Q1: 문법 제약은 NVFP4 조건에서 실제 완주를 늘렸지만 BF16의 완주수는 늘리지 않았습니다. 형식 제약으로 상태값·최신 검증까지 보장하지 못했습니다. E는 N보다 점추정이 높지만 탐색 구간에0이 있고 독립 자유 정책 비교입니다. Q2: 같은 BF16 가중치여도 두 serving 구성의 결과는 달랐습니다. G에서 차이가 작아졌지만 비열등성·동등성이나 일반 업무 우위를 인증하지 않습니다. 다음 실험은 별도 승인 후 동일 실패 관측의 N/E 진단이나 상태 변경 뒤 검증 계획을 다루어야 합니다. 이번에는 추가 실행하지 않았습니다.

## <a id="reproduce"></a>재현과 참고자료

[CPU/GPU reproduction](REPRODUCE.md) · [Protocol](specs/protocol.json) · [Reuse map](REUSE_MAP.md) · [Notice](NOTICE.md) · [Roadmap](ROADMAP.md)

[P0 source freeze](provenance/execution_freeze_p0_r2.json) · [P1 source freeze](provenance/execution_freeze_p1.json) · [Model identity](provenance/model_identity.json)

Codex 보조로 구현·문서화했고 별도 CPU 계산 코드로 검산했습니다. 이를 사람의 독립 심사나 외부 GPU 재현으로 표시하지 않습니다. 공개 CPU 감사는 저장된 원문·SSE·token ID·사건을 재생합니다. 모델 가중치 없이 새 모델 추론을 재현한 것은 아닙니다.
