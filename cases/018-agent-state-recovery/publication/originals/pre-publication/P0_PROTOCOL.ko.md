# Case018 P0 — 복구와 후속 업무의 CPU 계약

이 문서는 [protocol.json](specs/protocol.json)과 공개 [tools](specs/tools.json) / [grammar](specs/grammar.json)를 설명합니다. 모델 관측 전의 CPU 준비 계약이며, 실행 소스 identity는 각 run의 `execution_freeze.json`입니다. Git base와 미커밋 소스 hash는 다른 식별자입니다.

## 업무와 복구 목표

레코드 5개: 같은 SKU의 출발/도착 재고 2개, 다른 SKU의 재고 2개, 변경 금지 재고 1개입니다. 수량은 정수이며 금액·통화는 없습니다. 처리 원장에는 작업 ID, 출발/도착 ID, 수량, 단계, 보상 이력이 있습니다. 활성 `stock-lane`은 후속 이동에 영향을 줍니다. ID는 고정 seed의 무작위 값이며 수량·정답을 인코딩하지 않습니다.

| family | 오류 주입 전 checkpoint | 주입 | 복구 | F1 |
|---|---|---|---|---|
| A WRONG_TARGET | 의도한 이동 전 | 다른 SKU에 이동 효과 + 활성 처리 | 그 효과만 보상, 처리 해제 | 원래 의도한 이동 |
| B DUPLICATE | 정상 이동 1회 완료 | 별도 ID로 같은 효과 추가 1회 + 활성 처리 | 추가 1회만 보상, 정상 1회 유지 | 다음 이동 |
| C PARTIAL | 의도한 이동 전 | 출발 재고만 차감 + 활성 처리 | 차감만 보상, 처리 해제 | 원래 전체 이동 |

오류 주입은 CPU 환경 조작이며 자연 모델 오류율을 뜻하지 않습니다. DEV6은 A/B/C 각2개입니다. 같은 task의 4조건은 checkpoint/error/fault/public input이 같습니다.

## 상태 세 부분과 보호 시점

- `business_projection`: 모든 `records.<id>.quantity`, `sku`, `warehouse`. checkpoint와 비교합니다.
- `protected_data`: 레코드 집합; 각 레코드의 `sku`, `warehouse`, `owner`, `protected`; 보호 레코드 수량; `scope_id`; `stock_totals`; 복구 시작 이전 `audit` prefix. 복구 중 새 감사 항목도 이후 삭제·변경할 수 없습니다.
- `operational_state`: `ledger.<operation_id>.{source_id,destination_id,quantity,phase,compensations}`, `dedup.<operation_id>`, `active_operations.stock-lane`.

revision 증가와 감사 이력 추가는 정상입니다. 이전 revision이나 전체 JSON byte 복원을 요구하지 않습니다. 보상 완료된 incident 원장/dedup tombstone도 남깁니다. 후속 비교에서는 이 tombstone, revision 및 감사 suffix를 제외하고 살아 있는 원장·dedup·활성 처리를 비교합니다. 이는 해당 두 경로와 등록 상태의 적합성 검사이며 모든 미래 동등성 증명은 아닙니다.

P는 초기 상태 및 매 도구 실행 전후를 검사합니다. 주입이 만든 C의 재고 부족분은 시작 피해입니다. 각 SKU 부족분은 시작값에서 0까지 단조롭게 해소할 수 있지만 새로운 부족·초과·음수·보호 필드/기록 피해는 허용하지 않습니다. 나중에 되돌려도 새 피해를 지우지 않습니다. 보상 `source_restore`와 `destination_remove`를 독립 적용하므로 중간 상태의 허용 범위를 명시한 것입니다. 최종 B/Q는 별도로 검사합니다.

## 공개 API

모든 응답은 단일 `{"tool":...,"arguments":...}` object입니다. 이름은 정확히 등록된 것만 허용합니다.

| API | 인자 | 의미 |
|---|---|---|
| read_records | record_ids | 지정 레코드와 현재 revision |
| read_history | 없음 | opening 및 사건의 before/after 이력·revision |
| read_operation | operation_id | 원장·중복 방지·활성 처리·revision |
| apply_transfer | operation_id, source_id, destination_id, quantity, expected_revision | 원자적 이동. 같은 완료/보상 ID 재전달은 무시. 활성 처리 중 새 이동은 오류 |
| compensate_stock | operation_id, source_restore, destination_remove, expected_revision | 실제 전달한 수량을 해당 출발에 더하고 도착에서 뺌. 자동 역산/정답 교정 없음 |
| release_operation | operation_id, expected_revision | 보상 tombstone과 이력을 남기고 활성 처리를 해제. 수량 정답을 확인하지 않음 |
| validate_inventory | scope_id | 비음수 정수·보호 데이터·SKU 총합·원장 참조 불변식만 검사 |
| final | task_id, recovered, quantities, validation_event_id | 수량 목록과 검증 사건을 포함한 종료. 실제 파일 변경 없음 |

잘못된 schema-valid 값은 hidden gold로 교정하지 않습니다. 권한·음수 방지·현재 revision 등 공개 API 제약 위반은 정상 도구 오류입니다. 모델이 임의 경로/코드/URL을 넘기는 API는 없습니다. 저장 파일은 실행기가 만든 독립 root의 고정 `state.json`뿐입니다. symlink root/파일을 거절합니다.

모델 입력은 `public.initial_payload`의 명시적 key allowlist로 만듭니다. `checkpoint`, `oracle`, `continuations`, 정답 script, 내부 판정은 넣지 않습니다. 정상 read로 받은 수량·ID·원장은 정당한 관측입니다. 전체 checkpoint snapshot/reset/gold diff API는 없습니다.

## 관측·R0/R1·예산

공통 system은 마지막 변경 뒤 현재 상태의 성공 검증을 보고 final을 생성하도록 요구합니다. 생성 시작 시 available_result_event_ids를 고정합니다. 그 생성에서 나온 호출 결과는 다음 생성에 전달된 뒤에만 관측으로 셉니다. 한 응답에 여러 JSON object는 거절합니다.

R0는 형식에 맞는 final을 종료로 받고 같은 평가기가 사후 검사합니다. R1은 올바른 scope의 `SUCCESS`, strict bool `valid:true`, 현재 digest/version, 생성 전 관측 ID만 확인합니다. gold 복구 여부나 수량 정답은 보지 않습니다. 없으면 `FRESH_VALIDATION_REQUIRED`를 user feedback으로 최대2번 보냅니다. 세 번째 부적격 final은 `GUARD_LIMIT`입니다. 거절은 도구 사건을 만들지 않습니다. 정확한 final 내용은 별도 채점합니다.

상한은 requests16 / response512 tokens / episode8192 output tokens / parsed tool-or-final attempts16 / request180초 / episode900초 / context24576입니다. 파싱 불가 원문은 request/output만, JSON object의 이름/schema 오류는 parsed attempt도 소모하고 종료합니다. 정상 도구 오류·final·guard 거절은 공통 예산을 씁니다. 16번째 final은 허용하며, 16번째 non-final은 실행 후 TOOL_CALL_LIMIT입니다. 17번째 요청은 없습니다. 오류를 silently repair/coerce하지 않습니다.

P0는 UTF8 bytes/4 올림을 **simulated units**로만 사용해 cap/context 회계 경로를 검사합니다. 실제 tokenizer·문법 mask·서버 deadline/취소 통합은 NOT_RUN입니다. 실제 토큰 길이나 GPU 비용이 0이라는 뜻이 아닙니다.

## 두 독립 continuation과 지표

F1: A/C는 원래 이동, B는 다음 이동 → 결과 조회 → 공개 검증. F2: A/C는 활성 처리 채널을 사용하는 원래 남은 이동 → 다음 이동, B는 이미 완료한 원래 ID 재전달 → 다음 이동 → 조회·검증입니다. A/C는 활성 처리 잔여를, B는 중복 무시와 다음 처리를 드러내는 경로입니다. 양쪽의 업무 인자는 task 입력에 고정됩니다. mutation마다 각 상태의 공개 read가 반환한 현재 revision을 넣습니다. 반환 오류를 보고 수리하거나 경로를 바꾸지 않습니다.

terminal snapshot과 checkpoint 각각에서 F1/F2를 독립 복제해 실행합니다. 실제 상태/보호 계약, 이동/중복 무시 효과 및 살아 있는 처리 상태를 비교합니다. 따라서 종료 시 관련 business와 operational 상태를 전부 비교한다면 이 결정적 환경의 후속 검사는 상당 부분 논리적으로 중복됩니다. 별도 독립 표본·모델 자율 업무로 더하지 않습니다.

| 기호 | 정의 |
|---|---|
| D | schema-valid final에서 recovered가 bool true인 선언. 가드 거절 선언도 포함 |
| B | terminal business_projection = checkpoint |
| P | 전 과정의 보호·허용 중간 상태 계약 |
| F | 현재 scope/digest/version의 성공 validation을 final 생성 전에 실제 관측 |
| C1/C2 | 각각 고정 후속 절차의 올바른 결과·보호 계약 |
| Q | B ∧ P ∧ C1 ∧ C2: 종료 상태·후속 업무 적합 |
| W | Q ∧ accepted final ∧ final 내용 정확 ∧ F: 검증된 복구 workflow 완주 |

final이 없어도 terminal을 평가하고 B/P/C/Q를 계산합니다. W는0, F는 마지막 실제 generation 직전의 관측을 사용하는 진단으로 표시합니다. final 관측으로 잘못 명명하지 않습니다. terminal reason은 배타적 하나, 실패 labels는 중복 가능입니다.

손 계산: (출발20, 도착5)에서3 이동 →(17,8). 잘못된 이동이면 (+3,−3)로(20,5), 정상1회 뒤 중복이면(14,11)에서(+3,−3)로(17,8), 출발만 차감했으면(17,5)에서(+3,0)로(20,5)입니다. 모든 경우 활성 처리 해제는 별도입니다.

## 단계·통계와 보존

P0 gate는 계약·독립 audit·control/반례·보존·호출 예산 가능성입니다. P1에는 실제 모델 identity, 서버 adapter, tokenizer/grammar 및 KV 저장/읽기, DEV24 입력 identity와 실행 완료 gate가 필요합니다. **model_success_threshold=null**입니다. 모델 점수 바닥/천장도 공개하되 P0에서 예상 점수는 만들지 않습니다.

HOLDOUT24(각 family8)는 generator/seed/frozen fields만 등록했습니다. 생성은 P1 완료·freeze·기술 gate 뒤에만 가능합니다. gate 실패 시 코드가 거절하며 성공수 0도 그 자체로 gate 실패가 아닙니다. 실제 입력·gold 생성/봉인·모델 실행은 이번 범위 밖입니다. 사후 입력/채점 변경은 새 실험 버전이 필요합니다. 로컬 freeze 시간은 자기기록입니다.

향후 주 비교는 KV별 Q의 R1−R0, 보조는 W/구성요소/KV 차이입니다. family-stratified paired task bootstrap10000, seed61022018, 4조건 동시 재표집, 탐색적 pointwise 구간, DEV/HOLDOUT 별도 분모로 고정했습니다. KV block K0→K1, block 안 task/policy의 seeded shuffle, 새 fixture, prefix caching OFF, 최초 실제 token IDs 대응을 요구합니다. 이후 이력 분기는 허용합니다. P0 모델 통계·P1/P2·P3는 NOT_RUN입니다.
