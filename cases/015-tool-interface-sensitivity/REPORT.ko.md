# Case015 — 같은 도구, 다른 이름

## 요약

API 이름 변경 전후의 조회·계산·기록 수정·검증을 자동 비교하는 도구를 구현했습니다. 생성 전에 받은 결과와 실제 JSON 상태를 따로 기록하고, 저장 응답을 새 fixture에서 재실행해 성공 판정을 검산합니다.

새 A1 96개 실측에서 BF16은 15/16→16/16, RTN은 16/16→16/16, D16은 12/16→12/16이었습니다. BF16 회복 1건, 이름 변경에 따른 손실 0건입니다. D16의 동일 4개 fixture 실패는 두 API에 남았습니다.

[1. 배경](#section-1) · [2. 질문과 가설](#section-2) · [3. 원리와 실행 계약](#section-3) · [4. 방법](#section-4) · [5. 실험](#section-5) · [6. 결과](#section-6) · [7. 실패와 회복 분석](#section-7) · [8. 결론](#section-8) · [9. 재현과 출처](#section-9)

<a id="section-1"></a>
## 1. 배경

이름 문자열이 바뀌어도 같은 도구라면 업무를 끝낼 수 있을까? 기존 P1은 long-context DEV의 이름·순서를 바꾸어 실패와 회복을 관측했습니다. A1은 그 결과를 본 뒤 설계한 새 응용시험이며 실제 회사나 결제 시스템에 연결하지 않았습니다.

<a id="section-2"></a>
## 2. 질문과 가설

주 질문은 모델별 API_V1→API_V2의 유지 성공·손실·회복·양쪽 실패입니다. API_V2가 더 좋아야 한다는 가설은 없습니다. D16 변화−BF16 변화는 보조 비교이며 한 업무·4개 표현의 탐색 범위입니다.

<a id="section-3"></a>
## 3. 원리와 실행 계약

| Semantic ID | API_V1 | API_V2 |
|---|---|---|
| order.read | read_order | readOrder |
| amount.multiply | multiply_amount | multiplyAmount |
| order.write | write_order_total | writeOrderTotal |
| order.validate | validate_order | validateOrder |

이름만 lowerCamelCase로 바꾸고 설명·인자·순서를 유지했습니다. registry는 전단사이며 옛 이름·오탈자를 거절합니다. 생성 응답 안의 모든 호출은 동일한 immutable available_result_event_ids를 공유합니다. 곱셈은 read 결과 수신 후, write는 올바른 곱셈 결과 수신 후 생성해야 합니다. write와 validate는 같은 응답에서 순서대로 실행할 수 있습니다. 최종 true는 현재 state version에 대응하는 validation을 이미 받은 경우만 허용합니다.

<a id="section-4"></a>
## 4. 방법

주 endpoint VERIFIED_WORKFLOW_SUCCESS는 실행된 read→multiply→write→validate와 관측 계약, 실제 정수 total, 전 과정의 타 주문/불변 필드 보존, 현재 검증 수신, 정확한 세 필드 JSON을 모두 요구합니다. bool을 int와 동치로 세지 않습니다. schema가 맞는 잘못된 정수는 실제 계산·저장되며 oracle이 고치지 않습니다. STATE_ONLY_SUCCESS와 FINAL_ANSWER_CORRECT는 별도입니다.

<a id="section-5"></a>
## 5. 실험

seed6101621로 4 template×4fixture=16 scenario와 별도 smoke1을 만들고, ID와 금액을 독립 생성했습니다. 수량2..40, 단가100..20,000의 정수이며 초기 total은 null 또는 -1입니다. 초기 요청은 수량/단가를 노출하지 않습니다. 토큰 길이는 506..518로 기존 P1의4K/8K와 다릅니다. 16×2API×3방법=96 본 episode, BF16 두 API smoke2를 별도 실행했습니다. 순서 seed6101622와 BF16→D16→RTN을 등록했습니다.

같은 revision cdbee75f17c01a7cc42f958dc650907174af0554, greedy192token, enable_thinking=False, 최대8모델 요청·6도구 시도·900초, context24576, BF16/SDPA, logits_to_keep=1, within-request KV ON/turn cache OFF입니다. ENUM·추가 feedback·alias는 OFF. 252개씩의 원 RTN/D16 저장본을 읽었고 D16 beta_inverse/W_pre 경로를 유지했습니다.

<a id="section-6"></a>
## 6. 결과

### A1

| Model | API_V1 verified /16 | API_V2 verified /16 | Both success | Loss | Gain | Both fail |
|---|---|---|---|---|---|---|
| BF16 | 15 | 16 | 15 | 0 | 1 | 0 |
| RTN | 16 | 16 | 16 | 0 | 0 | 0 |
| D16 | 12 | 12 | 12 | 0 | 0 | 4 |

부가 D16−BF16 API 변화: -6.25 pp; 95% exploratory CI [-18.75, +0.00] pp

구간은 seed6101623의 10,000회 template 묶음 재표집입니다. 4개 template와 고정 이름 변경 하나에 조건화한 탐색 구간이며 API 모집단의 배포 승인·동등성 구간이 아닙니다.

### P1

| Model | I0 original /8 | I1 order /8 | I2 lowerCamel /8 | I3 opaque /8 |
|---|---|---|---|---|
| BF16 | 8 | 8 | 8 | 7 |
| RTN | 8 | 8 | 8 | 8 |
| D16 | 6 | 6 | 5 | 8 |

원 P1 주 interaction +8.33pp, 95% CI[-8.33,+25.00]pp; 손실과 회복을 별도 보존했습니다. 두 단계의 96개를192개의 같은 점수로 합산하지 않습니다.

### 비용

687.452s; 463 model requests; 378 attempted / 368 executed tools; 16927 generated tokens. Smoke: 10 requests / 8 tools / 360 tokens. Peak allocated 7.802 GiB; reserved 8.021 GiB.

Worker wall은 모델 hash·load·decode·smoke·실패·I/O를 포함합니다. Torch allocator peak와 Windows 배경 사용을 구분한 장치 free/total 표본을 응답별로 남겼습니다. serving 속도나 native INT4 가속 측정이 아닙니다.

<a id="section-7"></a>
## 7. 실패와 회복 분석

BF16 / A1-06: V1은191444를 올바르게 저장했지만 validate_order의 주문 ID를 ...3657 대신 ...3659로 생성했습니다. 오류 결과를 받고 validated:false로 끝났습니다. V2는 원 대상 ID로 검증하고 성공했습니다. D16은 A1-00/06/08/14에서 두 API 모두 write 호출 JSON을 완성하지 못해 저장 전에 끝났습니다. 미등록 이름과 관측 순서 위반은0건입니다. API 변화가 D16의 이 parse 실패를 회복시키지 않았습니다.

기록 결함을 사후 주석으로 분리했습니다. malformed-only 응답에서 조기 종료한8건의 원 terminal_reason은 초기값 BUDGET_EXHAUSTED로 남았으나 실제 모델/도구 예산은 소진하지 않았습니다. 별도 주석은 PARSE_ERROR로 설명합니다. 잘못된 주문 ID1건은 원 summary의 ARGUMENT_VALUE_WRONG에 빠졌지만 원 tool event에 UNKNOWN_ORDER/BAD_ARGUMENTS로 남아 있어 주석에서 값 오류로 집계합니다. 원 응답·코드 동결·성공 판정은 그대로이며 독립 endpoint 재계산의 영향은0건입니다. 조기 parse-only 종료 후의 recovery는 관측하지 않았습니다.

P1은 실행 직전 같은 fixture와 생성 전 같은 관측을 별도로 고정하지 않았습니다. 새 표시 계층은 그 둘을 구분하며 A1 계약으로 옛 점수를 고치지 않습니다.

<a id="section-8"></a>
## 8. 결론

한 주문 워크플로에서 이름 변경 회귀를 추적·재실행하는 도구를 완성했습니다. API_V2에서 손실은 없고 BF16 회복1건이 있었지만, D16의4개 fixture 실패는 남았습니다. 기존 P1의 인터페이스 민감도와 같은 현상으로 단정할 수 없습니다. 다음 연구는 P2 복구 진단을 별도 승인·등록하고 동일 관측을 주는 대조를 포함하는 것입니다.

<a id="section-9"></a>
## 9. 재현과 출처

[CPU / GPU commands](REPRODUCE.ko.md) · [A1 raw](followups/order-api-regression-v1/results/a1_r1/episodes.jsonl) · [Independent audit](followups/order-api-regression-v1/results/a1_r1/independent_audit.json) · [Diagnostic annotation](followups/order-api-regression-v1/results/a1_r1/diagnostic_annotations.json) · [Source freeze](followups/order-api-regression-v1/provenance/source_freeze.json) · [Original P1](publication/originals/p1_review/REPORT.ko.md) · [NOTICE](NOTICE.md)

27개 원 P1 회귀와18개 새 A1 고유 검사를 구분합니다. CPU 감사는 원 응답96개·실제 호출368개·요청463개의 대응과 상태/관측 판정을 재구성했으며 GPU 추론 자체를 재현하지 않습니다. Codex/ChatGPT 보조와 별도 CPU 계산 검토이며 사람의 독립 심사나 저자 승인이 아닙니다.
