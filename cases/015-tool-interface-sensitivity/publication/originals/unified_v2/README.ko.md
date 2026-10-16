# Case015 — 같은 도구, 다른 이름

API 이름 변경 전후의 조회·계산·기록 수정·검증을 자동 비교하는 도구를 구현했습니다. 생성 전에 받은 결과와 실제 JSON 상태를 따로 기록하고, 저장 응답을 새 fixture에서 재실행해 성공 판정을 검산합니다.

새 A1 96개 실측에서 BF16은 15/16→16/16, RTN은 16/16→16/16, D16은 12/16→12/16이었습니다. BF16 회복 1건, 이름 변경에 따른 손실 0건입니다. D16의 동일 4개 fixture 실패는 두 API에 남았습니다.

| Model | API_V1 verified /16 | API_V2 verified /16 | Both success | Loss | Gain | Both fail |
|---|---|---|---|---|---|---|
| BF16 | 15 | 16 | 15 | 0 | 1 | 0 |
| RTN | 16 | 16 | 16 | 0 | 0 | 0 |
| D16 | 12 | 12 | 12 | 0 | 0 | 4 |


한 합성 주문 업무 · 4개 영어 지시문 표현 · 16개 fixture · Qwen3-4B-Instruct-2507. 기존 P1의 96개와 분모를 합치지 않습니다. 품질은 저장 INT4를 dense BF16으로 복원해 측정했습니다.

[Pages](https://munsik-kim.github.io/inference-lab/ko/case015.html#overview) · [REPORT](REPORT.ko.md) · [CPU / GPU](REPRODUCE.ko.md) · [Stage map](STAGE_MAP.md) · [A1 runner](followups/order-api-regression-v1/src/order_regression/runner.py) · [Tests](followups/order-api-regression-v1/tests/test_order_contracts.py)

## 세 가지 구현

1. 정확한 이름↔의미 mapping과 두 API의 실제 동작 동등성 검사.
2. 조회 결과 수신→계산 결과 수신→저장·검증→최종 답의 관측 계약.
3. 원 응답·실제 상태 재실행과 모델별 손실·회복 회귀표.

## 기존 P1도 보존했습니다

기존 8개 DEV task×3방법×4인터페이스의 signed interaction은 +8.33pp, 95% 구간 [-8.33,+25.00]pp로 불확정입니다. [원 보고서](publication/originals/p1_review/REPORT.ko.md)와 [원형 복원 mapping](publication/provenance/p1_original_map.json)을 유지합니다.
