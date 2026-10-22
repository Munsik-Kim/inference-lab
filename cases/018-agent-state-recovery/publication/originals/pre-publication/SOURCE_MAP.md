# 원문 접근과 독립 구현 범위

접근일: 2026-10-22 KST. 공식 arXiv HTML v1의 아래 절을 직접 열어 읽었습니다. 전문·PDF·그림·저자 코드는 이 case payload에 복사하지 않았습니다. 논문 수치/실험을 Case018 분모로 옮기지 않습니다.

| 원문 | 실제 읽은 범위 | 가져온 원리 | Case018 대응과 차이 |
|---|---|---|---|
| [Lost in Recovery / ReTrace, 2610.27954v1](https://arxiv.org/html/2610.27954v1) | §5.1 Transition-Consistent Recovery, §5.2의 on-policy rollout/perturbation/goal-oriented replay (HTML lines158–194) | 겉으로 복원된 상태와 후속 행동 결과를 함께 확인 | B와 두 고정 continuation을 결합한 Q. GUI/시각 oracle가 아닌 재고 JSON, 모델별 on-policy 성공 경로 대신 공통 합성 checkpoint와 CPU 주입 오류를 사용. 현재 revision을 public read로 얻는 프로그램이므로 원 action의 문자 그대로 replay가 아님. 유한 두 경로는 모든 미래 동등성 증명이 아님 |
| [ACE-Compression, 2610.27913v1](https://arxiv.org/html/2610.27913v1) | §4.1 및 Appendix B.1 IFEval 설명 (lines157–165,257–270) | 개별 제약 충족과 전체 동시 성공의 구별 | D/B/P/F/C와 Q/W를 함께 보고. IFEval 데이터/점수, pruning/weight compression 실험, 전체 ability taxonomy를 재현하지 않음. 이번 KV 비교 자체는 아직 미실행 |

API·오류 family·가드·예산·통계·HOLDOUT gate의 구체 설계는 사용자가 전달한 Case018 P0 지시에 근거합니다. 새 이론/알고리즘을 제안한 것이 아닙니다. 접근 성공은 논문 실험이나 저자 코드 검증 완료를 뜻하지 않습니다. [접근 receipt](provenance/source_access.json)에 범위를 남겼습니다.
