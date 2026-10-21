# Roadmap / 다음 단계

현재 승인 범위는 Case017의 P0 기술 확인, DEV P1, 봉인한 HOLDOUT P2와 로컬 검토본입니다. 실제 완료 범위는 실행 receipt와 보고서에 기록합니다. 이 문서는 성공률이나 성능 향상을 예상한 실행 지시가 아닙니다.

## 이번 연구의 단계

| 단계 | 질문 | 자료와 분모 | 다음 단계로 가는 근거 |
|---|---|---|---|
| P0 | 같은 출력 계약·실제 캐시·원생성 계측이 동작하는가? | 별도 기술 probe와 smoke; 주 성공률에서 제외 | 실제 storage/attention, 입력 대응, raw token, 서버 문법 적용·공통 실행 설정 gate |
| P1 | 작은 DEV에서 정책과 비용이 어떻게 달라지는가? | DEV 8개 × 2 KV × 4 정책 = 64 episode | 등록한 셀의 완료·관측·감사. 점수 상승을 통과 조건으로 삼지 않음 |
| P2 | 같은 계약이 새 합성 fixture와 표현에서도 작동하는가? | HOLDOUT 24개 × 2 KV × 4 정책 = 192 episode | 봉인·불변 검증 후 실행, 전체 paired 표와 결측 장부 제출 |
| P3 | 비용용 업무를 반복·동시 실행하면 무엇이 달라지는가? | COST 6개(A/B/C 각 2), 현재 GPU 실행 없음 | 별도 승인·예산·사전 등록 필요 |

DEV의 관측을 본 뒤 HOLDOUT의 parser·정답·도구·정책을 바꾸지 않습니다. 구현 오류가 확인되면 원 attempt와 실행 소스를 보존하고 영향 범위를 먼저 판정합니다. CPU scripted trace의 성공을 모델의 회복 수로 채우지 않습니다.

## P3 — 설계만 보존

[COST fixture](data/cost_tasks.json)는 이번 GPU 실행 분모에 넣지 않습니다. 후속 비용 실험을 승인할 경우 다음 항목을 새로 고정해야 합니다.

1. 어느 정책·KV·요청 동시성을 비교할지, 동일 업무·도착 순서와 round 순서.
2. 전체 workload의 첫 요청부터 마지막 종료·취소 확정까지의 makespan.
3. 성공·실패·timeout·대기·retry·서버 drain을 모두 포함한 비용 회계.
4. 성공한 업무 수/분과 업무당 토큰·시간, 거짓 완료 수. 성공 0일 때 비용/성공은 undefined로 표시.
5. 엔진 queue·preemption과 요청 latency의 관측 범위, 서버 시작·compile·warmup의 별도 비용.
6. 표본 수와 반복 수의 구분. 같은 여섯 업무를 반복했다고 독립 업무 수가 늘어난 것으로 세지 않음.

현재 코드의 episode wall 합계는 순차 실행 비용입니다. 이를 미래 동시성 workload makespan이나 serving speedup으로 바꾸지 않습니다. 이번 결과만으로 P3의 유용 구간이나 native KV의 일반적인 품질 우위를 판정하지 않습니다.

## 결과를 보고 검토할 진단 후보

| 관측 | 검토할 후속 | 필요한 구분 |
|---|---|---|
| 형식만 회복하고 업무는 실패 | 인자·자료 선택·관측 순서·최신 validation의 동일 관측 진단 | 유효 JSON과 완주를 분리 |
| N/E가 다른 경로에서 다르게 실패 | 사전 선택한 공통 실패 원문·공통 관측을 사용하는 별도 replay | 자유 실행 결과에 합치지 않고 피드백 정보 효과를 조사 |
| 두 KV가 다른 경로로 진행 | 첫 공통 관측 prefix까지 비교하고 그 이후는 자유 rollout으로 표시 | 압축 반올림만의 인과효과로 단정하지 않음 |
| 도구·model 예산에 도달 | 최초 초과 경계와 불필요 호출, 정책의 final-as-attempt 한도 검토 | 예산 변경은 새 대조 실험; 기존 점수를 소급 변경하지 않음 |
| 모든 조건의 점수가 같음 | 범위·비용·실패 위치를 그대로 기록 | 차이 없음이나 폭 0 CI를 동등성 보장으로 바꾸지 않음 |

## 자동으로 확대하지 않을 항목

모델·데이터 다운로드, weight INT4/D16/RTN 비교, 학습/CIPR/LoRA, 새 문법·retry·prompt sweep, 다른 모델·엔진, 실제 서비스 연결은 이번 로드맵의 자동 실행 대상이 아닙니다. GitHub·Pages 공개도 로컬 검토 뒤 별도 승인이 필요합니다.

English scope: P3 is design-only. The current study does not authorize concurrency sweeps, new models, training, new weight quantization, real business integrations or remote publication. Further diagnostics require their own frozen comparison and do not alter the original free-policy scores.
