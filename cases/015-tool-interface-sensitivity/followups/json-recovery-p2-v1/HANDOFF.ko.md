# 완료 상태와 인계

P2는 실제 GPU 실행과 CPU 재계산을 완료했습니다. D16 C0=0/8, G=8/8, N=0/8, E=0/8이며 N은 저장 상태1/8만 회복했습니다. 4개의 선택된 fixture×API2종입니다. 원 P1/A1 판정은 그대로입니다.

- 실행 소스 commit: `828ac7358b653626bed7e0a609448e73d120f1bb`; 초기 CUDA 실패 source447bda9도 보존했습니다.
- [보고서](REPORT.ko.md), [재현](REPRODUCE.ko.md), [decision](results/gpu_r2/decision.json), [검산](publication/verification/latest.json).
- 기존 CPU 준비 당시의 CLOSED_SCOPE/NOT_RUN 기록과 import 이력은 역사적 자료로 보존합니다. [원형 대응표](publication/preservation_map.json).
- 현재 CPU 검증은 publication/run_checks.py, GPU는 별도 source commit의 execution/run.py입니다. 옛 CPU CLI의 run 거절을 우회하지 않았습니다.
- 다음 무승인 실험을 이어가지 않습니다. 이번 원격 완료 지점은 검토 가능한 PR이며 main 병합·Pages 배포는 포함하지 않습니다.
