# 모델 없는 P0 실행과 감사

Python3.10+ 표준 라이브러리를 사용합니다. 패키지 설치·GPU·모델·tokenizer·네트워크가 필요하지 않습니다. case 디렉터리에서 실행합니다. 출력은 존재하지 않는 새 경로여야 합니다.

```bash
cd cases/018-agent-state-recovery
PYTHONPATH=src python3 -B -m agent_state_recovery p0 --output /tmp/case018-p0-new
PYTHONPATH=src python3 -B -m agent_state_recovery audit --run /tmp/case018-p0-new --output /tmp/case018-audit-new.json
```

`p0`는 새 계약 검사, DEV6×3 control=18 trace, 별도 반례8 trace, terminal/reference 각각 두 continuation, raw 독립 감사와 보호 manifest 검사를 실행합니다. `verification.json`, `tests.log/json`, `control_and_counterexamples.json`, `budget_feasibility.json`, `run_manifest.json`, raw `episodes/`, 고정 `inputs/`, `execution_freeze.json`, 파일 hash 목록이 남습니다. 가짜 client의 토큰/시간은 simulated로, 실제 CPU 시간은 별도 기록됩니다. 모델 요청 수는0입니다.

`audit`는 현재 소스와 run freeze의 대응, 18 control/8 진단의 누락·중복, 원응답 hash, 요청/관측/도구/상태/후속 프로그램과 지표를 다시 검산합니다. 새 `*-replayed-states/`에 상태를 씁니다. 결과 summary bool을 그대로 합산하지 않습니다. 코드를 고친 다른 source로 과거 run을 같은 동결 실행처럼 이어 붙이지 않습니다.

실제 이 작업의 최종 실행·감사 경로와 명령은 [P0_REPORT.ko.md](P0_REPORT.ko.md)에 있습니다. 첫 CPU 준비 run도 보존하며, 후속 장부 강화는 별도 amendment와 source로 구분합니다. 새 출력 경로에서 다시 실행하는 것은 새 CPU 계약 검사이며 새 모델 실험이 아닙니다.

향후 모델 실행 CLI는 아직 없습니다. `FutureModelAdapter`는 인터페이스뿐이며 P1에 실제 환경/artifact identity·tokenizer/template·structured grammar mask·BF16/NVFP4 저장/읽기·raw usage/timeout/abort 통합과 최초 token IDs 대조 gate가 필요합니다. 이번에 이 항목을 실행하거나 설치하지 마세요. P1/P2 생성·GPU·게시를 수행하는 명령을 이미 구현된 것처럼 제공하지 않습니다.
