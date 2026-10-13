작업 root: LOCAL_RESEARCH
결과 root: LOCAL_RESEARCH/results/paper_int4_reference_r2

모델/GPU 없는 저장 기록 검산:
cd LOCAL_RESEARCH
PYTHONDONTWRITEBYTECODE=1 .venv-reference/bin/python source/analyze_paper_results.py --run results/paper_int4_reference_r2

합성 CPU 검사:
PYTHONDONTWRITEBYTECODE=1 .venv-reference/bin/python -m unittest discover -s tests -v
기존29 AST tests는 pinned vendor source가 필요하다.

GPU 실제 명령은 attempts/suite_commands.json, additional_gpu_commands.json, packed03_command.json 및 agent16k_extension/commands.json에 있다. 완료 폴더의 재실행은 거절한다. 재현하려면 새로운 result root에 같은 frozen inputs/manifest를 복사하고 hash를 검증해 새 ACTIVE_RUN을 지정한다. 같은 모델 revision의 로컬snapshot이 필요하다.

.venv-reference는 기존 parent를 상속하며 standalone 환경 archive가 아니다. CPU artifact hash 대조에는 로컬 INT4 payload 파일, 독립 CE 대조에는 reference_logits arrays가 필요하다. 외부에 weights/환경/불필요한 full tensors를 자동 공유하지 않는다.

현재 packed CLI는 semantic BLOCKED에서exit1이다. 이전01/02의exit0을PASS로 쓰지 않는다. threshold와 이전실패를 유지하고 새 --output을 사용한다.
