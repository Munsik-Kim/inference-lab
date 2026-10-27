# Case021 P5 재현 안내

이 공개 후보에서 가능한 것은 **집계표→검산·그림**과 작은 합성 수식의 검산입니다. 실제 Z/code/scale/단일항 출력은 포함하지 않아 전체 원자료 계산이나 GPU 재실행은 공개본만으로 할 수 없습니다. P5는 로컬 후보이며 원격 CI나 다른 머신 검증을 수행하지 않았습니다.

검증한 환경은 Linux/WSL의 Python3.11.15, NumPy1.26.4, Matplotlib3.10.9입니다. 기존 CPU 환경을 읽기 전용으로 사용했고 새 설치·upgrade는 없었습니다. [의존성 목록](requirements-cpu.txt)은 식별용입니다. 이 명령은 설치하거나 Torch·모델·CUDA를 불러오지 않습니다. native Windows는 검증하지 않았습니다.

public_candidate 디렉터리에서 아래를 실행합니다. output은 존재하지 않는 외부 경로여야 합니다. `--quick`은 manifest, 에너지 항등식·부호 분류·후보 대응·C의 조건별 차이·작은 exact rounding 식을 확인합니다. 두 번째 명령은 같은 확인 후 PNG/SVG3개를 추가합니다. 두 tests 명령은9개 계약 검사와6개 공개 경로/변조 사례를 실행합니다. 반복 실행을 새 고유 검사 수로 세지 않습니다.

```bash
python -B source/public_cpu.py --root . --output ../p5-quick-new --quick
python -B source/public_cpu.py --root . --output ../p5-figures-new
python -B tests/test_contracts_v3.py --output ../p5-fixtures-new
python -B tests/test_public.py --root . --output ../p5-public-tests-new
```

실제로 새 외부 output에서 quick/full, 계약 검사와 변조 검사를 수행했습니다. [검증 기록](verification.json)과 [명령·output 대응](records/reproduction.json)에 실제 경로를 가린 논리 이름과 실행 범위를 기록합니다. 실제 새 경로와 원 명령은 로컬 root의 worker 장부에 남습니다.

공개 숫자 검사는 같은 분석자를 다른 사람의 독립 심사로 만드는 것이 아닙니다. 합성식은 Fraction 정수/유리수 nearest-even과 벡터화 경로를 대조합니다. 공개 집계는 원자료를 다시 계산했다고 주장하지 않습니다. 정확한 좌표의 로컬 GPU 출력은 weight를 드러낼 수 있어 제외했습니다.

전체 로컬 재계산은 P5 root의 `source/replay_cpu.py --root P5_ROOT --output NEW_EXTERNAL`로 A/B/C 원출력→표를 재생합니다. 원 Case021/P4 위치 manifest와 현재 두 artifact, 저장 Z/출력, 기존 CPU 환경이 필요합니다. 실제로 수행해 표/JSON20개의 bytes가 일치했습니다. 이 명령은 GPU를 호출하지 않습니다. local GPU runner는 이력 검토를 위해 보존했으며 신규 실행 명령은 제공하지 않습니다. 승인된 P5 GPU 진단은 종료했습니다.

실패한 초기 수치 검사/source는 그대로 남아 있고, 현재 추천은 v3 계약 검사 및 arithmetic_v2입니다. [수정 이력](REUSE_MAP.md)을 보십시오. PUBLIC_MANIFEST는 자신을 제외한 공개 후보 파일을 식별하며, 역사적 source bundle이나 Git commit을 대신하지 않습니다.
