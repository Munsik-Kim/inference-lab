# CPU 재현과 다음 실행 경계

저장소 루트에서 실행합니다. 기본 경로는 Python 표준 라이브러리만 사용합니다. 실제 모델과 모델 실행 프레임워크는 불러오지 않습니다. 기존 원자료가 있는 Case015 전체 상대 경로를 유지해야 하며 이 후속 폴더만으로 독립 실행되는 pip 패키지는 아닙니다.

## 1. 기본 검증과 새 외부 출력

아래와 같은 명령을 실제로 실행해 PASS를 확인했습니다. 매번 새 출력 경로를 사용합니다.

```bash
P2=cases/015-tool-interface-sensitivity/followups/json-recovery-p2-v1
CPU_OUTPUT=$(mktemp -d /tmp/case015-p2-cpu-XXXXXX)
export PYTHONPATH="$P2/src" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=''

python3 -B -m json_recovery_p2.cli --help
python3 -B -m json_recovery_p2.cli validate
python3 -B -m json_recovery_p2.cli audit --output "$CPU_OUTPUT/replay"
python3 -B -m json_recovery_p2.cli baseline --tests --output "$CPU_OUTPUT/baseline"
python3 -B -m unittest discover -s "$P2/tests" -p test_cpu_contracts.py -v
python3 -B -m json_recovery_p2.cli dry-run --output "$CPU_OUTPUT/plan.json"
```

`audit`는 summary의 숫자를 정답으로 삼지 않고 원 96개 episode를 실제 fixture에서 재생해 checkpoint·분류·원 판정·원문 hash를 다시 만듭니다. 새 폴더의 생성물과 동결 data를 비교합니다. `prepare --output NEW_DIRECTORY`는 비교 전의 동일한 재구성 단계만 실행합니다.

`baseline --tests`는 원 Case015 457개 파일을 바이트 그대로 복사한 view에서 **기존 publication/scripts/run_checks.py --tests**를 실행합니다. 새 P2가 추가된 전체 tree에 과거 closed-set 검사를 억지로 적용하거나 원 manifest를 변경하지 않습니다. P1 27 / A1 18 / publication 6개의 실제 검사 이름과 수를 로그에서 확인합니다.

## 2. 선택적인 로컬 tokenizer·문법 검사

이미 설치된 `tokenizers 0.23.2`, `Jinja2 3.1.6`, `lm-format-enforcer 0.11.3`와 기존 snapshot을 사용했습니다. 패키지 설치·업그레이드는 하지 않았습니다. 아래의 `CPU_PYTHON`, `SNAPSHOT`, `RUNTIME`은 **자신이 보유한 기존 경로**를 설정하는 변수이며 다운로드 지시가 아닙니다. 이번 세션의 실제 경로는 외부 private handoff에 보존했습니다.

```bash
"$CPU_PYTHON" -B -m unittest discover -s "$P2/tests" -p test_grammar_contracts.py -v
"$CPU_PYTHON" -B -m json_recovery_p2.cli tokenizer-check \
  --snapshot "$SNAPSHOT" --output "$CPU_OUTPUT/retokenization.json"
"$CPU_PYTHON" -B -m json_recovery_p2.cli grammar-check \
  --snapshot "$SNAPSHOT" --output "$CPU_OUTPUT/grammar.json"
"$CPU_PYTHON" -B -m json_recovery_p2.cli local-inputs-check \
  --runtime "$RUNTIME" --output "$CPU_OUTPUT/local-identity.json"
```

`tokenizer-check`는 로컬 template 렌더와 native tokenizer JSON을 사용하여 8개 입력/24개 방법별 해시를 검사합니다. 추가로 463개 저장된 output ID를 raw/parser text로 복호화합니다. Transformers를 import하지 않습니다. `grammar-check`는 실제 Qwen token ID에 대한 allowed-token 목록을 확인하되 입력은 합성 계약 예제입니다. `local-inputs-check`는 snapshot 가중치 shard 및 RTN/D16 저장 파일의 bytes/hash만 읽고 tensor를 deserialize하지 않습니다.

이 의존성이나 tokenizer가 없는 일반 clone에서는 이 세 검사를 실행했다고 쓰지 않습니다. 기본 raw 재생/단위 검사와 선택적인 tokenizer/파일 검사의 범위는 다릅니다. 이 세션에서는 필요한 기존 파일과 의존성이 있어 실제 통과했습니다.

## 3. GPU dry-run과 거절 동작

```bash
python3 -B -m json_recovery_p2.cli run --dry-run --output "$CPU_OUTPUT/run-plan.json"
```

현재 실제 모델 실행 요청은 다음과 같이 **거절**됩니다. 의도한 exit code는 3입니다.

```bash
python3 -B -m json_recovery_p2.cli run
```

향후 실행에 필요한 입력과 조건부 요청 상한은 [dry-run](results/cpu_preparation/dry_run_plan.json), 남은 조건은 [gates.json](results/cpu_preparation/gates.json)에 있습니다. 별도 승인과 총 worker 예산, 검토한 GPU adapter 연결 없이 이 명령을 열지 않습니다. 원 A1 runner의 가드를 우회하는 명령은 제공하지 않습니다.

## 4. 결과를 읽는 방법

- [CPU 재생 요약](results/cpu_preparation/cpu_diagnostic_summary.json): 기존 기록에 대한 확인.
- [checkpoint summary](results/cpu_preparation/checkpoint_summary.csv): 공통 입력·관측·상태와 prefix 예산.
- [failure table](results/cpu_preparation/failure_table.csv): 8개 원 실패의 형식·EOS·예산.
- [새 복구 표](results/cpu_preparation/recovery_results.json): 조건부 계획행 96개, 전부 NOT_RUN/null.
- [신규/회귀 검사 명세](results/cpu_preparation/test_inventory.json): 고유 검사와 반복 실행을 구분.
- [실행 receipt](results/cpu_preparation/command_receipts.json): argv, exit, wall, log hash.

CPU scripted trace는 `synthetic_contract_test`입니다. 이것이 올바르게 복구되더라도 LLM의 복구 성공으로 합산하지 않습니다. 새 측정은 없으며 과거 기록은 `historical`임을 유지합니다.
