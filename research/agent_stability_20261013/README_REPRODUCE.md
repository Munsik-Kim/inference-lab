# 재계산과 재실행

## 모델 없는 CPU 검산

Python3.10+와 diova-compare0.1.2가 필요합니다. Torch·NumPy·GPU·모델은 필요하지 않습니다. 기존 연구 환경에 설치하지 말고 별도 CPU 환경을 사용하세요. 이 세션은 새 `/tmp` venv에 로컬 wheel을 `--no-index --no-deps`로 설치해 CLI와46개 package tests를 확인했습니다.

저장소 루트에서 새 CPU 환경에 소스를 설치하는 방법:

```bash
python -m pip install --no-deps --no-build-isolation ./packages/diova-compare
```

이 명령은 CPU 환경에 setuptools/wheel이 이미 있을 때 사용합니다. 로컬 검토 ZIP에 든 wheel을 사용하면 build 도구 없이 `python -m pip install --no-index --no-deps PATH_TO_WHEEL`로 설치할 수 있습니다. 기존0.1.1 wheel은 덮어쓰지 않았고 PyPI에는 올리지 않았습니다.

기존 Case013 기록을 비교하는 흐름은 [package 한국어 안내](../../packages/diova-compare/README.ko.md#case013-export)에 있습니다. 새 후속 기록은 다음처럼 검산합니다. 작업 디렉터리는 저장소 루트입니다.

```bash
python -B research/agent_stability_20261013/audit.py --result research/agent_stability_20261013/records --output /tmp/agent-followup-audit-new.json
diova-compare validate --input research/agent_stability_20261013/records/p2_compare_inputs/I0.jsonl
diova-compare compare --baseline research/agent_stability_20261013/records/p2_compare_inputs/I0.jsonl --candidate research/agent_stability_20261013/records/p2_compare_inputs/I2.jsonl --output /tmp/agent-followup-report-new
```

출력 경로는 모두 새 경로여야 합니다. audit는 이 완료된r1 기록의 inventory·입력 IDs/hash·도구 실행 조건·paired counts·저장된 logprob 합·선택 기준을 확인합니다. 미완료 run에는 정확한 누락/실패 오류를 내며 성공으로 메우지 않습니다. 전체 logits에서 확률을 다시 측정하거나 excluded activation vectors의 SSE를 복구하지 않습니다. 기본 bootstrap은 이 작은 DEV의 기술적 구간이며 비열등성/동등성 판정이 아닙니다.

## 회귀와 동결 자료 검사

새 CPU 환경에 package 설치 후:

```bash
python -B -m unittest discover -s packages/diova-compare/tests -v
```

진단의 작은 CPU fixture4개는 Torch가 있는 기존 Python을 읽기만 해서 실행했습니다. 원 모델을 로딩하지 않습니다.

```bash
python -B -m unittest discover -s research/agent_stability_20261013 -p 'test_*.py' -v
```

Case009/011은 공통 package0.1.1까지 옛 hash에 묶여 있습니다. 현재 package를 옛 manifest에 덮어쓰지 않습니다. 아래 wrapper가 원 ZIP의 package를 scratch에 복원하고 원 과학적 파일의 live hash를 확인한 뒤, 변경하지 않은 옛 검산기를 실행합니다. scalar 경로에는 NumPy 등 원 CPU requirements가 필요합니다. 복원 scratch는 Case011 전체 protected support를 포함하므로 약0.6GB가 필요합니다.

```bash
python -B tools/run_case009_frozen_checks.py --root . --output /tmp/case009-frozen-checks-new
python -B tools/run_case011_frozen_checks.py --root . --output /tmp/case011-frozen-checks-new
```

## 기존 모델이 필요한 GPU 재실행

외부 원 연구 root의 `.venv-reference`, Qwen3-4B revision `cdbee75f17c01a7cc42f958dc650907174af0554`, `results/paper_int4_reference_r2`의 RTN/D16 artifact 각252개와 원 입력이 필요합니다. 모델·artifact는 이 공개 준비에 포함하지 않습니다. `prepare.py`는 외부 r2/model 경로를 별도 runtime manifest에 기록하고, 새 result 경로와 분리합니다. Worker는 누락 artifact를 solver나 다운로드로 보충하지 않습니다.

이 세션의 실제 연구 root는 로컬 HANDOFF/identity에 기록돼 있습니다. 아래 변수는 그 외부 root와 새 빈 결과 위치로 지정합니다. 이미 본 결과를 새 held-out TEST라고 부르지 않습니다.

```bash
DIAGNOSTIC_PYTHON="$RESEARCH_ROOT/.venv-reference/bin/python"
mkdir -p "$DIAGNOSTIC_RESULT/attempts"
"$DIAGNOSTIC_PYTHON" -B research/agent_stability_20261013/prepare.py --research "$RESEARCH_ROOT" --result "$DIAGNOSTIC_RESULT" --case cases/013-dualquant-reference-quality
for method in BF16 RTN D16 D4; do
  "$DIAGNOSTIC_PYTHON" -B research/agent_stability_20261013/launch.py --result "$DIAGNOSTIC_RESULT" --stage first --method "$method"
done
"$DIAGNOSTIC_PYTHON" -B research/agent_stability_20261013/p1_finish.py --result "$DIAGNOSTIC_RESULT" --phase common
for method in BF16 RTN D16 D4; do
  "$DIAGNOSTIC_PYTHON" -B research/agent_stability_20261013/launch.py --result "$DIAGNOSTIC_RESULT" --stage common --method "$method"
done
"$DIAGNOSTIC_PYTHON" -B research/agent_stability_20261013/p1_finish.py --result "$DIAGNOSTIC_RESULT" --phase finish
```

`p1_gate.json`의 gate가 true이고, 맞는 대조 projection과 남은 예산이 있을 때만 다음을 실행합니다. I0의 episode 실패가 재현되지 않으면 worker가 I1/I2를 중단합니다.

```bash
"$DIAGNOSTIC_PYTHON" -B research/agent_stability_20261013/launch.py --result "$DIAGNOSTIC_RESULT" --stage p2 --method D16
```

명령은 실제 실행한 CLI입니다. 관측 worker wall475.752초는 이 host의 기록으로 완료시간 예측이 아닙니다. Torch peak allocated9.935GiB/reserved12.113GiB였고 device/Windows 전체 peak는 미측정입니다. 다른 작업을 종료하거나 CPU offload·truncate·KV quantization으로 protocol을 바꾸지 않습니다. 예산은7,200worker wall초이며 loader/import/idle도 포함합니다. 이번 경로는 복원된 BF16 참조 계산이고 packed serving speedup/최대 문맥 결과가 아닙니다.

## 원본·공개 경로

로컬 결과는 원 hash/명령/개인 경로를 그대로 보존합니다. `records/`는 [export 규칙](export_public.py)으로 경로만 치환한 공개 준비입니다. 원 runtime file의 hash는 external identity이며 공개 redacted bytes의 hash와 다릅니다. 각각 export map과 별도 manifest로 구분합니다. 원 Case013 manifest를 수정하지 않았고 GitHub push·PR·merge·Pages는 하지 않았습니다.
