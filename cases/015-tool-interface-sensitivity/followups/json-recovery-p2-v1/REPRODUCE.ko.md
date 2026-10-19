# P2 재현: 저장 기록 확인과 모델 실행을 구분합니다

## 1. 모델 없이 결과 확인

저장소 루트에서 실행합니다. 표준 라이브러리만으로 원 P1/A1 closed view, CPU 준비/실행 경계, 새 raw fixture 재생과 표 재계산을 확인합니다. 기존 성공 파일을 덮어쓰지 않는 새 외부 출력이 필요합니다.

```bash
P2=cases/015-tool-interface-sensitivity/followups/json-recovery-p2-v1
OUT=$(mktemp -d /tmp/case015-p2-review-XXXXXX)
python3 -B "$P2/publication/run_checks.py" --output "$OUT/checks" --tests
python3 -B "$P2/publication/derive.py" --results "$P2/results/gpu_r2" --output "$OUT/tables"
```

문법5개까지 확인하려면 `--grammar-python "$CPU_PYTHON"`을 추가합니다. CPU_PYTHON은 lm-format-enforcer0.11.3과 [고정 CPU 의존성](publication/requirements-cpu.txt)이 **이미 있는** Python 경로입니다. 로컬 연구 환경 설치/업그레이드를 지시하지 않습니다. CI의 깨끗한 CPU runner만 이 lock을 설치합니다.

`run_checks.py`는 원 manifest의 파일을 정확히 복원한 두 view를 만듭니다. `legacy_case015_view`는 원457개 파일로 과거 publication/scripts/run_checks.py --tests를 그대로 실행합니다. `cpu_execution_view`는 CPU 준비 당시71개와 실제 GPU 실행 소스를 복원하여 CPU20/문법5/실행경계9개를 검사합니다. 복원 view 밖에서 과거 닫힌 manifest를 느슨하게 만들지 않습니다. 결과에 기존 검사51개와 신규 검사를 구분해 남깁니다.

`derive.py`는 raw를 재파싱·fixture에 재실행한 후 원 assess/independent_endpoint를 확인해 표·paired counts·decision을 다시 만듭니다. 저장 success bool만 합산하는 경로가 아닙니다. 산출 표는 [CSV/JSON](results/gpu_r2/tables)에 있습니다. 예전 A1/P1의 모델 추론과 P0 Monte Carlo는 다시 실행하지 않습니다.

## 2. 로컬 tokenizer가 있을 때의 추가 CPU 검사

MODEL_SNAPSHOT은 기존 revision의 로컬 snapshot입니다. 가중치나 tokenizer 다운로드는 이 명령에 없습니다.

```bash
"$CPU_PYTHON" -B "$P2/publication/token_audit.py" \
  --results "$P2/results/gpu_r2" --snapshot "$MODEL_SNAPSHOT" --output "$OUT/token-audit.json"
```

이번 실행에서는175개 입력을 다시 토큰화했고175개 output ID를 raw/parser text로 복호화했으며 G의24개 focal,1118개 token mask를 독립 CPU 구현으로 다시 계산했습니다. 일반 clone에 tokenizer가 없으면 이 검사는 미실행입니다. 공개 scalar/fixture audit가 모델 logits 또는 GPU inference 자체를 재현한다고 표시하지 않습니다.

## 3. GPU 실험을 재현할 조건

모델 추론 소스 commit은 **828ac7358b653626bed7e0a609448e73d120f1bb**입니다. 결과/문서 commit과 다릅니다. 이 commit을 새 checkout/worktree로 열어 실행해야 하며 현재 통합 안내를 source commit에 소급 적용하지 않습니다. 모델 원본과 RTN/D16 저장본을 read-only로 사용합니다. 필요한 private runtime schema는 기존 Case015의 실행 receipt/모델 adapter와 같습니다. private runtime 자체를 공개하지 않습니다.

```bash
# 위 source commit의 별도 checkout 안에서, 실제로 보유한 경로를 지정합니다.
P2=cases/015-tool-interface-sensitivity/followups/json-recovery-p2-v1
python3 -B "$P2/execution/run.py" --help
python3 -B "$P2/execution/run.py" --runtime "$EXISTING_RUNTIME" \
  --gpu-python "$EXISTING_GPU_PYTHON" --output "$NEW_PRIVATE_RESULT_ROOT"
```

runtime은 model.path, artifact_root, original_result_root를 기존 로컬 입력에 연결합니다. 모델 revision/tokenizer/template/config·snapshot12파일·RTN/D16 각252파일·actual coverage/W_pre/beta/small weights·dtype/flags를 검증합니다. GPU 장치 접근이 가능한 실행이 필요합니다. 이번 sandbox 실패처럼 장치가 보이지 않으면 환경을 바꾸어 설치하지 말고 실제 접근 조건을 구분합니다.

pre-response 인프라 attempt가 있다면 보존된 receipt를 `--prior-receipt`로 전달해 같은7200초 cap에서 차감합니다. 모델 응답이 이미 있는 다른 source의 결과를 이어 붙이는 옵션은 없습니다. 새 output root는 비어 있어야 합니다. C0재현24/24가 안 맞으면 G/N/E를 멈추고 원인 해석을 보류합니다. 새 실패를 좋은 seed/설정으로 재시도하지 않습니다.

GPU worker는 Torch/Transformers를 명시적으로 import하는 별도 프로세스입니다. CPU 경로의 import 금지는 그대로입니다. 원 solver/encode/apply_artifacts/run_all은 금지하고 기존 Adapter.generate를 재사용합니다. G만 FiniteBudget 뒤에 실제 grammar mask를 연결합니다. [추가 실행 등록](execution/spec.json)과 [소스 동결](execution/source_freeze.json)을 확인합니다.

## 4. private 기록에서 공개본 재생성

이번 수집은 [collect.py](publication/collect.py)로도 별도 출력에 재현하여 **163개 파일의 bytes/hash 일치**를 확인했습니다. private-root에는 실제 run_r1/run_r2와 CPU 감사/검산 기록이 필요합니다. 이는 모델을 다시 실행하는 명령이 아닙니다.

```bash
python3 -B "$P2/publication/collect.py" --private-root "$PRIVATE_P2_ROOT" \
  --runtime "$EXISTING_RUNTIME" --gpu-python "$EXISTING_GPU_PYTHON" --output-p2 "$NEW_PUBLIC_COPY_ROOT"
```

원문 응답은 그대로이며 개인 경로가 포함된 metadata/log만 label로 정리합니다. [sanitization.json](publication/sanitization.json)에 원본/공개 SHA를 구분합니다. 초기 collector의 패키지 경로 label 탐색 실패2회도 게시 준비 기록에 남겼고 GPU 실행 소스/측정 결과에는 영향이 없습니다.
