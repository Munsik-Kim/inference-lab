# Case018 P1 — 저장 원자료 CPU 검산과 실제 실행 기록

아래 CPU 명령은 실제 실행했습니다. 새 output은 존재하지 않아야 하며 덮어쓰기를 거절합니다. case root에서 Python 표준 라이브러리로 저장된24개 DEV를 재파싱·상태 재적용·후속 두 branch 재계산합니다. 모델/서버/torch/vLLM/xgrammar import나 GPU 요청이 없습니다. 실제 tokenizer 재인코딩은 별도 CPU 검사입니다.

```bash
PYTHONPATH=src python3 -B -m agent_state_recovery audit \
  --run results/p1_r5 \
  --output <LOCAL_HOME>/Codex_main/case018-p1-evidence-20261022-101706/p1_final_audit.json
```

이 output은 이미 존재합니다. 재검산할 때는 `--output`을 새 경로로 바꿉니다. 독립 ReferenceState 결과·원 판정·raw trace 차이는24개 모두0이었습니다. 원문은 `results/p1_r5/episodes`, `DEV_K0_transport`, `DEV_K1_transport`, `inputs`에 있으며 source/입력/예정 key는 `execution_freeze.json`, `initial_inputs.json`, `plan.json`에 있습니다. 이 로컬 재생은 freeze에 등록된 읽기 전용 dependency 경로·runtime identity 파일도 검사하므로 해당 로컬 파일이 필요합니다. 이 안내를 외부 머신에서 완성된 재현으로 표현하지 않습니다.

집계는 같은 audit의 `independently_recomputed`와 실제 raw 회계에서 계산했습니다.

```bash
PYTHONPATH=src python3 -B -c 'from agent_state_recovery.p1_analysis import analyze; print(analyze("results/p1_r5", "<LOCAL_HOME>/Codex_main/case018-p1-evidence-20261022-101706/p1_final_audit.json", "results/p1_r5/analysis")["totals"])'
```

이미 있는 `analysis`도 다시 쓰지 않습니다. 새 재계산은 새 directory를 지정합니다. CSV는24행 episode ledger,4행 cell,12행 family,8행 Q/W paired 표입니다. 성공값을 수기로 입력하지 않았습니다.

## 기존 runtime의 별도 CPU token 검사

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  <LOCAL_HOME>/Codex_main/case016-kv-agent-private-20261019/venv_r2/bin/python -B \
  <LOCAL_HOME>/Codex_main/case018-p1-evidence-20261022-101706/retokenize_saved_r2.py \
  --runtime <LOCAL_HOME>/Codex_main/case018-p1-evidence-20261022-101706/runtime.json \
  --run <LOCAL_HOME>/Codex_main/inference-lab-case018-state-recovery-20261022-093529/cases/018-agent-state-recovery/results/p1_r5 \
  --output <LOCAL_HOME>/Codex_main/case018-p1-evidence-20261022-101706/p1_retokenization_r2.json
```

기존 tokenizer/xgrammar만 사용하며 모델을 로드하지 않습니다. 180개 요청의 render→encode·special token decode·모든 생성 prefix/EOS가 일치했습니다. 원 schema bytes의 순서와 실행 전 compiled grammar를 대조합니다. 최초 검사기의 실패도 저장했습니다. 실행 후 작성한 검사기이므로 실험 전 source freeze에는 포함하지 않습니다.

## 실제 GPU 실행 명령 (기록이며 이번에는 재실행하지 않음)

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  <LOCAL_HOME>/Codex_main/case016-kv-agent-private-20261019/venv_r2/bin/python -B \
  -m agent_state_recovery p1 \
  --output results/p1_r5 \
  --runtime <LOCAL_HOME>/Codex_main/case018-p1-evidence-20261022-101706/runtime.json \
  --preparation <LOCAL_HOME>/Codex_main/case018-p1-evidence-20261022-101706/preparation02 \
  --evidence <LOCAL_HOME>/Codex_main/case018-p1-evidence-20261022-101706
```

승인된 로컬 CPU/GPU 환경과 loopback 실행 권한이 필요합니다. 이 명령 기록은 추가 실행이나 HOLDOUT/P2 승인을 뜻하지 않습니다.

실행 입력JSON, snapshot/runtime/dependency hash, 실제 launch command/PID/start-time/log/정상 shutdown, append-only request/worker journal은 외부 evidence에 저장했습니다. `specs/p1_execution.json`과 amendment01–04가 실행 계약입니다. weight/환경/원 KV tensor를 새로 배포하지 않습니다. 모델 weight 전체의 현재 hash는 재계산하지 않았으며 작은 설정/tokenizer 파일 hash와 weight 존재/크기/역사적 identity 대조 범위입니다.

P0는 확장 전 source의 exact-byte snapshot으로 `p0_frozen_replay.json`에26trace를 재생했습니다. 현재 source를 과거 P0의 실행 source로 부르지 않습니다. P1 실행 source identity와 Git base는 별개입니다. CPU mock·tokenizer 검사·technical forward·생성API·모델 episode는 각각 구분해 셉니다.
