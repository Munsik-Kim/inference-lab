# Case018 P2 — 실제 실행과 모델 없는 재생 명령

명령의 cwd는 다음 case root입니다.

`<LOCAL_HOME>/Codex_main/inference-lab-case018-state-recovery-20261022-093529/cases/018-agent-state-recovery`

## 모델 없는 전체 endpoint 재생

```bash
PYTHONPATH=src python3 -B -m agent_state_recovery.p2 audit \
  --run results/p2_r1 --output <LOCAL_HOME>/Codex_main/case018-p2-evidence-20261022-114400/p2_final_audit.json
```

실제 실행해96/96, endpoint/raw 차이0을 확인했습니다. output과 `*-replayed-states`는 이미 있으므로 재실행은 새 경로를 지정해야 합니다. 표준 라이브러리 경로이며 torch/vLLM/tokenizer/GPU/서버를 사용하지 않습니다. 동결 source·원 tool response와 상태/두 branch를 다시 계산하고 SSE를 별도 조립합니다. strict parser/serialization/공개 builder는 공유합니다.

로컬의8개 읽기 전용 helper 파일 bytes, frozen spec이 참조하는 외부 request journal과 run 자료는 필요합니다. 이 TXT만으로 외부 GPU 재현이 가능한 것은 아닙니다. source가 달라지면 audit는 hash 오류로 거절하며, 원본은 외부 `frozen-execution-source/`에 있습니다. 원 P1 audit receipt/source는 수정하지 않았습니다.

```bash
PYTHONPATH=src python3 -B -m agent_state_recovery.p2 analyze \
  --run results/p2_r1 --audit <LOCAL_HOME>/Codex_main/case018-p2-evidence-20261022-114400/p2_final_audit.json \
  --output results/p2_r1/analysis
```

실제로96행 장부·4셀·12family행·8짝 표와 등록 bootstrap을 생성했습니다. 이미 있는 analysis는 덮어쓰지 않습니다. 재검산은 새 directory를 사용합니다. 통계는 원 task의4조건을 함께 재표집하며 성공수/비용을 수기로 넣지 않습니다.

## 기존 runtime의 CPU tokenizer/grammar 검사

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  <LOCAL_HOME>/Codex_main/case016-kv-agent-private-20261019/venv_r2/bin/python -B \
  -m agent_state_recovery.p2 tokencheck --run results/p2_r1 \
  --runtime <LOCAL_HOME>/Codex_main/case018-p2-evidence-20261022-114400/runtime.json --output <LOCAL_HOME>/Codex_main/case018-p2-evidence-20261022-114400/p2_tokencheck.json
```

모델을 로드하지 않고 실제 local tokenizer와 원 schema의 속성 순서/compiled grammar로728개 원 token prefix/EOS를 검사해 PASS였습니다. sorted freeze JSON을 schema compile 입력으로 쓰지 않습니다. TECH3개의 별도 CPU 원문 검사도 `tech_raw_tokencheck.json`에 있습니다. 원장상 실제 모델 요청은 검산 중0입니다.

## 생성·GPU 실행의 정확한 기록 (재실행 지시가 아님)

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  <LOCAL_HOME>/Codex_main/case016-kv-agent-private-20261019/venv_r2/bin/python -B \
  -m agent_state_recovery.p2 prepare --output results/p2_r1 \
  --runtime <LOCAL_HOME>/Codex_main/case018-p2-evidence-20261022-114400/runtime.json --evidence <LOCAL_HOME>/Codex_main/case018-p2-evidence-20261022-114400 \
  --parent <LOCAL_HOME>/Codex_main/case018-p1-evidence-20261022-101706

PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  <LOCAL_HOME>/Codex_main/case016-kv-agent-private-20261019/venv_r2/bin/python -B \
  -m agent_state_recovery.p2 run --output results/p2_r1 \
  --runtime <LOCAL_HOME>/Codex_main/case018-p2-evidence-20261022-114400/runtime.json --evidence <LOCAL_HOME>/Codex_main/case018-p2-evidence-20261022-114400 \
  --parent <LOCAL_HOME>/Codex_main/case018-p1-evidence-20261022-101706
```

prepare는 P1 gate·source·원자료 identity 뒤 등록 generator를 한 번 호출하고72 control+24 guard 예산 script·실제 토큰화를 거쳐 input seal을 씁니다. 기존 HOLDOUT/data·generation journal 또는 output은 재생성을 거절합니다. run은 그 source/seal을 확인해3개 owned worker/TECH3/모델96을 실행했습니다. 기존 attempt root는 자동 resume/retry하지 않습니다. 다음 실행 허가로 확대하지 않습니다.

실행에는 원 P1의 기존 로컬 Qwen snapshot·vLLM0.32.0 환경과 localhost/GPU 권한, GPU 가용 자원이 필요합니다. 모델·패키지 다운로드/설치나 다른 서비스 종료는 없었습니다. 바뀐 runtime 항목은 소유 private 경로와 port8197입니다. 원형/새 source/입력/원장/기술 receipt는 execution_freeze/input_seal 및 외부 evidence에 있으며 모델 weight나 큰 profiler를 배포하지 않습니다.

원 P1의186시도/184응답·미관측 취소2건·worker439.692244164초를 읽기 전용 parent로 유지합니다. P2는731시도/731응답·worker855.069382600초이고 누적1294.761626764초입니다. 입력/출력과 episode/CPU/worker 비용을 구분합니다. 정확한 누적 output은 과거 미관측2건 때문에 null이며 관측 하한57987 tokens입니다.

문서·공통표는 [facts.json](results/p2_r1/analysis/facts.json)에서 만들었고, 출력 후 추가한 수량 관계 진단은 별도 파일입니다. P0/P1 source/raw/보고서606개 기존 파일과 선택 보존 범위는 동일합니다. source는 미커밋 working bundle이며 Git base와 구분합니다. UI/사이트/Git 원격 게시/추가 모델 실험은 하지 않았습니다.
