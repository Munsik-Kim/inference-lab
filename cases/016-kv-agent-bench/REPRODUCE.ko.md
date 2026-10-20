# Case016 재현: 모델 없는 검산과 별도 GPU 실측

저장소 root에서 실행합니다. CPU 경로는 Python3 표준 라이브러리만 필요하며 Torch/Transformers/vLLM을 import하거나 서버에 접속하지 않습니다. 출력은 새 외부 디렉터리여야 합니다.

```bash
python3 -B cases/016-kv-agent-bench/publication/run_checks.py --output /tmp/case016-check-new --tests
```

이 명령은 공개 manifest→원형447개 복원→11개 과거 source view260개 hash 검사→기존96 episode/478 SSE 재생→새 기술 raw SSE 재조립·용량/처리량 재계산→표시 JSON 일치 순서입니다. 과거33+3개 검사와 감사15개를 회귀 재실행하며 EVAL 생성 검사1개는 제외합니다. 신규 기술 검사16개와 과거 검사 수를 분리합니다. CPU replay는 GPU inference 재현이 아닙니다.

```bash
TECH=cases/016-kv-agent-bench/followups/nvfp4-technical-r1
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TECH/src" python3 -B -m nvfp4_technical.audit --root "$TECH" --output /tmp/case016-raw-new
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TECH/src" python3 -B -m nvfp4_technical.plots --analysis /tmp/case016-raw-new --output /tmp/case016-figures-new
```

CSV와8개 SVG는 위 명령으로 재생성합니다. 기존 analysis/figure는 덮어쓰지 않습니다.

```bash
python3 -B cases/016-kv-agent-bench/publication/prepare_gpu_copy.py --output /tmp/case016-gpu-input-new
```

위 CPU 명령은 모델을 실행하지 않고 재현용 source/input 사본만 만듭니다.

## 별도 GPU 재현 조건

원 model/환경/payload는 포함하지 않습니다. Python3.12.14,vLLM0.32.0,FlashInfer0.7.0.post1,Torch2.13.0/CUDA13.0,driver610.88을 사용한 기존 격리 환경과 고정 revision의 로컬 Qwen snapshot이 필요합니다. source build나 driver 변경은 수행하지 않았습니다. 원 환경 lock·wheel SHA는 [기록](provenance/environment.json)에 있습니다.

완료된 run root를 다시 실행하면 overwrite를 거절합니다. 새 scratch case 사본에 준비된 protocol/data와 읽기 전용 원 src를 두고, private runtime JSON에 다음 필드를 지정합니다. 경로는 사용자 장비의 실제 위치입니다.

```json
{"python":"<EXISTING_GPU_PYTHON>","snapshot":"<PINNED_LOCAL_SNAPSHOT>","private_root":"<NEW_OWNED_SCRATCH>","original_case":"<SCRATCH_CASE>","technical_root":"<SCRATCH_CASE>/followups/nvfp4-technical-r1","port":8152}
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TECH/src" "$CASE016_GPU_PYTHON" -B -m nvfp4_technical.runner --runtime "$CASE016_PRIVATE_RUNTIME"
```

위 GPU 명령은 실행 가능한 별도 경로이며 CPU 확인에서 호출하지 않습니다. server command는 runner/server.py에 고정되어 있습니다. 새 GPU 재현에는 기존 results/와 execution_freeze를 원형에서 분리한 빈 출력 사본이 필요하며 새 결과는 새 freeze를 가집니다. 원 run과 같은 결과라고 자동 간주하지 않습니다. 원 snapshot·다른 서비스·기존 환경은 변경하지 않고 소유한 loopback 서버만 관리합니다.
