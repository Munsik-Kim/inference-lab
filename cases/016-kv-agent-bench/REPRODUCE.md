# Case016 reproduction: model-free checks and separate GPU measurement

Run from the repository root. The CPU route needs only Python3 standard library, without Torch/Transformers/vLLM imports or server requests. Choose a new external output directory.

```bash
python3 -B cases/016-kv-agent-bench/publication/run_checks.py --output /tmp/case016-check-new --tests
```

The route verifies the public manifest,restores447 originals,checks260 hashes in11 historical source views,replays96 episodes/478 SSE records,reassembles new technical SSE and recomputes capacity/throughput,then matches display JSON. It reruns33+3 historical tests and15 audit tests; one EVAL-generating test is excluded. The16 new technical tests are counted separately. CPU replay does not reproduce GPU inference.

```bash
TECH=cases/016-kv-agent-bench/followups/nvfp4-technical-r1
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TECH/src" python3 -B -m nvfp4_technical.audit --root "$TECH" --output /tmp/case016-raw-new
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TECH/src" python3 -B -m nvfp4_technical.plots --analysis /tmp/case016-raw-new --output /tmp/case016-figures-new
```

These commands regenerate CSVs and8 SVGs into new directories; retained analysis/figures are not overwritten.

```bash
python3 -B cases/016-kv-agent-bench/publication/prepare_gpu_copy.py --output /tmp/case016-gpu-input-new
```

This CPU command prepares only a source/input copy, without running a model.

## Separate GPU conditions

Model/environment/payloads are not included. Reproduction requires the pinned local Qwen snapshot and isolated runtime used here:Python3.12.14,vLLM0.32.0,FlashInfer0.7.0.post1,Torch2.13.0/CUDA13.0,driver610.88. No source build or driver change was performed. The [environment record](provenance/environment.json) retains the lock and wheel SHA.

The runner refuses to overwrite a completed run. Use a new scratch case copy containing the frozen protocol/data and read-only original src. Supply these fields in a private runtime JSON,using actual local paths:

```json
{"python":"<EXISTING_GPU_PYTHON>","snapshot":"<PINNED_LOCAL_SNAPSHOT>","private_root":"<NEW_OWNED_SCRATCH>","original_case":"<SCRATCH_CASE>","technical_root":"<SCRATCH_CASE>/followups/nvfp4-technical-r1","port":8152}
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TECH/src" "$CASE016_GPU_PYTHON" -B -m nvfp4_technical.runner --runtime "$CASE016_PRIVATE_RUNTIME"
```

The GPU command is a separate executable path,never called by CPU verification. server.py pins the launch command. A new GPU rerun needs a scratch input/source copy with an empty results directory and a separately retained historical execution freeze; its new freeze is a new run identity. It is not automatically equivalent to this run. Keep the snapshot,other services and existing environment unchanged; manage only the owned loopback server.
