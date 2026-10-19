# Reproduce recorded results separately from model inference

## Model-free CPU path

Run from repository root with a fresh external output:

```bash
P2=cases/015-tool-interface-sensitivity/followups/json-recovery-p2-v1
OUT=$(mktemp -d /tmp/case015-p2-review-XXXXXX)
python3 -B "$P2/publication/run_checks.py" --output "$OUT/checks" --tests
python3 -B "$P2/publication/derive.py" --results "$P2/results/gpu_r2" --output "$OUT/tables"
```

The stdlib path restores exact older closed views, runs original P1/A1/publication checks and CPU/execution boundaries, reparses raw responses into fresh fixtures and checks both original assess and independent_endpoint before recomputing tables/decision. Old hash manifests are not edited. Add `--grammar-python "$CPU_PYTHON"` for the five grammar tests using existing [pinned CPU dependencies](publication/requirements-cpu.txt). No local research-environment installation/upgrade is performed. A hosted ephemeral CPU runner installs this lock.

With an existing local snapshot/tokenizer:

```bash
"$CPU_PYTHON" -B "$P2/publication/token_audit.py" \
  --results "$P2/results/gpu_r2" --snapshot "$MODEL_SNAPSHOT" --output "$OUT/token-audit.json"
```

This locally passed 175 input tokenizations/output decodes and 24 G masks / 1,118 generated tokens. Without the tokenizer, mark it NOT_RUN. Public CPU replay does not regenerate model logits or independently reproduce GPU inference. P0 MC and the old P1/A1 GPU matrix are not rerun.

## Actual GPU reproduction

Use a separate checkout at execution source **828ac7358b653626bed7e0a609448e73d120f1bb**, not the later result/guide commit. Reuse the existing local snapshot and saved RTN/D16 files read-only, with the original private runtime schema:

```bash
P2=cases/015-tool-interface-sensitivity/followups/json-recovery-p2-v1
python3 -B "$P2/execution/run.py" --help
python3 -B "$P2/execution/run.py" --runtime "$EXISTING_RUNTIME" \
  --gpu-python "$EXISTING_GPU_PYTHON" --output "$NEW_PRIVATE_RESULT_ROOT"
```

runtime.model.path, artifact_root and original_result_root must point to the held original inputs. Gates hash 12 snapshot files and 252projections per quantized method, check actual shapes/W_pre/beta and unchanged parameters, exact tokenizer/template/config/numerical flags, then 24 new C0 input/output references. A mismatch blocks G/N/E. Existing GPU device access is required; do not install a driver/model or terminate other work to force access.

Use `--prior-receipt` for preserved zero-response infrastructure failures, debiting the same 7,200 s cap. Outputs are fresh and complete cells are never overwritten. CPU imports remain guarded; the isolated GPU process explicitly imports Torch/Transformers. Model-free replay is a different verification scope.

## Recreate the public copy from held private records

```bash
python3 -B "$P2/publication/collect.py" --private-root "$PRIVATE_P2_ROOT" \
  --runtime "$EXISTING_RUNTIME" --gpu-python "$EXISTING_GPU_PYTHON" --output-p2 "$NEW_PUBLIC_COPY_ROOT"
```

The original records are not edited. Path-bearing metadata/logs are separately normalized with original/public hashes. Collector reproduction matched 163 files byte-for-byte. Two earlier collector path-label failures are retained and did not affect inference or measurement. Model weights, packed payloads, environment, private runtime and full logits are excluded.
