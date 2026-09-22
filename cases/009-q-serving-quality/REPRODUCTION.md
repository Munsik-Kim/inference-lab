# Reproduction and reanalysis

Run from the repository root. Use a **new external output directory** each time. Existing cases, model artifacts and their manifests must remain unchanged.

## Three separate reproduction routes

### 1. Model-free scalar reanalysis

[Current publication instructions](publication/README.md) verify current identity and restore the original reviewed tree into a new external directory before running its frozen verifier/tests. `publication/posthoc/analyze.py` recomputes the new descriptive tables from unchanged public scalars. Original intervals, input freezes and full process exits are retained. NumPy is needed for original study tests; identity/posthoc/CLI core use the standard library.

### 2. Corrected wheel and actual records

From the publication ZIP or repository root, create a dedicated Python environment and new output path:

```bash
python -m venv /path/to/new-cpu-env
/path/to/new-cpu-env/bin/python -m pip install --no-index --no-deps downloads/diova_compare-0.1.1-py3-none-any.whl
/path/to/new-cpu-env/bin/python -B cases/009-q-serving-quality/publication/check_cli.py --root . --python /path/to/new-cpu-env/bin/python --output /path/to/new-record-check
```

This exports metric/filter-specific public inputs, executes the installed CLI outside the checkout, and verifies point metrics/counts and original nonzero exit codes. Item-bootstrap intervals from the generic CLI are not the original MMLU subject-cluster intervals. Full vectors, private benchmark text and generated rationales are not reconstructed.

### 3. Optional bounded GPU lifecycle diagnostic

[Lifecycle v1](supplemental/lifecycle-v1/README.md) uses the frozen existing artifacts/environment and short synthetic inputs. Its runner takes `--private-config` and a nonexistent `--output` directory. Read the frozen budget and protocol first. Twenty-four clean probes did not reproduce historical full-task termination failures; the original failures remain. This is separate from full benchmark reproduction below.

## Recorded GPU path

### Environment boundaries

The benchmark reused the existing runtime unchanged. Evaluation used a new Python 3.12.14 venv, installed the official harness checkout at the recorded commit, then referenced the two existing runtime site-package directories through a local `.pth` file. Its own dependencies take precedence over those shared runtime packages. This produced the versions in `package_versions.json`; it is a recorded local setup, not a portable lockfile or a recommendation to upgrade an existing environment.

That evaluation environment emitted an optional cuteDSL/Numba import warning with NumPy 2.5.3 and later showed abnormal native finalization in some complete calculations. The cause is unresolved. The benchmark's NumPy 2.3.5 environment was not changed. Reproducing a clean lifecycle needs a separately reviewed environment diagnostic; it is not established by the stored scalar checks.

The CPU package is independent: a clean venv can install its wheel with no model/runtime dependencies. Figure/analysis tools use the separately recorded NumPy and Matplotlib versions. Do not combine these three environments into one unpinned root install.

### Existing artifacts and measured commands

The original BF16 and W4 checkpoints are required locally, with the hashes in [artifacts.json](provenance/artifacts.json). No weights are included. If W4 is absent, stop with `BLOCKED_ARTIFACT`; do not silently rebuild or requantize it. The serving environment reuses installed vLLM0.29.0/Torch2.13.0/Transformers5.17.0. The harness environment separately installs pinned official task code without upgrading the original environment.

The study retained installed `serve --help`, `bench serve --help=all` and `lm-eval run --help` snapshots privately; [command provenance](provenance/commands.json) contains sanitized executed commands and help hashes. The custom loopback client uses the installed benchmark's timing convention while validating exact returned token IDs. It does not claim to have invoked `vllm bench serve` for the measured requests.

Private configuration has this shape, with absolute paths supplied by the operator:

```json
{"runtime_python":"/path/to/existing-runtime/bin/python","models":{"BF16":"/path/to/bf16-snapshot","W4":"/path/to/existing-w4-artifact"},"requests":"/path/to/frozen-requests.json"}
```

```bash
"$GPU_PY" -B "$CASE009/scripts/serving.py" sweep \
  --private-config "$PRIVATE_CONFIG" \
  --protocol "$CASE009/configs/serving_protocol.json" --work "$NEW_PRIVATE_RUN/serving"
```

Fresh server processes run sequentially. Completed directories are retained; an existing incomplete/failed directory blocks automatic retry. Do not run another GPU job concurrently. Request timeouts and process-start limits are in the runner; the outer study wall budget is monitored by orchestration. Only the process groups started by this study are stopped.

For quality, first freeze official task contexts with the pinned tokenizer and harness. This uses no model forward, but requires cached official datasets. DEV validates ARC/WikiText validation inputs before the full protocol is sealed. The full loop follows the frozen task/model order; one command per task/model is:

```bash
"$EVAL_PY" -B "$CASE009/scripts/freeze_quality_inputs.py" \
  --tokenizer "$BF16_SNAPSHOT" --output "$NEW_PRIVATE_FREEZE"
```

Compare the generated request/task file hashes with `quality_input_freeze.json` before using that directory as `PRIVATE_QUALITY_FREEZE`. Different task code, cached dataset revisions or tokenizer bytes require a new study identity; they must not silently replace the recorded freeze.

```bash
"$EVAL_PY" -B "$CASE009/scripts/quality.py" \
  --model "$MODEL_ARTIFACT" --tokenizer "$BF16_SNAPSHOT" \
  --task "$TASK" --arm "$ARM" --freeze "$PRIVATE_QUALITY_FREEZE" \
  --output "$NEW_PRIVATE_RUN/full-$TASK-$ARM"
```

Set HF caches to the verified private cache, HF/Transformers offline flags, `VLLM_USE_V2_MODEL_RUNNER=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_WORKER_MULTIPROC_METHOD=spawn` and a separate `VLLM_CACHE_ROOT`. The task arguments, generation limits and chat template are checked against the retained request manifest before forward calls. Full-task runs require no `--limit`; the script's `--dev` mode is explicitly a two-validation-document smoke.

Private measured records can be reanalyzed without another model run:

```bash
python -B "$CASE009/scripts/analyze_serving.py" --work "$PRIVATE_RUN/serving" \
  --protocol "$CASE009/configs/serving_protocol.json" \
  --requests "$CASE009/configs/requests.json" --output "$OUT/serving"
python -B "$CASE009/scripts/analyze_quality.py" --quality "$PRIVATE_RUN/quality" \
  --freeze "$CASE009/configs/quality_input_freeze.json" --output "$OUT/quality"
```

The [run state](RUN_STATE.json) and reports identify what actually completed. A missing pair or failed attempt remains visible. Regenerating this new case's inventory is an editorial operation, not a replacement for original artifact/protocol checks. Old scientific checksums are not regenerated.
