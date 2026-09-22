# Reproduction and reanalysis

Run from the repository root. Use a **new external output directory** each time. Existing cases, model artifacts and their manifests must remain unchanged.

## Public CPU route

Python3.12 was used. `verify_public.py` and the installable comparison core use the standard library; analysis/tests additionally use NumPy, and plots use Matplotlib. [Recorded versions](provenance/package_versions.json) distinguish the original serving environment, the separate evaluation environment and CPU analysis. There is no root command that installs or upgrades every GPU dependency.

```bash
REPO="$(pwd)"
OUT="$(mktemp -d)"
CASE009="$REPO/cases/009-q-serving-quality"
python -B -m unittest discover -s "$CASE009/tests" -v
python -B "$CASE009/scripts/verify_public.py" \
  --case "$CASE009" --output "$OUT/scalar-audit.json"
python -B "$CASE009/scripts/report.py" \
  --serving "$CASE009/results/derived/serving_summary.json" \
  --quality "$CASE009/results/derived/quality_summary.json" \
  --output "$OUT/report"
```

The scalar audit is a second implementation of point metrics, request pairing and public file checks. It does not independently rerun GPU inference, bootstrap intervals, private generation extraction or full-vocabulary logits. WikiText denominators can be recomputed from private official samples; public scalars retain those denominators. Rendering a figure again does not create new measurements. Matplotlib SVG creation metadata can prevent byte identity even when coordinates and values agree.

The [standalone CLI](../../packages/diova-compare/README.md) installs into a separate CPU environment. It can run outside this repository. A small historical demo is:

```bash
python -m pip install "$REPO/packages/diova-compare"
python -B "$REPO/packages/diova-compare/examples/export_case008.py" \
  --case "$REPO/cases/008-build-reconstruct-reload" --output "$OUT/historical"
cd "$OUT"
diova-compare compare --baseline historical/Q-BF16.jsonl \
  --candidate historical/Q-W4.jsonl --output paired-report
```

These 192 pairs are marked **historical Case008**. They are not Case009 quality results. The new official-task adapter keeps benchmark/filter/subject identities and official metric names.

For the newly retained official-task scalars, export each metric/filter separately and compare matching files:

```bash
python -B "$REPO/packages/diova-compare/examples/export_case009.py" \
  --case "$CASE009" --output "$OUT/official"
diova-compare compare \
  --baseline "$OUT/official/arc_challenge-none-acc-BF16.jsonl" \
  --candidate "$OUT/official/arc_challenge-none-acc-W4.jsonl" \
  --output "$OUT/arc-pairs"
```

The optional process-exit field survives in the CLI output. Stored metrics from failed finalization remain qualified evidence. Generic item-bootstrap intervals use a separate seed and 2,000 draws; they are not advertised as byte-identical to the study's 5,000-draw or MMLU subject-cluster intervals.

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
