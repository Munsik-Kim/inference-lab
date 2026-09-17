# Reproduction

All commands run from this case directory. Output paths must be new and outside the source case. Existing successful ledger cells are immutable; use the same work directory and identical tokens/configuration to resume. A conflicting identity is an error, not a silent merge.

## CPU analysis of retained measurements

The tested CPU environment used Python 3.12.14, NumPy 2.3.5, Matplotlib 3.10.8 and Pillow 12.3.0. `requirements-cpu.txt` describes these versions, not a guarantee that every platform recreates the GPU environment. No model or GPU library is imported by the CPU analysis path.

```bash
python -B scripts/verify_publication.py --output /tmp/case006-publication-check-new.json
python -B scripts/analyze_study.py --records results/raw/paired/*.json --timing results/raw/timing/*.json --secondary results/raw/secondary/*.json --output /tmp/case006-analysis-new
python -B scripts/audit_study.py --records results/raw/paired/*.json --summary /tmp/case006-analysis-new/summary.json --output /tmp/case006-audit-new.json
python -B scripts/audit_timing.py --timing results/raw/timing/model-timing.json --summary /tmp/case006-analysis-new/summary.json --output /tmp/case006-timing-audit-new.json
python -B scripts/verify_inputs.py --evaluation-inputs inputs/evaluation --output /tmp/case006-input-audit-new.json
python -B scripts/plot_study.py --summary /tmp/case006-analysis-new/summary.json --pairs /tmp/case006-analysis-new/pairs.json --output /tmp/case006-figures-new
python -B scripts/write_report.py --summary /tmp/case006-analysis-new/summary.json --output /tmp/case006-narrative-new
python -B scripts/build_study_demo.py --summary /tmp/case006-analysis-new/summary.json --pairs /tmp/case006-analysis-new/pairs.json --inputs inputs/prompts.json inputs/evaluation/standard-*.json inputs/evaluation/boundary_pool-*.json inputs/evaluation/length-*.json --gold inputs/gold.json inputs/evaluation/gold.json --figures /tmp/case006-figures-new --subsets inputs/evaluation/subsets.json --secondary results/raw/secondary/secondary.json --output /tmp/case006-demo-new
python -B scripts/package_publication.py --zip /tmp/case006-public-new.zip --record /tmp/case006-package-new.json
```

The HTML embeds its data and figures. Open `index.html` directly in a browser. No server, external font, CDN, telemetry or runtime network request is required. File previewers that disable JavaScript may show only the static shell; tablet browser/file-preview support has not been verified. The Markdown, PNG and JSON files remain usable separately.

The historical `reproduce_analysis.py`, `verify_results.py` command and `build_demo.py` reproduce the CPU-only preparation state retained at the top level of `results/`. They are not the completed-study entry points. `audit_study.py` reuses only the historical verifier's independent scalar primitives, not its empty-state CLI.

## Pinned local GPU replay

GPU reproduction is a separate hardware experiment. It requires an already available pinned snapshot and the recorded working environment: PyTorch 2.13.0+cu130, Transformers 5.17.0, Triton 3.7.1 and SageAttention 2.2.0 at commit `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`. Source and extension hashes are in provenance. No script downloads weights, installs a kernel or repairs an unsupported backend.

Set `GPU_PY` to that environment's Python, `SNAPSHOT` to the verified local Qwen revision, and `CASE006_RUN` to a new private directory. These variables contain local paths and are not credentials. Do not substitute a different model/backend and claim the same experiment.

```bash
"$GPU_PY" -B scripts/run_case006.py --stage smoke --snapshot "$SNAPSHOT" --output "$CASE006_RUN/smoke.json"
"$GPU_PY" -B scripts/validate_integration.py --snapshot "$SNAPSHOT" --output "$CASE006_RUN/integration.json"
"$GPU_PY" -B scripts/verify_inputs.py --evaluation-inputs inputs/evaluation --snapshot "$SNAPSHOT" --output "$CASE006_RUN/input-audit.json"
"$GPU_PY" -B scripts/measure_scores.py --stage eval --snapshot "$SNAPSHOT" --inputs inputs/evaluation/standard-*.json --gold inputs/evaluation/gold.json --semantic "$CASE006_RUN/integration.json" --freeze configs/eval_manifest_freeze_v1.json --work-dir "$CASE006_RUN/standard-eval"
"$GPU_PY" -B scripts/measure_scores.py --stage eval --snapshot "$SNAPSHOT" --inputs inputs/evaluation/boundary_pool-*.json --gold inputs/evaluation/gold.json --semantic "$CASE006_RUN/integration.json" --freeze configs/eval_manifest_freeze_v1.json --selection configs/boundary_selection_v1.json --work-dir "$CASE006_RUN/boundary-eval"
"$GPU_PY" -B scripts/measure_scores.py --stage length --snapshot "$SNAPSHOT" --inputs inputs/evaluation/length-*.json --gold inputs/evaluation/gold.json --semantic "$CASE006_RUN/integration.json" --freeze configs/eval_manifest_freeze_v1.json --work-dir "$CASE006_RUN/length"
"$GPU_PY" -B scripts/run_secondary.py --snapshot "$SNAPSHOT" --inputs inputs/evaluation/secondary-*.json --gold inputs/evaluation/gold.json --freeze configs/eval_manifest_freeze_v1.json --work-dir "$CASE006_RUN/secondary"
for round in 0 1 2; do
  "$GPU_PY" -B scripts/measure_model_cost.py --snapshot "$SNAPSHOT" --inputs inputs/evaluation/standard-*.json --subsets inputs/evaluation/subsets.json --freeze configs/eval_manifest_freeze_v1.json --process-round "$round" --work-dir "$CASE006_RUN/timing" || break
  sleep 3
done
```

The short pause permits the just-finished process's activity sample to settle; it does not terminate or preempt another job. The resource gate still checks five utilization/free-memory samples. GPU jobs must be serial. Preserve failure records and stop on numerical/semantic failure rather than changing precision, masks, prompts or data.

These commands replay already published inputs. They are not fresh held-out confirmation. To reproduce the original preparation order, use `dev_presentation.py`, `measure_scores.py --stage dev-b|dev-paired|pool`, `freeze_protocol.py` and `prepare_evaluation.py` with new output files. The original freezes must not be overwritten or renamed as a new preregistration.

Model captures, reference calculations and full-vocabulary vectors stay in private work directories. Only scalar payloads should be exported using `collect_public.py`. Public scalars support arithmetic auditing; full GPU replay is required to independently reproduce kernel outputs.


## Connecting original replay to the readout supplement

`CASE006_DIR` is the absolute path to this published case directory. `CASE006_RUN`
is the private root holding the original measurement outputs; `CASE006_READOUT_RUN`
is a separate, new private directory for readout outputs. The commands above create:

```text
$CASE006_RUN/
  standard-eval/private_vectors/
  boundary-eval/private_vectors/
  length/
  secondary/
  timing/
```

Pass the root `CASE006_RUN` to `--original-run`, not either dataset subdirectory.
Run the following example from the supplement directory:

```bash
cd "$CASE006_DIR/supplemental/readout-ties-v1"
"$GPU_PY" -B scripts/run_readout.py \
  --original-case "$CASE006_DIR" \
  --original-run "$CASE006_RUN" \
  --snapshot "$SNAPSHOT" \
  --work-dir "$CASE006_READOUT_RUN"
```

This is a command for a future pinned-local GPU reproduction, not a command run
for the documentation fix. It requires the complete private vectors and the same
pinned model/environment. Public scalar records cannot reconstruct hidden states.
The CPU documentation path-contract check verifies directory agreement only; it
does not establish fresh-install GPU reproduction or numerical identity.

## Publication package verification

The original 59-test suite was rerun for publication in a byte-verified original-only scratch tree extracted from the historical reviewed archive (SHA256 `79a215e73cf6ab30931392d67bb658cc18deee1034fe3b488ba6a78d7d12220d`). It intentionally checks that original exact inventory. Its frozen tests/helpers were not changed to accept new supplemental files or edited presentation documents. Running that inventory assertion on the combined tree is a scope mismatch, not an instruction to weaken the original check.

For the combined tree, run the new additive package verifier, original measured-study analysis/auditors above, and the supplement's CPU tests and independent checker. Use new output paths:

```bash
python -B scripts/verify_publication.py --output /tmp/case006-publication-verification-new.json
python -B -m unittest discover -s publication_tests -v
python -B scripts/package_publication.py --zip /tmp/case006-publication-new.zip --record /tmp/case006-package-new.json
cd supplemental/readout-ties-v1
python -B -m unittest discover -s tests -v
python -B scripts/readout_analysis.py --case . --output /tmp/case006-readout-analysis-new
python -B scripts/verify_scalar.py --case . --summary /tmp/case006-readout-analysis-new/summary.json --output /tmp/case006-readout-audit-new.json
```

The combined verifier checks both original and supplemental measured-file identities against the approved snapshots, exact protocol/raw hashes, the presentation-change allowlist and the complete publication manifest. Original `SHA256SUMS` and historical status records remain preserved as historical evidence. `PUBLICATION_SHA256SUMS` is the separate current packaging manifest. Neither contains a hash of itself or of the surrounding ZIP.

CPU reanalysis checks saved scalar arithmetic and intervals; reproducing full-vocabulary/hidden checks requires excluded private tensors or the original pinned capture procedure. No new GPU experiment was run for publication. `provenance/github_publication_validation_20260917.json` records tests and analysis actually rerun during this publication preparation.
