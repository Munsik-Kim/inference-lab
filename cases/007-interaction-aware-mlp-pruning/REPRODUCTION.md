# Reproduction

Reading the reports and offline `demo/index.html` needs no GPU or network service. GPU reproduction is separate from checking scalar arithmetic. Existing Cases001–006 and their packages remain unchanged.

## CPU

Use an existing Python environment with NumPy (tested versions in provenance). Torch is optional for the core unit suite; if absent, its one CPU structural-slicing test is explicitly skipped. Matplotlib is needed only to recreate figures. This task installs nothing.

From the case directory, with a new output path outside the repository:

```bash
python -B -m unittest discover -s tests -v
python -B scripts/analyze.py --raw results/raw --output "$ANALYSIS_NEW"
python -B scripts/verify.py --raw results/raw --summary "$ANALYSIS_NEW/summary.json" --output "$VERIFY_NEW_JSON"
python -B scripts/verify_completion.py --case . --summary "$ANALYSIS_NEW/summary.json" --output "$INTERVAL_VERIFY_NEW_JSON"
python -B scripts/report.py --case . --summary "$ANALYSIS_NEW/summary.json" --output "$REPORT_NEW"
python -B scripts/completion_notes.py --case . --output "$NOTES_NEW_JSON"
```

All output variables must designate new paths outside the case. `ANALYSIS_NEW` and `REPORT_NEW` are directories; the other outputs are JSON files. Compare regenerated summary/pairs against `results/derived/`, and generated figures/demo against their public counterparts. Generated README/ANALYSIS are numerical table drafts; the public versions add reviewed explanatory text. Scalars reproduce aggregate calculation, not hidden tensors or GPU outputs. Full-vocabulary KL requires excluded complete logit vectors to verify independently. Public token diagnostics retain the worst sampled token per method/prompt; prompt sums and all method-level values are preserved. Private originals contain all 32 token diagnostics.

## GPU — fixed local setup only

Set `GPU_PY` to the existing tested environment, `SNAPSHOT` to the existing verified model snapshot, and `RUN_NEW` to a new private directory outside the case. No script downloads a model. All stages check the fixed source/input hashes and refuse to overwrite an existing stage. Use the published input/protocol files for a replay; `prepare.py` documents original CPU preparation and intentionally refuses to overwrite those files.

```bash
"$GPU_PY" -B scripts/run.py --stage smoke --snapshot "$SNAPSHOT" --work "$RUN_NEW"
"$GPU_PY" -B scripts/run.py --stage calibrate --snapshot "$SNAPSHOT" --work "$RUN_NEW"
"$GPU_PY" -B scripts/run.py --stage development --snapshot "$SNAPSHOT" --work "$RUN_NEW"
"$GPU_PY" -B scripts/run.py --stage heldout --snapshot "$SNAPSHOT" --work "$RUN_NEW"
"$GPU_PY" -B scripts/run.py --stage timing --round 0 --snapshot "$SNAPSHOT" --work "$RUN_NEW"
"$GPU_PY" -B scripts/run.py --stage timing --round 1 --snapshot "$SNAPSHOT" --work "$RUN_NEW"
"$GPU_PY" -B scripts/run.py --stage timing --round 2 --snapshot "$SNAPSHOT" --work "$RUN_NEW"
```

Run GPU stages serially, only while resources are available. A previous-stage validity failure blocks later stages. Do not interpret an existing failed-stage directory as permission to overwrite it or tune on held-out results. Investigate and record a correction before starting a distinct run. Published held-out inputs are no longer unseen in a reproduction. No generation, other layer/model or retraining stage exists in v1.

The offline explorer uses embedded scalar records, escaped text and local figures. GitHub HTML source preview is not an executing demo. Real iPad operation is not validated by this task.

## Completed resume and historical partial package

The initial resource gate blocked held-out access before any held-out forward. The preserved partial package has SHA256 `2cdb4d929a9503aa7ce2a3bedde62b9055cb93aaf1b4dcb9587b63efdb4ea5d6`. Its original partial-only CPU commands apply to that archive, not to the now-completed combined tree. `results/partial_summary.json`, historical validation records and the blocked `results/raw/heldout/status.json` remain unchanged as historical evidence. The current stage outcome is `results/raw/heldout/summary.json`; current study state is `RUN_STATE.json`.

The completed run used `prepare_resume.py` once to copy/hash-check the existing private calibration/development stages into a new private root, then executed the held-out stage and timing rounds 0–2. No recalibration or new selection was performed. There are no remaining authorized GPU stages. [Completion history](provenance/completion_history.json) records the previous state and execution counts.

`export_completed.py --private-run "$COMPLETED_PRIVATE_RUN" --case "$PARTIAL_CASE_COPY"` is an additive scalar-export helper for a copy of the historical partial case. It checks the selection and refuses existing targets. Do not run it against the already completed public tree. It preserves all prompt-level statistics and exports only the worst of 32 token diagnostics; tensors remain private.

The frozen protocol and evaluation-freeze code hashes remain unchanged. Additional completion checks/export/packaging code is recorded separately; the unfrozen artifact coverage test now checks 960 completed cells. Historical test counts are not substituted for a current test run. Existing full-model score/timing commands were executed on the recorded environment, not independently reproduced on another installation.

## Reviewed original and current publication

The reviewed v2 archive has 1,282 files and SHA256 `811c392b7efd36c4ed186a5766a396b7ce70adcb68363cde4342f9da6fbcfa53`. Its original `SHA256SUMS` remains unchanged: it describes that historical exact inventory, not the edited publication tree. Original `scripts/package.py` applies to that snapshot only. Do not rewrite the old manifest to make the publication tree pass it.

The [publication snapshot](provenance/publication_snapshot.json) records all original hashes, the five editable documents with old/new hashes and reasons, and an exact added-file allowlist. [PUBLICATION_SHA256SUMS](PUBLICATION_SHA256SUMS) covers the current case tree excluding itself. The new verifier checks the pinned historical manifest and all protected original bytes before checking current coverage; merely rehashing modified measurements is rejected.

From the case directory, use the existing CPU Python environment and new outputs outside the case:

```bash
python -B scripts/verify_publication.py --case . --output "$PUBLICATION_AUDIT_NEW_JSON"
python -B -m unittest discover -s publication_tests -v
python -B scripts/package_publication.py --case "$CASE007_DIR" --zip "$PACKAGE_NEW_ZIP" --record "$PACKAGE_NEW_JSON"
```

`CASE007_DIR` is the absolute path to the case directory. All three output paths must be new. The scientific CPU commands above remain unchanged and work on extracted publication files. Original tests are not weakened. `report.py` recreates numerical-table drafts, figures and HTML; the edited README/ANALYSIS prose is not claimed byte-identical. Figure/HTML bytes are checked separately in new scratch output.

The original RUN_STATE and past validation reports record the reviewed experimental snapshot, including the earlier block/resume and local-review status. They are not live PR/publication state. Current branch/PR information is in GitHub and the publication handoff. This publication pass performs zero GPU runs and no new browser or real-iPad test; existing browser evidence remains historical. No private full tensor audit is represented as a new public-scalar check.

The [new ZIP](../../downloads/case007_mlp_pruning_reviewed_publication_v1.zip) is case-only, not a repository backup. Download metadata is [outside the ZIP](../../downloads/case007_mlp_pruning_reviewed_publication_v1.json) to avoid circular hashes. Case-internal links work after extraction; repository-relative downloads and NOTICE's Case006 attribution link require the repository. The latter source is also available at the [existing Case006 permalink](https://github.com/Munsik-Kim/inference-lab/blob/24ba3310dfd9fc14ed7f88d5ce91d31f0c176ab1/cases/006-attention-decision-stability/src/tasks.py). No model assets or previous-case ZIPs are bundled.
