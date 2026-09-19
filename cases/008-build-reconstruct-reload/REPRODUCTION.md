# Read, recalculate and reload

[Structured report](REPORT.md) · [한국어 보고서](REPORT.ko.md) · [Publication and historical identities](publication/README.md)

Commands run from the repository or extracted publication ZIP root, which contains `tools/modelpack/`, `cases/008-build-reconstruct-reload/` and `LICENSE`. The ZIP distributes code, recipes, inputs and measured evidence; **large model weights and private H/Y/logit vectors are excluded**. Cross-case links and download links need the full repository, but the CPU commands below are included in this ZIP.

## A. CPU checks on the current public tree

The analysis environment is Python 3.12, NumPy 2.3.5 and Matplotlib 3.10.8. The historical 19-test suite additionally needs Torch 2.13.0, Transformers 5.17.0 and safetensors 0.8.0 for three tiny CPU model/artifact fixtures. Use existing compatible interpreters. Publication preparation reran 19 tests in that fixture environment; plotting used the existing CPU environment with Matplotlib. No package installation or GPU research run is part of publication verification.

Set `CPU_PY` to your compatible CPU analysis interpreter and `FIXTURE_PY` to the interpreter with the listed Torch/Transformers packages (they may be the same). The commands below were run against new external outputs. Do not overwrite frozen files.

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CASE=cases/008-build-reconstruct-reload
CHECK_PARENT=$(mktemp -d)
export MPLCONFIGDIR="$CHECK_PARENT/matplotlib"
"$CPU_PY" -B "$CASE/publication/verify_publication.py" --root "$PWD" \
  --output "$CHECK_PARENT/publication.json"
"$CPU_PY" -B -m unittest discover -s "$CASE/publication/tests" -v
"$CPU_PY" -B -m tools.modelpack.analysis --case "$CASE" --output "$CHECK_PARENT/analysis"
"$CPU_PY" -B -m tools.modelpack.audit --case "$CASE" \
  --summary "$CHECK_PARENT/analysis/summary.json" --output "$CHECK_PARENT/scalars.json"
"$CPU_PY" -B "$CASE/scripts/check_timing.py" --case "$CASE" \
  --summary "$CHECK_PARENT/analysis/summary.json" --output "$CHECK_PARENT/timing.json"
"$CPU_PY" -B "$CASE/scripts/additional_tables.py" --case "$CASE" --output "$CHECK_PARENT/additional.json"
"$CPU_PY" -B "$CASE/scripts/local_tables.py" --case "$CASE" --output "$CHECK_PARENT/local.json"
"$CPU_PY" -B "$CASE/publication/posthoc/recalculate.py" --case "$CASE" --output "$CHECK_PARENT/posthoc.json"
"$CPU_PY" -B "$CASE/publication/posthoc/review_metrics.py" --case "$CASE" --output "$CHECK_PARENT/review"
"$CPU_PY" -B -m tools.modelpack.report --case "$CASE" --output "$CHECK_PARENT/report"
"$CPU_PY" -B "$CASE/scripts/plot_score_boundaries.py" \
  --summary "$CHECK_PARENT/analysis/summary.json" --output "$CHECK_PARENT/q_score_boundaries.png"
```

`analysis`, `report` and the adapted review script expect a **nonexistent child output directory**. `mktemp` creates only the parent. The publication verifier/tests check current documents and package inventory; the original scientific checker expects the historical exact inventory instead.

### Restore the original review snapshot

```bash
"$CPU_PY" -B "$CASE/publication/restore_original.py" --root "$PWD" \
  --output "$CHECK_PARENT/historical"
(
  cd "$CHECK_PARENT/historical"
  "$FIXTURE_PY" -B -m unittest discover -s cases/008-build-reconstruct-reload/tests -v
  "$CPU_PY" -B -m tools.modelpack.package verify
)
```

Restoration takes edited documents from hash-verified `publication/original_docs/`, all other original case files unchanged, the original 19 modelpack sources, and `LICENSE`. It omits new publication files. Original `SHA256SUMS` is unchanged; `PUBLICATION_SHA256SUMS` covers the current case, its shared modelpack sources and root license, excluding itself. Rehashing modified raw data cannot satisfy the separately anchored original inventory.

CPU audits reconstruct option scores, transitions, paired intervals and ratio-of-sums recovery. Full-vocabulary normalizers and KL remain retained GPU-derived evidence; four option logits cannot recreate full vectors. Private H/Y fit checking is a different operation and was not repeated for publication. Historical GPU/browser results are in `validation.json`; a regenerated HTML file alone is not a new browser test. Real iPad remains NOT_TESTED.

## B. Future GPU reproduction with the original environment

**Documentation examples only: none of these model/build/reload/timing commands ran during publication preparation.** Use the same pinned snapshots and source/binary versions in [provenance](provenance/environment.json). Local offline flags do not create an OS-level network sandbox. New machine GPU reproduction or numerical identity is not established by the path-contract CPU fixtures.

| Variable | Meaning and required environment |
|---|---|
| `Q_SNAPSHOT`, `R_SNAPSHOT` | Existing local snapshots at the exact revisions in the report; no automatic downloads |
| `BUILD_PY` | Existing Q conversion environment: Torch 2.13.0, LLM Compressor 0.13.0, Transformers 5.14.1, compressed-tensors 0.18.0 |
| `ARTIFACT_PY` | Compatible artifact-inspection interpreter with Torch, safetensors and the pinned serializer; may use the measured serving environment |
| `GPU_PY` | Original vLLM 0.29.0 / Transformers 5.17.0 serving environment; restore normal CUDA visibility for this separate GPU session |
| `NEW_Q_BUILD`, `NEW_Q_FINALIZED`, `PRIVATE_Q_COPY` | Three distinct, nonexistent external child directories: build output, finalized artifact, verified reload copy |
| `NEW_Q_INSPECTION_JSON`, `NEW_Q_RELOAD`, `NEW_Q_EVAL`, `NEW_Q_TIMING` | New external output file/directory for each distinct operation |
| `NEW_R_SMOKE`, `NEW_R_CAL`, `NEW_R_DEV`, `NEW_R_ARTIFACTS`, `NEW_R_CAL_READOUT` | New external output directories for the R stages |
| `PRIVATE_R_COPY_ROOT` | New external root containing seven arm directories, not a single artifact |
| `NEW_R_INSPECTION_JSON`, `NEW_R_RELOAD_JSON`, `NEW_R_BASELINE`, `NEW_R_CANDIDATE`, `NEW_R_TIMING` | New external R result paths; choose fresh paths for each arm/process/round |

Set absolute interpreter paths. Choose an external parent, then assign child paths without creating the final children. All examples quote paths, including paths containing spaces. Successful previous directories must not be moved, deleted or overwritten.

### Q: build → finalize → copy → inspect → fresh-process smoke

```bash
"$BUILD_PY" -B -m tools.modelpack build-q --snapshot "$Q_SNAPSHOT" --output "$NEW_Q_BUILD"
"$ARTIFACT_PY" -B - "$NEW_Q_BUILD" "$NEW_Q_FINALIZED" <<'PY'
from pathlib import Path
import sys
from tools.modelpack.qartifact import finalize
report = finalize(Path(sys.argv[1]), Path(sys.argv[2]))
print(report['build'])
PY
"$ARTIFACT_PY" -B - "$NEW_Q_FINALIZED" "$PRIVATE_Q_COPY" <<'PY'
from pathlib import Path
import shutil, sys
from tools.modelpack.common import ROOT, verify_checksums
source, target = map(Path, sys.argv[1:])
if target.exists() or target.resolve().is_relative_to(ROOT):
    raise ValueError('Use a new external copy directory')
before = verify_checksums(source)
shutil.copytree(source, target)
if verify_checksums(target) != before:
    raise ValueError('Q copy checksum mismatch')
print('Q COPY PASS')
PY
"$ARTIFACT_PY" -B -m tools.modelpack inspect-artifact --artifact "$PRIVATE_Q_COPY" \
  --output "$NEW_Q_INSPECTION_JSON"
"$GPU_PY" -B -m tools.modelpack evaluate --track Q --artifact "$PRIVATE_Q_COPY" \
  --split smoke --output "$NEW_Q_RELOAD"
```

`build-q` creates `checkpoint/` inside the build root. The actual Python function `qartifact.finalize(build, output)` validates its checksum/tensor copy and writes `artifact.json`. There is no invented `finalize` CLI command. Q uses `evaluate --track Q --split smoke`; the dense-only `reload-test` command is for R. Repeat smoke with a different output in a second fresh process and compare reports before evaluating.

```bash
"$GPU_PY" -B -m tools.modelpack evaluate --track Q --artifact "$PRIVATE_Q_COPY" \
  --split heldout --output "$NEW_Q_EVAL"
"$GPU_PY" -B -m tools.modelpack benchmark --track Q --artifact "$PRIVATE_Q_COPY" \
  --round 1 --output "$NEW_Q_TIMING"
```

The frozen implementation checks the recorded local evaluation freeze. A changed build hash or environment must be diagnosed/versioned, not silently accepted by rewriting that freeze. Use rounds 1, 2 and 3 with distinct output paths for the original timing design.

### R: fixed structure → reconstruction → seven standalone artifacts → copy

```bash
"$GPU_PY" -B -m tools.modelpack.r_study smoke --snapshot "$R_SNAPSHOT" --output "$NEW_R_SMOKE"
"$GPU_PY" -B -m tools.modelpack.r_study calibrate --snapshot "$R_SNAPSHOT" --output "$NEW_R_CAL"
"$GPU_PY" -B -m tools.modelpack.r_study develop --snapshot "$R_SNAPSHOT" \
  --calibration "$NEW_R_CAL" --output "$NEW_R_DEV"
"$GPU_PY" -B -m tools.modelpack.r_study export --snapshot "$R_SNAPSHOT" \
  --calibration "$NEW_R_CAL" --development "$NEW_R_DEV" --output "$NEW_R_ARTIFACTS"
"$ARTIFACT_PY" -B - "$NEW_R_ARTIFACTS" "$PRIVATE_R_COPY_ROOT" <<'PY'
from pathlib import Path
import shutil, sys
from tools.modelpack.common import ROOT, verify_checksums
source, target = map(Path, sys.argv[1:])
arms = ('R-B', 'I25', 'I25-R', 'P25', 'P25-R', 'S50', 'S50-R')
if target.exists() or target.resolve().is_relative_to(ROOT):
    raise ValueError('Use a new external copy root')
before = {arm: verify_checksums(source / arm) for arm in arms}
shutil.copytree(source, target)
for arm in arms:
    if verify_checksums(target / arm) != before[arm]:
        raise ValueError('R copy checksum mismatch: ' + arm)
print('R SEVEN-ARM COPY PASS')
PY
"$ARTIFACT_PY" -B -m tools.modelpack inspect-artifact --artifact "$PRIVATE_R_COPY_ROOT/I25-R" \
  --output "$NEW_R_INSPECTION_JSON"
"$GPU_PY" -B -m tools.modelpack reload-test --artifact "$PRIVATE_R_COPY_ROOT/I25-R" \
  --input "$CASE/inputs/R/smoke.json" --device cuda --output "$NEW_R_RELOAD_JSON"
"$GPU_PY" -B -m tools.modelpack evaluate --track R --artifact "$PRIVATE_R_COPY_ROOT" \
  --arm R-B --output "$NEW_R_BASELINE"
"$GPU_PY" -B -m tools.modelpack evaluate --track R --artifact "$PRIVATE_R_COPY_ROOT" \
  --arm I25-R --baseline "$NEW_R_BASELINE" --output "$NEW_R_CANDIDATE"
"$GPU_PY" -B -m tools.modelpack benchmark --track R --artifact "$PRIVATE_R_COPY_ROOT" \
  --round 1 --output "$NEW_R_TIMING"
"$GPU_PY" -B -m tools.modelpack.r_study calibration-readout \
  --calibration "$NEW_R_CAL" --development "$NEW_R_DEV" --output "$NEW_R_CAL_READOUT"
```

`export` creates `R-B/`, `I25/`, `I25-R/`, `P25/`, `P25-R/`, `S50/`, `S50-R/` and `export.json`. Inspection/reload-test receive **one arm**; R evaluate/benchmark receive the **root**. Complete the frozen comparison with all seven arms, two fresh reload processes per arm, and three timing rounds, using separate outputs. The example names do not imply those operations were repeated during publication.

Calibration-readout reuses private H and runs the smaller projection without another transformer pass. `scripts/check_private_fit.py --help` describes the additional CPU FP64 H/Y check. Public norms cannot reconstruct those vectors. Existing `RUN_STATE.json`, artifact export-time statuses and correction records retain their historical values.
