# Reproduce the Case 010 evidence

[한국어](REPRODUCTION.ko.md) · [Case home](README.md) · [Version map](VERSION_MAP.md)

Choose the path for what you want to verify. The unified case ships compact recorded predictions and both original research subtrees. Model checkpoints, the upstream checkout and full private state histories are separate artifacts.

## 1. Read the implementation and results

Start with the [unified report](REPORT.md), [derived summary](summary/project.json), [failure-aware OnlineAdapter](versions/v2/source/online_v2.py), and [fresh-process contract tests](versions/v2/tests/test_online_v2.py). The current execution contract is v2. Stage A's source remains the historical reference.

The [snapshot manifest](provenance/snapshot_manifest.json) maps every original path to its preserved path with bytes and SHA256. It covers 1,075 v1 files and 387 v2 files. Historical manifests remain unchanged inside those trees. A historical `LOCAL_REVIEW` or unpublished status describes that snapshot; the integration's branch/PR state is a separate record.

## 2. Verify without a model

Use an existing Python environment with NumPy. The wrapper does not require Torch, CUDA, a model download or a trained checkpoint. Disable bytecode writes and keep every output outside the preserved case tree. In a repository clone:

```bash
cd cases/010-ckda-finite-precision-memory-horizon
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=
python -B scripts/verify_unified.py
python -B -m unittest discover -s tests -v
python -B scripts/check_cpu.py --output /PATH/TO/NEW_CASE010_CHECK
```

Replace the final path with a **nonexistent external directory**. The CPU wrapper verifies source identity, restores the old workspace layout there, runs the original v1 publication/scalar/supplement auditors and the v2 independent fresh-result auditor, then exercises the original synthetic failure-aware byte-codec tests. These tests include separate child processes and require no trained model. The original synthetic fixture is not a learned memory-performance example.

The verifier rejects changes to an original snapshot even if a display summary is regenerated. Scalar audits read saved predictions and integer gold, check actual file identity and pairing, and recompute first error/RMST/horizon values. They do not reproduce the original forward passes or reread the private final recurrent tensors.

For separate steps, run from the unified case directory:

```bash
python -B scripts/derive_summary.py --output /PATH/TO/NEW_PROJECT.json
python -B scripts/restore_workspace.py --output /PATH/TO/NEW_RESTORED_WORKSPACE
```

The first command regenerates compact display data from the original records; optional `--figure /PATH/TO/NEW_FIGURE.png` also needs Matplotlib. The second recreates historical paths without editing either snapshot:

```text
NEW_RESTORED_WORKSPACE/
  cases/010-ckda-finite-precision-memory-horizon/   # v1
  research/case010-failure-aware-v2/              # v2
```

Run old source and relative-path checks in that restored layout. For example, after setting `RESTORED` to the new workspace path and `AUDIT` to a different new output path:

```bash
python -B "$RESTORED/cases/010-ckda-finite-precision-memory-horizon/scripts/verify_publication.py"   --root "$RESTORED/cases/010-ckda-finite-precision-memory-horizon"
python -B "$RESTORED/research/case010-failure-aware-v2/analysis/audit_v2.py"   --results "$RESTORED/research/case010-failure-aware-v2/results/fresh"   --output "$AUDIT"
```

Do not run a historical exact-inventory verifier against the outer unified tree. Do not regenerate an old scientific checksum to accommodate the new location. The original [v1 guide](versions/v1/REPRODUCTION.md) and [v2 guide](versions/v2/REPRODUCTION.md) retain their own historical path assumptions; restore first.

### Small fresh-process example

```bash
python -B scripts/restart_demo.py
```

This NumPy-only example injects one numerical fault into two synthetic stochastic streams, saves the actual bytes and resumes the suffix in a fresh Python process. It reports one active and one terminal stream, suffix equality and final-byte equality. The child checks that Torch was not loaded. This is a byte-codec contract demonstration, not a learned CKDA evaluation.

## 3. Re-run the actual research separately

The original guides describe model-dependent replay, training history, fixed-checkpoint precision diagnostics, fresh input evaluation and timing. They require the pinned upstream `ef9d108d1692387cae37f5b2d539a71826a127c1`, a compatible environment and the three exact private checkpoint hashes in the [v2 protocol](versions/v2/protocol_v2.json). No weights are distributed in this unified case. Model inference, training, calibration, TEST re-evaluation and GPU execution are outside this integration.

The complete historical suites include model/upstream-dependent tests. Do not infer that these ran merely because the model-free wrapper passes; report actual count/skip/error output. The previous 141 v1 tests and 76 v2 tests in historical receipts are historical executions, not this integration's test total.

## Evidence boundaries

- v1 scalar reanalysis covers its recorded 125 toy, 78 original learned and 12 supplementary records; these are record counts, not independent model or sequence counts.
- v2 fresh analysis has 15 cells, each with 1,024 inputs and length 2,048. All settings share input IDs and the three original checkpoints. Historical and diagnostic records do not enlarge fresh N.
- Public payload sizes and hashes validate identity and accounting. Private final hidden-state bodies and checkpoint-based restart evidence require the separate artifacts to re-execute.
- Browser emulation, CPU checks and any remote CI results describe their own scope. None is a new GPU experiment or an actual iPad test.

## Checks actually re-executed for integration

The [integration receipt](provenance/integration_checks.json) records the current run: 18 unified contract tests and 17 original v2 codec-only synthetic tests passed, with no skips. The fresh-process demo retained one active and one terminal stream with identical suffix and final bytes. Original v1 publication checks passed before and after scalar audits (125 toy, 78 learned, 12 supplementary records); the v2 independent auditor verified all 15 fresh cells. These are stored-data and synthetic CPU checks. The complete historical 141/76 suites, trained-checkpoint replay, fresh inference and timing remeasurement were not re-executed.
