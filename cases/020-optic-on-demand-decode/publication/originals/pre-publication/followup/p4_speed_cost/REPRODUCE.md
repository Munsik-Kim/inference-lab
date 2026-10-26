# Case020 P4 — CPU-only raw → tables → figures

These runtime paths were verified on this machine. Aggregation requires no model, GPU or Torch import. Existing runtimes were reused read-only; this is not a clean-environment claim.

```bash
cd /home/mskim/Codex_main/inference-lab-case020-optic-decode-20261026-1215/cases/020-optic-on-demand-decode/followup/p4_speed_cost
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1   /home/mskim/inference-lab/.venv-baseline/bin/python src/audit.py verify
```

Regenerate into fresh external output directories. If these already exist, choose a new destination.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1   /home/mskim/inference-lab/.venv-baseline/bin/python src/audit.py summarize   --out /tmp/case020-p4-summary-new
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/case020-p4-mpl-new   OPENBLAS_NUM_THREADS=1 /home/mskim/miniforge3/envs/ai_lab/bin/python   src/audit.py figures --out /tmp/case020-p4-figures-new
PYTHONDONTWRITEBYTECODE=1 /home/mskim/inference-lab/.venv-baseline/bin/python   -m unittest discover -s tests -v
```

When the complete Case is copied elsewhere, use explicit `--portable`. Recorded paths inside the Case are mapped to the current Case; missing outside-case originals are reported as unavailable, never verified. P4 alone without original artifacts, IDs and historical raw is insufficient.

The audit independently recomputes hashes, shapes, dtypes, ID M/U,16-layer error sums, sampled BF16/FP64 affine results, ordering, K/units, H/D/E statistics, lifecycle ns intervals/N and the full GPU debit. It does not treat saved PASS flags as truth. Fixtures reject unit errors, event/K conversion errors, fake numerical PASS and manifest content/coverage mutation.

## Unchanged verifier over the historical manifest scope

The original verifier does not expect added P4 coverage. This helper creates a temporary external copy of exactly747 original files and runs the unchanged verifier, then removes the copy. It temporarily uses about81MiB.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1   /home/mskim/inference-lab/.venv-baseline/bin/python src/verify_historical.py   --tmp-parent /tmp/case020-p4-historical-new   --receipt /tmp/case020-p4-historical-receipt-new.json
```

The recorded execution is `provenance/original_verifier_restored_scope.json`. The original manifest remains SHA256 f4aa25642fdaa16a7d2e943da8ae903d057a0ebe52d01c337dcfa68764e801a2.

## Evidence flow and execution history

- Original P2: `../../results/20261026-1235/` → `summary/legacy_*.csv`.
- P4 immutable raw: `results/p4-20261026-1451/` + append-only `worker_ledger.jsonl`.
- Design/IDs/arrays: `PROTOCOL.json`, `provenance/plans.json`, `inputs.json`, `artifacts.json`, `freeze.json`, `main_freeze.json`, `preflight_index.json`.
- Independent CPU: `src/audit.py` → `summary/` → `src/figures.py` → four PNG/SVG pairs.
- Local regeneration/test/preservation/cleanup evidence: `provenance/local_validation.json`, `preservation_after.json`, `resource_closeout.json`.
- New file identity: `PUBLIC_MANIFEST_P4.json` excludes itself; the original manifest stays historical.

GPU work in this ledger is complete. Do not rerun prepare.py/run.py to overwrite plans or reset the ledger. Frozen source/order and raw preserve the executed experiment. A new GPU replay needs its own run identity/freeze, remaining budget, exclusive compute check and preflight. Remote CI and cross-machine validation are NOT_RUN.

Three initial P4A decision/paired/memory-stage snapshots are preserved byte-identically in `results/p4a_initial/`. External CPU replay exposed a scope mismatch between those snapshots and generated summary files; no numerical disagreement or GPU rerun occurred. Original paths, the observed error and preserved hashes are in `provenance/p4a_snapshot_preservation.json`.
