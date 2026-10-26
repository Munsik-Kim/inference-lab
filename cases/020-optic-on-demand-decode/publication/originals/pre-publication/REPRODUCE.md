# Reproduce Case020

## CPU audit and reconstruction of artifacts

Use Python with NumPy 2.2.6 or2.3.5; plotting additionally needs Matplotlib 3.10.8. The CPU audit does not import Torch, initialize CUDA or need a model. --help is also CPU-only. The observed rendering runtime was Python 3.10.19; the official operator example requires Python≥3.12.

```bash
# From this case directory. These commands need no model, Torch, or CUDA.
python src/audit_v2.py verify
python src/audit_v2.py summarize --out /tmp/optic-summary
MPLCONFIGDIR=/tmp/optic-mpl python src/audit_v2.py figures --out /tmp/optic-figures
python tests/test_audit_v2.py -v
```

Pipeline: `results/20261026-1235/*.json` plus `inputs/` and preserved BF16 words → `src/audit_v2.py` → `results/summary/{correctness,memory,latency,paired}.csv` and `decision.json` → `src/figures.py` → `figures/01_memory,02_latency,03_unique_cost.(png,svg)`. A/B/C timing samples are never pooled across blocks. The independent auditor recomputes statistics and storage formulas rather than trusting stored PASS/summary fields.

The raw B block0 NPZ files hold requested BF16 uint16 words, not dense tables. Decode words with `(words.astype(uint32)<<16).view(float32)`. P1 raw outputs cover all methods/seeds/controls. NPZ loading always uses allow_pickle=False.

## Re-run the official CPU example / P1

These commands require the tested operator runtime or another Python≥3.12 runtime with NumPy and compatible Torch. They do not run generation/download entry points. Write into a new output location to retain original evidence.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=vendor/optic/src python vendor/optic/examples/decode_optic.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B src/cpu_checks.py --out /tmp/optic-p1
```

P1 rewrites deterministic `inputs/p1_*.npz` and fixture_manifest.json; run in a disposable copy if preserving the candidate exactly. The minimal original source is vendored with hashes/license and needs no editable install or vLLM/Transformers.

## GPU re-execution protocol

The original run is complete and must not be overwritten or appended as another replicate. For a new GPU run, make a separate worktree/output copy, assign a new run ID and ledger, and update/freeze paths explicitly before running prepare→freeze→probe→preflight→smoke→sweep. The supplied scripts intentionally bind the original run ID and1800s ledger; their current commands are for audit/review, not a one-command reset of the original evidence. The worker supports individual --out locations for source review, but launching it alone bypasses the parent budget ledger and is not the recommended measured workflow.

Use a free GPU, Python≥3.12 and sm_120-supporting Torch with actual FP16 mm(out_dtype=FP32) support. Keep2GiB allocated-delta and1800s cumulative worker caps. Never replace the official arithmetic with a fallback. No package upgrade into a serving environment is needed. No weights/datasets are part of this case.

Observed operator executable: `/home/mskim/inference-lab/.venv-baseline/bin/python`; rendering executable: `/home/mskim/miniforge3/envs/ai_lab/bin/python`. Both were read-only reused. No clean install, remote CI or other-machine result is claimed.

## Verify delivered bytes

`PUBLIC_MANIFEST.json` records every case file except itself, by relative path/bytes/SHA256. It separates Git base, upstream source and uncommitted benchmark bundle. `provenance/reproduction_receipt.json` records same-environment summary/image regeneration and visual QA. No wheel/cache/model/dense table is public.
