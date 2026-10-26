# Case020 — Check recorded evidence without a model or GPU

[한국어](REPRODUCE.ko.md) · [Report](REPORT.md)

Tested on Linux/WSL in a fresh Python3.12.14 venv. Dependencies are NumPy2.3.5, Matplotlib3.10.8 and their pinned small dependencies. No Torch, CUDA or model download is needed. Existing GPU environments were not upgraded. Native Windows is untested.

```bash
git clone https://github.com/Munsik-Kim/inference-lab.git
cd inference-lab
python3.12 -m venv /tmp/case020-cpu-env
/tmp/case020-cpu-env/bin/python -I -B -m pip install \
  -r cases/020-optic-on-demand-decode/publication/requirements-cpu.txt
/tmp/case020-cpu-env/bin/python -I -B -m pip check
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/case020-cpu-env/bin/python -I -B \
  cases/020-optic-on-demand-decode/publication/run_checks.py \
  --mode quick --tests --output /tmp/case020-quick-new
```

Choose new names if the venv or output already exists. Output must be a nonexistent directory outside the checkout. The script runs directly without editable installation or private absolute paths.

## Quick checks versus full recorded-evidence audit

Quick verifies public/mapped/frozen bytes, all original P2/P4 timing mean/median/p95, storage, actual sessions/budget, document facts, a small independent FP64 slice and unit contracts. It does not claim full numerical checking of large stored GPU outputs. Reading all file hashes is distinct from full numerical audit.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/case020-mpl-new \
  /tmp/case020-cpu-env/bin/python -I -B \
  cases/020-optic-on-demand-decode/publication/run_checks.py \
  --mode full --tests --output /tmp/case020-full-new
```

Full adds NPZ CRC/member/magic checks, unchanged P2/P4 auditors, all preserved P1 tokens/layers/seeds/controls, saved P2 BF16 outputs and P4 preflight checks, an alternative statistics cross-check, and four PNG/SVG pairs. Historical manifest coverage is checked on an exact 747-file temporary copy, removed afterward. No verifier expectation or numerical tolerance is weakened.

P4 portable mode discloses unavailable original-machine external paths. Exact public copies of 19 external originals are independently verified by preservation_map. Recorded absolute paths remain unchanged; the wrapper selects the archive within the current checkout. This is not new GPU/model reproduction.

## Inputs and outputs

- Original Case: `publication/originals/pre-publication/` (1020 exact files including P4).
- External analysis/handoffs: `publication/originals/external/` (19 exact files).
- Path/bytes/hash mapping: `publication/preservation_map.json`.
- New raw-derived tables: `summary/`; each run regenerates them under `OUTPUT/summary/`.
- Full audit outputs: `OUTPUT/historical_P2_summary/`, `OUTPUT/historical_P4_summary/`, `OUTPUT/figures/`.
- Success/failure: `OUTPUT/verification.json` or `failure.json`, `tests.log` / `failure.log`.

All evidence is in ordinary Git files: no Release asset or implicit evidence download. Checks enforce25 MiB/file and 128 MiB/Case. Full temporarily needs about 81 MiB for the historical copy, plus its external outputs.

The performed fresh-environment validation is recorded in publication/receipts/local_cpu.json. Remote CI runs Case020 CPU quick checks, not the historical GPU experiment. Post-publication self-checks are reported against distinct PR/main SHAs in the final handoff. Historical NOT_RUN receipts are unchanged.

Preserved prepare/worker/run scripts are historical execution source and are not launched by these checks. New GPU work requires a separately scoped run/freeze/budget/preflight.
