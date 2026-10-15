# Reproduce Case014 P1 / 재현

Use the case directory as cwd. This run reused an existing WSL snapshot read-only under the user's **2026-10-15 P1-specific authorization**. D-origin verification is not a gate for this run. No model copy, download, installation or P2/P3 execution occurred.

The exact model/revision is `Qwen/Qwen3-0.6B @ c1899de289a04d12100db370d81485cdf75e47ca`. [Authorization and actual local file hashes](provenance/p1_r2_authorization.json) distinguish full local payload identity from a newly fetched official digest. Runtime versions are in [runtime manifest](results/p1_r2/runtime_manifest.json). The review ZIP replaces private path values with labels; original local evidence is preserved.

## CPU-only saved-result audit

No weights, GPU or new data are needed. `PY_CPU` is an existing Python with NumPy/Torch for the optional bootstrap and regression tests.

```bash
python3 analysis/p1_r2/audit_public.py --case . --output /tmp/case014-public-audit-NEW.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PY_CPU" -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PY_CPU" analysis/p1_r2/audit_public.py --case . --bootstrap --output /tmp/case014-public-bootstrap-NEW.json
```

The independent auditor reconstructs scalar terms, selection, paired denominators, NLL/PPL, sampled KL/flip aggregates and all120 summary rows. Optional bootstrap repeats the same10,000 document resamples, not model inference. It cannot recompute GPU logits from scalar records. Thirty artifact reload checks and eight private full-array checks were performed locally; their inputs are excluded from the ZIP.

**The15 regression tests are the existing P0 tests rerun. P0's2,000-seed MC was not rerun.** Its original lambda is0 (direct full-Gram solve without ridge in the dither control). Original P0 data/documents/code and closed manifest are preserved in [historical snapshot](publication/original_pre_p1/REPORT.ko.md).

## Actual P1 GPU entry point

An existing compatible environment is required: observed Python3.12.14, Torch2.13.0+cu130, Transformers5.17.0, NumPy2.3.5, RTX5080 SM120. There are no installation steps. `--help` and the actual entry point were checked.

Prepare a **fresh private run directory**, linking its `inputs/` to the original frozen private input directory. Do not extract new calibration documents. That directory must contain `documents.private.json` and `input_manifest.json`; their hashes are verified before the model loads. Public hashes do not reconstruct original text or token arrays.

```bash
mkdir "$FRESH_PRIVATE_ROOT"
mkdir "$FRESH_PRIVATE_ROOT/attempts"
ln -s "$FROZEN_INPUT_DIR" "$FRESH_PRIVATE_ROOT/inputs"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 python3 scripts/launch.py --python "$PY_GPU" --model "$MODEL_SNAPSHOT" --private "$FRESH_PRIVATE_ROOT" --output results/p1_FRESH
```

The actual completed run used `results/p1_r2`. A new attempt must use a fresh output and preserve failed/partial records. The cumulative budget is5,400 worker seconds, including load, CPU fit, pack/reload, metric/I/O and cleanup. This run used358.877s; the inner runner recorded344.235s. GPU busy time was not measured.

The runner applies only the fixed6×5 target/strength grid. It preserves full-model residual/norm/MLP paths, uses the same full256-position head, and freezes the CAL selector before held-out forwards. It installs **BF16 weights restored from actual INT4 files**; this is a quality reference, not a native low-bit serving benchmark. Six alpha1 continuous controls use FP32 weights/inputs and BF16 output casts. All controls are separately labeled.

## Local private checks and CPU analysis

These commands were actually used with external local assets. `PRIVATE_P1_ROOT` identifies the completed private root; `MODEL_SNAPSHOT` is the existing model path. They require the omitted payloads/traces and do not run a new GPU experiment.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PY_CPU" analysis/p1_r2/reload_artifacts.py --private "$PRIVATE_P1_ROOT" --manifest results/p1_r2/artifact_manifest.json --output /tmp/case014-reload-NEW.json
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PY_CPU" analysis/p1_r2/check_private_arrays.py --case . --private "$PRIVATE_P1_ROOT" --model "$MODEL_SNAPSHOT" --output /tmp/case014-arrays-NEW.json
```

For reanalysis, create a fresh external view containing `p1` → this run directory, `p0.json`/`p0_full.json` → historical P0 result files. Then:

```bash
PYTHONDONTWRITEBYTECODE=1 "$PY_CPU" scripts/analyze.py --results "$FRESH_ANALYSIS_VIEW"
MPLCONFIGDIR=/tmp/case014-mpl PYTHONDONTWRITEBYTECODE=1 "$PY_PLOT" scripts/plot.py --results "$FRESH_ANALYSIS_VIEW" --figures /tmp/case014-figures-NEW
```

`PY_PLOT` is an existing NumPy/Matplotlib environment. Derived outputs refuse overwriting. Original P0 figures remain unchanged. All metrics use original document pairs; repeated targets/strengths are not independent documents. Intervals are exploratory pointwise estimates.

## Scope and omissions

The review contains code, all lightweight scalar rows, specs, identities, figures and receipts. It omits original/INT4 model weights, full document text, K/Gram/activation arrays and full-vocabulary logits. The private manifest records their local sizes/hashes. The old root `PUBLIC_MANIFEST.json` remains historical; the new `P1_PUBLIC_MANIFEST.json` covers the current review file set.

P2/P3, 4B/agent expansion, native packed serving, commit/push/PR/merge/Pages and external upload were not run.
