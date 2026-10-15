# Reproduce / 재현

Run in this case directory. Use existing compatible environments; this review does not install packages. Observed versions and source identities are in [provenance](provenance/identity.json), [model metadata](provenance/model.json), and [source audit](references/source_audit.json).

## CPU checks (no model or GPU)

Set `PY_CPU` to an existing Python with NumPy and Torch CPU support. The experiment used Python3.12.14, NumPy2.3.5, Torch2.13.0+cu130; CUDA is disabled for these checks.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PY_CPU" -m unittest discover -s tests -v
python3 scripts/audit.py --results results/local_r1 --output /tmp/case014-audit-NEW.json
```

The independent stdlib auditor recomputes Fraction identities and MC statistics from per-seed SSE. When actual P1 records exist, it also validates scalar decompositions, pairing, CAL selection, NLL denominators and PPL. Without model matrices/logits this does not re-evaluate their GPU computation.

To repeat the same 2,000-seed MC with a new output:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY_CPU" scripts/fixtures.py --seeds 2000 --output /tmp/case014-p0-NEW.json
```

Plots use an existing NumPy/Matplotlib environment:

```bash
MPLCONFIGDIR=/tmp/case014-mpl "$PY_PLOT" scripts/plot.py --results results/local_r1 --figures /tmp/case014-figures-NEW
```

## Existing-cache inputs (CPU, private)

Requires a separate existing environment with PyArrow and Transformers. No network is used. `MODEL_SNAPSHOT`, `PG19_ARROW`, `PILE_ARROW` are external existing assets; no dataset text or model weights are supplied in this ZIP.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 "$PY_DATA" scripts/prepare_data.py --model "$MODEL_SNAPSHOT" --pg19 "$PG19_ARROW" --pile "$PILE_ARROW" --output "$NEW_PRIVATE_INPUT_DIR"
```

`NEW_PRIVATE_INPUT_DIR` must be empty. PG19 test has book URL boundaries. Pile ArXiv records use content grouping because original article IDs are unavailable. The manifest records selection, offsets, exact hashes, limited near-duplicate checks and uncertain prior use outside the bounded audit. Source cards: [PG19](https://huggingface.co/datasets/pg19), [mirror](https://huggingface.co/datasets/emozilla/pg19), [Pile subset](https://huggingface.co/datasets/NeelNanda/pile-10k). Text remains private.

## P1 GPU command — prepared, execution status in decision

This is an implemented entry point, **not a verified GPU reproduction command until the storage gate and actual run are recorded**. Existing snapshot and tokenizer revision must be `c1899de289a04d12100db370d81485cdf75e47ca`. No new download or installation is authorized. The current D-original location is unconfirmed; do not treat the WSL cache as a verified D mirror.

```bash
python3 scripts/launch.py --python "$PY_GPU" --model "$MODEL_SNAPSHOT" --private "$PRIVATE_RUN_ROOT" --output results/local_r1/p1
```

`PRIVATE_RUN_ROOT/inputs/` must contain the frozen private token records and manifest from input preparation. Use a fresh P1 output directory. The runner does not resume failed attempts or refit on DEV/EVAL. It records a 5,400-second conservative worker budget; earlier complete records remain intact. A new environment/model/config or missing artifact requires a new identity, not a silent retry.

The path is packed INT4 storage followed by dense BF16 quality execution. Continuous controls use FP32 weights and input, returning BF16. Neither provides native low-bit serving speed or capacity measurements.

After a completed grid:

```bash
python3 scripts/audit.py --results results/local_r1 --output /tmp/case014-full-audit-NEW.json
"$PY_CPU" scripts/analyze.py --results results/local_r1
```

Analysis refuses overwriting derived files. CPU audit reproduces scalar arithmetic and aggregation, not missing private W/X/Z/full-vocabulary logits. P2/P3 and remote publication remain outside this run.
