# Reproduction and supported boundaries

English | [한국어](REPRODUCTION.ko.md)

Run from the repository root. Every result/receipt output must use a new path.

## 1. Recalculate recorded measurements without models

Use Python3.12 and NumPy2.3.5, as recorded in [CPU requirements](publication/requirements-cpu.txt). An isolated verification environment can be prepared without touching any research environment:

```bash
VERIFY_ENV=$(mktemp -d)
python3 -m venv "$VERIFY_ENV/venv"
"$VERIFY_ENV/venv/bin/python" -m pip install -r cases/013-dualquant-reference-quality/publication/requirements-cpu.txt
"$VERIFY_ENV/venv/bin/python" -B cases/013-dualquant-reference-quality/analysis/audit.py --output "$VERIFY_ENV/scalars.json"
"$VERIFY_ENV/venv/bin/python" -B -m unittest discover -s cases/013-dualquant-reference-quality/tests -v
"$VERIFY_ENV/venv/bin/python" -B cases/013-dualquant-reference-quality/publication/verify.py --output "$VERIFY_ENV/publication.json"
```

The auditor recomputes NLL/PPL from per-token losses, flips from predicted IDs, exact long-task scores, free-agent final answers/actions and paired counts. It also sums recorded per-layer weight SSE/energy and byte ledgers. This authenticates metadata aggregation, not new weight-error measurements or private payload hashes. It aggregates selected-position full-vocabulary KL; excluded full logits prevent new KL calculation.

## 2. Synthetic INT4 checks on CPU

If an existing compatible Torch environment is available:

```bash
CUDA_VISIBLE_DEVICES='' python3 -B -m unittest discover -s cases/013-dualquant-reference-quality/synthetic -v
```

This uses tiny CPU matrices, not model weights. It checks packing, rounding, scale transport, full-row reductions, actual byte ledgers and failure handling. The original39-test receipt is preserved under [records](records/verification.json); those historical tests are not the number run by the new NumPy job. Original source-equation AST tests additionally need the pinned vendor file, which is not redistributed.

## 3. Actual model reruns: separate requirements

No model reruns occur during this publication. The unchanged [model runner](source/paper_model_runner.py) needs the pinned local Qwen snapshots and original WikiText token arrays, plus compatible Torch/Transformers/CUDA. Weight artifacts and dataset text are not downloadable from this case. The private block-format library is unnecessary for this INT4 variant.

Use the existing permitted local files; inspect CLI options before preparing a fresh directory:

```bash
python3 -B cases/013-dualquant-reference-quality/publication/prepare_rerun.py --help
python3 -B cases/013-dualquant-reference-quality/source/paper_model_runner.py --help
```

`prepare_rerun.py` accepts `--model06b`, `--model4b`, `--wiki06b`, `--wiki4b`, `--output`. It checks config/tokenizer/input hashes, copies owned frozen fixtures and writes a local execution manifest. It loads no model and runs no inference. Large model-weight identity is still a separate responsibility; config matching alone does not authenticate weights.

Then `paper_model_runner.py --run NEW_RUN --model 06b --method BF16` is the real GPU command shape. Run BF16 first for each model to create the required reference arrays; RTN/D16 follow, and D4 reuses the saved D16 artifact. `--model 4b` also runs the frozen long/agent DEV. These commands are documented GPU paths, not a promise that a new environment or private checkpoint is available. The completed public records are never an output directory for reruns.

Final packed probe receipts are included, but a validated serving path is not supplied. Neither official packed W4A4 nor a stock GPTQ benchmark is reproduced by this package.

## 4. Build the bilingual local display

```bash
PREVIEW=$(mktemp -d)
python3 -B tools/showcase/build.py --output "$PREVIEW/site" --base-path /inference-lab/
python3 -B tools/showcase/check.py --site "$PREVIEW/site" --output "$PREVIEW/site-check.json"
```

Open `en/case013.html` or `ko/case013.html` in the built site. It displays saved results and source copies; it does not execute a model. Deployment is a separate action from this PR.

## 5. Regenerate descriptive figures on CPU

With Matplotlib and NumPy in a separate existing CPU environment:

```bash
FIGURES=$(mktemp -d)
python3 -B cases/013-dualquant-reference-quality/publication/plot_results.py --output "$FIGURES/plots"
```

The recorded plot environment versions and sources are in [figure metadata](figures/figure_manifest.json). This is not an inference command; raw CSV/JSON remains unchanged. [Figure captions](figures/README.md) state denominators and normalization.
