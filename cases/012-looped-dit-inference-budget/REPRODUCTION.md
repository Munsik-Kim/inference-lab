# Case 012 reproduction

## 1. Inspect and recalculate saved records — no models

Run from the Case 012 directory. Keep the CPU environment outside the research subtree so the exact publication inventory contains only study files:

```bash
python3 -m venv ../../.venv-case012-cpu
../../.venv-case012-cpu/bin/python -m pip install -r requirements-cpu.txt
../../.venv-case012-cpu/bin/python analysis/audit.py
../../.venv-case012-cpu/bin/python publication/verify.py
```

The commands above read saved records and print receipts without changing the study files. `analysis/analyze.py` rebuilds timing and pairing data; run it only in a disposable restored copy as shown below. Without evaluator JSON, quality remains `ANNOTATION_PENDING` and `primary=null`. `analysis/audit.py` independently checks PNG bytes, dimensions, noise hashes, 192 MAIN / 24 SMOKE coverage and saved median times. It does not run the model or judge image correctness.

Open `demo/viewer.ko.html` or `demo/viewer.en.html` directly in a browser. Full-resolution PNGs are reached by clicking images. The 256px JPGs in `demo/thumbs/` are review thumbnails, not originals. `demo/gallery.*.html` lists every MAIN and SMOKE image without JavaScript.

## 2. Blind human annotation

Open `demo/annotation.ko.html` or `demo/annotation.en.html` before inspecting the unblinded comparison viewer. The annotation screen has 216 opaque image IDs and no loop/step/time/seed labels. It contains the English model input, Korean display text, the frozen constraint checklist and three choices per constraint.

Enter an evaluator name or anonymous ID. Export JSON to a file; localStorage is only supplementary. Import checks IDs, image hashes, rubric identity and constraint keys. CPU auditing checks actual PNG hashes. Annotation exports may be partial; missing MAIN labels do not become scores. SMOKE and MAIN labels are validated together but only the 192 MAIN images enter the primary comparison.

In a disposable restored copy with the CPU environment activated (section 5), apply the exported labels and rebuild the viewer. This changes that copy, not the published records:

```bash
python analysis/analyze.py --annotations /your/export/case012_annotations.json
python demo/build.py
```

`uncertain` is not satisfied for the primary metric, with an explicitly separate optimistic sensitivity. Primary is **C_time minus A_time**, paired over the same prompt–seed pairs. Bootstrap clusters are the 16 prompts, with four seeds and three settings retained together. The four template families limit generalization. No independent human judgement or inter-rater agreement is claimed before annotation.

## 3. Synthetic runtime contracts

The saved-record tests need only CPU requirements:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ../../.venv-case012-cpu/bin/python -m pytest -q -p no:cacheprovider tests --ignore=tests/test_adapter_cpu.py
```

`tests/test_adapter_cpu.py` additionally needs PyTorch and Transformers. It checks arithmetic/RNG/error contracts on a synthetic tiny model, not Looped-DiT quality or RTX support. In the tested isolated GPU environment:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest -q tests
```

## 4. Actual GPU re-execution requirements

GPU execution requires the pinned **B/32 EMA checkpoint** and **FLAN-T5-Large encoder/tokenizer** identified in `provenance/models.json`. Weights and environments are excluded from the review ZIP. Tensor-identical encoder-only extraction avoids loading T5 decoder weights. `provenance/models.json` records the complete source safetensors hash, retained tensor hashes and encoder-only file hash. Do not substitute a different model, EMA revision or T5 dtype.

Tested environment: Python 3.12.14, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, Transformers 4.44.2. The denoiser uses BF16; original T5 default loading produced FP32. Both reside on the GPU throughout every condition. No compile, offload, VAE or quantizer is added.

The current guarded runner command is (it reuses a completed phase without loading models):

```bash
python -m source.safe_run --help
python -m source.safe_run --checkpoint /your/looped-dit-b32.pt \
  --encoder /your/t5-encoder --phase parity
```

Model paths are user supplied. The runner checks artifact hashes, frozen inputs, serial ownership and the shared generation budget. Completed attempts are verified and reused; they are never overwritten. Unfinished `.partial` records are not successful checkpoints. Failed starts count toward the budget. Do not manually reset `results/budget.json` to obtain additional research runs.

The original measured runner is preserved unchanged in `source/run.py`. Use `source.safe_run` for readers: it prevents repeating and overwriting a completed parity receipt. Its CPU guard was added after generation and does not change measured data.

Phases already performed are `parity`, `pilot`, `trace`, `dev`, `verify`, `smoke`, `main`, and `repeat --round 0/1/2`. Selection commands are `python -m source.select_time predict` and `python -m source.select_time freeze`. They refuse to overwrite frozen selection files. A future independent rerun needs a new result root and a separately approved protocol, not overwriting this completed study.

Measured settings are exported in `configs/main_settings.json`. These are time-selected presets, not quality-ranked “fast/balanced/best” recommendations. The 5% DEV time target was missed by A and B; this is a quality–time comparison near a target budget, not an exact equal-time experiment.

ComfyUI use is limited to supplying PNGs and reading the preset JSON in an existing workflow. This Case adds no custom node and changes no ComfyUI or StyleBridge files.

## 5. Public GitHub edition and historical documents

The public edition adds beginner-facing bilingual README text, every MAIN image in GitHub-readable Markdown, and explicitly post-hoc analysis of saved records. No new generation or quality labels were added. `publication/original_docs/` preserves prior document bytes; `publication/original_inventory.json` maps every original case file to its original size and hash.

```bash
../../.venv-case012-cpu/bin/python publication/verify.py
../../.venv-case012-cpu/bin/python publication/restore_original.py --output /your/new-scratch/case012-original
```

The output must not already exist. The restored directory reproduces the original Case012 subtree, not the entire repository or its original review ZIP. Run `analysis/audit.py` and CPU tests from that restored directory to inspect the original code and saved measurements. Publication metadata records current scope separately from historical `LOCAL_REVIEW` fields.

To rebuild derived data or apply annotations, activate the same CPU environment, then change to the restored directory. Use your actual new scratch path in place of the example:

```bash
source ../../.venv-case012-cpu/bin/activate
cd /your/new-scratch/case012-original
python analysis/analyze.py
python demo/build.py
python analysis/audit.py
```

Keep generated receipts, environments and annotation exports outside the published study subtree. Its exact-inventory verifier checks the preserved public files, not a modified annotation workspace.

The paired image Markdown can be read on GitHub. The HTML viewer/annotation UI runs locally after cloning; no Pages deployment or browser-hosted model execution is implied. The post-hoc illustration is non-blind AI visual inspection, not human labels or a new primary score. The raw CPU wall and CUDA event values are both retained; do not subtract them to estimate preprocessing overhead.
