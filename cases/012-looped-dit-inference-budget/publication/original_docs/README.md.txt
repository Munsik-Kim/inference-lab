# Case 012 — Allocating an Image-Generation Budget: Loops vs Steps

Built a runner that allocates time between internal loop depth and generation steps in one frozen Looped-DiT checkpoint. It saves three images from identical initial noise and prepares blinded annotation of count, color and left/right constraints.

[Images and results](demo/viewer.en.html) · [Blind annotation](demo/annotation.en.html) · [Adapter code](source/adapter.py) · [Full report](REPORT.md)

## Implementation and current observations

- **Paired generation and measurement:** parity with the official Euler sampler, initial-noise hashes, separate complete-request time, GPU sampling time and memory records.
- **Time-based selection:** L1/S89 · L2/S66 · L4/S50 frozen using DEV time alone. Completed **192 MAIN images** (16 prompts × four seeds × three settings) plus **24 SMOKE images**.
- **Evaluation and inspection:** bilingual blind annotation, JSON export/import, all paired images and model-free CPU auditing.

The DEV reference, L4/S50, had median **4.536 seconds**. A/B were **7.09%/7.25%** longer, missing the 5% target. This compares nearby measured budgets, not exactly equal wall time. The report gives actual MAIN measurements over 64 requests per setting.

**Quality annotations pending (`ANNOTATION_PENDING`).** Generated images are not automatically correct images. Constraint pass rates, paired gains/losses and quality-ranked presets await evaluator annotations.

## Inspect directly

```bash
python -m pip install -r requirements-cpu.txt
python analysis/analyze.py
python analysis/audit.py --output audit-receipt.json
```

[All original images](demo/gallery.en.html) · [CPU/GPU reproduction](REPRODUCTION.md) · [Frozen presets](configs/main_settings.json) · [Attribution](NOTICE.md)

Scope: RTX 5080, one B/32 EMA checkpoint, 512px, Euler, CFG=6, BF16 denoiser / FP32 T5, batch one. B/32 denotes patch size. L=1 still uses the checkpoint trained at four loops.
