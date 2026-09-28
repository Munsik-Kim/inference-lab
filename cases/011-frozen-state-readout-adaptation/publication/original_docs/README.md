# Case 011 — Frozen-State Readout Adaptation and Length Extrapolation

[한국어](README.ko.md) · [DIOVA](../../README.md)

Built a shared-state comparison runner and a two-tensor save/reload path for adapting only the final Linear. The study reads 1,024 new sequences through 2,048 group tokens on each of three existing CKDA checkpoints, keeping Native and INT8 storage fixed.

## Working components

- [Frozen feature adapter](source/adapter.py): Exact existing 192-dimensional GELU features; no backbone gradients.
- [Head fitting](source/fitting.py) · [patch loader](source/head_patch.py): Fit 1,158 final parameters in CPU FP64, then copy two FP32 tensors into their original slots.
- [Paired runner](source/run_eval.py) · [independent audit](analysis/audit.py): Compare first error, gold scores and controls for three readouts over one cache.
- [Fresh-process reload](source/reload_check.py): Compare logits, features and cache bytes on the actual small checkpoints.

Stack: Python · NumPy · PyTorch. Public scalar reanalysis needs neither a model nor a GPU.

## Recorded results

Under INT8, the MIXED refit reduced uninterrupted correct lifetime on all three checkpoints. MIXED−ORIGINAL changes for seeds 0/1/2 were -9.89 / -10.40 / -13.49 tokens, with all three primary intervals below zero. Over positions 257–2,048, mean gold CE nevertheless decreased in 3/3 and token accuracy increased in 2/3. Better gold scores did not translate into a longer interval of correctness from the start.

| Checkpoint | ORIGINAL | SHORT | MIXED | Δ MIXED−ORIGINAL | 98.333% interval |
| --- | --- | --- | --- | --- | --- |
| 0 | 285.84 | 285.84 | 275.96 | -9.89 | [-14.59, -5.41] |
| 1 | 264.59 | 264.59 | 254.19 | -10.40 | [-15.71, -5.16] |
| 2 | 239.80 | 239.80 | 226.31 | -13.49 | [-17.66, -9.38] |

Values are RMST0: mean consecutive correct group tokens before the first error. Each row pairs the same 1,024 sequences. The 98.333% intervals use 5,000 bootstrap resamples with an approximate Bonferroni family target of 95% across three primary contrasts.

SHORT_REFIT retained the original FP32 tensors. MIXED_REFIT changed both tensors; seeds 0/1 reached the predeclared 200-iteration limit with finite accepted iterates. Solver status and scores by length remain visible in the report.

## Check the saved results on CPU

From this Case011 directory, run with Python and NumPy and choose an output directory that does not already exist.

```bash
python -B analysis/aggregate.py --input-only-fit results/input_only_fit.json \
  --output /tmp/case011-new-audit --no-figures
```

This recalculates public predictions, gold and scalar sums. Model fitting/replay separately requires the original checkpoints, upstream and head patches. Original model weights and fitted head tensors are excluded from the review ZIP; exact manifests are included.

[Results and analysis](REPORT.md) · [Methods](METHODS.md) · [Reproduction](REPRODUCTION.md) · [Protocol](configs/protocol.json) · [Local result explorer](demo/en.html) · [Attribution](NOTICE.md)

Status: local research and review candidate. No remote publication, PR or Pages deployment was performed.
