# Full-image AI assessment

[한국어](README.ko.md) · [Case overview](../../README.md)

### AI assessment of all 192 saved MAIN images

A code agent inspected each original PNG individually: **one AI rater, zero human raters**. Settings, time and seed labels were masked. The session had previously seen 48 example images, so this is not independent fully blind evaluation. The original human-pending record is preserved; this supplement has its own [protocol](protocol.json) and [item judgments](annotations.json).

| Loops / Steps | Median request time (s) | Every listed constraint met — AI assessment |
|---|---:|---:|
| 1 / 89 | 4.534 | 51/64 (79.69%) |
| 2 / 66 | 4.385 | 53/64 (82.81%) |
| 4 / 50 | 4.762 | 53/64 (82.81%) |

Each setting has 64 images from the same 64 prompt–noise pairs. A pass requires every frozen checklist item to be `satisfied`. This does not score aesthetics or every phrase in the prose prompt. Count prompts score count only; other categories score their explicit color, relation and compound items.

**Primary L4−L1: +3.12 percentage points**, with a 95% paired prompt-cluster bootstrap interval [-3.12, +10.94] pp. The interval includes zero. It resamples 16 prompts 5,000 times and does not include AI judgment error. There are four related template families in this small experiment.

Across 64 paired inputs: both pass 49, L1 only 2, L4 only 4, neither 9. Individual constraints show 7 gains and 6 losses. These compare final images; they are not observations of one image being corrected within a loop trajectory.

| Category | L1/S89 | L2/S66 | L4/S50 |
|---|---:|---:|---:|
| Count | 11/16 | 12/16 | 13/16 |
| Object colors | 15/16 | 15/16 | 15/16 |
| Left/right | 16/16 | 16/16 | 16/16 |
| Compound | 9/16 | 10/16 | 9/16 |

Category denominators are 16 images per setting. The 7 `uncertain` constraint labels count as failures for the primary score. Treating all of them as passes gives the optimistic sensitivity: L1 81.25%, L2 84.38%, L4 87.50%. Inspect every image and its evidence in [the full comparison](../IMAGES.md).

These are quality–time observations near a budget. DEV missed the ±5% time target, and MAIN times differ too. L2 reached the same pass count as L4 at a shorter median time, without establishing general quality superiority or a quality-ranked preset. [Shared result JSON](summary.json) · [Independent CPU arithmetic audit](audit.py)

```bash
python publication/assessment_v1/audit.py
```

Run from the Case 012 directory. This checks scoring arithmetic and hashes; it does not independently re-rate the AI judgments.
