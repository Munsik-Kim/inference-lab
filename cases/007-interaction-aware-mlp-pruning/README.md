# Interaction-Aware Structured MLP Pruning

English | [한국어](README.ko.md)

At 25% structured pruning of one Qwen3-0.6B MLP, pairwise-aware selection produced a small held-out reduction in local reconstruction error and stayed closer to the baseline model’s distribution and choices. At 50%, both selectors chose the same removal set, leaving no distinct structural comparison. Under the frozen two-budget rule, the overall result remains **COMPLETED_NO_CLEAR_TRANSFER**. Closer baseline fidelity did not provide the best gold NLL, and no whole-model prefill speedup was established.

**Scope:** RTX 5080, native BF16 dense MLP at zero-based layer 13, 512-token prompts, no retraining. B is the unpruned BF16 baseline. INDEPENDENT uses separate group importance; PAIRWISE also uses signed interactions. The deletion budgets remove 4 or 8 of 16 contiguous groups. Calibration/development/held-out contain 96/48/192 synthetic prompts, with six separate smoke prompts. Local error pools 32 fixed token outputs per prompt; prompts, not tokens, are the uncertainty unit. These held-out inputs were unseen during selection; this publication reanalysis is not fresh evaluation.

## Fixed random references

Both importance-based selections had lower **mean** held-out local error than each of the 20 frozen random sets at the same budget. Pairwise information supplied a smaller additional advantage at 25%.

<!-- publication-table: random -->
| Mean held-out local error | 25% deletion | 50% deletion |
|---|---|---|
| Minimum among 20 frozen random sets | 32.5468% | 48.0447% |
| Median of the 20 set means | 48.8427% | 66.9175% |
| Maximum among 20 frozen random sets | 71.4458% | 81.5654% |
| INDEPENDENT | 29.8291% | 45.1326% |
| PAIRWISE | 29.6288% | 45.1326% |
| Random sets beating either selection | 0/20 | 0/20 |
<!-- /publication-table: random -->

For each removal set, first average relative error over the same 192 prompts; min/median/max then summarize the 20 set means. The minimum is only the best of these 20, not the optimum over all removal sets. This descriptive comparison establishes neither universal statistical superiority nor global optimality. Random sets were measured only at the local MLP boundary, not as complete models for quality or speed. The 20×192 repetitions are not 3,840 independent prompts. [Original analysis](ANALYSIS.md#all-random-references) · [retained summary](results/derived/summary.json).

## Small local gain, unchanged overall decision

At 25%, mean paired relative-error change (PAIRWISE−INDEPENDENT) is **−0.200277 percentage points**, with a pointwise 95% interval **[−0.284779, −0.114472]**; error is lower on **141/192** prompts. The errors themselves are about 29.8%, so the added reduction is small. It is not a 0.2% accuracy improvement.

INDEPENDENT_4 removes `[8,12,13,14]`; PAIRWISE_4 removes `[8,13,14,15]`. Both 50% settings remove `[7,8,9,11,12,13,14,15]` and have identical weights and recorded outputs. Their [0,0] paired interval reflects the same structure, not independent non-inferiority evidence. The frozen rule requires both budgets’ interval upper bounds below zero; it is unchanged. [Protocol](configs/protocol.json) · [selection](results/raw/selection.json) · [paired results](results/derived/pairs.json).

![Held-out local errors: a small 25% difference and identical 50% selections](figures/01_transfer.png)

## Preserving B is different from scoring the correct answer

INDEPENDENT_4 and PAIRWISE_4 each remove 4/16 groups. Gold is the independently computed task answer. A regression means a B-correct answer became wrong; a gain means a B-wrong answer became correct.

<!-- publication-table: quality -->
| Metric · 192 held-out prompts | INDEPENDENT_4 | PAIRWISE_4 |
|---|---|---|
| Mean local relative Frobenius error | 29.8291% | 29.6288% |
| Mean full-vocabulary KL(B ∥ candidate), nats | 0.125937 | 0.042244 |
| Choice flips versus B | 48/192 | 12/192 |
| B correct → candidate wrong (regression) | 15/94 | 2/94 |
| B wrong → candidate correct (gain) | 14/98 | 3/98 |
| Wrong → different wrong | 19/192 | 7/192 |
| Correct choices | 93/192 | 95/192 |
| Mean Δgold-choice NLL (candidate−B), nats | -0.336159 | -0.266299 |
<!-- /publication-table: quality -->

Regression is conditional on B’s **94 correct** prompts; gain is conditional on its **98 wrong** prompts. Flips and wrong-to-different-wrong changes use all 192. Lower local error and KL indicate closer computations/distributions to B; fewer flips preserve B’s decisions, which are not always correct. NLL is negative log probability of the gold choice, conditional on four labels. ΔNLL is candidate−B, so **negative is better**.

Lower reconstruction error and closer baseline fidelity did not imply the best gold NLL. Baseline preservation and gold-based quality were distinct evaluation objectives in this study. Full-vocabulary KL(B ∥ candidate) was computed from complete native logits and retained as a scalar; four public option logits/norms cannot independently reconstruct it. [Measurement implementation](scripts/run.py) · [metric definitions](src/core.py).

The existing **post-hoc** direct PAIRWISE−INDEPENDENT comparison gives gold-choice NLL **+0.069860 nats [0.025925, 0.111997]**, worse for PAIRWISE on this metric. Accuracy changes by **+1.04 percentage points [−4.17, +6.25]**. These pointwise intervals did not select groups or change the original decision. [Post-hoc records](results/derived/posthoc_completion_notes.json) · [scalar checker](scripts/verify_completion.py). No component-causal explanation was isolated.

B answers retrieval 63/64, comparison 15/64 and code 16/64 correctly. Both 25% candidates retain all retrieval choices and score 63/64 there. Near-chance baseline performance on comparison/code limits utility claims; this is forced-choice scoring, not evidence of improved coding ability or free generation.

## Actual smaller modules and measured cost

Gate/up rows and down columns were physically sliced. Width/parameters/BF16 bytes for the chosen MLP are 3072 / 9,437,184 / 18,874,368 before pruning, 2304 / 7,077,888 / 14,155,776 at 25%, and 1536 / 4,718,592 / 9,437,184 at 50%. Other model layers remain unchanged; these are not whole-model compression percentages. [Structure evidence](provenance/structural_sizes.json).

Recorded local MLP speedups span **1.176–1.412×**. All model-prefill speedup intervals include 1. The same 50% module’s method-labelled timing differences reflect measurement variation. Six fixed prompts, three processes and five-call block means are a bounded same-device measurement. Allocator peaks include resident comparison modules; whole-device peak was not sampled. [Timing and memory](ANALYSIS.md#timing-variability-and-memory).

## Inspect and reproduce

- [Reviewed case ZIP](../../downloads/case007_mlp_pruning_reviewed_publication_v1.zip) · [size and SHA256 metadata](../../downloads/case007_mlp_pruning_reviewed_publication_v1.json).
- [CPU reproduction and publication checks](REPRODUCTION.md) · [methods](METHODS.md) · [limitations](LIMITATIONS.md).
- [Offline explorer source](demo/index.html): open the downloaded local HTML; GitHub source preview is not a live demo. Browser evidence is historical; real iPad is NOT_TESTED.
- [Portfolio](PORTFOLIO.md) · [review-to-publication snapshot](provenance/publication_snapshot.json).

Deployment is **NOT_ASSESSED**. This publication adds no GPU runs. Qwen/Transformers supply the model; HOPE is motivation from a separate MoE setting. This project contributes controlled selection, real structural surgery, measurements and CPU-auditable evidence, with OpenAI Codex assistance. It is not a new pruning algorithm or HOPE reproduction. [Attribution and license](NOTICE.md).
