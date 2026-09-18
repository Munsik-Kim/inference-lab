# Analysis

| Budget / method | Removed groups | HELD_OUT median / p95 error |
|---|---|---|
| 4/16 INDEPENDENT | [8, 12, 13, 14] | 30.139% / 32.230% |
| 4/16 PAIRWISE | [8, 13, 14, 15] | 29.755% / 32.103% |
| 8/16 INDEPENDENT | [7, 8, 9, 11, 12, 13, 14, 15] | 46.064% / 50.108% |
| 8/16 PAIRWISE | [7, 8, 9, 11, 12, 13, 14, 15] | 46.064% / 50.108% |

## Transfer and selection

The calibration objectives are different functions, so their optimum values must not be compared directly as if they measured the same cost. The shared quadratic objective below evaluates both choices on the same Q.

- INDEPENDENT_4: diagonal objective 13.803061; shared pairwise objective 13.859998.
- PAIRWISE_4: diagonal objective 14.234686; shared pairwise objective 13.776041.
- INDEPENDENT_8: diagonal objective 30.857525; shared pairwise objective 31.236273.
- PAIRWISE_8: diagonal objective 30.857525; shared pairwise objective 31.236273.

The 50% selectors choose the identical set. A zero paired interval there describes equal recorded computations, not a population non-inferiority study. No new group search was added after this observation.

## All random references

For each of the 20 removal sets per budget frozen in the original protocol, average relative error over the same 192 held-out prompts. Then summarize those 20 set means. This is a distribution over sets, not 3,840 independent prompts. Values below are percentages of local relative Frobenius error, not accuracy loss.

| Mean held-out local error | 25% deletion | 50% deletion |
|---|---|---|
| Minimum among 20 frozen random sets | 32.5468% | 48.0447% |
| Median of the 20 set means | 48.8427% | 66.9175% |
| Maximum among 20 frozen random sets | 71.4458% | 81.5654% |
| INDEPENDENT | 29.8291% | 45.1326% |
| PAIRWISE | 29.6288% | 45.1326% |
| Random sets beating either selection | 0/20 | 0/20 |

Both importance-based selections beat each of these fixed local references on the recorded mean. The random minimum is not an optimum over all sets; this is a descriptive finite comparison, not universal superiority. Random full-model quality and speed were not measured. [Original indices](configs/protocol.json) and [retained summary](results/derived/summary.json) supply the inputs to this reaggregation.

## Gold and model cost


B is the unpruned native BF16 control. At 25%, both candidates remove four groups from the same sixteen. Regressions and gains use B-correct and B-wrong denominators, respectively; other fractions use all held-out prompts.

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

Local error and KL(B ∥ candidate) describe closeness to B, and fewer flips preserve more baseline choices. B is not the gold oracle. Gold-choice NLL is conditional on the four valid labels; candidate−B is better when negative. PAIRWISE preserves B more closely, but INDEPENDENT has the lower gold NLL. This does not isolate the downstream cause of that difference. The KL implementation uses full-vector probabilities from B; the stored scalar can be aggregated here, but cannot be independently reconstructed from public four-option logits. [Source](src/core.py) · [original call](scripts/run.py).

The table below retains the all-budget summaries and unconditional event counts for comparison. See the later post-hoc section for the direct between-selector NLL/accuracy intervals.

| Method | Correct / 192; regressions / gains / flips | Paired choice NLL change [95% interval] |
|---|---|---|
| B | 94/192; 0 / 0 / 0 | +0.00000 [+0.00000, +0.00000] |
| INDEPENDENT_4 | 93/192; 15 / 14 / 48 | -0.33616 [-0.42165, -0.25036] |
| PAIRWISE_4 | 95/192; 2 / 3 / 12 | -0.26630 [-0.31896, -0.21283] |
| INDEPENDENT_8 | 96/192; 12 / 14 / 42 | -0.03322 [-0.09859, +0.03229] |
| PAIRWISE_8 | 96/192; 12 / 14 / 42 | -0.03322 [-0.09859, +0.03229] |

| Method | MLP speedup [95% interval] | Model prefill speedup [95% interval] |
|---|---|---|
| INDEPENDENT_4 | 1.176× [1.092, 1.205] | 0.985× [0.870, 1.147] |
| PAIRWISE_4 | 1.193× [1.168, 1.211] | 1.001× [0.915, 1.108] |
| INDEPENDENT_8 | 1.412× [1.199, 1.477] | 0.955× [0.801, 1.137] |
| PAIRWISE_8 | 1.392× [1.155, 1.433] | 0.988× [0.911, 1.038] |

Held-out model comparison measures the complete physically sliced layer, not a masked matrix. Accuracy and local reconstruction are different outcomes. Shared input families and a single model/layer limit transfer claims. All timing intervals are pointwise, not hardware-general guarantees.

## Post-hoc interpretation

[Split-Q stability](results/derived/posthoc_Q.json) was computed only after primary results were fixed. Eigenvalues and signed pair counts describe the recorded matrix; they are not a new selection rule. No held-out oracle, new layer, new pruning budget or recovery training was run.

## What does and does not transfer

At 25% deletion, the primary held-out mean error contrast is −0.2003 percentage points (PAIRWISE minus INDEPENDENT), with a pointwise 95% interval of [−0.2848, −0.1145]. This is evidence of a limited local reconstruction advantage on this frozen input family. At 50% the two selected modules have identical weight hashes and identical recorded model scores. The pre-specified overall decision therefore remains COMPLETED_NO_CLEAR_TRANSFER: it is not a claim that the 25% result disappeared.

A post-hoc direct model-score contrast from the same 192 prompts gives PAIRWISE minus INDEPENDENT gold-choice NLL +0.069860 nats [0.025925, 0.111997] at 25% (higher is worse), despite 95 versus 93 correct choices and fewer baseline regressions (2 versus 15). Its paired accuracy difference is +1.04 percentage points [−4.17, +6.25]. These are different metrics, not a uniform quality improvement. Neither selector was selected or changed using these outcomes. [Descriptive contrasts and task/process tables](results/derived/posthoc_completion_notes.json) are explicitly post-hoc; the frozen primary endpoint is unchanged.

The baseline answers 63/64 retrieval, 15/64 comparison and 16/64 code items correctly. Near-chance performance in the latter tasks limits utility claims. Conditional choice NLL is not free-generation quality. Both 25% candidates preserve all 64 retrieval choices (63/64 correct); the 50% common structure scores 58/64. No generation, perplexity or downstream application was assessed.

## Timing variability and memory

Local MLP median paired speedups span 1.176–1.412×. All model-prefill intervals include 1, with measured medians 0.955–1.001×. Slow blocks are retained. Same-shape/same-weight 50% modules were timed separately; their differing measured costs reflect timing variation, not distinct compression designs. Three process rounds on one GPU, six repeated prompts and warm-cache blocks do not establish cross-device performance.

Held-out diagnostics peaked at 1,776,835,584 allocated bytes and 1,845,493,760 reserved bytes. Each timing process peaked at 1,292,528,640 allocated and 1,325,400,064 reserved bytes, starting from 1,240,873,984 allocated bytes. These include the baseline and resident comparison modules; they are not isolated compressed-model memory figures. Whole-device peak was NOT_SAMPLED. Before/after temperature, clock, device-used memory and utilization are retained per process; continuous background activity was not monitored.

## Completion history and audit

The earlier resource-blocked package remains a historical snapshot. Once the GPU became free, only the outstanding frozen held-out and timing stages ran. The 96 calibration and 48 development measurements, group selection, inputs, protocol and frozen runner/analysis code stayed byte-identical. This completion adds 960 held-out model forwards and three timing processes, with no new calibration, layer search or recovery training.

The independent scalar audit recomputes 29,568 retained norm records, four selectors, 960 model cells, transfer intervals and timing ratio medians. A second checker independently reconstructs 15 model-score intervals and 10 timing intervals, and checks observed full-output validity. Neither checker reproduces excluded GPU tensors or full-vocabulary KL from four-option logits. Private-full and public-scalar summary/pairs match byte-for-byte.

## Publication revision

This revision adds explanations and reaggregates existing public scalars; it runs no new GPU experiment. Original protocol, source, inputs, results, plots, HTML, RUN_STATE and historical validation records remain byte-identical to the reviewed v2 snapshot. The historical private-full audit above is not a new private-tensor audit in this publication pass. [Publication preservation checks](REPRODUCTION.md#reviewed-original-and-current-publication) distinguish those scopes.
