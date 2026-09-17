# Analysis

## What the controlled study found

On an RTX 5080, replacing only Qwen3-0.6B layer 13's prompt-prefill attention changed 5 of 192 standard-set choices with A_PUBLIC and 8 with V4. These included 2 and 4 gains against independently computed gold, with no standard-set regressions; a separately selected boundary stress set did contain regressions. The task-balanced mean gold-NLL changes had intervals spanning zero. This case retains the score changes, decision transitions and actual model-prefill costs rather than treating similar aggregate accuracy as equivalent behavior.

| Arm | Correct / 192 | Gold choice NLL change, nats [95% CI] | Regressions | Gains | All choice flips |
|---|---:|---|---:|---:|---:|
| B | 104 / 192 | Reference | — | — | — |
| A_PUBLIC | 106 / 192 | +0.00109 [-0.01198, +0.01404] | 0 / 104 B-correct | 2 / 88 B-wrong | 5 / 192 |
| V4 | 108 / 192 | -0.00718 [-0.01920, +0.00446] | 0 / 104 B-correct | 4 / 88 B-wrong | 8 / 192 |


The mean paired NLL effects are small compared with their intervals. V4's four gains are observations from this fixed input family, not a broadly validated improvement. No arm is selected as a winner or approved for deployment. The secondary generation floor is an important negative result: all three arms failed the exact JSON schema on all 24 selected scenarios.

## Standard, selected stress and dependent lengths

| Set / length | Independent base scenarios in this table | A_PUBLIC regressions / gains / flips | V4 regressions / gains / flips |
|---|---:|---|---|
| boundary_pool/L4096 | 46 | 2 / 1 / 3 | 1 / 2 / 3 |
| dev/L4096 | 96 | 1 / 0 / 2 | 2 / 0 / 4 |
| standard/L512 | 48 | 0 / 0 / 1 | 0 / 0 / 1 |
| standard/L2048 | 48 | 0 / 0 / 0 | 0 / 1 / 1 |
| standard/L4096 | 192 | 0 / 2 / 5 | 0 / 4 / 8 |

The 48 base scenarios at each shorter length are reused standard IDs, not 96 new independent samples. The 46 stress scenarios were selected from a separate 192-item BF16-only pool before candidate access. The pool had 12 exact ties, no nonzero gap below 0.1, and sparse retrieval bins. Descriptively, the smallest positive observed gap was 0.125; native BF16 logits limit score resolution, and FP64 analysis cannot recover discarded bits. No bins were refilled. Stress regressions show why the zero standard-set count is not a safety guarantee.

## Task utility and format mass

| Task | BF16 correct | A_PUBLIC correct | V4 correct | A_PUBLIC NLL change [95% CI] | V4 NLL change [95% CI] |
|---|---:|---:|---:|---|---|
| RETRIEVAL | 59/64 | 59/64 | 59/64 | +0.00887 [-0.00179, +0.02070] | -0.00222 [-0.01708, +0.00990] |
| COMPARISON | 30/64 | 31/64 | 32/64 | -0.00626 [-0.03131, +0.01679] | -0.02767 [-0.04842, -0.00799] |
| CODE | 15/64 | 16/64 | 17/64 | +0.00067 [-0.02709, +0.02840] | +0.00834 [-0.01899, +0.03495] |

A_PUBLIC: mean paired Brier change +0.00095 [-0.00624, +0.00813]; gold-margin change -0.01042 [-0.03971, +0.01953]. Mean allowed-label mass B/candidate: 0.99986922/0.99986279; full-vocabulary argmax was an allowed label for 192/192 of 192. Conditional choice normalization is therefore visible rather than silently replacing format compliance.

V4: mean paired Brier change -0.00303 [-0.00929, +0.00310]; gold-margin change -0.00260 [-0.03060, +0.02604]. Mean allowed-label mass B/candidate: 0.99986922/0.99986533; full-vocabulary argmax was an allowed label for 192/192 of 192. Conditional choice normalization is therefore visible rather than silently replacing format compliance.

## Local error and score propagation

| Arm | Median / p95 pooled local error versus FP32 | Mean full-vocabulary KL(B || candidate) |
|---|---|---:|
| A_PUBLIC | 2.191% / 2.482% | 0.0028955 |
| V4 | 1.879% / 2.223% | 0.0028879 |

These item-level errors pool 32 sampled queries across 16 heads. They are not Case005's distribution of document×head units, and no 1%/3% gate is imported. The reference uses common BF16 inputs with FP32 arithmetic; its output is not injected into the model. Native-BF16 differences, last-query differences and hidden-boundary absolute/reference RMS are retained separately. Different boundary denominators prevent reading a percentage ratio as an absolute amplification factor.

The local-error/scoring scatter and gap/flip bins are exploratory associations. R requires candidate as well as B logits; it cannot predict risk before execution or save inference cost. Exact BF16 ties remain flagged. No predictive classifier or AUC claim is made.

## Model cost and generation

| Arm | Complete model-prefill speedup [95% CI] | Wall median / block-mean p95 (ms) |
|---|---|---|
| B | 1.000x (control) | 87.595 / 88.460 |
| A_PUBLIC | 1.0041x [1.0033, 1.0052] | 87.231 / 88.046 |
| V4 | 1.0014x [1.0007, 1.0023] | 87.440 / 88.328 |

B: strict generation schema 0/24; scored answer/evidence 0/0; median output length 46.0; median warmed request 1.2678 s. Diagnostic and uninstrumented greedy tokens matched on 24/24.

A_PUBLIC: strict generation schema 0/24; scored answer/evidence 0/0; median output length 46.0; median warmed request 1.2316 s. Diagnostic and uninstrumented greedy tokens matched on 24/24.

V4: strict generation schema 0/24; scored answer/evidence 0/0; median output length 46.0; median warmed request 1.2086 s. Diagnostic and uninstrumented greedy tokens matched on 24/24.

A_PUBLIC: common-B-prefix argmax disagreement in 1/24; free-running divergence in 1/24; eight-token re-alignment in 0/24. A later token match would not establish state/cache recovery.

V4: common-B-prefix argmax disagreement in 1/24; free-running divergence in 1/24; eight-token re-alignment in 0/24. A later token match would not establish state/cache recovery.

Generation prompts differ from one-token scoring prompts. Invalid formatting is not proof that every underlying factual choice was wrong; no post-hoc repair converts it into a successful task. Responses at the 64-token cap are censored. Different lengths make elapsed-generation comparisons unequal-work observations. No isolated decode-time estimate was collected.

## Validity and evidence limits

All completed primary arm records passed full-output validity checks. The original overstrict LM-head shape diagnostic and its resolution remain in provenance. Frozen measurement code was not changed after evaluation began. The independent checker recalculates scores, norm errors, correctness transitions and 5000-draw intervals from public scalar records; a separate checker repeats timing calculations. Neither is an independent third-party GPU reproduction. Full-vocabulary KL and omitted hidden vectors have a narrower public audit boundary.

The initial task family, one model and one replaced layer bound this conclusion. BF16's low comparison/code accuracy and the failed JSON-generation interface prevent a broad utility claim. No non-inferiority margin, deployment tolerance or production verdict was specified. The appropriate next decision is whether to design a separate, more useful task/interface validation study—not to promote either bundle from these results.
