# Case014 P1b — Transfer of compensation-strength selection

2026-10-15 KST · p1b_r2

[Overview](#section-1) · [Problem and hypotheses](#section-2) · [Changed and fixed elements](#section-3) · [Data and selection rules](#section-4) · [Results](#section-5) · [Mechanisms and uncertainty](#section-6) · [Decision before P2](#section-7) · [Reproduction and sources](#section-8)

<a id="section-1"></a>

## 1. Overview

Separated the documents used to fit K from those used to choose alpha, then evaluated the fixed INT4 candidates on new documents. Reused six existing K matrices and30 candidates for Qwen3-0.6B, froze two selectors using32 new books, and evaluated64 different PG19 books and16 ArXiv content groups. **Execution: COMPLETED_P1B.**

**Q1:** SELECT_SSE reduced fresh-ID local SSE versus OLD_FIT for all six targets. Primary D_macro=−0.360780,95% CI[−0.385582,−0.338209], or−36.078[−38.558,−33.821] percentage points of alpha0 SSE.

**Q2:** The additional local-SSE reduction versus fixed0.25 was−1.122[−1.307,−0.933] baseline-SSE percentage points. The NLL difference−0.000275 nats/token had95% CI[−0.000628,+0.000077], including zero. Auxiliary intervals against fixed0.5 were negative for both SSE and NLL.

**Q3:** Selectors differed for5/6 targets. SELECT_NLL minus SELECT_SSE fresh-ID NLL was−0.000514[−0.001174,+0.000128], inconclusive. NLL selection increased macro local SSE versus alpha0.

**SHIFT limitation:** Against alpha0, SELECT_SSE increased SSE by11.578% and NLL by0.005248 nats/token; SELECT_NLL increased them by64.595% and0.011454. Independent selection did not remove cross-domain overcompensation.

| scope | status |
| --- | --- |
| K fit / target quantization | P1 artifacts reused;0 new calls |
| candidate rows / policy aliases | 3,360 /4,704; aliases are shared measurements |
| documents | SELECT32 / fresh ID64 / fresh SHIFT16; smoke reused |
| runtime | stored INT4 → FP32 decode → dense BF16 full hybrid forward |
| inference unit | one target intervention; different prefixes for blocks4/12/20 |
| P2 / P3 / remote publication | NOT_RUN_SCOPE / NOT_RUN_SCOPE / NOT_RUN |



[Reproduce](REPRODUCE.md) · [Frozen spec](specs/p1b.json) · [Frozen selection](../../results/p1b_r2/selection_frozen.json) · [All policy summaries](../../results/p1b_r2/policy_summary.csv) · [All paired contrasts](../../results/p1b_r2/paired_contrasts.csv) · [All3360 raw rows](../../results/p1b_r2/cells) · [Independent audit](../../results/p1b_r2/independent_audit.json) · [Preservation](../../results/p1b_r2/preservation.json) · [Historical P1 motivation](../../results/p1b_r2/historical_P1_reference.json)


<a id="section-2"></a>

## 2. Problem and hypotheses

P1 CAL-FIT choices increased local SSE on its EVAL set; post-hoc alpha0.25 reduced it for all six targets. P1b was designed after observing that result. It separates beating OLD_FIT, adding value beyond a fixed weak strength, and the difference between SSE and gold-NLL objectives. Old EVAL minima and alpha_cont_star did not feed either selector.

<a id="section-3"></a>

## 3. Changed and fixed elements

Targets are zero-based blocks4/12/20, o_proj[1024,2048] and down_proj[1024,3072]. Grid=[0,.25,.5,.75,1]. All six K/lambda identities,140 prefixes and30 targets match P1 full-file hashes. Prefix replacement affects only q/k/v/o/gate/up/down in blocks<l; other target-block modules and the suffix stay original. Common Z across alpha was checked on every row.

Used BF16/batch1/seq256/SDPA/use_cache=False, TF32=False, BF16/FP16 reduced-precision reduction=False and the native full256-position head. No chat/BOS/EOS. Scored255 gold next tokens and full-vocabulary KL(BF16||candidate) at16 fixed positions. Reused P1 identity tolerance1e−10 and sign scale1e−9×SSE. Runtime guards reject fitting and encoding calls.


| policy | rule |
| --- | --- |
| BASE_ZERO | 0 (prefix+target RTN hybrid) |
| OLD_FIT | P1 CAL-FIT selected value |
| FIXED_QUARTER / HALF / FULL | .25 / .5 /1 |
| SELECT_SSE | SELECT raw SSE argmin |
| SELECT_NLL | SELECT gold next-token NLL argmin |



<a id="section-4"></a>

## 4. Data and selection rules

Selected96 books by stable canonical-ebook-ID hash within the first existing PG19 train shard; seed6101517 assigns32 SELECT and64 EVAL books from that pool. P1 used81 books from test. The source split differs; strict identical distributions are not established. Used16 new Pile ArXiv content groups after excluding its prior16. Article IDs are unavailable. New transfer:0 bytes.

Checked canonical ID/content/token/normalized-window identity and token5gram overlap against97 known prior inputs and among112 new inputs. Across17,080 pairs, maximum Jaccard=0.010460; none reached0.8. Full-source near duplicates, other unobserved past usage and pretraining exposure remain unverified. All model inputs contain256 tokens with the original masks/positions.

SELECT_SSE minimizes summed saved-weight local SSE on SELECT; SELECT_NLL minimizes sumNLL/sumValidTokens on the same SELECT records. Exact ties choose the smaller alpha. All12 choices froze with zero fresh-evaluation forwards. EVAL64 and SHIFT16 ran afterwards. Same-alpha policies alias identical rows/artifacts, with unchanged core bytes.

| target | OLD_FIT | SELECT_SSE | SELECT_NLL | SSE artifact SHA prefix | NLL artifact SHA prefix |
| --- | --- | --- | --- | --- | --- |
| 4/o_proj | 1 | 0.5 | 1 | c07a339024a6 | 2e890ab723ad |
| 4/down_proj | 1 | 0.25 | 0.25 | 7f066ce3fb9e | 7f066ce3fb9e |
| 12/o_proj | 1 | 0.25 | 0.75 | ffea19addbf6 | 31d0c99e7bc6 |
| 12/down_proj | 1 | 0.25 | 0 | f85c528d2363 | bad898512d21 |
| 20/o_proj | 1 | 0.5 | 1 | 70a65a6cf833 | 837da86808e0 |
| 20/down_proj | 0.75 | 0.25 | 0.5 | 06f7096db00a | 0bd88e9338f3 |



<a id="section-5"></a>

## 5. Results

S0 is alpha0 SSE for the same target/documents. ΔSSE/S0 divides candidate-minus-comparator by alpha0, not by OLD_FIT. Values above average six fixed independent interventions equally. Per-target intervals, paired improvement/loss/tie counts and medians are in the complete CSV. PPL is computed separately for each target hybrid, never averaged across targets.

| cohort | contrast | ΔSSE/S0 ×100 | 95% CI | ΔNLL nats/token | 95% CI |
| --- | --- | --- | --- | --- | --- |
| FRESH-EVAL-ID | SELECT_SSE − OLD_FIT | -36.078 | [-38.558, -33.821] | -0.003735 | [-0.004858, -0.002605] |
| FRESH-EVAL-ID | SELECT_SSE − BASE_ZERO | -10.172 | [-10.712, -9.613] | -0.001055 | [-0.001743, -0.000416] |
| FRESH-EVAL-ID | SELECT_SSE − FIXED_QUARTER | -1.122 | [-1.307, -0.933] | -0.000275 | [-0.000628, 0.000077] |
| FRESH-EVAL-ID | SELECT_SSE − FIXED_HALF | -3.173 | [-3.698, -2.711] | -0.000608 | [-0.001158, -0.000069] |
| FRESH-EVAL-ID | SELECT_NLL − BASE_ZERO | 3.404 | [2.002, 4.869] | -0.001570 | [-0.002602, -0.000534] |
| FRESH-EVAL-ID | SELECT_NLL − SELECT_SSE | 13.576 | [12.683, 14.521] | -0.000514 | [-0.001174, 0.000128] |
| FRESH-EVAL-ID | SELECT_NLL − FIXED_HALF | 10.403 | [9.910, 10.913] | -0.001122 | [-0.001929, -0.000358] |
| FRESH-SHIFT | SELECT_SSE − OLD_FIT | -154.292 | [-172.864, -136.662] | -0.021617 | [-0.027313, -0.016272] |
| FRESH-SHIFT | SELECT_SSE − BASE_ZERO | 11.578 | [8.693, 14.547] | 0.005249 | [0.003806, 0.006684] |
| FRESH-SHIFT | SELECT_SSE − FIXED_QUARTER | 5.598 | [4.691, 6.508] | 0.001607 | [0.000875, 0.002363] |
| FRESH-SHIFT | SELECT_SSE − FIXED_HALF | -29.302 | [-33.877, -25.075] | -0.004185 | [-0.005875, -0.002424] |
| FRESH-SHIFT | SELECT_NLL − BASE_ZERO | 64.595 | [55.514, 74.349] | 0.011454 | [0.008136, 0.014990] |
| FRESH-SHIFT | SELECT_NLL − SELECT_SSE | 53.017 | [46.772, 59.847] | 0.006205 | [0.003701, 0.008656] |
| FRESH-SHIFT | SELECT_NLL − FIXED_HALF | 23.715 | [21.406, 26.152] | 0.002020 | [0.000022, 0.004155] |


![All target alpha-SSE curves](../../figures/p1b_r2/01_alpha_sse.svg)

![All target alpha-NLL curves](../../figures/p1b_r2/02_alpha_nll.svg)

![Paired policy effects](../../figures/p1b_r2/04_paired_effects.svg)

Tiny horizontal symbol offsets reveal ties; exact alphas are in the choice table.

### FRESH-EVAL-ID /64 books

#### 4/o_proj

| policy | α | ΔSSE/S0 % | NLL | ΔNLL | PPL | KL | flip % | core bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE_ZERO | 0 | 0.000 | 3.873390 | 0.000000 | 48.1052 | 0.152956 | 17.237 | 1179684 |
| OLD_FIT | 1 | 10.204 | 3.872694 | -0.000696 | 48.0717 | 0.156363 | 17.610 | 1179684 |
| FIXED_QUARTER | 0.25 | -10.712 | 3.872189 | -0.001201 | 48.0474 | 0.152732 | 17.053 | 1179684 |
| FIXED_HALF | 0.5 | -12.653 | 3.872121 | -0.001269 | 48.0442 | 0.151412 | 16.967 | 1179684 |
| FIXED_FULL | 1 | 10.204 | 3.872694 | -0.000696 | 48.0717 | 0.156363 | 17.610 | 1179684 |
| SELECT_SSE | 0.5 | -12.653 | 3.872121 | -0.001269 | 48.0442 | 0.151412 | 16.967 | 1179684 |
| SELECT_NLL | 1 | 10.204 | 3.872694 | -0.000696 | 48.0717 | 0.156363 | 17.610 | 1179684 |


#### 4/down_proj

| policy | α | ΔSSE/S0 % | NLL | ΔNLL | PPL | KL | flip % | core bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE_ZERO | 0 | 0.000 | 3.872847 | 0.000000 | 48.0791 | 0.152909 | 17.279 | 1769508 |
| OLD_FIT | 1 | 47.077 | 3.881571 | 0.008724 | 48.5004 | 0.176804 | 18.971 | 1769508 |
| FIXED_QUARTER | 0.25 | -5.924 | 3.872658 | -0.000190 | 48.0700 | 0.152164 | 17.169 | 1769508 |
| FIXED_HALF | 0.5 | -0.115 | 3.873519 | 0.000672 | 48.1114 | 0.156825 | 17.567 | 1769508 |
| FIXED_FULL | 1 | 47.077 | 3.881571 | 0.008724 | 48.5004 | 0.176804 | 18.971 | 1769508 |
| SELECT_SSE | 0.25 | -5.924 | 3.872658 | -0.000190 | 48.0700 | 0.152164 | 17.169 | 1769508 |
| SELECT_NLL | 0.25 | -5.924 | 3.872658 | -0.000190 | 48.0700 | 0.152164 | 17.169 | 1769508 |


#### 12/o_proj

| policy | α | ΔSSE/S0 % | NLL | ΔNLL | PPL | KL | flip % | core bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE_ZERO | 0 | 0.000 | 3.936941 | 0.000000 | 51.2616 | 0.231821 | 24.246 | 1179684 |
| OLD_FIT | 1 | 44.008 | 3.936462 | -0.000479 | 51.2370 | 0.235120 | 24.320 | 1179684 |
| FIXED_QUARTER | 0.25 | -6.920 | 3.935726 | -0.001215 | 51.1993 | 0.230117 | 24.112 | 1179684 |
| FIXED_HALF | 0.5 | -2.047 | 3.935915 | -0.001026 | 51.2090 | 0.230502 | 24.026 | 1179684 |
| FIXED_FULL | 1 | 44.008 | 3.936462 | -0.000479 | 51.2370 | 0.235120 | 24.320 | 1179684 |
| SELECT_SSE | 0.25 | -6.920 | 3.935726 | -0.001215 | 51.1993 | 0.230117 | 24.112 | 1179684 |
| SELECT_NLL | 0.75 | 15.417 | 3.935728 | -0.001213 | 51.1994 | 0.232002 | 24.112 | 1179684 |


#### 12/down_proj

| policy | α | ΔSSE/S0 % | NLL | ΔNLL | PPL | KL | flip % | core bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE_ZERO | 0 | 0.000 | 3.937767 | 0.000000 | 51.3039 | 0.232531 | 24.161 | 1769508 |
| OLD_FIT | 1 | 36.963 | 3.949020 | 0.011253 | 51.8845 | 0.246556 | 25.496 | 1769508 |
| FIXED_QUARTER | 0.25 | -7.896 | 3.938848 | 0.001081 | 51.3594 | 0.232247 | 24.259 | 1769508 |
| FIXED_HALF | 0.5 | -4.019 | 3.943200 | 0.005432 | 51.5834 | 0.235857 | 24.724 | 1769508 |
| FIXED_FULL | 1 | 36.963 | 3.949020 | 0.011253 | 51.8845 | 0.246556 | 25.496 | 1769508 |
| SELECT_SSE | 0.25 | -7.896 | 3.938848 | 0.001081 | 51.3594 | 0.232247 | 24.259 | 1769508 |
| SELECT_NLL | 0 | 0.000 | 3.937767 | 0.000000 | 51.3039 | 0.232531 | 24.161 | 1769508 |


#### 20/o_proj

| policy | α | ΔSSE/S0 % | NLL | ΔNLL | PPL | KL | flip % | core bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE_ZERO | 0 | 0.000 | 3.978386 | 0.000000 | 53.4307 | 0.301128 | 27.855 | 1179684 |
| OLD_FIT | 1 | 2.597 | 3.973023 | -0.005364 | 53.1449 | 0.297084 | 27.537 | 1179684 |
| FIXED_QUARTER | 0.25 | -16.501 | 3.975430 | -0.002956 | 53.2730 | 0.298955 | 27.831 | 1179684 |
| FIXED_HALF | 0.5 | -21.290 | 3.973848 | -0.004538 | 53.1888 | 0.296659 | 27.641 | 1179684 |
| FIXED_FULL | 1 | 2.597 | 3.973023 | -0.005364 | 53.1449 | 0.297084 | 27.537 | 1179684 |
| SELECT_SSE | 0.5 | -21.290 | 3.973848 | -0.004538 | 53.1888 | 0.296659 | 27.641 | 1179684 |
| SELECT_NLL | 1 | 2.597 | 3.973023 | -0.005364 | 53.1449 | 0.297084 | 27.537 | 1179684 |


#### 20/down_proj

| policy | α | ΔSSE/S0 % | NLL | ΔNLL | PPL | KL | flip % | core bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE_ZERO | 0 | 0.000 | 3.977987 | 0.000000 | 53.4094 | 0.301802 | 27.947 | 1769508 |
| OLD_FIT | 0.75 | 14.585 | 3.980627 | 0.002640 | 53.5506 | 0.299831 | 27.616 | 1769508 |
| FIXED_QUARTER | 0.25 | -6.349 | 3.977787 | -0.000201 | 53.3987 | 0.299447 | 27.610 | 1769508 |
| FIXED_HALF | 0.5 | -1.873 | 3.976032 | -0.001956 | 53.3051 | 0.299008 | 27.702 | 1769508 |
| FIXED_FULL | 1 | 42.716 | 3.981113 | 0.003125 | 53.5766 | 0.304467 | 27.966 | 1769508 |
| SELECT_SSE | 0.25 | -6.349 | 3.977787 | -0.000201 | 53.3987 | 0.299447 | 27.610 | 1769508 |
| SELECT_NLL | 0.5 | -1.873 | 3.976032 | -0.001956 | 53.3051 | 0.299008 | 27.702 | 1769508 |


### FRESH-SHIFT /16 content groups

#### 4/o_proj

| policy | α | ΔSSE/S0 % | NLL | ΔNLL | PPL | KL | flip % | core bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE_ZERO | 0 | 0.000 | 2.667016 | 0.000000 | 14.3969 | 0.139474 | 13.015 | 1179684 |
| OLD_FIT | 1 | 98.084 | 2.678320 | 0.011304 | 14.5606 | 0.153362 | 13.750 | 1179684 |
| FIXED_QUARTER | 0.25 | 0.264 | 2.669307 | 0.002291 | 14.4300 | 0.141895 | 13.088 | 1179684 |
| FIXED_HALF | 0.5 | 16.319 | 2.672786 | 0.005770 | 14.4803 | 0.142927 | 13.284 | 1179684 |
| FIXED_FULL | 1 | 98.084 | 2.678320 | 0.011304 | 14.5606 | 0.153362 | 13.750 | 1179684 |
| SELECT_SSE | 0.5 | 16.319 | 2.672786 | 0.005770 | 14.4803 | 0.142927 | 13.284 | 1179684 |
| SELECT_NLL | 1 | 98.084 | 2.678320 | 0.011304 | 14.5606 | 0.153362 | 13.750 | 1179684 |


#### 4/down_proj

| policy | α | ΔSSE/S0 % | NLL | ΔNLL | PPL | KL | flip % | core bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE_ZERO | 0 | 0.000 | 2.666970 | 0.000000 | 14.3963 | 0.139638 | 12.500 | 1769508 |
| OLD_FIT | 1 | 257.519 | 2.716914 | 0.049944 | 15.1335 | 0.211419 | 17.843 | 1769508 |
| FIXED_QUARTER | 0.25 | 11.641 | 2.673579 | 0.006609 | 14.4917 | 0.159106 | 13.603 | 1769508 |
| FIXED_HALF | 0.5 | 58.470 | 2.680998 | 0.014028 | 14.5997 | 0.171326 | 14.338 | 1769508 |
| FIXED_FULL | 1 | 257.519 | 2.716914 | 0.049944 | 15.1335 | 0.211419 | 17.843 | 1769508 |
| SELECT_SSE | 0.25 | 11.641 | 2.673579 | 0.006609 | 14.4917 | 0.159106 | 13.603 | 1769508 |
| SELECT_NLL | 0.25 | 11.641 | 2.673579 | 0.006609 | 14.4917 | 0.159106 | 13.603 | 1769508 |


#### 12/o_proj

| policy | α | ΔSSE/S0 % | NLL | ΔNLL | PPL | KL | flip % | core bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE_ZERO | 0 | 0.000 | 2.718075 | 0.000000 | 15.1511 | 0.201826 | 17.549 | 1179684 |
| OLD_FIT | 1 | 131.526 | 2.729219 | 0.011144 | 15.3209 | 0.215492 | 18.407 | 1179684 |
| FIXED_QUARTER | 0.25 | 2.033 | 2.720406 | 0.002331 | 15.1865 | 0.202429 | 17.377 | 1179684 |
| FIXED_HALF | 0.5 | 24.920 | 2.720232 | 0.002157 | 15.1838 | 0.204196 | 17.721 | 1179684 |
| FIXED_FULL | 1 | 131.526 | 2.729219 | 0.011144 | 15.3209 | 0.215492 | 18.407 | 1179684 |
| SELECT_SSE | 0.25 | 2.033 | 2.720406 | 0.002331 | 15.1865 | 0.202429 | 17.377 | 1179684 |
| SELECT_NLL | 0.75 | 68.371 | 2.724932 | 0.006857 | 15.2554 | 0.210709 | 17.819 | 1179684 |


#### 12/down_proj

| policy | α | ΔSSE/S0 % | NLL | ΔNLL | PPL | KL | flip % | core bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE_ZERO | 0 | 0.000 | 2.718607 | 0.000000 | 15.1592 | 0.202827 | 17.328 | 1769508 |
| OLD_FIT | 1 | 164.908 | 2.737573 | 0.018966 | 15.4494 | 0.218471 | 19.338 | 1769508 |
| FIXED_QUARTER | 0.25 | 3.915 | 2.717974 | -0.000633 | 15.1496 | 0.198836 | 17.475 | 1769508 |
| FIXED_HALF | 0.5 | 32.711 | 2.721828 | 0.003221 | 15.2081 | 0.200325 | 17.525 | 1769508 |
| FIXED_FULL | 1 | 164.908 | 2.737573 | 0.018966 | 15.4494 | 0.218471 | 19.338 | 1769508 |
| SELECT_SSE | 0.25 | 3.915 | 2.717974 | -0.000633 | 15.1496 | 0.198836 | 17.475 | 1769508 |
| SELECT_NLL | 0 | 0.000 | 2.718607 | 0.000000 | 15.1592 | 0.202827 | 17.328 | 1769508 |


#### 20/o_proj

| policy | α | ΔSSE/S0 % | NLL | ΔNLL | PPL | KL | flip % | core bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE_ZERO | 0 | 0.000 | 2.757428 | 0.000000 | 15.7593 | 0.239492 | 21.471 | 1179684 |
| OLD_FIT | 1 | 110.122 | 2.777213 | 0.019785 | 16.0742 | 0.247573 | 21.691 | 1179684 |
| FIXED_QUARTER | 0.25 | -4.026 | 2.758526 | 0.001098 | 15.7766 | 0.238910 | 21.250 | 1179684 |
| FIXED_HALF | 0.5 | 13.509 | 2.764687 | 0.007259 | 15.8741 | 0.242520 | 21.078 | 1179684 |
| FIXED_FULL | 1 | 110.122 | 2.777213 | 0.019785 | 16.0742 | 0.247573 | 21.691 | 1179684 |
| SELECT_SSE | 0.5 | 13.509 | 2.764687 | 0.007259 | 15.8741 | 0.242520 | 21.078 | 1179684 |
| SELECT_NLL | 1 | 110.122 | 2.777213 | 0.019785 | 16.0742 | 0.247573 | 21.691 | 1179684 |


#### 20/down_proj

| policy | α | ΔSSE/S0 % | NLL | ΔNLL | PPL | KL | flip % | core bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE_ZERO | 0 | 0.000 | 2.758914 | 0.000000 | 15.7827 | 0.238076 | 20.858 | 1769508 |
| OLD_FIT | 0.75 | 233.057 | 2.808963 | 0.050049 | 16.5927 | 0.284965 | 22.574 | 1769508 |
| FIXED_QUARTER | 0.25 | 22.050 | 2.769068 | 0.010155 | 15.9438 | 0.245546 | 21.176 | 1769508 |
| FIXED_HALF | 0.5 | 99.350 | 2.783083 | 0.024169 | 16.1688 | 0.254569 | 21.936 | 1769508 |
| FIXED_FULL | 1 | 423.746 | 2.844071 | 0.085157 | 17.1856 | 0.314905 | 23.750 | 1769508 |
| SELECT_SSE | 0.25 | 22.050 | 2.769068 | 0.010155 | 15.9438 | 0.245546 | 21.176 | 1769508 |
| SELECT_NLL | 0.5 | 99.350 | 2.783083 | 0.024169 | 16.1688 | 0.254569 | 21.936 | 1769508 |



<a id="section-6"></a>

## 6. Mechanisms and uncertainty

C=alpha K_old Z; K was not refit. Verified Delta=T_inherited+T_cancellation+T_requant using saved FP32 What lifted to FP64, with A=W(X−Z), B=(What−W−alpha K)Z, B0=(What0−W)Z. New-input alpha_cont_star is an explanatory continuous diagnostic, not a candidate generator/selector. Comparisons below repeat each document across six targets and four nonzero strengths; labels may overlap.

A: inherited error falls but cancellation loss makes fixed-B T_cont positive. B: T_cont<0 but actual re-quantized Delta>0. C: the old CAL continuous correction improves there but worsens on the new document. D: lower local SSE with worse suffix gold NLL. Re-quantization includes scale changes, rounding, clipping and codec; it is not isolated rounding causality.
| cohort | independent docs | comparisons | A | B | C | D |
| --- | --- | --- | --- | --- | --- | --- |
| SELECT-ID | 32 | 768 | 0 | 10 | 344 | 188 |
| FRESH-EVAL-ID | 64 | 1536 | 1 | 23 | 685 | 366 |
| FRESH-SHIFT | 16 | 384 | 0 | 1 | 360 | 17 |



Measured BF16 direction agreement and conservative arithmetic-bound certification are separate. Of1,536 EVAL comparisons,1,535 agreed in direction;139 agreed but were uncertified by the bound. One had a different measured direction. Lack of certification does not mean unmeasured or reversed. Rare A/B counts do not establish prevalence or a unique whole-model cause.
| cohort | comparisons | measured same direction | bound certified | same direction / uncertified |
| --- | --- | --- | --- | --- |
| SELECT-ID | 768 | 768 | 700 | 68 |
| FRESH-EVAL-ID | 1536 | 1535 | 1396 | 139 |
| FRESH-SHIFT | 384 | 384 | 351 | 33 |



Only the registered primary D_macro is primary; other intervals are auxiliary/exploratory pointwise.10,000 paired document resamples, seed6101519, use the same indices for every target/policy and recompute ratio numerators/denominators. Selectors stay frozen. Intervals condition on old K and this SELECT, excluding selection/refit/model-population uncertainty. Including zero means inconclusive.

![All target document SSE/NLL changes](../../figures/p1b_r2/03_local_sse_vs_nll.svg)


| target | cohort | α_cont raw | α_cont clipped | pooled cosine A,U | ||A||/||U|| |
| --- | --- | --- | --- | --- | --- |
| 4/o_proj | SELECT-ID | 0.4020 | 0.4020 | 0.3619 | 1.1108 |
| 4/down_proj | SELECT-ID | 0.2563 | 0.2563 | 0.2545 | 1.0072 |
| 12/o_proj | SELECT-ID | 0.2854 | 0.2854 | 0.2763 | 1.0329 |
| 12/down_proj | SELECT-ID | 0.2931 | 0.2931 | 0.2727 | 1.0748 |
| 20/o_proj | SELECT-ID | 0.4598 | 0.4598 | 0.4481 | 1.0261 |
| 20/down_proj | SELECT-ID | 0.2858 | 0.2858 | 0.2637 | 1.0839 |
| 4/o_proj | FRESH-EVAL-ID | 0.4289 | 0.4289 | 0.3700 | 1.1591 |
| 4/down_proj | FRESH-EVAL-ID | 0.2571 | 0.2571 | 0.2541 | 1.0119 |
| 12/o_proj | FRESH-EVAL-ID | 0.2773 | 0.2773 | 0.2703 | 1.0260 |
| 12/down_proj | FRESH-EVAL-ID | 0.2935 | 0.2935 | 0.2744 | 1.0699 |
| 20/o_proj | FRESH-EVAL-ID | 0.4856 | 0.4856 | 0.4673 | 1.0391 |
| 20/down_proj | FRESH-EVAL-ID | 0.2886 | 0.2886 | 0.2645 | 1.0912 |
| 4/o_proj | FRESH-SHIFT | 0.1191 | 0.1191 | 0.1420 | 0.8388 |
| 4/down_proj | FRESH-SHIFT | 0.0380 | 0.0380 | 0.0664 | 0.5726 |
| 12/o_proj | FRESH-SHIFT | 0.0913 | 0.0913 | 0.1175 | 0.7767 |
| 12/down_proj | FRESH-SHIFT | 0.0797 | 0.0797 | 0.1133 | 0.7035 |
| 20/o_proj | FRESH-SHIFT | 0.1629 | 0.1629 | 0.2152 | 0.7568 |
| 20/down_proj | FRESH-SHIFT | 0.0234 | 0.0234 | 0.0509 | 0.4601 |



<a id="section-7"></a>

## 7. Decision before P2

Independent selection transferred local fidelity within ID, while extra gold-NLL value over fixed0.25 and cross-domain usefulness remain unresolved/negative. Do not adopt these six alphas as a full-model default. Any P2 requires separate authorization/registration, each method’s own input trajectory, fresh evaluation and output metrics. Sequential earlier-layer changes alter Z, so these six effects cannot be summed into a P2 prediction. P2/P3 were not run.

<a id="section-8"></a>

## 8. Reproduction and sources

Reran15 existing tests;10 new unique tests passed. Repeats are not extra unique tests. The original P1 saved-record auditor checked2,880 rows/120 summaries; P0 MC and the original P1 GPU sweep were not rerun. Independently recomputed3,360 fresh raw rows,216 summaries,12 choices,4,704 aliases and462 intervals. Also checked9 private matrix rows with an independent nibble decoder/Torch CPU FP64 and a small actual SMOKE slice identity. Public scalar audit does not re-evaluate GPU NLL/KL.

Preserved hashes for236 original Case014 files,6,351 tracked files,33 P0-private files,2,148 P1-private files and6 model snapshot files, including the full local weight file. No new official external digest or D-origin comparison. No install/upgrade/stage/commit/push/PR/merge/Pages/Drive actions.

First worker exited1 at a comparison with a historical SMOKE prefix array that was never saved:3 forwards/7.042s, preserved source/log. r2 changed availability handling only and exited0:3,813 forwards/512.949s. This correction preceded all new SELECT outputs; spec/inputs/K/quantizer stayed unchanged.

Torch peak allocated1,613,842,432 B, reserved1,845,493,760 B. Device usage is sampled after forwards, not a continuous Windows peak. Actual core bytes are1,179,684 for o_proj and1,769,508 for down_proj, identical across alpha. This is a dense BF16 quality reference, without packed-serving speed/VRAM benefit measurements.


| timing scope | seconds |
| --- | --- |
| data_prepare (outside worker) | 33.492 |
| validation (outside worker) | 1.956 |
| load (inside worker) | 1.612 |
| SELECT_forward + local metrics/I/O | 152.325 |
| select_compute | 0.004995 |
| EVAL_forward + local metrics/I/O | 244.339 |
| SHIFT_forward + local metrics/I/O | 106.597 |
| metric (included in phase totals) | 202.912 |
| saved scalar analysis (outside worker) | 1.473 |
| independent scalar audit (outside worker) | 0.471 |
| failed worker attempt | 7.042 |
| successful worker subprocess | 512.949 |
| all worker attempts total | 519.991 |




| identity | value |
| --- | --- |
| branch | research/case014-quantization-compensation |
| Git HEAD/base | 09194c3f2e2c991feff5b6abab0cf8ab81a07052 |
| GPU execution source SHA256 | 19ed11ef4246a36b1d5a68c3f6ba62146c0cea15c556dae8c6879eb1c0bca8eb |
| P1b spec SHA256 | 0d11679fbecef6c077cd55c85715a6daf1656a063f4f39171bc1028b58d46a69 |
| input manifest SHA256 | 1e08e292c742f20df04325739fc71ab2adbb892f347eb318a955987ff4c879e9 |
| selection file SHA256 | 82bd0827bb05f649d1ea71893449db15f46fb4f7262176fa900a759fa0f3738e |
| model revision | c1899de289a04d12100db370d81485cdf75e47ca |
| model full local weight SHA256 | f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b |



[Reproduce](REPRODUCE.md) · [Frozen spec](specs/p1b.json) · [Frozen selection](../../results/p1b_r2/selection_frozen.json) · [All policy summaries](../../results/p1b_r2/policy_summary.csv) · [All paired contrasts](../../results/p1b_r2/paired_contrasts.csv) · [All3360 raw rows](../../results/p1b_r2/cells) · [Independent audit](../../results/p1b_r2/independent_audit.json) · [Preservation](../../results/p1b_r2/preservation.json) · [Historical P1 motivation](../../results/p1b_r2/historical_P1_reference.json)

Sources: [PG19](https://huggingface.co/datasets/pg19), [cached PG19 mirror](https://huggingface.co/datasets/emozilla/pg19/tree/c021754c8e01c5b1cc83a1f549c1f97fbbb756b8), [Pile10k](https://huggingface.co/datasets/NeelNanda/pile-10k/tree/127bfedcd5047750df5ccf3a12979a47bfa0bafa), [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca), [Guo2026v1](https://arxiv.org/html/2610.16020v1). Dataset license cards were read from pinned local caches; raw text is excluded. This is a fixed-artifact ridge/RTN follow-up, not an official full QEP reproduction or a new quantizer. Codex assisted implementation, analysis and documentation; no independent human review is claimed.
