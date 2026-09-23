# Finite-precision memory horizons in Complex KDA

**Case010 · Scientific report · 23 September 2026**

Contents: [Background](#background) · [Hypothesis](#hypothesis) · [Theory](#theory) · [Methods](#methods) · [Experiment](#experiment) · [Results](#results) · [Analysis](#analysis) · [Conclusion](#conclusion) · [References](#references)

## Background

Complex KDA's expressive finite-group constructions do not establish arbitrary-length IEEE floating-point robustness. The official artifact reports final-quarter token accuracy; we ask how long **every prefix remains correct** under a persistent-state byte budget. [Siems et al.](https://arxiv.org/html/2609.24797v1)

Generic error feedback and the expressivity/robustness distinction are prior art. This CKDA-specific comparison provides auditable horizons and storage accounting. Full v1 references were checked, without reproducing their results. See [NOTICE](NOTICE.md).

## Hypothesis

Transporting a finite residual through the complete recurrence may improve failure horizons at equal total bytes. Fixed low-rank correction should depend on error geometry. Floating residuals test the mechanism while paying their full storage cost.

## Theory

For each head, the physical state follows

```
A_t = (I − beta_t k_t k_tᵀ) Diag(alpha_t)
S_t = A_t S_(t−1) + beta_t k_t v_tᵀ
```

With decoded stored state Z and residual R, the compensated update is `U_t=A_t(Z_(t−1)+R_(t−1))+B_t`. After quantizing U, its local write-back error becomes the next residual. Exact arithmetic and an exact, uncompressed residual reconstruct the same affine update; this is a standard accounting identity. The untransported ablation adds the old residual after applying A, differing by `(I−A)R`.

For candidate-minus-reference state error e, exact affine computation gives `e_t=A_t e_(t−1)+eta_storage,t`. Actual FP32 paths also require the arithmetic-roundoff difference `rho_candidate,t−rho_reference,t`; the same dtype does not make that term cancel for different input states. Under ideal unit-key normalization, |alpha|≤1 and beta∈[0,2] imply an operator norm no greater than1. An additive worst-case norm bound then uses all error terms, including roundoff. This bound does not certify labels from the learned nonlinear MLP. Nor does a finite symbolic group require memory growing with sequence length: an exact group-ID tracker is a separate capacity control.

Define tau as the first wrong scored label, `F(L)=Pr(tau≤L)`, and survival `1−F(L)`. We report grid-restricted T_epsilon, restricted mean failure-free length, and exact one-sided binomial bounds. A later correct label cannot undo first failure; a right-censored observation does not imply infinite memory.

## Methods

The [pinned official model](https://github.com/OpenEuroLLM/ComplexKDA/tree/ef9d108d1692387cae37f5b2d539a71826a127c1) has one layer, width 48, twelve 16×16 heads, signed gates, extended beta, the original MLP decoder, 56,530 parameters, and 3,072 recurrent FP32 values. We preserve its weights, normalization, gating, projection, residual embedding, and classifier.

Seven input IDs are projected separately at **B=1,T=1**, forming a shared 29,100-byte coefficient table charged to every arm. Torch uses the physical transition and official `einsum` read. Two DEV sequences of length 2048, with zero/nonzero initial states, passed unchanged atol 2e-5 / rtol 2e-4; final states matched the official sequential cache bitwise. An earlier NumPy route failed 413/24,588 logit comparisons, maximum absolute difference 0.007014. Its [development evidence](provenance/development_attempts.json) remains retained; tolerances were not relaxed.

Serialized caches include codes, scales, residuals, RNG, and cursor. Shared configuration, bases, and tables count toward `shared + N × per-stream`, for N=1,16,128; common model weights are separate. Decoded FP arrays are temporary, not persistent cache. Candidates receive neither gold nor an external native-state error. Readout follows write-back from the represented state. BOS is quantized but unscored; invalid logits count as wrong, while nonfinite recurrent states cause absorbing failure.

The original 26 arms cover native FP32, uniform 2–16, stochastic 4, full residual 4+4/8/FP32, low-rank 4+8 at ranks 1/2/4, untransported 4+4, and mixed 4/8 or 6/8. CAL fixes centered covariance bases and ranks mixed channels using paired residual energy times the next complete transition's squared column norm. This generic comparator is not DAMP reproduction.

## Experiment

The [protocol](configs/protocol.json) separates CAL 64×32, DEV 128×128, and TEST 512×2048, with seeds 1001/2001/3001. Independent uniform S3 elements and integer left-composition supply inputs/gold. All 512 TEST sequences supply seven horizons 32–2048. One-sided 95% simultaneous bounds use Bonferroni family 26×7×3=546; model seeds remain separate strata. A supported 5%-risk horizon requires the Clopper–Pearson survival lower bound to reach 0.95.

Three models completed 20,000 updates, batch 128, upstream Muon/AdamW and OneCycleLR, curriculum 4/6/8/16/32. Final scheduled checkpoints alone were used. Training took 459.9/533.6/567.7 seconds; each achieved 100% DEV32 survival/final-quarter accuracy. Seed 0 passed the predefined admission rule for seeds 1/2. This local schedule does not reproduce the official 60,000-update, batch 1024 study.

Five toy geometries isolate lattice, rotation, noncommuting, and moving/switching-plane effects. [Original records](results/toy-run/) and [versioned repairs](results/toy-repair-v1b/) retain numerical failures and the stochastic seed-key correction.

## Results

All 78 original learned evaluations produced full-horizon records. Every arm/model combination failed all 512 sequences by 2048. Nonfinite-state absorption affected uniform 2 in 248/483/496 sequences and uniform 3 in nine seed-0 sequences; these failures remain included. The table gives amortized bytes at N=128, restricted mean failure-free length (RMST), and empirical/confidence-supported 5%-risk horizons. Triplets follow seeds 0/1/2, using the original 546 family.

| Arm | Bytes/stream | RMST | Empirical horizon | Supported horizon |
|---|---:|---|---|---|
|Native FP32|12,525.77|283.10 / 271.24 / 239.87|128 / 64 / 128|64 / 64 / 64|
|Uniform 4|1,823.45|147.93 / 124.32 / 185.51|64 / 32 / 64|32 / 32 / 64|
|Uniform 5|2,207.45|235.82 / 211.86 / 226.48|64 / 64 / 128|64 / 64 / 64|
|Uniform 6|2,591.45|273.94 / 258.02 / 235.45|128 / 64 / 128|64 / 64 / 64|
|Uniform 8|3,359.45|283.44 / 270.31 / 239.62|128 / 64 / 128|64 / 64 / 64|
|Full 4+4|3,409.86|282.25 / 270.02 / 238.89|128 / 64 / 128|64 / 64 / 64|
|Full 4+8|4,945.86|282.88 / 271.47 / 240.16|128 / 64 / 128|64 / 64 / 64|
|Full 4+FP32|14,112.17|283.10 / 271.24 / 239.87|128 / 64 / 128|64 / 64 / 64|
|Low-rank 1|2,071.90|202.09 / 176.75 / 208.71|64 / 64 / 64|64 / 32 / 64|
|Low-rank 2|2,269.90|282.89 / 218.40 / 236.77|128 / 64 / 128|64 / 64 / 64|
|Low-rank 4|2,665.90|241.79 / 235.82 / 240.89|128 / 64 / 128|64 / 64 / 64|

Native sequences all fail by 1024; seed 2 by 512. At 128, native survival is 96.48/89.65/97.66%, despite final-quarter accuracy 99.73/98.44/99.90%. At 2048, token accuracy remains 42.94/50.82/39.81% despite zero survival. BOS accuracy is 100/0/100%; seed 1's unscored class 3 prediction is retained. See [all native horizons](results/summary/native_reference.csv), [all arms](results/summary/all_arms.csv), and [failure curves](figures/learned_first_failure.png).

In C31 rotation, toy INT4 RMST is 11.16; full 4+4 has zero failures through 2048, versus 24.25 for fixed low-rank and 9.79 for untransported correction. However, native FP32 also has zero failures and fits inside the full 4+4 N=128 cap: 4,167 versus 4,944 bytes. Signed-lattice INT4 already survives 2048; moving-plane full 4+4 fails 106/512. Geometry and feasible native controls prevent a universal residual claim. See [toy curves](figures/toy_first_failure.png).

## Analysis

At N=16/128, low-rank 1 improves RMST over the strongest feasible original-menu baseline by 54.16/52.43/23.20 tokens; low-rank 2 by 47.07/6.54/10.28. Only low-rank 1 seed 0 raises the supported horizon, 32→64; rank 2's supported horizon ties feasible baselines. At N=1, shared basis overhead removes both ranks' RMST advantages. These TEST-selected comparisons are descriptive, not paired superiority tests. [Budget table](results/summary/budget_comparisons.csv); [N=128 frontier](figures/learned_budget_horizons_N128.png).

Full 4+4 falls below the best feasible RMST in every seed. Full 4+8 exceeds it only in seed 2 by 0.16 tokens, without extending supported horizons. Floating residuals match native RMST while costing more than native FP32. Stochastic 4 and untransported 4+4 have lower RMST than uniform 4 in every seed. No arm supports a 1%-risk horizon under the simultaneous bound.

Low-rank 2 shows large long-rollout diagnostics, especially seed 2. Original TEST records:

| Seed | Write-back MSE | Projection leakage MSE | State norm mean | Native state norm mean |
|---:|---:|---:|---:|---:|
|[0](results/learned-seed0/TEST/LOWRANK_4_8_R2.json)|1.064e+11|1.064e+11|2.077e+07|[2.936e+03](results/learned-seed0/TEST/NATIVE_FP32.json)|
|[1](results/learned-seed1/TEST/LOWRANK_4_8_R2.json)|9.166e+10|9.166e+10|5.178e+07|[1.671e+03](results/learned-seed1/TEST/NATIVE_FP32.json)|
|[2](results/learned-seed2/TEST/LOWRANK_4_8_R2.json)|2.536e+25|2.536e+25|1.042e+14|[1.462e+03](results/learned-seed2/TEST/NATIVE_FP32.json)|

These averages cover all 2,049 writes, including BOS and continued rollout after first label failure. MSE measures the candidate's own write-back and discarded residual projection, **not error against native state**. State norm is the mean per-sequence Frobenius norm. These aggregate diagnostics do not isolate the cause of first failure.

Isolated CPU costs below are medians of three DEV 16×128 repeats, seed 2002, two threads, serial fresh processes. Units are milliseconds per scored group token, with BOS overhead included. [Timing records](results/summary/timings.csv) retain concurrent embedded measurements separately.

| Model seed | Native | Uniform 4 | Uniform 8 | Low-rank 2 | Full 4+4 | CAL fit seconds |
|---:|---:|---:|---:|---:|---:|---:|
|0|0.0234|0.0973|0.0951|0.1442|0.1578|0.19483|
|1|0.0258|0.0963|0.0907|0.1466|0.1615|0.19487|
|2|0.0245|0.1009|0.0987|0.1472|0.1510|0.19769|

Complete-call timing includes allocation, initial packing, table lookup, decoding, transition, residual projection, encoding, and original FP32 readout; it excludes diagnostics, gold metrics, loading, table construction, and I/O. Post-hoc cost-only calibration timing covers one unchanged INT4 rollout/covariance/eigendecomposition/mixed-score fit per seed, excluding input/model setup. Every returned array matched the retained calibration byte-for-byte. Returned statistics occupy 35,712 bytes, **not peak memory**. [Calibration receipts](results/calibration-cost/). These are CPU reference costs, not GPU-kernel speedups or whole-model memory measurements.

## Conclusion

### Completed budget supplement

The separate [four-arm audit](configs/budget_supplement_v1.json) adds TOP2/TOP5 mixed 4/8 and TOP1/TOP4 mixed 6/8. Byte feasibility and existing CAL rankings fixed these allocations before any low-rank TEST results, although other TEST results already existed. All twelve evaluations completed. This is `POST_HOC_BUDGET_AUDIT_SAME_TEST`: the same TEST inputs, unchanged original candidates, no new calibration, and no fresh independent confirmation. Original 546-family results remain unchanged; the [joint view](results/supplement-summary/) re-expresses all thirty arms with a 630-comparison family. Supplementary timing remains unmeasured.

| Additional baseline | Total bytes at N=1 / 16 / 128 | RMST, seeds 0 / 1 / 2 |
|---|---|---|
|Mixed 4/8 TOP2|31,815 / 58,575 / 258,383|209.71 / 183.43 / 211.55|
|Mixed 4/8 TOP5|32,103 / 63,183 / 295,247|228.73 / 217.30 / 220.18|
|Mixed 6/8 TOP1|32,439 / 68,559 / 338,255|273.41 / 259.17 / 233.39|
|Mixed 6/8 TOP4|32,583 / 70,863 / 356,687|283.78 / 268.66 / 238.99|

Every added baseline supports a 5%-risk horizon of at least64 in each seed. At N=128, TOP2 costs258,383 bytes versus rank1's265,203 and exceeds rank1 RMST in all three seeds by7.62 / 6.68 / 2.85 tokens. The original rank1 seed0 supported-horizon advantage disappears against this feasible allocation. Rank2 still has higher RMST than the best feasible baseline at N=128 by47.07 / 6.54 / 10.28 tokens, while both support the same horizon64.

Across **54 residual-budget comparisons**—six full/low-rank arms × three seeds × three stream counts—no candidate has a higher confidence-supported 5%-risk horizon than the best feasible baseline in this recorded menu. This is a descriptive comparison of supported lower bounds, not a statistical superiority test, a global allocation optimum, or proof of no possible benefit. [All budget comparisons](results/supplement-summary/budget_comparisons.csv).

The study implements actual packed storage, causal correction, all-prefix failure records and exact restart checks. It **does not demonstrate an equal-budget improvement in the primary supported-horizon endpoint** for these small trained models and tested allocations. Some mean failure-free lengths improve; native extrapolation is limited, and the packing-inclusive CPU paths are slower than native. These are separate observations, preserved alongside the implementation.

## References

- Siems et al., [Complex KDA: Understanding and Enhancing the Expressivity of Kimi Delta Attention](https://arxiv.org/html/2609.24797v1), 2026, v1.
- Zhang et al., [DAMP: Decay-Aware Mixed-Precision Recurrent-State Quantization](https://arxiv.org/html/2608.27513v1), 2026, v1; its allocation/storage method is not reproduced.
- Chung, Choi and Kim, [Rethinking State Tracking in Recurrent Models Through Error Control Dynamics](https://arxiv.org/html/2605.07755v1), 2026, v1.
- Erbas, Intes and Pandey, [When Quantization Breaks Memory: Recurrent-State Write-Back in Low-Precision Temporal Inference](https://arxiv.org/html/2609.04490v1), 2026, v1. Its clipped residual follows the raw update; our transported correction and untransported diagnostic do not reproduce its full clipping/grid/read conventions.
