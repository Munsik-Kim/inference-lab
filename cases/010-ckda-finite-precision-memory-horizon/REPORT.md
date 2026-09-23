# Case 010 — Finite-Precision Recurrent Memory: Storage, Restart, and Readout Horizon

[한국어](REPORT.ko.md) · [Case home](README.md) · [CPU reproduction](REPRODUCTION.md)

One project connects real low-bit state storage, residual transport, failure-preserving restart and readout-lifetime measurement. Stage A explores the storage menu; Stage B repairs a restart contract and evaluates a smaller, fixed menu on fresh inputs.

## Contents

[1. Background](#background) · [2. Hypotheses and questions](#questions) · [3. Theory](#theory) · [4. Methods](#methods) · [5. Experiments](#experiments) · [6. Results](#results) · [7. Analysis](#analysis) · [8. Conclusion](#conclusion) · [9. References](#references)

<a id="background"></a>
## 1. Background

A recurrent model writes its state repeatedly. Unlike storing fixed quantized weights, each write changes the input to later transitions. A small state mean-square error need not preserve a symbolic decision, and a correct final token can conceal an earlier failure. This project therefore measures **the first incorrect symbolic read**, the continuous correct prefix before it, and the actual bytes needed to store state between calls.

The task draws arbitrary elements from the six-element permutation group S3. An independent integer multiplication table computes the left cumulative product, `g_t = x_t … x_1`. The original model's MLP must identify that product after every token. The official final-quarter scaled accuracy and this project's all-prefix survival answer different questions.

<a id="questions"></a>
## 2. Hypotheses and verification questions

Stage A asks whether transporting a candidate's own write-back residual can extend readable lifetime under the same total byte cap. Full residuals are mechanism controls; low-rank residuals spend fewer bytes but discard directions. Uniform and CAL-based mixed precision provide stronger comparisons than a fixed low-bit state alone.

Stage B follows observations from Stage A. Its software question is whether finite, terminal and stochastic streams retain their execution meaning after serialization and fresh-process restart. Its fresh-input comparison is Rank2 versus a CAL-selected Mixed5/6 allocation at the Rank2 N=128 cap. Fixed-coefficient precision and late-state traces are diagnostics. The two stages were not one experiment preregistered before any observations; the original and follow-up freezes remain separate.

<a id="theory"></a>
## 3. Theory and measurement definitions

For each head, the physical-coordinate convention is

```text
A_t = (I − beta_t k_t k_tᵀ) Diag(alpha_t)
B_t = beta_t k_t v_tᵀ
S_t = A_t S_(t−1) + B_t
attention read = q_tᵀ S_t / sqrt(16)
```

The original normalization, signed gates, output gating/projection and MLP readout are retained. The explicit physical-coordinate path avoids mixing the upstream cumulative sign gauge with a residual in another coordinate frame.

A stored candidate represents `Z + Rhat`:

```text
U_t = A_t (Z_(t−1) + Rhat_(t−1)) + B_t
Z_t = Decode_state(Encode_state(U_t))
E_t = U_t − Z_t
Rhat_t = Decode_residual(Encode_residual(E_t))
```

`E_t` is its own write-back residual; the candidate never receives gold, reference state or future error. A low-rank variant stores coefficients in a CAL-fixed per-head basis. Full transition transports the reconstructed residual before projection. The untransported ablation instead adds the old residual after the transition; it is a placement control, not a representation of every published feedback method.

Exact full-residual reconstruction recovers the affine update as an accounting identity. With finite storage and arithmetic, error also includes the candidate/reference arithmetic-roundoff difference. Under normalized keys, `|alpha|≤1` and `beta∈[0,2]`, an ideal nonexpansive-transition norm bound accumulates these errors. It does not certify the nonlinear MLP's labels. A finite-group ID and transition table are a separate exact capacity control, precluding a universal sequence-length-dependent storage lower bound for this task.

| Quantity | Definition |
| --- | --- |
| First error `tau` | First scored group-token position with a wrong label or INVALID prediction; later recovery does not erase it |
| Survival `S(t)` | Fraction with `tau > t` |
| RMST0 | `mean(min(tau−1, Tmax))`; no observed failure contributes `Tmax`; units are consecutive correct group tokens |
| Empirical `T0.05` | Largest evaluated position with observed first-failure fraction ≤0.05; v1 originally uses its seven-point grid, v2 also reports every-token empirical values |
| Supported grid lower horizon | Largest declared grid point whose one-sided exact binomial upper failure bound is ≤0.05, using Bonferroni `alpha/family` |
| Storage | `B_total(N)=B_shared+N*B_stream`; shared basis, codec config and actual token table are charged |

A censored sequence indicates only that failure was not observed by `Tmax=2048`. Distinct grid lower bounds do not themselves test a between-method horizon difference. Input-level paired intervals are calculated separately within each fixed checkpoint; tokens, arms and the three checkpoints are not pooled as independent inputs.

<a id="methods"></a>
## 4. Methods

The trained model is the pinned S3/a11_b02 Complex KDA: one layer, width 48, 12 heads with 16×16 state, no short convolution, and a 48→192→6 MLP decoder. Native coefficients and learned recurrence use FP32. Every method for a checkpoint uses identical weights, token coefficients, inputs and readout. Each token writes back its represented state before readout; this is a sequential CPU reference implementation.

A single BOS is write 0; the first scored group token is write 1. BOS label correctness is excluded from tau, while terminal failure during BOS persists into scored tokens. Ties choose the lowest label index. The entire output is checked for finiteness.

The current [failure-aware adapter](versions/v2/source/online_v2.py) distinguishes three events:

| Event | State action | Scoring action |
| --- | --- | --- |
| Finite but wrong label | Continue updating | Record first error, allow later token recovery |
| Nonfinite readout of finite state | Commit the finite state and continue | INVALID `−1` at that position |
| Numerical reconstruction/transition/storage failure | Mark that stream terminal; retain the last committed state/residual/RNG body | INVALID at every later position; cursor still advances |

A malformed checkpoint, wrong shape, programming error or process failure is an execution error, not an algorithmic terminal outcome. The row-level update commits only after all storage/read-after-write checks pass. Terminal rows consume no more RNG.

The v2 header stores a little-endian cursor (8 B), terminal code (1 B), and first-terminal write (8 B): 17 B versus v1's 8 B cursor. Shared schema identity is also charged. Gold, tau and output history belong to the separate [evaluation checkpoint](versions/v2/source/checkpoint.py), not model state. Its temporary-directory verification and same-filesystem rename provide atomic visibility, without asserting fsync crash durability.

### The five Stage B settings

| Display name | Original identifier | What is stored |
| --- | --- | --- |
| Native FP32 | `NATIVE_FP32` | Full FP32 state plus the common v2 failure envelope |
| INT8 | `UNIFORM_8` | All state channels at 8 bits, with stored scales |
| INT5 | `UNIFORM_5` | All state channels at 5 bits, with stored scales |
| Rank2 | `LOWRANK_4_8_R2` | INT4 state plus INT8 coefficients in a frozen rank2 residual basis |
| Mixed5/6 | `MIXED_5_6_BUDGET` | Start with 5-bit state; promote 29 CAL-ranked head/key channels to 6 bits under the Rank2 N=128 cap |

Native and INT8 are capacity/fidelity references; they do not fit the Rank2 cap. All five methods include the same failure-envelope contract and use every recurrent head.

<a id="experiments"></a>
## 5. Experiments and evidence scope

| Condition | Stage A / v1 | Stage B / v2 |
| --- | --- | --- |
| Model | S3/a11_b02; three final training checkpoints | The same three checkpoints; no new training |
| CAL | 64 × 32; seed 1001 | Reuse the same CAL statistics and bases |
| DEV | 128 × 128; seed 2001; training gate DEV32 | Retained DEV; separate diagnostic cohort below |
| Main TEST | 512 × 2048; seed 3001 | Fresh 1024 × 2048; seed 50101 |
| Methods | 26 original + 4 supplementary mixed allocations | Native, INT8, INT5, Rank2, Mixed5/6 |
| Confidence family | 26×7×3=546; joint 30×7×3=630 | 5×13×3=195 |
| Scored grid | 32, 64, 128, 256, 512, 1024, 2048 | 32, 48, 64, 80, 96, 112, 128, 160, 192, 256, 512, 1024, 2048 |
| Diagnostic inputs | Five separate toy geometry conditions | 32 × 2048; seed 40101; precision and traces |
| Timing inputs | Separate isolated CPU timing | 16 × 128; seed 2002; 3 repeats; 2 threads |
| Record status | Original exploratory and supplementary evidence | Failure-aware follow-up informed by v1 |

The three local checkpoints were originally trained for 20,000 updates each, batch 128, using the upstream Muon/AdamW policy and curriculum 4/6/8/16/32. The final scheduled update is the checkpoint selection rule; the seed 0 DEV32 gate and remaining training budget allowed seeds 1/2. No TEST-based checkpoint selection is used. Exact policy and source identity are in the [v1 protocol](versions/v1/configs/protocol.json), [checkpoint provenance](versions/v1/provenance/checkpoints.json) and [v2 protocol](versions/v2/protocol_v2.json).

Stage A's four supplementary allocations reuse its TEST and are `POST_HOC_BUDGET_AUDIT_SAME_TEST`. Stage B's narrower menu uses the already explored v1 results; rank and basis are unchanged. Mixed5/6 promotes 29 head/key channels using frozen CAL importance and actual serialized lengths, stopping before the next promotion exceeds the cap. The common 1,024-input evaluation batch is not the N=128 storage scenario.

Original execution receipts report model replay, precision and timing already performed. This integration runs only preservation checks, recorded-scalar recomputation and synthetic CPU contracts. Current integration results are recorded separately from [v2's historical validation receipt](versions/v2/provenance/validation_receipt.json). The independent reviewer documents/probe named in earlier requests were unavailable in the retained evidence; [the availability receipt](versions/v2/results/historical-reanalysis/review_artifact_availability.json) records that limitation. No unseen independent-review test count is claimed.

<a id="results"></a>
## 6. Results

### A. Stage A: exploration and a stronger baseline menu

The original 26-arm/546-family comparison has 2/54 residual-budget comparisons with a larger supported horizon. Adding four feasible mixed allocations and recomputing the joint 30-arm/630-family view gives 0/54. This count describes the declared candidate caps and tested comparator menu; it is not a proof that every codec fails or that improvement is impossible. Rank2's historical mean-length signal and the native model's finite lifetime remain in the [original report](versions/v1/REPORT.md) and [historical recalculation](versions/v2/results/historical-reanalysis/HISTORICAL_ANALYSIS.ko.md).

Native's historical all-token empirical T0.05 values are 139/108/144; its original seven-grid values are 128/64/128. The simultaneous supported grid value is 64 for all three checkpoints. These v1 bounds cannot be subtracted from the v2 bounds to claim a memory improvement: inputs, sample size, grid and family changed.

<a id="restart"></a>
### B. Failure and restart

| Evidence | v1 behavior | v2 behavior |
| --- | --- | --- |
| Request-recreated synthetic fault probe | Function-local failure could be lost when bytes were reopened | Explicit terminal metadata prevents revival |
| 19 retained historical cells | Uninterrupted outputs define the comparison; 3 legacy U2 split runs revived after write 1024 | All 19 full/split/fresh-process prediction and final-hash comparisons match |
| Four previously observed numerical failures | Recorded writes 838, 1741, 623, 867; selected after observation | Each replayed at the same write; immediate post-failure child continuations preserve terminal body/RNG and cursor |
| Finite and stochastic contract fixtures | Original bodies/RNG form the reference | Healthy rows, corrupted headers, chunk boundaries and fresh-process continuation have dedicated tests |

[Historical summary](versions/v2/results/historical-summary/summary.json) · [Actual failure-boundary receipts](versions/v2/results/historical-failure-boundaries/index.json) · [Synthetic probe](versions/v2/results/v1_failure_probe_from_request.json) · [Current contract tests](versions/v2/tests/test_online_v2.py).

These historical actual-model checks are retained evidence, not rerun inference in this integration. The synthetic probe was reconstructed from the described defect, not the unavailable reviewer's exact script. Private final state bodies are not included; public checks compare recorded hash relationships and compact predictions. A terminal v2 body deliberately differs from the v1 placeholder.

<a id="native-int8"></a>
### C. Fresh inputs: native and INT8

Native serializes 12,305 B per stream; INT8 serializes 3,137 B, a 74.5% reduction. Shared costs remain separate below. RMST0 is the mean number of consecutive correct group tokens before the first error, not total token accuracy.

| Seed | Native RMST0 | INT8 RMST0 | INT8 − Native (tokens) |
| --- | --- | --- | --- |
| 0 | 288.57 | 288.12 | -0.45 |
| 1 | 271.15 | 271.57 | +0.42 |
| 2 | 238.28 | 237.49 | -0.79 |

INT8−native pointwise 95% paired intervals are [−2.64,1.85], [−2.14,3.01] and [−2.00,0.38] tokens. The small point differences are observations on these checkpoints, not a predefined equivalence/non-inferiority result. [Original paired table](versions/v2/results/fresh-summary/paired_table.csv).

<a id="fresh"></a>
### D. Fresh inputs: five settings and equal-budget comparison

| Seed | Setting / 설정 | RMST0 (tokens) | Empirical T0.05 | Supported grid T0.05 ≥ | Terminal / 1024 |
| --- | --- | --- | --- | --- | --- |
| 0 | Native FP32 | 288.57 | 135 | 112 | 0 |
| 0 | INT8 | 288.12 | 136 | 112 | 0 |
| 0 | INT5 | 242.85 | 123 | 96 | 0 |
| 0 | Rank2 | 277.47 | 129 | 112 | 0 |
| 0 | Mixed5/6 | 268.20 | 132 | 112 | 0 |
| 1 | Native FP32 | 271.15 | 118 | 96 | 0 |
| 1 | INT8 | 271.57 | 118 | 96 | 0 |
| 1 | INT5 | 220.35 | 96 | 80 | 0 |
| 1 | Rank2 | 219.97 | 112 | 96 | 0 |
| 1 | Mixed5/6 | 240.73 | 110 | 96 | 0 |
| 2 | Native FP32 | 238.28 | 142 | 128 | 0 |
| 2 | INT8 | 237.49 | 141 | 128 | 0 |
| 2 | INT5 | 222.77 | 133 | 112 | 0 |
| 2 | Rank2 | 235.60 | 141 | 128 | 0 |
| 2 | Mixed5/6 | 230.69 | 139 | 112 | 0 |

Terminal count 0 means no numerical state termination; every stream still had a first symbolic readout error by 2048. The denominator remains 1,024 for each checkpoint/setting. The same inputs run through five settings are paired observations, not 5,120 independent sequences.

| Seed | Rank2 − Mixed RMST0 (tokens) | Pointwise 95% interval |
| --- | --- | --- |
| 0 | +9.27 | [+1.07, +17.35] |
| 1 | -20.76 | [-27.41, -14.20] |
| 2 | +4.91 | [+0.78, +8.94] |

The intervals use 5,000 input-paired bootstrap resamples within each checkpoint and are pointwise. Rank2's supported lower horizon is larger than Mixed5/6's in one of three checkpoints; that is a descriptive bound comparison. Uniform5 remains a secondary reference, not a substitute for the stronger budget-matched Mixed5/6 baseline. [Original fresh table](versions/v2/results/fresh-summary/fresh_table.csv) · [Derived integration summary](summary/project.json).

![Fresh v2 survival for each checkpoint; no v1/v2 before-after pooling](versions/v2/figures/fresh_survival.png)

### E. Actual storage and CPU cost

| Setting / 설정 | Stream bytes | Shared bytes | Total N=1 | Total N=16 | Total N=128 |
| --- | --- | --- | --- | --- | --- |
| Native FP32 | 12305 | 30344 | 42649 | 227224 | 1605384 |
| INT8 | 3137 | 30560 | 33697 | 80752 | 432096 |
| INT5 | 1985 | 30560 | 32545 | 62320 | 284640 |
| Rank2 | 2033 | 32409 | 34442 | 64937 | 292633 |
| Mixed5/6 | 2043 | 30965 | 33008 | 63653 | 292469 |

The cap comparison is Rank2 versus Mixed5/6 at N=128: 292,633 versus 292,469 B. The shared-cost trade-off means this cap relation does not hold automatically at N=1,024. All sizes include the failure envelope, actual code/scale/RNG buffers, basis, mixed map, codec configuration and canonical token table. V2 adds 9 B per stream and 934 shared B to the corresponding old envelope. The unchanged common model has 226,120 parameter-value bytes and a 233,203 B checkpoint file; neither is part of the candidate-specific cache reduction. Process peak RAM and GPU VRAM were not measured. [Buffer-level ledger check](versions/v2/results/ledger_validation.json).

The following values are median complete-call **ms per group token**, CPU 2 threads, 16 sequences × 128 group tokens, 3 fixed repeats. BOS work is inside the call; the rate denominator is 16×128. Transition, residual projection, packing, status handling and original readout are timed. Loading, disk I/O and gold/shadow diagnostics are outside the timer.

| Seed | Native FP32 | INT8 | INT5 | Rank2 | Mixed5/6 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0306 | 0.0939 | 0.1209 | 0.1438 | 0.1185 |
| 1 | 0.0288 | 0.0975 | 0.1206 | 0.1491 | 0.1160 |
| 2 | 0.0306 | 0.0946 | 0.1090 | 0.1419 | 0.1119 |

These prototype codec calls are slower than native. They are not GPU speed measurements or single-request latency. Timing ran serially after study workers completed; system-wide background load was not fully controlled. [Original timing records](versions/v2/results/timing/).

### F. Fixed-coefficient precision and late instability

| Diagnostic | Retained observation | Interpretation boundary |
| --- | --- | --- |
| D00 FP32 recurrence / FP32 readout | RMST0: 290.38 / 279.69 / 236.75 on 32 diagnostic inputs | Different cohort from fresh 1,024 |
| D10 FP64 recurrence / cast→FP32 readout | Same complete predictions and tau as D00 in each seed | Retained FP32 coefficients promoted, not regenerated |
| D01 FP32 recurrence / FP64 readout; D11 FP64 / FP64 | Same complete predictions and tau as D00 | Fixed represented weights; explicit dtype/parity checks |
| Rank2 late state norm | Maxima ≈ 1.2832e9 / 1.5151e9 / 4.4855e16, each after first readout error | Continued rollout and error order, not identified cause |

[Precision pairs](versions/v2/results/diagnostic-summary/precision_pairs.csv) · [Dtype/readback checks](versions/v2/provenance/diagnostic_readback.json) · [Trace extrema](versions/v2/results/diagnostic-summary/trace_extrema.csv).

The same 32 diagnostic inputs supply per-time norm, self-writeback error, projection leakage and readout margins. At t=512, all 32 are already after first error for each Rank2 checkpoint. Pre/post curves use changing conditional subsets; valid counts accompany the statistics. Terminal scratch zeros are excluded, and gold/native/FP64 shadows are evaluator-only. The arithmetic promotion does not establish exact ground truth or isolate every cause of short native lifetime.

<a id="analysis"></a>
## 7. Analysis

Packing and restart solve an execution problem that a state MSE plot cannot detect: a stream's failure history must survive the same boundary as its numeric payload. The new terminal envelope achieves that contract; it does not itself lengthen symbolic memory.

Rank2 has favorable mean differences in two checkpoints and an unfavorable difference in the third under the stronger N=128 comparator. Mean length and a 5%-risk tail criterion can move differently. Native already loses all-prefix correctness at finite lengths, so candidate errors cannot all be assigned to quantization. Fixed-FP32-coefficient arithmetic promotion leaves the observed diagnostic labels unchanged, while large late norms remain in continued rollout. These observations do not prove model undertraining, representational impossibility or a universal precision requirement.

The strongest readable conclusion is therefore specific: the tools support reproducible storage/lifetime comparisons; INT8 substantially reduces serialized state with near-native mean point estimates; the tested rank2 correction has no consistent equal-budget advantage across these three checkpoints. The original negative cases and supplemental baselines remain accessible beside the favorable means.

<a id="conclusion"></a>
## 8. Conclusion

- Implemented real low-bit state storage, causal residual transport and failure-/cursor-/RNG-preserving fresh-process restart.
- Completed the original fresh-input comparison on the same three checkpoints; this integration recomputes retained evidence without new inference.
- Rank2 did not establish a consistent equal-budget memory-horizon advantage over Mixed5/6.
- INT8 used 74.5% fewer serialized per-stream bytes than native, with less than one token difference in mean-lifetime point estimates for each checkpoint.

[Run the CPU audits](REPRODUCTION.md) · [Read the current adapter](versions/v2/source/online_v2.py) · [Track original versions](VERSION_MAP.md).

<a id="references"></a>
## 9. References and attribution

The bibliography is inherited from the retained [v1 NOTICE](versions/v1/NOTICE.md) and [v2 NOTICE](versions/v2/NOTICE.md). This integration adds no new literature results or claims of reproduction. Complex KDA supplies the model/task/native recurrence; recurrent-state quantization, corrected-state feedback and state-error analysis have prior work. DIOVA contributes the adapters, byte contracts, comparison protocols and audit/display tools. Codex assisted implementation and documentation. [Unified source and license notice](NOTICE.md).
