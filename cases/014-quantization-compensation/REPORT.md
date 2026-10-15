# Case 014 — Improving Model Quality at the Same INT4 Storage Budget

Quantization Compensation: Cancellation, Re-quantization, and Generalization

[한국어](REPORT.ko.md) · [Stage map](STAGE_MAP.md) · [CPU/GPU reproduction](REPRODUCE.md)

Implemented a path-aware compensation pipeline that refits projection weights and stores them in the same INT4 format. On Qwen3-0.6B, fixed quarter-strength compensation reduced perplexity by 12.92% versus RTN on 256-token excerpts from 64 fresh PG19 books, with unchanged projection storage.

The gain did not transfer to the ArXiv evaluation. Quality was measured after dense BF16 reload; serving speed and VRAM savings were not measured.

[1. Background](#background) · [2. Questions and hypotheses](#questions) · [3. Theory and implementation contract](#contract) · [4. Methods](#methods) · [5. Experiments](#experiments) · [6. Results](#results) · [7. Analysis](#analysis) · [8. Conclusion](#conclusion) · [9. Reproduction and sources](#references)

<a id="background"></a>
## 1. Background

The study asks when correcting inherited input error improves actual outputs. Existing cancellation, re-quantization and new documents can change the benefit. P1 motivated P1b independent selection; those observations motivated P2 fixed weak full-model comparisons. This sequence was not one experiment preregistered from the outset.

<a id="questions"></a>
## 2. Questions and hypotheses

Q1: can inherited-error correction destroy useful cancellation? Q2: does recalculating stored scales/codes reverse a benefit? Q3: do fitted corrections and selected strengths transfer to unused documents? P2 separately registers whole-model QUARTER minus RTN on fresh PG19 gold NLL as primary. HALF is auxiliary and does not replace the primary after observing results.

<a id="contract"></a>
## 3. Theory and implementation contract

```text
W[out,in], X/Z[in,tokens], Wstar=W+alpha*K
A=W(X-Z); C=alpha*KZ
B=(What-Wstar)Z; B0=(What0-W)Z
R=A-C-B; R0=A-B0
Delta = ||R||²-||R0||²
T_inherited = ||C||²-2<A,C>
T_cancellation = 2<B0,C>
T_requant = ||B-B0||²-2<R0-C,B-B0>
Delta = T_inherited+T_cancellation+T_requant
```

CAL fitting uses the full-input Gram G=ZZᵀ and λ=.001 trace(G)/d_in with CPU FP64 Cholesky/triangular solves. No explicit inverse or block-diagonal approximation is used. Ridge H is generally not a projector. CAL K stays fixed on evaluation; AP_eval is not refitted.

The fixed-B counterfactual A−C−B0 differs from actually executing a high-precision Wstar control. Algebra uses saved FP32 What lifted to FP64; BF16 module errors, weight casts and arithmetic residuals are separate. P2 B0 is local RTN on each method’s current Z, not the separate whole-RTN model error.

<a id="methods"></a>
## 4. Methods

The common encoder is signed INT4[−8,7], input group64, per-output-row/group FP32 absmax/7, nearest-even, unit scale only for true-zero groups, low-nibble-first, header36B. Pack→disk→independent decode→BF16 installation is checked. K is not a runtime GEMM. Common embedding/lm_head ties, norms, attention and RoPE are preserved.

P2 captures actual inputs with the native entire CausalLM forward after each group: q/k/v → actual attention → o → residual/norm → gate/up → SiLU product → down. Only shared-input groups share Gram factors. QUARTER/HALF independently start from original weights and create their own Z/K.

<a id="experiments"></a>
## 5. Experiments

Stage documents, choices and originals are mapped in [STAGE_MAP](STAGE_MAP.md). P1 uses CAL32/DEV16/EVAL32/SHIFT16 with 6 targets×5 strengths. P1b keeps6 K matrices and 30 artifacts, using new SELECT32/EVAL64/SHIFT16. P2 reuses original CAL32 but evaluates separate fresh PG19 64 books and ArXiv16 content groups with all 196 projections. Known prior inputs are excluded by manifest union; Pile has content-group identity without article IDs. Pretraining non-exposure is not established.

All LM metrics use256 input tokens without chat/BOS/EOS and 255 original next-token targets. KL uses16 fixed full-vocabulary BF16∥candidate positions; argmax flips use255. P2 uses BF16/batch1/SDPA/use_cache=False/full256 head/TF32 and reduced reduction OFF. Every artifact was frozen at zero fresh forwards, followed by evaluation.

<a id="results"></a>
## 6. Results

### P2 — Whole-model results

PG19 ·64 book excerpts ·16,320 valid next tokens/method

| Method | NLL ↓ (nats/token) | PPL ↓ | KL(BF16∥method) ↓ | Argmax flip |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 3.815629 | 45.405313 | 0.000000 | 0.000% |
| RTN | 4.053521 | 57.599922 | 0.438809 | 31.740% |
| COMP_QUARTER | 3.915147 | 50.156465 | 0.290688 | 25.447% |
| COMP_HALF | 3.911792 | 49.988432 | 0.236665 | 24.179% |

ArXiv ·16 content groups ·4,080 valid next tokens/method

| Method | NLL ↓ (nats/token) | PPL ↓ | KL(BF16∥method) ↓ | Argmax flip |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 2.348638 | 10.471299 | 0.000000 | 0.000% |
| RTN | 2.521869 | 12.451842 | 0.320093 | 22.549% |
| COMP_QUARTER | 2.540615 | 12.687474 | 0.310653 | 23.162% |
| COMP_HALF | 2.633587 | 13.923621 | 0.318018 | 24.975% |

| Cohort | Candidate−RTN | ΔNLL (nats/token) | 95% paired CI | PPL change |
| --- | --- | ---: | --- | ---: |
| EVAL-ID | COMP_QUARTER | -0.138374 | [-0.149414, -0.127515] | -12.923% |
| EVAL-ID | COMP_HALF | -0.141730 | [-0.156537, -0.126159] | -13.214% |
| SHIFT | COMP_QUARTER | +0.018747 | [-0.005723, +0.044885] | +1.892% |
| SHIFT | COMP_HALF | +0.111718 | [+0.072058, +0.153853] | +11.820% |

The12.92% primary PPL reduction is relative to RTN, not accuracy or speed.64/64 means lower NLL on each fixed book excerpt. QUARTER recovers58.17% of RTN’s extra NLL gap above BF16 on PG19; it does not beat BF16.

The original seed6101523 and 10,000 paired document bootstrap resamples are recomputed. Intervals condition on this model/CAL32/fixed strengths. HALF/SHIFT are auxiliary pointwise. Publication post-hoc HALF−QUARTER point estimates are−.003356 on PG19 and+.092972 nats/token on ArXiv. Reviewer code/seed were unavailable, so unsupported historical CIs were not attached.

![PG19](figures/p2_r1/01_pg19.svg)

![ArXiv](figures/p2_r1/02_arxiv.svg)

### P0→P1→P1b observations

P0 rational fixed-error examples,2bit4-code examples and infinite-lattice dither are distinct models. Dither is a ridge0/no-clipping positive control with 2,000 stored seeds per condition. Publication recalculates saved SSE/SE; Monte Carlo is not rerun.

P1 CAL-selected strengths increased EVAL local SSE in 6/6 targets; post-hoc quarter strength reduced it in 6/6. The following sums are local squared energies over 32 documents for that target. Continuous transfer loss is the large positive term; cancellation is not established as the main cause.

| Target | CAL alpha | EVAL selected ΔSSE | EVAL alpha0.25 ΔSSE | T_inherited | T_cancellation | T_requant |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 12.mlp.down_proj | 1 | +36.68% | -7.78% | 1.15e+05 | -1.25e+03 | 9.48e+03 |
| 12.self_attn.o_proj | 1 | +43.26% | -6.86% | 7.17e+04 | 2.31e+02 | 7.79e+03 |
| 20.mlp.down_proj | 0.75 | +14.29% | -6.45% | 4.08e+05 | -7.39e+02 | 1.96e+05 |
| 20.self_attn.o_proj | 1 | +5.11% | -16.07% | 9.29e+04 | -1.23e+04 | 1.72e+04 |
| 4.mlp.down_proj | 1 | +45.40% | -5.97% | 2.92e+04 | 5.51e+01 | 2.95e+03 |
| 4.self_attn.o_proj | 1 | +10.98% | -10.59% | 2.91e+03 | -5.52e+01 | 2.71e+02 |

P1b separates selection documents while retaining K and artifacts. These are mean ratios over six independent interventions, not one whole-model result.

| Target | OLD_FIT | SELECT_SSE | SELECT_NLL |
| --- | ---: | ---: | ---: |
| 12.mlp.down_proj | 1 | 0.25 | 0 |
| 12.self_attn.o_proj | 1 | 0.25 | 0.75 |
| 20.mlp.down_proj | 0.75 | 0.25 | 0.5 |
| 20.self_attn.o_proj | 1 | 0.5 | 1 |
| 4.mlp.down_proj | 1 | 0.25 | 0.25 |
| 4.self_attn.o_proj | 1 | 0.5 | 1 |

| SELECT_SSE versus | ΔSSE / alpha0 SSE, target mean | ΔNLL, target mean | NLL95% pointwise CI |
| --- | ---: | ---: | --- |
| OLD_FIT | -36.078 baseline-SSE pp | -0.003735 | [-0.004858, -0.002605] |
| BASE_ZERO | -10.172 baseline-SSE pp | -0.001055 | [-0.001743, -0.000416] |
| FIXED_QUARTER | -1.122 baseline-SSE pp | -0.000275 | [-0.000628, +0.000077] |

Extra local SSE benefit over fixed quarter is concentrated in block4/20 o_proj selecting.5; the other four targets alias the same quarter artifact. The additional NLL CI crosses 0. Both selectors worsen mean SSE/NLL against alpha0 on ArXiv.

<a id="analysis"></a>
## 7. Analysis

Reducing local SSE and lowering gold next-token NLL are different objectives. Six P1b effects were not summed to predict P2. P2 directly observes PG19 gains in actual full models fitted on their own trajectories, alongside ArXiv losses/uncertainty.

Algebraic sign, measured BF16 sign and conservative-bound certification remain distinct. P1 has measured agreement in all 2,304 saved comparisons and 2,195 bound certifications. P1b fresh EVAL has 1,535/1,536 measured agreement and 1,396 certifications. Lack of certification does not mean unmeasured or reversed direction. Post-hoc selection sensitivity is not new training or a guarantee.

<a id="conclusion"></a>
## 8. Conclusion

Fixed weak own-trajectory compensation works at the same INT4 format in this PG19 whole-model comparison. The same gain is not established on ArXiv, so domain limits accompany the result. P2 completed without extra selection or experiments to seek a favorable outcome.

| Axis | Scope |
| --- | --- |
| Capture/fit/serialization/reload |196 projections/model,588 historical files,392 K fits; implementation completed |
| PG19 whole-model quality |Registered QUARTER−RTN LOWER_NLL |
| ArXiv |QUARTER higher point estimate/CI crossing0; HALF higher NLL |
| Native packed serving / VRAM savings / agent / other model |NOT_EVALUATED |
| Publication |New GPU inference/K fits/requantization0; CPU scalar/interval/unit checks only |


<a id="references"></a>
## 9. Reproduction and sources

Six original top-level documents were preserved byte-for-byte before replacement. Of579 research files,578 are byte-identical in retained public locations; one environment receipt uses the earlier review ZIP’s sanitized model-path field. The private original is unchanged. See the [mapping](publication/provenance/ORIGINAL_MAP.json) and [stage restorer](publication/scripts/restore_stage.py).

[CPU/GPU reproduction](REPRODUCE.md) · [Stage map](STAGE_MAP.md) · [Publication checks](publication/verification/cpu-recalculation.json) · [Raw-derived display](publication/summary/project.json)

Distinguish Guo’s [v1 paper](https://arxiv.org/html/2610.16020v1), the [historical source audit](references/source_audit.json) and [official code identity](https://github.com/tellsgo/quantization-compensation/tree/1e5dea93dfd9466ce0569fc8dde55ef8c2fcb9e0). Section checks are historical HTML checks; PDF pages were not separately rendered. HALF and new-document selection transfer are prior author results, not claimed discoveries here. Project contributions are capture, fit orchestration, codec/reload, decomposition, evaluation and audit tools. Qwen/PyTorch/Transformers/NumPy and compensation ideas are upstream/prior work. Codex assistance and separate CPU calculation paths are disclosed, without claims of independent human peer review, external GPU reproduction or official author approval.


Storage ledger: RTN=QUARTER=HALF247,733,136B; headers7,056 + scales27,525,120 + codes220,200,960B. BF16 projection tensors880,803,840B; unchanged common tensors311,296,000B (ties counted once). Core reduction71.874%.

Offline construction: RTN9.898s / QUARTER546.999s / HALF515.160s. Includes capture/FP64 solve/diagnostics/I/O, not serving time.
