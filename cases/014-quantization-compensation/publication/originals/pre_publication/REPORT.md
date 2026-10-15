# Case014 — Quantization Compensation: Cancellation, Re-quantization, and Generalization

[한국어](REPORT.ko.md)

1. [Background](#s1)
2. [Questions](#s2)
3. [Theory and implementation contract](#s3)
4. [Method](#s4)
5. [Experiments](#s5)
6. [Results](#s6)
7. [Analysis](#s7)
8. [Conclusion](#s8)
9. [Reproduction and references](#s9)

<a id="s1"></a>
## Background

Quantization stores model numbers using fewer bits. Compensation adjusts later weights to reduce error inherited from earlier layers. This tool checks what survives after those adjusted weights are stored as INT4 again, and whether the benefit transfers to other documents.

Executed correction fitting, INT4 save/reload, error decomposition and common-input full-model comparison. CAL-selected strengths reduced local error on CAL, but increased aggregate local error on new EVAL documents at all six targets. Next-token scores moved differently across targets.

<a id="s2"></a>
## Questions

Assessed Q1 cancellation, Q2 actual re-quantization and Q3 CAL-to-new-document transfer separately. Scalar counterexamples/dither remain historical P0 evidence, separate from these LLM observations.

<a id="s3"></a>
## Theory and implementation contract

W[out,in], X/Z[in,tokens]. CAL only: G=ZZᵀ, λ=0.001 trace(G)/d; (G+λI)Kᵀ=[W(X−Z)Zᵀ]ᵀ; δW=αK; W*=W+δW.

A=W(X−Z), C=δWZ, B=(What−W*)Z, B₀=(What₀−W)Z, R=A−C−B, R₀=A−B₀.

Δ=||R||²−||R₀||²=T_inherited+T_cancellation+T_requant.

T_inherited=||C||²−2⟨A,C⟩; T_cancellation=2⟨B₀,C⟩; T_cont=||C||²−2⟨R₀,C⟩; T_requant=||B−B₀||²−2⟨R₀−C,B−B₀⟩.

FP32(W*) was re-quantized to signed INT4[-8,7], input group64, output-row FP32 absmax/7 scales and nearest-even rounding. The format uses low-nibble-first packing, a36-byte header and unit scales only for zero groups. Algebra uses the actual reloaded FP32 What; model execution uses BF16(What). Quantization, fitting and scoring remained unchanged from preparation.

This is packed INT4 storage plus dense BF16 quality reference execution. K is not an extra runtime operation. Continuous controls use FP32 weights/inputs and BF16 output casts; they are separate diagnostics, not upper bounds.

<a id="s4"></a>
## Method

Captured teacher projection inputs X and hybrid-prefix inputs Z using actual full-model forwards. Targets are o_proj/down_proj in blocks4/12/20; strengths are0/0.25/0.5/0.75/1. Each target is intervened on separately, preserving its other block operations and suffix.

CAL fitting used CPU FP64 full-input Gram, Cholesky, triangular solves and row chunks128. All six K/lambda values,30 artifacts and selected strengths froze before non-CAL forwards. DEV performed no reselection; EVAL-best strengths remain post-hoc oracles.

Each256-token document contributes255 next-token NLL/flip positions and16 fixed full-vocabulary KL positions. No added chat/BOS/EOS; batch1, use_cache=False, SDPA, TF32 and BF16 reduced reduction disabled, with the same full256-position LM head.

<a id="s5"></a>
## Experiments

Distinct PG19 book URLs define CAL32/DEV16/EVAL32;16 Pile ArXiv content groups define SHIFT. One separate smoke input is excluded from evaluation denominators. Document/token hashes, offsets, masks and positions were retained. Original Pile article IDs are unavailable; article-disjointness, prior use beyond the bounded audit and pretraining exposure remain unverified.

Thirty core settings ×96 documents produce2,880 rows, not2,880 independent documents. EVAL has32 books; SHIFT has16 content groups. The3,845 forwards include teacher, prefix, smoke and continuous controls.

Executed on RTX5080 SM120, Torch2.13.0+cu130 and Transformers5.17.0. Conservative worker wall: **358.877s**; inner runner interval344.235s. GPU busy time, TTFT and TPOT were not measured.

Actual core files: **1,179,684B** for o_proj and **1,769,508B** for down_proj, identical across five strengths at each target. Torch peak allocated:1,646,827,520B; peak reserved:1,845,493,760B. Maximum post-forward CUDA used-memory sample:3,331,915,776B, not a continuously monitored Windows peak.

<a id="s6"></a>
## Results

Deltas compare against alpha0 at the same target. Each baseline is a hybrid with RTN preceding blocks and target, not whole-model RTN or the historical Case013 baseline. SSE is aggregate local error; NLL is nats/token. Negative changes improve the respective metric.

CAL-selected EVAL aggregate SSE increased at6/6 targets. EVAL NLL decreased at the three o_proj targets and increased at the three down_proj targets. SHIFT NLL point estimates increased at6/6; some intervals include zero. Intervals are exploratory pointwise95% document-paired bootstrap intervals,10,000 resamples, seed6101516.

|Block / projection|CAL α|CAL ΔSSE %|EVAL ΔSSE %|EVAL ΔNLL [95% CI]|SHIFT ΔNLL [95% CI]|
|---|---:|---:|---:|---|---|
|4 / down_proj|1|-53.34|+45.40|+0.017570 [+0.009267, +0.025416]|+0.047380 [+0.028125, +0.070095]|
|4 / o_proj|1|-52.06|+10.98|-0.006263 [-0.012331, -0.000537]|+0.010130 [+0.002822, +0.018001]|
|12 / down_proj|1|-55.72|+36.68|+0.011532 [+0.006298, +0.017117]|+0.016129 [+0.005145, +0.027515]|
|12 / o_proj|1|-55.25|+43.26|-0.004538 [-0.009533, +0.000430]|+0.005138 [-0.001055, +0.011251]|
|20 / down_proj|0.75|-45.33|+14.29|+0.003528 [-0.000984, +0.008394]|+0.056429 [+0.043199, +0.071002]|
|20 / o_proj|1|-66.15|+5.11|-0.007653 [-0.011391, -0.003748]|+0.023418 [+0.014054, +0.033364]|

![All six target curves](figures/p1_r2/fig2_components.svg)

[All120 target/strength/split summaries](results/p1_r2/summary.csv) · [Raw2880 rows](results/p1_r2/cells)

![CAL/EVAL ranks](figures/p1_r2/fig3_transfer.svg)

<a id="s7"></a>
## Analysis

A denotes inherited-error improvement overturned by fixed-B cancellation loss; B is a re-quantization reversal; C is frozen continuous-correction transfer loss; D combines local SSE improvement with worse NLL. Labels overlap. Denominators count document × target × nonzero-strength comparisons.

The single EVAL A pattern is block4/down_proj/alpha0.5, document hash `4f835d09…be32`: T_inherited=−3.673586, T_cancellation=+4.965093, T_cont=+1.291507, T_requant=+4.018703, Δ=+5.310211. Actual BF16 module ΔSSE was+5.491078. The conservative arithmetic bound67.454096 exceeds Δ, so its runtime sign guarantee remains unresolved.

A supplementary post-hoc diagnostic applies |SSE_run−SSE|≤2||R||||e||+||e||² to both arms. Of2,304 nonzero-strength rows,2,195 exceed the Δ-sign bound;109 remain unresolved. This does not relabel the original FP64 patterns or establish a unique cause. Only1 of the12 B rows exceeds that Δ-sign bound.

|Split|Comparisons (nonzero α)|A|B|C|D|
|---|---:|---:|---:|---:|---:|
|CAL-FIT|768|0|0|0|205|
|DEV-SELECT|384|0|8|169|81|
|EVAL-ID|768|1|4|339|172|
|SHIFT|384|0|0|367|14|

![All target local error versus downstream NLL](figures/p1_r2/fig4_local_nll.svg)

<a id="s8"></a>
## Conclusion

In this fixed setup, strengths chosen using CAL local error did not preserve the EVAL local-error ordering. The result connects compensation, actual re-quantization, new inputs and downstream scores through saved artifacts and full forwards. It does not imply every correction is harmful or estimate prevalence on other models. Execution completion is separate from quality improvement.

<a id="s9"></a>
## Reproduction and references

The existing15 unique P0 tests passed again; they are not15 new tests. P0 MC was not rerun; its lambda was0. The independent stdlib auditor checked2,880 scalar rows, selection, pairing and denominators. A fresh CPU process independently decoded30 artifacts;8 private-array samples were checked with Torch CPU FP64. The review without private weights/activations/logits does not rerun GPU NLL/KL computation.

User authorization covers read-only reuse of the existing WSL snapshot. Full local weight SHA256 matches the existing cache blob identity; this is distinct from newly obtaining an external official digest. The original snapshot, P0/history and environment are preserved. P2/P3 and remote publication were not run.

Execution source SHA256: `da2a0368bb4a284a0af1a5896121ca0e828a59d9392f49b954092664c1cab960`. Base Git SHA: `09194c3f2e2c991feff5b6abab0cf8ab81a07052`. These identify different objects.

[Reproduce](REPRODUCE.md) · [Independent audit](results/p1_r2/independent_audit.json) · [Source audit](references/source_audit.json) · [Original P0 report](publication/original_pre_p1/REPORT.md).

[Guo2026 v1](https://arxiv.org/html/2610.16020v1) motivates this independent ridge/RTN diagnostic. This is not official full QEP reproduction or a new quantizer. Codex assisted implementation and reporting; no independent human review is claimed.

