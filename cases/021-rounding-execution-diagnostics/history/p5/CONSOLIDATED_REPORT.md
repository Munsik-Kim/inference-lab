# Case021 together — Representation, GPU execution and downstream sensitivity

**Single terms agreed; combinations of actual values differed.** This P5 finding covers two q_proj projections and fixed masks of two existing DEV documents. Reconstructing weights, executing them on the GPU and propagating their differences are separate questions.

| Question | Comparison and evidence | Interpretation |
|---|---|---|
| What changes with representation? | D0 `(codes×gamma)×alpha_inv` versus D1's BF16 combined scale and BF16 weight | Multiplication reassociation and scale/weight rounding boundaries both change |
| What remains after matching representation? | D1 versus actual Marlin P; P4 actual-input differences and P5 agreement for single terms | A local execution difference is supported; accumulation and scale handling are not isolated |
| Does higher intermediate precision improve agreement? | Common-input R0/R1 norm–RoPE with native cos/sin, checked against a local FP64 expression | Individual reference accuracy differs from agreement between execution paths |

D0 is a dense reference reconstructed from the D16 artifact, not the original uncompressed checkpoint. D1/P share codes, BF16 scale and a single input beta application. R1 retains Q/K norm–RoPE intermediates in FP32 before the final BF16 Q/K; it does not convert the whole model to FP32. Since R0 was already common, this is not a reproduction of RL-stack alignment.

The original24 EVAL documents at length512 and16 full-vocabulary positions gave E0=0.0006952504144, E1=0.0007815328033, delta=-0.0000862823889. The paired-document95% interval [-0.0002068221560, 0.0000390900519] includes zero: **INCONCLUSIVE** remains. P5 neither generated new full-model outputs nor turned the old EVAL into a new holdout. The historical32-token D0–P discrepancy of about0.02605433 remains outside the original0.02 gate.

In P4, both individual FP64-reference errors summed to a smaller value for all96 EVAL local combinations, while the pair gap increased34 times, decreased57 and stayed equal5. P5A separates the34 increases into13/17/4 sign classes for ΔN/ΔC. This explains local cancellation patterns; it does not allocate causal shares of full-model KL.

P5B matched candidate A in both implementations, including all406,626 coordinates where A/B differed. P5C agreed at K1/2 for all four combinations, with differences first observed at K8 or64. K1024 inputs/outputs matched P4. GPU diagnostics ended without adding K values, seeds, models or kernel changes.

![Fixed sparse replay](figures/03_sparse_replay.png)

[Detailed observations, counterexamples, failures and costs](P5_REPORT.md) · [Shared facts](facts.json) · [Reproduction scope](REPRODUCE.md) · [Sources/identity](SOURCE_MAP.md)

The case contributes fixtures, adapters, measurement, CPU audits and explanation, with Codex assistance. GEMM, Marlin, the allocator and the existing quantized artifact are reused. No model-capability, training-stability or serving-speed improvement is established. Public files support aggregate checks; weight-revealing raw evidence remains local.
