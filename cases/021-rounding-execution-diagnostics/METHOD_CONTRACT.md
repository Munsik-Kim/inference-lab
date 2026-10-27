# Method contract

D0 PRIMARY_DENSE: `blocks=codes.to(scale_dtype).reshape(O,I/64,64); pre=(blocks*gamma[...,None])*alpha_inv[...,None]; W0=pre.reshape(O,I).to(BF16)`. The multiplication order is retained.

D1 PACKED_FORMAT_DENSE: `s32=FP32(alpha_inv)*FP32(gamma); s16=BF16(s32); W1=BF16(codes.float()*s16.float()[...,None])`, with correct output/group/input axes. P uses signed-code→c+8 offset encoding, transpose[I,O], eight nibbles/int32, Marlin repack and group64 scale permutation. Actual projection bias is NONE. Every module applies `z=BF16(FP32(x)*beta_inv)` once. P5 direct cores receive z and do not apply beta again.

D0–D1 combines reassociation and format/rounding changes. D1–P includes dequantization/rounding/epilogue/reduction possibilities; it is not a pure reduction-order estimator. P is actual `torch.ops._C.marlin_gemm`, INT4 storage/BF16 MMA, FP32 reduce=true, atomic=false, TF32/reduced reductions OFF. Runtime reuse: Python3.12.14/Torch2.13.0+cu130/Transformers5.17.0/vLLM0.29.0, RTX5080 sm120, driver621.01. Not a clean-environment claim.

R0 is native HF Q/K RMSNorm–RoPE. R1 keeps that region's intermediates FP32, reuses native BF16 cos/sin values, and returns BF16 Q/K with matching layout/stride. Q/K norm weight, epsilon, positions, head mapping and rotate-half are retained. No whole-model FP32, changed V/attention/GEMM or recomputed frequencies.

Selected positions are floor((j+1)*N/17),j=0..15. Full lm_head runs at every input position first. A logit at position i predicts token i+1. KL(reference||candidate) and JSD use natural log/full vocabulary in the original saved-logit audit; public scalar checks reaggregate its recorded results. Positions average within a document; documents get equal weight. relative_RMSE=sqrt(SSE/reference_energy); zero energy is UNDEFINED with absolute errors.

P5 local identity: yP−yD=d+n; G=||d||²+N+C. New sign tolerance=max(1e-12,1e-10×max|G/N/C|); energy identity relative tolerance1e-10. Near-zero changes stay separate. Single-term A=R(a×R(c×s)), B=R(a×c×s), exact BF16 nearest-even; ±0 numerically equal, raw bits retained privately. Representable-step distance respects negative ordering. No overflow/subnormal generalization from normal power-of-two controls.

P5 sparse K budgets1/2/8/64/256/1024 share per-row nested permutations. Retained actual BF16 bits stay unchanged. K1024 equals saved P4 Z bytes. K/nonzero count/norm vary together. First-observed K is limited to this fixed grid. P5B's record-coordinate count includes across-fixture coefficient reuse and excludes process repeats.

| Term | Plain meaning |
|---|---|
| artifact | Saved codes/scales/metadata sufficient to reconstruct the defined execution |
| INT4 /D16 | Four-bit weight code storage /historical W4A16 setting name, not16-bit weights |
| dense /packed | Reconstructed weight matrix /codes and scales supplied to the kernel |
| BF16 /FP32 | Number formats; intermediate precision differs from final tensor dtype |
| activation | Values passed from one model calculation to another |
| norm /RoPE | Vector normalization /position-dependent Q/K rotation |
| KL /JSD | Distribution differences; smaller means closer, not more correct |
| ULP | Steps between representable floating-point values, not rounding-operation count |
| INCONCLUSIVE | This study did not establish the direction of mean KL reduction |

[Source evidence](SOURCE_MAP.md) · [Recorded P5 contract](history/p5/P5_PROTOCOL.json) · [Original contract snapshot](history/p0p3/METHOD_CONTRACT.md.txt)
