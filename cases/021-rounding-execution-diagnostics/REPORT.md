# Case021 integrated report — Representation, execution and propagation

[Overview](README.md) · [한국어](REPORT.ko.md) · [Reproduction](REPRODUCE.md)

## Question and observations

The case separates reconstruction-format changes from execution differences after matching codes/scales. The final P5 probes agreed for single terms but differed when actual values were combined. Smaller local reference errors did not consistently improve agreement between the final model distributions.

## 1. Contracts and inputs

The historical base is Qwen3-0.6B revision c1899de289a04d12100db370d81485cdf75e47ca, with196 group64 D16 projection artifacts. D16 is the existing Case013 W4A16 setting: INT4 weight codes and BF16 activations, not16-bit weights. Actual projection biases are absent. Input beta is applied once. D0 is reconstructed from the artifact, not the uncompressed checkpoint.

| Path | Meaning |
| --- | --- |
| D0 | Dense artifact reference reconstructed in the original arithmetic order |
| D1 | Preconstruct BF16 weights from codes and BF16 scales, then use dense GEMM |
| P | Supply the same packed codes/scales to the Marlin GPU operation |
| R0 /R1 | Native shared Q/K norm–RoPE rounding /FP32 intermediates until final BF16 Q/K |

D0–D1 changes the original `(codes×gamma)×alpha_inv` order and combined-scale/weight rounding. D1–P captures remaining execution differences after matching signed codes and BF16 scales. These are not isolated scale-cast or reduction-order effects. Actual Marlin symbols were profiled: INT4 storage/BF16 MMA, not native INT4 MMA or a new kernel.

R1 retains Q/K norm (normalizing vector size) and RoPE (position-dependent rotation) intermediates in FP32 until final BF16 Q/K. It reuses native BF16 cos/sin, leaving V, attention, GEMM and other model parts unchanged. This is neither whole-model FP32 nor a reproduction of RL-stack alignment: R0 was already shared.

DEV8/EVAL24 are32 authored synthetic documents: explanation, key-value, code and JSON. Length128 is a prefix of the same512-token document, not an independent sample. Batch1 teacher-forcing prefill uses no cache/generation; lm_head computes all positions before16 fixed full-vocabulary rows are retained. The384 main full forwards,54 repeats,384 local block replays and32DEV/24EVAL replacements have different denominators.

## 2. Original primary result and counterexamples

| EVAL512 /24 documents | Value (nats) |
| --- | --- |
| E0: KL(D1_R0 ∥ P_R0) | 0.0006952504144 |
| E1: KL(D1_R1 ∥ P_R1) | 0.0007815328033 |
| delta = E0−E1 | -0.0000862823889 |
| Paired95% /seed61027023 | [-0.0002068221560, 0.0000390900519] |
| Verdict | INCONCLUSIVE |

KL uses natural logarithms, in nats. It is distribution disagreement, not answer accuracy. Positive delta means reduction, but the observed negative mean has an interval containing zero. The original paired-document10,000-replicate bootstrap keeps seed61027023 and linear quantiles. tau_abs remains1e-10. Identical retained logits across54 same-path repeats do not establish an error bound for every input.

Historical32-token D0–P relative_RMSE about0.02605433 remains outside the original0.02 gate. Probe02/03 did not have controlled identical reduction settings; matched-format dense was already present there. The DEV-selected layer25 block-output replacement reduced the DEV mean KL but increased the EVAL mean. Sensitivity is not a general improvement. [Historical report](history/p0p3/REPORT.md.txt)

## 3. P4A: which endpoint is fixed?

Existing24 EVAL documents, length512,16 positions each. Within a row, the D1 reference is fixed and only P changes.

| Fixed reference | P_R0: KL /JSD | P_R1: KL /JSD |
| --- | --- | --- |
| D1_R0 | 0.0006952504 / 0.0001740831 | 0.0008521777 / 0.0002127459 |
| D1_R1 | 0.0009130131 / 0.0002282488 | 0.0007815328 / 0.0001951464 |

P_R1 moves farther from fixed D1_R0, but closer to fixed D1_R1. K00−K10 changes the reference itself and is not a one-sided improvement. Same-mode proximity does not establish the sole cause of the original disagreement.

| Fixed reference | Effect K0−K1 | Posthoc95% /seed61027025 |
| --- | --- | --- |
| D1_R0 | -0.0001569272545 | [-0.0003351114788, -0.0000262346295] |
| D1_R1 | 0.0001314802600 | [0.0000142299660, 0.0002537434723] |

The original primary interval and posthoc bootstrap retain distinct seeds/document ordering, independently reaggregated on CPU. Old EVAL is not a new holdout. Drift from fixed D0_R0 is a separate comparison:

| Fixed D0_R0 reference | Mean KL (nats) |
| --- | --- |
| D1_R0 | 0.0009939798 |
| D1_R1 | 0.0010561329 |
| P_R0 | 0.0009719356 |
| P_R1 | 0.0010312041 |

![Cross precision](figures/02_cross_precision.png)

## 4. P4B/P5A: individual accuracy and cancellation

Saved Q/K coordinates share the D0_R0 block input. The96 local combinations are24 documents×two layers(25/27)×Q/K, not96 documents. Both individual errors summed against a local FP64 formula decreased in all96, while the pair gap increased34 times, decreased57 and stayed equal5.

Let d be the difference between FP64 formula outputs from the two projection inputs, and n the difference in their calculation errors. Then G=||d||²+N+C, N=||n||², C=2⟨d,n⟩; ΔG=ΔN+ΔC when inputs stay fixed. C can be negative cancellation or positive reinforcement.

| Increasing-gap sign class | Count | Weaker original negative cross |
| --- | --- | --- |
| ΔN≤−tolerance, ΔC>tolerance | 13 | 12 |
| ΔN>tolerance, ΔC>tolerance | 17 | 12 |
| ΔN>tolerance, ΔC≤−tolerance | 4 | 0 |
| Near tolerance/other | 0 | 0 |

Tolerance=max(1e-12,1e-10×the cell's largest absolute G/N/C) was frozen before P5 derived tables. The30 cases with rising cross terms are not all disappearing negative cancellation: that earlier pattern covers24 cases and differs from this classification. No division by tiny ΔG or causal attribution of full-model KL is made.

## 5. P5B: arithmetic candidates for one term

A=R_BF16(a×R_BF16(c×s)); B=R_BF16(a×c×s). Here c is a signed INT4 code, s the actual BF16 combined scale and a the post-beta BF16 input. A rounds the weight first; B rounds at the end of the single product. They discriminate arithmetic hypotheses, not model every kernel detail.

Synthetic c=3,s=1.0078125 gives equal3.03125 for a=1 and equal0.37890625 for a=0.125. For a=1.5, A=4.5625 and B=4.53125. Exact Fraction/nearest-even checks reproduce these values; no claim is made that this illustrative triple occurs in the artifact.

Layer0/25 q_proj used five512-row fixtures each:10,485,760 first-block output coordinates. D1/P and D1/A mismatches were zero. All406,626 A/B discriminators matched P's A_only;10,079,134 were both, with B_only/neither zero. Non-discriminating coordinates are not evidence distinguishing the candidates.

406,626 counts (projection,fixture,row,output-axis) records once in the first block, without multiplying by three process repeats. The5,120 row occurrences contain4,102 unique (layer,axis,coefficient) triples and1,018 reused occurrences across fixtures. These are neither independent workloads nor deduplicated mathematical input-output pairs. The1,024 selected challenge coordinates are a subset of ALL_VALID. Actual-coefficient GPU rows came from dev_code_00 under the frozen ordering, even though the CPU pool included two DEV documents.

## 6. P5C: nested masks retain actual values

Only stable single-term equality activated frozen branch D. Two projections×the same two DEV documents, M512/I1024/O2048/group64 and K={1,2,8,64,256,1024}. Retained BF16 bits are unchanged and other coordinates become zero, without renormalization.

| Layer /document | First observed K | K1024 relative_RMSE | K1024 mismatched % /max steps |
| --- | --- | --- | --- |
| 0 / dev_code_00 | 64 | 0.0001055513265 | 0.033665% / 17 |
| 0 / dev_code_01 | 8 | 0.0001047640272 | 0.031662% / 2 |
| 25 / dev_code_00 | 64 | 8.487991352e-05 | 0.034809% / 33 |
| 25 / dev_code_01 | 64 | 7.582970442e-05 | 0.032234% / 44 |

| Layer /document | Nonzero elements at K=1/2/8/64/256/1024 (512 rows) |
| --- | --- |
| 0 / dev_code_00 | 512 / 1024 / 4096 / 32768 / 131072 / 524288 |
| 0 / dev_code_01 | 512 / 1024 / 4096 / 32768 / 131072 / 524288 |
| 25 / dev_code_00 | 511 / 1023 / 4094 / 32758 / 131010 / 524043 |
| 25 / dev_code_01 | 512 / 1024 / 4096 / 32756 / 131021 / 524058 |

Each output denominator is512×2048. Actual nonzero counts and input norm change with K; first-observed K is not a universal threshold. All96 repeat hashes match. At K1024, input bytes and all8 method outputs match P4 across separate times/processes.

P5A re-read25,165,824 original DEV8×three-layer coordinates:7,823 mismatched, with representable distances1/2/>2 equal7,496/133/194 and maximum27,892. ULP denotes steps between representable values, not internal rounding operations; signed ordering and numeric ±0 are handled.

Both oracles remain: dense64 promotes actual D1 BF16 weights; exact64 computes codes times actual BF16 scales. Neither is uncompressed-model ground truth. Across24 conditions, raw SSE ordering is P lower11/D1 lower2/equal11 for dense64 and7/6/11 for exact64. At K1024, exact64 favors D1 for layer0's two documents and P for layer25's two. These small raw orderings are not general superiority claims. The768 exact dyadic audit rows include264 midpoint ties and zero final-rounding disagreements on that subset. Another231,110 oracle-condition coordinates remain UNRESOLVED under the bounded rounding audit.

## 7. Costs and scope

| Phase | Recorded GPU parent wall (s) | Recorded CPU worker wall (s) |
| --- | --- | --- |
| P0–P3 | 1971.504192 | See original CPU-cost receipt |
| P4 | 82.207145 | 789.183164 |
| P5 | 229.252607 | 61.077458 |
| Campaign | 2282.963944 | Not added to GPU debt |
| Publication | 0 | Separate delivery receipt |

| Path /mode | Blocks ×calls | Host median of block medians ms [range] | Event median of block medians ms |
| --- | --- | --- | --- |
| D0/R0 | 3 ×10 | 26.993 [25.822, 27.246] | 26.948 |
| D0/R1 | 3 ×10 | 28.168 [25.710, 29.124] | 28.116 |
| D1/R0 | 3 ×10 | 26.913 [22.738, 28.597] | 26.860 |
| D1/R1 | 3 ×10 | 28.313 [23.265, 28.569] | 28.264 |
| P/R0 | 3 ×10 | 26.603 [26.289, 27.370] | 26.553 |
| P/R1 | 3 ×10 | 27.571 [26.747, 34.632] | 27.526 |

Original cost workers use one fixed DEV512 document. Host timing includes same-call event record/wait; event values are stream elapsed, not pure kernel time. R0 retains a no-op dispatch wrapper, and R1's first call follows R0 in the same process. These diagnostic forwards are not serving TTFT/TPOT. [Recorded samples](evidence/p0p3/timing_raw.csv)

P5 has12 compute workers,60B+144C main core calls,16 controls and4 profiler calls:224 total. New full-model/input-extraction forwards are zero. Its229.252607s GPU parent debit includes131.266670s waiting for a foreign PID, gates and closeout. This publication runs zero GPU work; its CPU/build/CI costs belong to a separate delivery ledger. Allocated/reserved memory, device VRAM, process CPU time and parent wall remain distinct.

## 8. History, limits and reproduction

Retained history includes driver610.88→621.01 drift; the first sequential large-vocabulary audit failure followed by math.fsum under the unchanged tolerance; and an idle shell guard failure whose following command still executed. The supervisor's independent gate is not retroactively the failed shell guard passing. P4 couples gate success directly to spawn and tests nonzero/exception/timeout with zero spawns. Its first P4A timeout348.003570s, corrected source/freeze and audit schema/sum fixes remain visible.

P5 scalar-shift/signed-zero audit failures, outer C dictionary-unpacking error and resource rejection retain their receipts and reasons. No other process was stopped. Endpoint samples do not prove continuous exclusivity. [Execution history](HISTORY.md)

P0–P5 source/raw/reports/ledgers remain read-only locally. Public files contain scalar tables, synthetic inputs, source and small receipts, excluding actual Z/weights/full logits/single-term raw outputs. Summary reaggregation is not fresh full-vocabulary KL from hidden logits. See [reproduction scopes](REPRODUCE.md), [sources](SOURCE_MAP.md), [reuse](REUSE_MAP.md) and [notices](NOTICE.md). No capability, RL-stability, serving-speed or all-input equivalence claim follows.

## Recorded execution denominators and memory

| Phase /scope | Actual count |
|---|---|
| P0–P3 main /repeat /local block | 384 /54 /384 |
| Original replacement DEV /EVAL | 32 /24 |
| Original timing workers /timed /cold /warm-up forwards | 9 /180 /18 /54 |
| Original preflight full /historical isolated block /replacement auxiliary block | 38 /9 /58 |
| P0–P3 GPU workers | 34 |
| P4 GPU /CPU workers | 10 /18 (one CPU timeout retained) |
| P5 compute workers /GPU accounting records /CPU workers | 12 /15 /21 |
| P5B /P5C main core calls | 60 /144 |
| P5 additional core controls /profiler | 16 /4 |
| Publication new GPU/model/GEMM | 0 /0 /0 |

Recorded P5 projection-only maximum PyTorch allocated peak was 50,668,032 bytes (48.321 MiB); reserved peak was 81,788,928 bytes (78 MiB). These are not device-wide VRAM or new publication memory measurements. Repeats, oracles and coordinates are not added as independent workloads.
