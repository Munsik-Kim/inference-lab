# Methods

**Execution status:** smoke, calibration, development, 192 held-out prompt comparisons and all three timing processes completed. The earlier resource block is retained in the historical partial package and completion provenance. The frozen methods below were not changed on resume.

## Question and frozen scope

At the same number of removed groups, compare independent importance with signed pairwise deletion cost. The pinned Qwen/Qwen3-0.6B snapshot is `c1899de289a04d12100db370d81485cdf75e47ca`. Zero-based decoder layer 13 is fixed. Hidden size is 1024 and intermediate size 3072. Gate/up weights are 3072×1024, down weights 1024×3072, all without bias. Native SwiGLU is `down_proj(silu(gate_proj(x)) * up_proj(x))`.

The input boundary is the native BF16 tensor entering the MLP after `post_attention_layernorm`. All attention, other MLPs and the output head remain native BF16. Attention is forced to the installed Flash SDPA backend in every arm. SageAttention is not called in this case. No training, all-layer search, weight downloads or recovery tuning is performed.

[Protocol](configs/protocol.json), [architecture](provenance/architecture.json) and [environment](provenance/environment.json) identify the arithmetic and source. The initial v1 resource check counted its own CUDA context; v2 excludes only the current PID. The v1 freeze and a [correction record](provenance/premeasurement_corrections.json) retain the change made before any model forward. This is a recorded local freeze, not external preregistration.

## Inputs and independent units

Fresh synthetic English retrieval, subtraction/comparison, and bounded integer-code scenarios are generated from facts. Each family supplies 32 calibration, 16 development and 64 held-out prompts: 96/48/192 independent base scenarios. Six separate smoke prompts do not enter those totals. Gold comes from facts or a closed-form integer calculation checked against a bounded AST interpreter. Distinct wrong options and balanced gold positions are constructed before inference.

Every prompt has 512 tokens including the non-thinking chat template. Only an explicitly irrelevant filler field adjusts length; essential facts and questions are never truncated. Text, facts, token hashes and exact token IDs are retained. Whole-prefix-plus-label tokenization verifies four distinct single-token choices. The input never contains a marked correct answer continuation. Shared templates limit generalization beyond this synthetic family.

Sample positions are `floor(i*(511)/31)` for i=0..31, including the final prompt position. A prompt is the uncertainty unit. Its tokens, groups and repeated timings do not add independent samples.

## Accounting and selection

Groups are contiguous sets `G_i = {192*i, ..., 192*(i+1)-1}`, i=0..15. Full index lists are frozen. For represented BF16 inputs and weights promoted to FP32, form `h = silu(W_gate x) * W_up x` and `c_i = W_down[:, G_i] h[G_i]`. There is no down-projection bias in this snapshot; the sliced adapter nevertheless preserves any supported bias semantics.

For removed groups P, the real-arithmetic identity is `||sum(c_i, i in P)||² = sum ||c_i||² + 2 sum_{i<j}<c_i,c_j>`. This is an accounting identity, not a new theorem. FP32 group outputs are transferred to CPU FP64 for inner products. Equal averaging over 32 tokens and over prompts gives Q. The diagonal is independently checked against group squared norms; pre-symmetry residuals are retained. Off-diagonal signs are preserved. Floating arithmetic does not make this identity bitwise exact for native BF16 execution.

`INDEPENDENT` minimizes the diagonal sum; `PAIRWISE` minimizes the complete selected Q submatrix sum. Exhaustive lexicographically ordered enumeration evaluates 1,820 four-group sets and 12,870 eight-group sets. Exact cost ties choose the first lexicographic set. Twenty distinct seeded random sets per budget are fixed before calibration. All selectors use calibration records only. Development checks validity and never changes the selected set. The immutable selection hash is checked before held-out access. No held-out oracle is used as a primary method.

## Structural surgery and validity

Retained indices slice gate/up rows and down columns into new dense `Linear` layers. The original module and its shared configuration stay intact. A context manager restores the original module even on exceptions. A full-size masked computation is only the reference for this test, not the compressed artifact.

CPU FP64 fixtures check exact index/bias mapping. GPU smoke checks six removed sets on each of six prompts, with relative Frobenius tolerance 0.01 between sliced and masked BF16 outputs. FP32 decomposition and direct deletion must agree within relative error 1e-5. Native model repeat outputs must match bitwise. Development repeats sliced/masked checks on all calibration and development sampled inputs. Any failure blocks held-out work; missing validity is never assumed true.

Diagnostic model forwards scan all 28 complete decoder outputs, the complete final-normalized hidden tensor, the pruned MLP output and full final-token logits. The complete layer-13 input hash must match the baseline across arms; input mutation is rejected. Integrated and standalone sampled sliced outputs use the same 0.01 arithmetic tolerance, acknowledging GEMM shape-dependent rounding.

## Local and model metrics

The primary local observation is a prompt's relative Frobenius error, pooling its 32 sampled token vectors. Standalone sliced modules consume the sampled native inputs; their reference is the sampled output of the original full-prompt BF16 MLP. Different matrix row counts can add native-rounding differences. The FP32 calibration objective and this actual BF16 reconstruction metric are distinct quantities. Absolute RMS, reference norm, cosine and all sampled token metrics are retained. A reference RMS ≤1e-6 yields undefined relative error and blocks the study outcome; epsilon is not added.

All 44 selected/random configurations are evaluated locally on all three splits. Only B and the four selected method/budget configurations receive held-out model forwards. Those forwards use the physically smaller modules on the complete prompt. Native last-position full-vocabulary logits (`logits_to_keep=1`) supply CPU FP64 option NLL, Brier (sum over four labels, no division), gold margin, allowed-label mass, vocabulary KL and argmax. Exact ties use the original first-index rule. Conditional choice scoring is not free-generation accuracy. Regressions (B correct, candidate wrong), gains and wrong-to-wrong changes are reported separately. Private full vectors are omitted from Git; retained options and normalizers permit scalar checks but cannot reconstruct full-vocabulary KL.

For each budget, the primary transfer estimand is the mean paired per-prompt relative-error difference, PAIRWISE minus INDEPENDENT, on 192 held-out prompts. A task-stratified paired prompt bootstrap uses 5,000 replicates and NumPy linear quantiles. Equal task sample sizes give equal task weights. Median/p95 are descriptive prompt statistics, not token statistics. Intervals are pointwise. Positive transfer requires both intervals entirely below zero; negative transfer requires both entirely above zero; all other valid results are `COMPLETED_NO_CLEAR_TRANSFER`. These are study descriptions, not deployment or non-inferiority gates. Model scores are secondary and do not follow from local error.

## Size and timing

Native MLP parameters total 9,437,184, or 18,874,368 BF16 bytes. Removing 4/16 or 8/16 groups retains width 2304 or 1536, respectively. Report actual module counts as well as formulas. Whole-model weights remain mostly unchanged; no model-wide 25%/50% reduction is implied.

Timing uses the first two frozen held-out IDs per task: six prompts, three separate processes on the same device. Each method/input has five warmups and five paired blocks of five calls. Seeded shuffled order is shared within a block. Wall timing includes a synchronization before starting, complete calls, and a final synchronization. CUDA events use separate corresponding blocks. Primary p95 describes block-mean per-call latency, not service-request p95.

Two boundaries are kept separate: GPU-resident full 512-token MLP input → MLP output; and pretokenized prompt → last-position model logits, with no cache. Imports, model load, input transfer, model surgery, diagnostics, reference calculation and disk I/O are outside timing. Required runtime calls/output allocation stay inside. No compile, graph, memoization or old-case speedup is reused. Bootstrap retains paired ratios while resampling prompt clusters within task and shared global process IDs, then blocks. Allocator peaks include resident comparison modules; whole-device sampled peak is not claimed. Telemetry before/after a process does not measure a continuous device peak.

## Attribution and audit limit

See [NOTICE](NOTICE.md) for Qwen, Transformers, HOPE, structured-pruning prior work and Case006 oracle reuse. The CPU verifier uses a distinct scalar calculation path for norms, selection and paired outcomes; this checks consistency of retained measurements, not third-party GPU replication. Split-Q stability is labelled post-hoc and does not modify selection.
