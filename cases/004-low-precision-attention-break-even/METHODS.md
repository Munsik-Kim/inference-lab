# Methods

This is an operator experiment on one RTX 5080, not a model-quality evaluation. The [local protocol](configs/experiment_spec.json) was fixed after development checks and before confirmation capture or measurement. Its [SHA256](configs/experiment_spec.sha256) covers the specification; `frozen_files` records the implementation, inputs, development evidence and environment used to make those choices. It is not an external preregistration.

## Interface and implementations

All complete adapters accept GPU-resident, contiguous BF16 Q/K/V in BHND layout and return contiguous BF16 O. Scale is `1/sqrt(head_dim)`, dropout is zero, and causal cases use square prefill alignment. Required `.contiguous()` operations and output conversions are inside the adapter. Inputs were hashed before and after use. No KV-head expansion, decode, unequal query/key length, arbitrary GPU mask, CUDA Graph, `torch.compile`, CPU offload or service server is involved.

The two BF16 adapters use the installed PyTorch `scaled_dot_product_attention`: default dispatch, or a per-call `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` context. Both were profiled on every development input and shape. Only actual fused executions qualified. For each shape, the path with the lower median development complete-adapter wall cost was fixed in [baseline_selection.json](provenance/baseline_selection.json). Exact ties choose default. Both paths remain in the confirmation records; the confirmation results do not reselect the faster baseline. Context-manager and Python dispatch overhead are part of the respective public adapters.

The candidate is official [SageAttention commit d1a57a5](https://github.com/thu-ml/SageAttention/tree/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5), package 2.2.0. The adapter calls `sageattn_qk_int8_pv_fp8_cuda` explicitly with:

```python
tensor_layout="HND", qk_quant_gran="per_warp",
pv_accum_dtype="fp32+fp16", smooth_k=True, smooth_v=False,
return_lse=False
```

The pinned source uses INT8 Q/K: 32-token query warp scales within a 128-token query tile, and 64-token key-block scales. K is centered by its sequence mean. V is transposed/padded/permuted and quantized per channel across the sequence to E4M3, with `scale_max=2.25` for this option. P is also converted to E4M3 inside attention. The source uses integer QK accumulation, an FP16 PV instruction buffer, and FP32 outer output state. Final normalization/scaling produces BF16 output. This is not an all-FP8 Q/K/V path or full-LLM INT8 inference.

[Source and API provenance](provenance/sources.json), the [runtime profiler](provenance/backend_probe.json), and [matching static instructions](provenance/instruction_evidence.json) provide different levels of evidence. The profiled specialization calls the authors' `accum_f16_fuse_v_scale_attn_inst_buf` operator. Matching compiled symbols contain `IMMA.16832.S8.S8` and `QMMA.16832.F16.E4M3.E4M3`. Static instruction occurrences are not dynamic call counts. This is stronger than checking the Python dispatch option, but no instruction-level performance attribution is made.

The idea of checking preprocessing amortization comes from [Attention Quantization for Tabular Foundation Models, v1](https://arxiv.org/html/2609.13031v1). Its pure-FP8 kernel, hardware, TabPFN evaluation and reported crossover are not reproduced here. SageAttention's published speed and quality results are upstream claims, separate from the measurements in this directory.

## Inputs and capture

The synthetic matrix has 24 conditions: batch 1, Hq=Hkv=8, head dimension 64 or 128, length 512/1024/2048/4096/8192/16384, causal or noncausal. Independent standard-Gaussian CPU FP32 tensors are rounded to BF16. Development and confirmation use disjoint seed bases, 400000 and 410000, plus the fixed shape index. Confirmation repeats the same tensor per shape across process rounds.

The real-input matrix has three additional conditions: Qwen3-0.6B, layer 13, batch 1, Hq=16, Hkv=8, head dimension 128, causal lengths 512/2048/4096. [Model provenance](provenance/model.json) pins revision `c1899de289a04d12100db370d81485cdf75e47ca`; all nine existing snapshot files were checked against their recorded hashes. No model was downloaded for this case.

Eight development documents reuse Case 003 development texts read-only. Sixteen new confirmation documents have distinct IDs and prefixes: five English, five Korean and six code documents. They are self-authored synthetic archives, not customer data or representative language benchmarks. [The manifest](inputs/manifest.json) preserves text hashes, tokenizer revision, all 4096 input token IDs, and prefix hashes. Tokenization adds no special tokens or chat template. Each document supplies nested length prefixes; these are dependent observations. The generators share a narrow template family, despite nonduplicate IDs and text hashes.

The capture hook intercepts the installed Qwen attention interface after QK normalization and RoPE. It saves **all** query positions, corresponding unexpanded K/V, and native BF16 attention output, one document/length at a time. Query head `h` maps to KV head `h//2`. Capture validates the native interface and compares a complete BF16 adapter replay against the native output. Relative Frobenius error above 1% invalidates a capture. The [development](provenance/dev_capture.json) and [confirmation](provenance/confirmation_capture.json) manifests report both tolerance and bitwise checks. Model forward time is excluded from operator timing; the capture process exits before benchmarking starts.

Full activations remain private. The public text, token IDs, model revision and capture script permit recapture. Case 003's 16 sampled query rows are never repeated, padded or treated as full-length Q. Real-Qwen timing runs the full square operator; only the numerical reference uses sampled rows.

## Cost and memory

Primary cost is the complete public BF16-input/BF16-output adapter: Python wrapper, smoothing, absmax/scale calculations, quantization, required layout/padding, attention, normalization/epilogue and output conversion. Preprocessing is repeated every call. Model loading, tokenization, QKV projections, CPU/GPU input copies, hashes, reference calculations, correctness checks and profiling are outside the timing boundary.

Each input/backend gets 20 warmup calls. Confirmation has five separate processes; each runs 20 paired blocks of 10 calls per backend. Primary wall timing synchronizes before and after the ten calls and divides the elapsed duration by ten. A separate ten-call block is enclosed by CUDA events; the end event is synchronized before reading device elapsed time. Wall and event blocks do not purport to be the same executions. The CPU mock tests check their boundaries; actual device values are retained alongside wall values.

Seeded balanced rotations interleave default, forced Flash and Sage blocks. Every round uses the fixed shape and document ordering. The same inputs and document weights are retained across backends and rounds. This is repeated-input, warm-cache steady state. Reported p50/p95 describe **ten-call block-mean per-call costs**, not independent service-request latency. First complete calls are recorded separately; they are neither warmed measurements nor a clean decomposition of model loading, driver initialization and JIT time. They may include first-call work for that input in an already initialized process.

Primary speedup is the median of paired BF16/Sage wall block-cost ratios. The ratio of median costs is also saved because these estimators need not coincide. A 2,000-replicate percentile bootstrap samples process rounds as outer clusters, then paired block indices within each retained document. These are shape-wise pointwise 95% intervals, without multiple-comparison coverage or an across-hardware guarantee. Five process rounds are a small cluster count; systematic host/driver effects are not removed by this interval.

Kernel-only results use the exact upstream internal call with prequantized operands and a preallocated output. They are collected after all complete-adapter blocks, only after bitwise equality with the public wrapper is checked for that input. Their prepared tensors are not live during primary timing. They retain Python/custom-op overhead, exclude preprocessing/allocation and are auxiliary, not the selection gate. Subtracting this time from complete cost does not identify pure quantization latency.

Memory records distinguish CUDA allocator allocated/reserved peaks, the process-start state and 100 ms whole-device NVML samples. Allocator peaks include live inputs, outputs and wrapper allocations, but may omit allocations outside the PyTorch allocator. Whole-device readings include WSL/Windows background use and are sampled peaks, not exact per-process peaks. Model-capture peak is separate from operator peak. The allocator budget is capped at 13 GiB; no memory or clock setting is changed globally. OOM, missing telemetry and unsupported execution must retain their status rather than zero values.

## Numerical checks and decisions

Small development fixtures use an independent NumPy FP64 stable-softmax oracle, including distinct KV heads, flipped causal masks, explicit scale changes and partially/all-masked rows. The oracle returns zero plus an invalid-row mask for all-masked rows. Production adapters accept causal/noncausal square attention only. GPU Gate 0 uses a coarse semantic tolerance (BF16 1%, Sage 5%); it is separate from the stricter confirmation screen.

For every confirmation input, 32 query positions `floor(i*(N-1)/31)` use **all causal-valid keys** (or all keys for synthetic noncausal inputs). QK, stable softmax and PV are calculated in FP32 with TF32 disabled, one head at a time. BF16 and Sage errors share this reference. It is a numerical reference, not exact real arithmetic or a task gold answer. The reference never materializes all heads' full N×N matrices.

The primary unit is one document × length × layer × head over its 32 sampled-query outputs. Relative error is `norm(O - R, 'fro') / norm(R, 'fro')`; absolute RMS error and output cosine are also stored. Invalid rows are counted. If reference RMS is at most 1e-6, relative error is null and the unit needs numerical review; no epsilon-based automatic pass is allowed. Per-row error/reference norms support independent recomputation of each unit's Frobenius ratio. Direct Sage-versus-BF16 differences are additional evidence, not a substitute for the common reference.

Confirmation round 0 alone supplies primary fidelity. At each real length, 16 documents × 16 heads give 256 dependent units, not 256 independent documents. Later rounds check repeatability. Numerical uncertainty resamples 16 document clusters, retaining all heads of each sampled document. The fixed local screen requires median error ≤1%, p95 ≤3%, no invalid rows, no near-zero unresolved units and complete coverage. These thresholds are this project's engineering choices, not a published quality tolerance.

`LOCAL_CANDIDATE` additionally requires verified matching interface/backend, median complete speedup ≥1.10 and its interval lower bound >1.00. Unknown fingerprints return BF16/UNKNOWN. Synthetic shapes remain `SYNTHETIC_ONLY`: their H8/H8 geometry has no matching real-Qwen fidelity measurement, even at shared lengths. Failure of the real screen is `NUMERICAL_REJECT`, even when timing is faster. No model or service dispatch is modified.

## Known causal approximation limitation

Permuting future V positions while preserving their channel ranges left the tested prefix unchanged, and the independent oracle checks confirmed mask semantics. A separate extreme stress test multiplied future V by 1000. Sage's earlier outputs then changed through whole-sequence quantization statistics, while BF16's did not. This is a nonlocal approximation effect, not evidence that masked values were directly summed by the attention kernel. It rules out a prefix-invariance or autoregressive-decode claim. The stress fixture is not pooled into ordinary-input accuracy, and its observation was recorded before protocol freeze.

## Reproduction and attribution

See [README](README.md) for commands, [INSTALL](provenance/INSTALL.md) for the exact isolated build method and [NOTICE](NOTICE.md) for attribution. Analysis consumes raw GPU records, rejects mock timings, and checks unit errors against retained row norms. It emits the tables and figures without selecting favorable rounds or changing the frozen thresholds.

`make_report.py`, `verify_artifacts.py` and `package_case.py` are post-measurement presentation, audit and packaging utilities. The report renderer only adjusts the speed figure's labels and axis limits; numerical analysis and the measurement code remain byte-identical to the frozen versions. The case-wide `SHA256SUMS` additionally covers these later files and the derived documents. [artifact_audit.json](provenance/artifact_audit.json) records independent scalar recomputation; [validation.log](provenance/validation.log) records commands and exit codes.
