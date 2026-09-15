# Methods

## Scope and two local freezes

The primary question concerns Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`, zero-based layer 13, B1/Hq16/Hkv8/D128, square causal prefill at L4096. The [Phase A protocol](configs/phase_a.json) was fixed before the real DEV numerical comparison. Its [hash](configs/phase_a.sha256) covers the design and 43 file hashes. This is local execution ordering, not external preregistration.

The finite candidate list is B, A_PUBLIC and V1–V4. A_MATCHED is the identical explicit anchor entry point and is not separately timed. V3 failed a small finite-output validity test; the failure and bounded diagnostic were retained. The other five configurations entered two DEV processes. Selection required a new V1–V4 setting with median error ≤0.01, p95 ≤0.03, no nonfinite/undefined outputs, and paired median speedup ≥1.50. Highest eligible speedup would win; ties would use lower p95, then config ID. There was no eligible candidate.

Phase B would have frozen a single finalist, fresh exact token IDs and all evaluation code before fresh capture. It was not created. Sixteen new synthetic documents (5 English, 5 Korean, 6 code), three lengths and five processes were conditional plans, not executed data. The fixed input-only generator is retained, but did not run. There is no undisclosed confirmation result, runner-up evaluation or threshold relaxation.

## Source audit and effective settings

The pinned [Sage core](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention/core.py) matches the installed source hash. Case004's three extension binary hashes also match. [source_audit.json](provenance/source_audit.json) and [support_table.json](provenance/support_table.json) distinguish inspected source, actual execution and validity. The compiler/runtime inventory is in [compiler_runtime.json](provenance/compiler_runtime.json). No existing environment or source was changed.

| Config | Explicit public function | PV accumulation option | P/V format | V scale_max | smooth_v | Status |
| --- | --- | --- | --- | --- | --- | --- |
| B | Torch SDPA default | Fused BF16 control | BF16 interface | Not applicable | Not applicable | Executed |
| A_PUBLIC | sageattn_qk_int8_pv_fp8_cuda | fp32+fp16 | E4M3 | 2.25 | false | Executed |
| V1 | sageattn_qk_int8_pv_fp8_cuda | fp32 | E4M3 | 448.0 | false | Executed |
| V2 | sageattn_qk_int8_pv_fp8_cuda | fp32+fp32 | E4M3 | 448.0 | false | Executed |
| V3 | sageattn_qk_int8_pv_fp8_cuda | fp32 | E4M3 | 448.0 | true | Small fixture executed; validity failure |
| V4 | sageattn_qk_int8_pv_fp16_cuda | fp32 | FP16 | Not applicable | false | Executed |

All low-precision settings use INT8 Q/K, per_warp scales, smooth_k=true, HND and return_lse=false. Query tile/warp lengths are 128/32 and key block length is 64. Complete calls include upstream centering, scale computation, quantization, V conversion/layout and output allocation/conversion. Upstream may repeat the K-mean vector for GQA, but full K/V are not expanded. V4 casts BF16 V to FP16 within the public call.

A routing-only spy confirmed that top-level sageattn() ignores extra precision kwargs on the SM120 branch and forwards its fixed options. Measurements therefore use explicit lower functions. Per-channel quantizer arguments and internal operand dtypes were instrumented in a separate smoke call; instrumentation was restored before timing. Representative profiler calls identify different runtime operator/kernel families. The [Case004 static instruction evidence](provenance/case004_instruction_evidence.json) is reused only for the same hashed anchor binary. Static instruction presence and an accumulation option name do not establish dynamic instruction counts or exact effective accumulator bits for every variant.

The fp32+fp16/fp32+fp32 FP8 paths ignore smooth_v=true; the FP16 fp32 path also ignores it. None of these ignored-flag-only combinations was counted as a new setting. V3 uses the path where smoothing is effective and encountered a zero-scale failure. This is a bundle comparison; changing accumulation also changes the V quantizer range relative to the anchor.

## Inputs, capture and reference

The eight DEV texts and exact token IDs are byte-identical to Case004. The 16 old confirmation IDs/text/prefix hashes remain [historical references](inputs/historical_index.json), with no claim that they are fresh. All inputs are self-authored synthetic English/Korean/code documents, not representative customer data. Document/head/query observations and different prefixes are dependent.

Eight existing full L4096 Q/K/V/native-output captures were verified by file hash, exact prefix hash, model revision, query count and capture metadata. They originate after QK normalization and RoPE in the installed Transformers Qwen3 attention interface; source order and module hash were checked again. Query head h maps to KV head floor(h/2). Every current B replay matched the stored native BF16 output bitwise. No model weights or activations were copied into this package, and no model capture ran during Case005.

Each measured complete call uses all query positions. Error reference selects positions floor(i*(N−1)/31), i=0…31, and all causal-valid keys. BF16 inputs are promoted to FP32 for QK, stable softmax and PV. TF32, reduced-precision reduction flags and float32-matmul policy were recorded. References are calculated one head and at most eight selected queries at a time, outside timing. Small semantic fixtures use a separate NumPy FP64 oracle. The FP32 reference is not exact arithmetic or downstream task gold.

For a document/head unit, relative error is ||O−R||_F / ||R||_F over 32 query vectors. Absolute RMS, cosine, reference norm and row norms are retained. Reference RMS ≤1e−6 gives undefined relative error and an inconclusive screen; no epsilon repair. Full operator outputs, not only selected rows, are checked for nonfinite values. NumPy linear quantiles define median/p95. Round 0 is primary fidelity; later identical metrics are repetition evidence only.

## Timing and statistics

B reuses Case004's development-frozen default fused SDPA choice. The current small and real-geometry profiler checks verify actual fused execution. It is not a math-fallback or an assertion of fastest BF16 across all libraries. Both B and candidates are measured in each new session; historical Case004 time is not the denominator.

Two fresh processes each run all eight documents and five settings. Each setting/document has 20 warmups and five paired blocks of ten calls. Balanced rotated orders use seed 505100 + round×10000 + length + document_index×37. Every wrapper recomputes preprocessing per call; no quantized-input cache, CUDA Graph or torch.compile. Wall timing synchronizes before starting the timer and after ten calls, then divides by ten. Separate CUDA-event blocks measure device time and await the end event. Capture, transfers, reference, profiler and I/O are outside both blocks. First-call time is separate, not a clean JIT decomposition.

Speedup is median over corresponding document/process/block BF16_time / candidate_time. Pointwise 95% intervals use 5,000 replicates (seed 505901): resample eight document clusters, resample the two GLOBAL process IDs once per replicate, and resample paired blocks in each selected cell. Process IDs are not independently redrawn for each document. All comparisons use the same index sequence. Error intervals use 5,000 document-cluster replicates (seed 505902) and retain all heads. This small repeated-input design does not estimate other hardware or document distributions precisely.

The 1%/3% numerical gate is a point-estimate rule. Confidence intervals are reported without silently replacing that rule. DEV speedup ≥1.50 selects entry to confirmation; a future final GO would additionally require its CI lower bound >1.00. No GO was issued. Point-estimate Pareto dominance requires no worse latency, median error and p95 error, with at least one strict improvement. Nondominance is not statistical superiority.

CUDA allocator allocated/reserved peaks and 100 ms whole-device samples have different scopes. Whole-device values include desktop/background activity. No clocks or memory settings were forced globally; a process allocator cap of 13 GiB was applied. No concurrent compute job was reported at process start. WSL telemetry cannot exclude every possible Windows background activity.

## Failure and prefix diagnostics

The first V3 fixture failed the generic GQA assertion. Follow-up retained the original record and found zero per-channel scales and nonfinite values after smoothing. The source mechanism and observed counts are in [v3_findings.json](provenance/v3_findings.json). No repair, new kernel, candidate replacement or repeated search was attempted.

The Case004 D128 fixture (seed 400135, Hq4/Hkv2/L32) compares B and anchor with future V multiplied by 1000 and with a range-preserving future permutation. Prefix changes are a separate diagnostic and never enter the ordinary-input error distribution. No finalist existed to test further. Exact prefix invariance, decode equivalence, teacher-forced evaluation suitability and security properties are outside the conclusion.

## Reproduction and unexecuted conditional path

See [README](README.md#reproduction) for CPU commands. In a disposable case copy, move existing results/dev/round-*.json aside to preserve them. Set CASE005_PYTHON to a compatible existing interpreter and DEV_TRACES to the directory holding the eight verified Case004 full DEV trace files. No package installation is implicit:

```bash
"$CASE005_PYTHON" -B scripts/benchmark.py --stage dev --round 0 --traces "$DEV_TRACES"
"$CASE005_PYTHON" -B scripts/benchmark.py --stage dev --round 1 --traces "$DEV_TRACES"
```

If full traces are unavailable, the supplied capture code and exact inputs allow recapture from an existing pinned model snapshot. Verify snapshot file hashes against [model provenance](provenance/case004_model.json); weights are not downloaded automatically. Capture in a private working directory and let that process exit before benchmarking. Newly serialized trace hashes can differ; preserve the distributed record and create a separately versioned reproduction manifest/protocol rather than silently substituting inputs under its hash. Example in that separate reproduction copy:

```bash
"$CASE005_PYTHON" -B scripts/capture_qwen.py --split dev --snapshot "$QWEN_SNAPSHOT" --work-dir "$CASE005_WORK"
```

fresh_inputs.py, freeze.py --phase b and capture_qwen.py --split fresh form the conditional path; no Phase B or fresh capture was exercised in this stopped experiment. The scripts reject absent DEV finalists and refuse to overwrite raw rounds. A different installed fingerprint needs its own separately frozen protocol. The CPU audit verifies consistency of retained scalar evidence, not independently reconstructed kernel output.

write_report.py, verify_results.py and tests/test_audit.py were added after measurement for presentation and evidence checks. They are covered by SHA256SUMS, separate from the frozen evaluation code. [reuse_changes.json](provenance/reuse_changes.json) distinguishes copied modules from the reference adaptation.

## Evidence categories and publication-only changes

The Phase A gate, DEV measurements, validity exclusion, timing/error definitions and STOP decision belong to the original experiment. The protocol, 43 frozen files, raw rounds and original result JSON/CSV remain byte-identical to the historical experimental package.

[POSTHOC_THRESHOLD_REVIEW](POSTHOC_THRESHOLD_REVIEW.md) is separate interpretation: paired V4 unit comparisons, threshold examples and external metric comparisons were performed after the result was known. It does not select a new candidate. Downstream quality, full-model latency and model integration are future work, not evidence in either category.

Publication finalization added a scalar post-hoc calculator and checks. The non-frozen `write_report.py` now exports the measured table and the existing figures only to a separate output directory; it no longer overwrites editorial documents or `results/decision.json`. The README and ANALYSIS tables are checked against that renderer. The packaging helper accepts the standalone LICENSE and verifies existing checksums instead of rewriting them. Original measurement and verification code was not changed. [publication_validation.json](provenance/publication_validation.json) records the new checks separately from the original [result audit](provenance/result_audit.json).

To create a separate case-only ZIP after checksum verification (both destinations must be new):

```bash
python -B scripts/package.py --zip /tmp/case005-public.zip --record /tmp/case005-package.json
```
