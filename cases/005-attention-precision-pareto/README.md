# Precision–Cost Trade-offs for Low-Precision Attention on RTX 5080

Four changed SageAttention configurations were investigated around the Case004 path on **RTX 5080**. Three completed the real-Qwen development comparison; one was excluded by a small finite-output test. None met the fixed fidelity and complete-call-speed gate, yielding **STOP_DEV_SCREEN** before fresh confirmation. The completed development screen records a finite fidelity–cost trade-off and a specific validity boundary, not a verdict on downstream Qwen quality or low-precision attention generally.

## Question

Can a changed setting meet the gate on **Qwen3-0.6B, zero-based layer 13, B1/Hq16/Hkv8/D128, causal L4096**, with BF16 Q/K/V input and BF16 output?

## Development decision

The pre-specified gate was **median local relative output error ≤1%, p95 ≤3%, no invalid/nonfinite outputs or undefined reference units, and complete-call speedup ≥1.50×**. These are project-specific engineering entry criteria, not industry standards or validated downstream-quality tolerances. They were not relaxed. No changed setting qualified; there was no finalist, Phase B, or fresh confirmation.

## Main results

<!-- BEGIN DEV TABLE -->
| Setting | Median error | p95 error | Complete speedup [95% CI] | DEV outcome |
| --- | --- | --- | --- | --- |
| B: fused BF16 | 0.169% | 0.176% | 1.000× [1.000, 1.000] | Control |
| A_PUBLIC: FP8, fp32+fp16 | 2.294% | 4.616% | 2.110× [2.073, 2.149] | Error limits exceeded |
| V1: FP8, fp32 | 2.293% | 4.539% | 1.891× [1.817, 1.944] | Error limits exceeded |
| V2: FP8, fp32+fp32 | 2.293% | 4.539% | 1.852× [1.829, 1.880] | Error limits exceeded |
| V3: FP8, fp32, smooth V | Not measured in DEV | Not measured in DEV | Not measured in DEV | Excluded at validity gate |
| V4: FP16 PV, fp32 | 1.958% | 4.070% | 1.338× [1.319, 1.346] | Error and speed limits missed |
<!-- END DEV TABLE -->

These are **DEV measurements**: 8 synthetic documents × 16 heads = 128 dependent units per setting. Full-query operators ran on all 4096 positions; error against a common FP32 reference covers **32 fixed queries per head**, each using every causal-valid key. Local numerical fidelity is not task accuracy.

Speedup is the median of paired BF16/candidate wall-block ratios from two processes. Complete calls include preprocessing, conversions, allocation and wrapper cost; capture, transfers and reference calculations are excluded. [ANALYSIS](ANALYSIS.md) reports error intervals and repeated-measurement limits.

## What changed between configurations

All use INT8 Q/K, per-warp quantization, fixed K smoothing and explicit APIs. `A_MATCHED` aliases `A_PUBLIC`.

| Setting | PV path / accumulation option | V scale_max / V smoothing |
| --- | --- | --- |
| A_PUBLIC | FP8 / `fp32+fp16` | 2.25 / off |
| V1 | FP8 / `fp32` | 448 / off |
| V2 | FP8 / `fp32+fp32` | 448 / off |
| V3 | FP8 / `fp32` | 448 / on; validity exclusion |
| V4 | FP16 / `fp32` | Not applicable / off |

V1/V2 retained speed without materially reducing aggregate local error; their intervals overlap. Accumulation and V range policy change together relative to the anchor. V4 changes operand format and includes a V cast. This does not isolate accumulator precision; see [effective settings](METHODS.md#source-audit-and-effective-settings).

## What the result supports

V4 is an observed higher-fidelity, higher-cost point: median/p95 error fell from 2.294%/4.616% to 1.958%/4.070%, while speedup fell from 2.110× to 1.338×. It remains outside the original acceptance region. The [post-hoc paired analysis](POSTHOC_THRESHOLD_REVIEW.md#paired-v4-observation) found lower V4 error in all 128 paired units; these come from only 8 independent documents and are not fresh confirmation.

V3 encountered zero centered V ranges and nonfinite outputs in small fixtures, a specific boundary of the [pinned implementation](provenance/v3_findings.json). The preliminary and corrected diagnoses are retained.

## What the result does not support

Answer correctness, extraction accuracy, perplexity, generation quality and end-to-end model latency were not evaluated. A separate future-V range stress showed sensitivity to whole-sequence statistics; it is not ordinary-input error or evidence of directly summed masked values. Prefix invariance and decode equivalence are not claimed.

## Reproduction

CPU analysis uses NumPy 2.3.5 and Matplotlib 3.10.8. From this directory:

```bash
python -B -m unittest discover -s tests -v
sha256sum -c SHA256SUMS
python -B scripts/analyze.py --stage dev --output /tmp/case005-reanalysis
python -B scripts/verify_results.py --summary results/dev_summary.json --output /tmp/case005-audit.json
python -B scripts/recalculate_posthoc.py --case . --output /tmp/case005-posthoc.json
python -B scripts/write_report.py --output /tmp/case005-presentation
```

The independent verifier recomputes 1,280 repeated unit metrics and intervals from scalar evidence, not kernel outputs. GPU reproduction requires the [pinned model and capture procedure](METHODS.md#reproduction-and-unexecuted-conditional-path). Publication finalization used no new GPU runs.

## Files / evidence

- **Original experiment:** [frozen protocol](configs/phase_a.json), [raw rounds](results/dev/), [summary](results/dev_summary.json), [decision](results/decision.json), [runtime support](provenance/support_table.json).
- **Post-hoc interpretation:** [threshold and project comparison](POSTHOC_THRESHOLD_REVIEW.md), [paired scalar calculations](results/posthoc_review.json), [official sources](provenance/posthoc_sources.json).
- **Methods and provenance:** [METHODS](METHODS.md), [execution](provenance/execution.json), [model revision](provenance/case004_model.json), [publication verification](provenance/publication_validation.json).

Weights, full Q/K/V, caches, environments and private reports are excluded. Exact synthetic DEV inputs and token IDs are included. Future downstream or model-integration work is unexecuted and requires a separate protocol.

## Attribution

SageAttention's kernels belong to its authors. This case contributes the bounded audit, measurements and analysis; [NOTICE](NOTICE.md) records Case004 reuse, upstream licensing and Codex assistance.
