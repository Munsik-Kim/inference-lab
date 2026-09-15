# Development results and interpretation

No tested new setting met the development gate at Qwen3-0.6B layer 13, causal length 4096. FP16 PV reduced local output error, but remained above both error limits and below the 1.50× speed threshold. The two valid FP8 alternatives retained speed but did not materially reduce the median error. The V-smoothing alternative failed a small finite-output check. The result is **STOP_DEV_SCREEN**: no finalist, no fresh confirmation and no deployment recommendation.

## Original experiment: pre-specified DEV results

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

## Error uncertainty and measured cost

| Setting | Median error 95% CI | p95 error 95% CI | Wall median / p95 (ms) | Process medians (ms) |
| --- | --- | --- | --- | --- |
| B | [0.168, 0.170]% | [0.175, 0.177]% | 0.8232 / 0.8420 | 0.8243, 0.8222 |
| A_PUBLIC | [2.201, 2.379]% | [3.899, 4.880]% | 0.3912 / 0.4654 | 0.3926, 0.3892 |
| V1 | [2.189, 2.396]% | [3.943, 4.731]% | 0.4353 / 0.4717 | 0.4450, 0.4318 |
| V2 | [2.190, 2.396]% | [3.943, 4.731]% | 0.4450 / 0.4737 | 0.4456, 0.4448 |
| V4 | [1.880, 2.061]% | [3.419, 4.229]% | 0.6165 / 0.6425 | 0.6188, 0.6129 |

Error intervals resample eight document clusters with all heads retained. Timing intervals resample document IDs and the two GLOBAL process IDs, then paired blocks inside each selected cell. The process draw is shared across documents. Identical indices are reused across settings. Two processes and eight documents provide limited uncertainty information; these are pointwise intervals for this local repeated-input design. Timing p95 describes ten-call block means, not service request p95.

## Post-hoc interpretation: measured trade-off

All five measured points are nondominated under the exact three-axis point-estimate rule: latency, median error and p95 error. This does not make all five useful candidates. V2 has a microscopically lower p95 than V1, despite slower median time and the same displayed median error; their intervals overlap. This is not evidence of a meaningful V2 quality advantage or a globally optimal frontier.

V4 reduces median/p95 error from 2.294%/4.616% to 1.958%/4.070% versus the current anchor, while speedup falls from 2.110× to 1.338×. It is an observed higher-fidelity, higher-cost point outside the original acceptance region. A post-hoc join by document ID and head found lower error in all 128/128 paired units (no ties); these are repeated measurements from only 8 independent documents, not fresh confirmation. See the [paired records and threshold review](POSTHOC_THRESHOLD_REVIEW.md). The tested V1/V2 bundles retained substantial speed without materially reducing aggregate local error, and their confidence intervals overlap. They change V scale and accumulation together relative to the anchor, while V4 changes P/V format and V conversion. Residual error cannot be assigned entirely to Q/K quantization from this experiment; no component-isolating ablation was run.

The historical Case004 L4096 confirmation result (2.100×, median 2.301%, p95 3.897%) used a different, already-observed 16-document split and five timing processes. The current anchor uses eight DEV documents and new paired timing. The two results are recorded side by side for context, not treated as repeated independent confirmation or subtracted to claim a regression.

## Original experiment: worst sampled units

| Setting | DEV document | Head | Relative error | Absolute RMS | Cosine |
| --- | --- | --- | --- | --- | --- |
| B | c004-dev-06 | 12 | 0.181% | 0.000674 | 0.999998 |
| A_PUBLIC | c004-dev-04 | 12 | 5.103% | 0.019326 | 0.998725 |
| V1 | c004-dev-04 | 12 | 5.179% | 0.019612 | 0.998686 |
| V2 | c004-dev-04 | 12 | 5.179% | 0.019612 | 0.998686 |
| V4 | c004-dev-02 | 12 | 4.652% | 0.016093 | 0.998969 |

All valid DEV outputs were finite, with no missing records or undefined reference units. Recorded numerical metrics were identical across the two rounds for all five settings. This is repeated output consistency on one installation, not task accuracy or long-term stability.

## Original experiment: separate validity and prefix diagnostics

V3 generated 8,192 nonfinite quantized V entries and 16,384 nonfinite output entries for constant channels. With a fixed small variation added, 64 quantized entries and 128 output entries were nonfinite. The centered per-channel range can still be zero after BF16 rounding. The [MeanScaleKernel at the tested Sage revision d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5](https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/csrc/fused/fused.cu#L385-L418) divides scale_max by that range without a zero guard. This is a specific numerical validity boundary in the pinned implementation, not a threshold failure or a general failure on Qwen. The initial probe called this a GQA mismatch; the retained follow-up narrows the diagnosis to nonfinite outputs rather than proving incorrect head indexing. V3 was excluded before the real-trace comparison.

For the fixed D128 prefix fixture, multiplying future V by 1000 changed the anchor prefix by relative norm 0.520670 and absolute RMS 0.310263. BF16 was unchanged. Range-preserving future permutation changed neither output prefix. This is a separate whole-sequence-statistics sensitivity, not evidence that masked values were directly summed, not a security classification, and not a normal-input quality score. There was no finalist, so no finalist stress was run.

## Memory and execution scope

Maximum operator peak allocated was 120.02 MiB and peak reserved was 206.00 MiB. The maximum 100 ms whole-device sample was 2785.24 MiB; temperatures ranged 37–51 °C. Whole-device data include desktop/background use and are not allocator or per-process values. Extension allocations may not all appear in Torch counters. No OOM or DEV execution failure occurred; V3's small-fixture validity failure remains separate.

## Decision

STOP_DEV_SCREEN applies to the three valid changed settings plus one excluded invalid setting in the requested four-variant list. No configuration met the local 1%/3% and 1.50× gate. Consequently no fresh confirmation documents were generated, no Phase B was frozen, and no L512/L2048 or additional model/layer runs were started. The useful result is a measured finite trade-off and a specific failure boundary, not a claim that no better setting exists in SageAttention generally.

## Future work, not measured

This case did not evaluate answer correctness, extraction accuracy, perplexity, generation quality or end-to-end model latency. Local numerical fidelity is not downstream task accuracy. Additional layers/models, prefill integration, new precision settings and task-level tolerances need a separate decision and protocol; none was started during publication finalization.
