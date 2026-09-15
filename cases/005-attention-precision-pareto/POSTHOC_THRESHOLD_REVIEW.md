# Post-hoc threshold and paired-result review

This review was written after the Case005 development result was known. It does not modify the frozen experiment protocol or **STOP_DEV_SCREEN** decision. The analyses below are descriptive, use the same observed DEV records, and neither select a finalist nor constitute fresh confirmation.

## Paired V4 observation

Recomputing relative errors from stored row norms and joining by **document ID and head**, V4 had lower error than A_PUBLIC in **128/128** paired DEV units, with no ties. The median paired reduction was **0.378 percentage points**. This differs from subtracting the two aggregate medians. All 128 comparisons come from only **8 independent synthetic documents**; heads, queries and rounds are dependent. No binomial significance test treats the units as independent samples.

V4 is an observed higher-fidelity, higher-cost point. Its median/p95 errors were 1.958%/4.070%, versus 2.294%/4.616% for A_PUBLIC, while complete-call speedup fell from 2.110× to 1.338×. It remains outside the original development acceptance region. This paired result does not establish downstream-quality improvement.

## How sensitive was the gate?

The following examples were chosen after seeing the result. They illustrate decision sensitivity, not recommended tolerances, a revised protocol, or eligibility for confirmation. Validity exclusions remain in force; A_PUBLIC is not a new candidate.

| Median error limit | p95 error limit | Speedup minimum | Changed settings meeting these DEV point conditions |
| --- | --- | --- | --- |
| 1% | 3% | 1.50× | None — original gate |
| 1% | 3% | 1.25× | None |
| 2% | 4.5% | 1.50× | None |
| 2% | 4.5% | 1.25× | V4 |
| 3% | 5% | 1.50× | V1, V2 |

Lowering the speed threshold alone would not change the absence of a qualifying candidate. V1/V2 exceeded 1% error in every one of their 128 units; V4 did so in 125 units. These counts describe the distribution; the original gate applied to median/p95, not a requirement that every unit be below 1%.

A 1.338× operator speedup corresponds to about 25.24% lower paired complete-call cost, while the 1.50× gate asks for at least 33.33%. The smaller gain is real, but its value for a whole model was not measured. Changing the criterion now would not supply the missing quality or confirmation evidence.

## Comparison with official projects

These are comparisons of evaluation definitions, not new runs of other backends. Sources were inspected on 2026-09-15; versions and hashes are recorded in [posthoc_sources.json](provenance/posthoc_sources.json).

| Source | What was checked or reported | Why it cannot replace this gate |
| --- | --- | --- |
| [SageAttention2++ v1](https://arxiv.org/html/2505.21136v1), Table 2 and Appendix A.2 | CogVideoX layer-averaged cosine 99.97% and relative L1 about 0.01862; separate downstream results in Table 3 | These are observations, not acceptance bounds. Relative L1 and layer averages are not document/head relative-Frobenius median/p95. |
| [FlashAttention-3 v1](https://arxiv.org/html/2407.08608v1), §4.3/Table 3 | Absolute RMSE against FP64 on synthetic inputs with outliers: 0.024 for the FP8 baseline, 0.0091 for FA3 FP8 | 0.0091 is not 0.91% relative error. The reported improvement is not a general tolerance. |
| [FlashAttention correctness test](https://github.com/Dao-AILab/flash-attention/blob/0dc2cb48e484894c48a6cc5c6503fa5cded350c3/tests/test_flash_attn.py#L702-L704) | A representative assertion bounds maximum absolute error by twice the low-precision PyTorch reference error | The norm, reference and purpose differ. [Hopper tests](https://github.com/Dao-AILab/flash-attention/blob/0dc2cb48e484894c48a6cc5c6503fa5cded350c3/hopper/test_flash_attn.py#L184-L285) also round FP8 reference inputs before attention, unlike our original-BF16 input boundary. |
| [TransformerEngine FP8 tests](https://github.com/NVIDIA/TransformerEngine/blob/51317b440be0378f570ca6cc7dec2962f6e3f8df/tests/pytorch/attention/test_attention.py) and [helper](https://github.com/NVIDIA/TransformerEngine/blob/51317b440be0378f570ca6cc7dec2962f6e3f8df/tests/pytorch/utils.py#L240-L266) | DPA: RMSE <0.11×combined output range; MHA coefficient 0.15. FP8 `assert_close` exceptions are logged; the range-based RMSE assertion remains | This normalizes by extrema, not reference RMS. It is not a relative-L2 tolerance of 11%/15%. That snapshot excludes SM120 from its FP8 attention availability gate; it is not evidence of execution on this setup. |

The Case005 limits are conservative engineering entry criteria, not an established industry standard or a validated mapping from local error to model quality. Different norms, aggregation units and workloads prevent translating the external tolerances numerically. Nothing here implies that our candidates pass those tests or that their criteria justify changing ours. Whether this gate discards useful model-level trade-offs remains untested.

High cosine also does not imply the 1% relative-error gate was met: the DEV unit-median cosine was approximately 0.999745 for V1/V2 and 0.999822 for V4. The private review included a norm/direction squared-error identity; it was a geometric description, not an attribution of error to QK, PV or accumulator components. No component-causal claim is made here.

## Recalculate this review

```bash
python -B scripts/recalculate_posthoc.py --case . --output /tmp/case005-posthoc.json
```

Compare with [results/posthoc_review.json](results/posthoc_review.json). The standard-library implementation reconstructs errors from row norms and timing ratios from paired blocks; it does not import the frozen analysis/decision modules. The JSON preserves all 128 joined comparisons, raw-file hashes and the descriptive threshold examples. It cannot independently reconstruct GPU outputs without the excluded full tensors.

## Future work, not completed evidence

A separate study could relate local error to downstream task quality and full-model cost under a new pre-specified protocol. Other models/layers, prefill integration and additional precision settings were not evaluated here. This review neither begins those experiments nor recommends deploying a measured point.
