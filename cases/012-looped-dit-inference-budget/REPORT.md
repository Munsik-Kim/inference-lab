# Case 012 — Allocating an Image-Generation Budget: Loops vs Steps

[한국어](REPORT.ko.md)

Built a measured-budget runner, paired records, blind annotation and bilingual inspection tools for one Looped-DiT B/32 checkpoint. Planned generation is complete and an AI has assessed all 192 saved MAIN images. L2/L4 each meet every listed constraint in 53/64 images, versus 51/64 at L1.

## Contents

1. [Background](#s1)
2. [Hypotheses and questions](#s2)
3. [Theory and cost model](#s3)
4. [Methods](#s4)
5. [Experiments](#s5)
6. [Results](#s6)
7. [Analysis](#s7)
8. [Conclusion](#s8)
9. [References and contributions](#s9)

<a id="s1"></a>
## 1. Background

Image generation spends computation across denoising steps and model blocks within each step. This Case measures complete requests and prepares evaluation of explicitly listed count, color-binding and image-coordinate relations. Looped-DiT authors already compare loops with steps; this work contributes a local measured-budget execution and inspection tool.

<a id="s2"></a>
## 2. Hypotheses and questions

The primary question is whether L4 satisfies all listed constraints more often than L1 near a measured time budget: **C_time−A_time** all-constraint pass rate. L2 provides an intermediate trade-off. Both constraint gains and losses are retained. LATENCY-DEV time alone selected settings; no image quality, new seed or rewritten prompt was used for selection.

<a id="s3"></a>
## 3. Theory and cost model

Joint visits per forward are `6+5L+6`. CFG6 uses conditional/unconditional calls separately: `2S` model calls and `2S(12+5L)` joint visits. Proxy L1/S94, L2/S73 and L4/S50 give 3196/3212/3200 visits (1598/1606/1600 before CFG). Text-only preamble runs twice per forward. Embeddings, output heads, T5, transfer and Python costs are outside this proxy; it is neither equal FLOPs nor equal latency. Actual module visits were recorded in diagnostic TRACE requests, excluded from performance aggregates.

<a id="s4"></a>
## 4. Methods

Checkpoint/EMA, tokenizer, prompt length256, CFG6, image size512, noise scale2, Euler equations, eager execution and TF32 policies remain fixed. Only L/S changes. No VAE or timestep conditioning is added. Actual denoiser dtype is BF16; the original T5 loading default produced FP32. Initial pixel noise is BF16; the original Euler write-back becomes FP32 after its first update.

Dedicated per-pair CUDA generators produce the same initial noise across settings, with shape/dtype and actual byte hashes. Model-load and warmup RNG consumption cannot change that noise. Different step schedules are not treated as matching intermediate trajectories.

Complete-request time includes string→tokenization→T5→sampling→CPU/PIL with CUDA completion synchronization. Downloads, cold load, PNG disk writes, evaluation and profiling are excluded. Sampler CUDA events use encoded text and exclude text encoding and noise construction. Finite checks/hashes run after performance timing; per-step checks/hooks run separately in TRACE/native parity.

Attempts are written to `.partial`, verified by inventory/hash, then renamed on the same filesystem. Completed results are verified and reused, not overwritten; failed starts charge the same budget.

<a id="s5"></a>
## 5. Experiments

RTX5080 SM12.0, driver610.43.02, Python3.12.14, PyTorch2.9.1+cu128, CPU2threads, batch1. Denoiser and T5 remain GPU-resident; no offload/compile/new attention path. Original environments are unchanged.

SMOKE is 4 prompts ×2 seeds ×3 block proxies (24 images). DEV has 3 timing prompts ×L1/2/4 ×S25/50/75 (27 requests), plus six one-estimate verification requests. MAIN is 16 prompts ×four seeds ×three settings (192 images /64 paired prompt–seed inputs). Timing repeats use two fixed pairs ×three settings ×three fresh processes (18 requests), adding zero independent quality samples.

Four categories/families cover count, color binding, left/right and composite constraints. English inputs remain fixed; Korean is display text. Prompts/rubric/protocol were frozen before generation and actual MAIN settings/order before MAIN.

DEV reference L4/S50 median was 4.535896s. A single integer estimate/verification per L1/L2 within S25–125 selected S89/S66. L1/S89 extrapolates beyond the initial measured grid ending at75; L2/S66 interpolates. The nearest actually measured settings still missed the5% target (+7.09%/+7.25%); no further search followed.

<a id="s6"></a>
## 6. Results

### Implementation and generation

Completed192/192 MAIN and24/24 SMOKE images. Official Euler pre-PIL tensors and generate PNG pixels were bitwise identical to the adapter at all three loops. Beyond the initial S2 probe, **full MAIN step counts89/66/50 also passed bitwise parity and every-step finite checks**. TRACE logs record actual call/block counts.

### Actual timing and memory

| Setting | L | S | Complete median (s) | Complete min–max (s) | Sampler median (ms) | Allocated peak (GiB) | Reserved peak (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_time | 1 | 89 | 4.534 | 3.519–5.354 | 4506.6 | 1.880 | 1.895 |
| B_time | 2 | 66 | 4.385 | 3.491–5.150 | 4357.2 | 1.880 | 1.895 |
| C_time | 4 | 50 | 4.762 | 3.689–5.506 | 4732.4 | 1.880 | 1.895 |

64 requests per setting. Error bars below are observed min–max, not confidence intervals. Device observations differ from torch allocated/reserved memory; text-encoder residency is identical. Three fresh-process repeat medians are in [summary](analysis/summary.json).

![MAIN complete-request medians and observed min–max](figures/complete-request-time.png)

### AI assessment of all 192 saved MAIN images

A code agent inspected each original PNG individually: **one AI rater, zero human raters**. Settings, time and seed labels were masked. The session had previously seen 48 example images, so this is not independent fully blind evaluation. The original human-pending record is preserved; this supplement has its own [protocol](publication/assessment_v1/protocol.json) and [item judgments](publication/assessment_v1/annotations.json).

| Loops / Steps | Median request time (s) | Every listed constraint met — AI assessment |
|---|---:|---:|
| 1 / 89 | 4.534 | 51/64 (79.69%) |
| 2 / 66 | 4.385 | 53/64 (82.81%) |
| 4 / 50 | 4.762 | 53/64 (82.81%) |

Each setting has 64 images from the same 64 prompt–noise pairs. A pass requires every frozen checklist item to be `satisfied`. This does not score aesthetics or every phrase in the prose prompt. Count prompts score count only; other categories score their explicit color, relation and compound items.

**Primary L4−L1: +3.12 percentage points**, with a 95% paired prompt-cluster bootstrap interval [-3.12, +10.94] pp. The interval includes zero. It resamples 16 prompts 5,000 times and does not include AI judgment error. There are four related template families in this small experiment.

Across 64 paired inputs: both pass 49, L1 only 2, L4 only 4, neither 9. Individual constraints show 7 gains and 6 losses. These compare final images; they are not observations of one image being corrected within a loop trajectory.

| Category | L1/S89 | L2/S66 | L4/S50 |
|---|---:|---:|---:|
| Count | 11/16 | 12/16 | 13/16 |
| Object colors | 15/16 | 15/16 | 15/16 |
| Left/right | 16/16 | 16/16 | 16/16 |
| Compound | 9/16 | 10/16 | 9/16 |

Category denominators are 16 images per setting. The 7 `uncertain` constraint labels count as failures for the primary score. Treating all of them as passes gives the optimistic sensitivity: L1 81.25%, L2 84.38%, L4 87.50%. Inspect every image and its evidence in [the full comparison](publication/IMAGES.md).

These are quality–time observations near a budget. DEV missed the ±5% time target, and MAIN times differ too. L2 reached the same pass count as L4 at a shorter median time, without establishing general quality superiority or a quality-ranked preset. [Shared result JSON](publication/assessment_v1/summary.json) · [Independent CPU arithmetic audit](publication/assessment_v1/audit.py)

<a id="s7"></a>
## 7. Analysis

Similar joint-block proxies do not imply equal requests: T5, preamble, output and per-step overhead differ. Because DEV missed the tolerance, results must be read as an actual quality–time comparison near a budget, not an exact equal-time superiority test.

AI scoring gives a positive L4−L1 point estimate, with an interval containing zero. Count-category passes increase and compound-category passes stay equal, while paired gains and losses coexist. The study compares final images, not an observed within-image loop-correction trajectory.

Cold first samples and load time are separate records; repeated timing adds no independent quality inputs. Full GenEval, CLIP/FID and external/large judges were not run.


### Initial publication analysis — before full AI assessment

The same 273 record JSON files support recalculation of 192 MAIN and 18 repeated-timing requests. New model executions and quality annotations are zero. This is **POST_HOC_SAME_RECORDED_SCALARS**. [Paired times and source hashes](publication/posthoc/analysis.json)

The L2/S66 MAIN median is 3.28% shorter than L1/S89 and 7.93% shorter than L4/S50. It is faster in 45 and 47 of the corresponding 64 pairs. MAIN shares one process; these are not 64 independent runtime experiments. L2 remains shortest in repeats of two fixed inputs across fresh processes, but the L1/L4 ranking changes.

In 45 of 192 MAIN records, the sampler CUDA event exceeds the surrounding CPU wall time, by up to about 116ms. Subtracting the two clocks cannot estimate T5, transfer or PIL cost. The cause remains undiagnosed and the original values are preserved. Complete-request summaries and repeats remain descriptive; small differences are not presented as a stable performance advantage.

Non-blind AI inspection of all 16 prompts at the first frozen seed72301 (48 images) finds five balloons where four were requested at every setting, and three requested cubes appearing as three at L1/2 but four at L4. These examples do not replace human quality annotations for all 192 images. [Post-hoc example identity](publication/example_identity.json) · [Every image](publication/IMAGES.md)

The present result is a recorded comparison for investigating which constraints different budget allocations gain or lose. Attractive appearance and correct requested composition must be assessed separately. Quality scores were unavailable at that stage. The separate full AI assessment is reported above, preserving the original pending summary.

<a id="s8"></a>
## 8. Conclusion

Implemented frozen-model L/S execution, measured-time selection, atomic records, blind annotation and bilingual inspection, and completed planned generation. The shared budget ledger records **302 full generations** and **1337.1s** of generation-process wall time, below320/14400s. **AI assessment is complete** for all 192 saved images. The primary difference is uncertain and no quality winner is declared. Human assessment and inter-rater agreement remain unmeasured.

Continue with [blind annotation](demo/annotation.en.html), [all paired images](demo/viewer.en.html) or [model-free auditing](REPRODUCTION.md). New training and image generation are zero. Current publication scope is a feature branch and review PR; main merge and Pages deployment are separate.

<a id="s9"></a>
## 9. References and contributions

- [OpenSenseNova/Looped-DiT](https://github.com/OpenSenseNova/Looped-DiT), pinned MIT model/code; [paper v1](https://arxiv.org/abs/2609.40305v1), architecture and evaluation sections including author loop/step comparisons. This is not independent reproduction of author tables.
- [B/32 model card](https://huggingface.co/sensenova/Looped-DiT-B32), [FLAN-T5-Large](https://huggingface.co/google/flan-t5-large), [GenEval](https://github.com/djghosh13/geneval). The local user subset is not the full official benchmark.
- Source/model/prompt identities: [provenance](provenance/models.json), [input freeze](configs/input_freeze.json). DIOVA implementation and Codex assistance are attributed in [NOTICE](NOTICE.md).
