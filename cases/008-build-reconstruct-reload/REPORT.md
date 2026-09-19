# Case 008 — Quantized Model Build/Reload and Pruned-MLP Reconstruction

**Qwen checkpoint integration and output reconstruction in a fixed smaller MLP**

[한국어](REPORT.ko.md) · [Short overview](README.md) · [Reproduction](REPRODUCTION.md)

Track Q builds a GPTQ W4A16 Qwen 4B checkpoint and runs a copy in fresh vLLM processes. Track R keeps an already smaller Qwen 0.6B MLP structure and fits only its output weights. Q's weight files became about 67.0% smaller; R recovered 94.1–95.4% of deletion-induced squared output error on short synthetic held-out prompts. Request cost and gold-based scores are separate results below.

Experiment records: [build freeze](configs/build_freeze.json), 2026-09-19 13:37 UTC (22:37 KST); [completion validation](validation.json), 14:57 UTC (23:57 KST). Report: 2026-09-20 KST. The supplied user review has no separate authored timestamp; none is inferred. Device: RTX 5080. Evidence snapshot: `case008_two_track_review_20260919.zip`, SHA256 `49cc2640b63d7662cf3d40c168eaf036505bbf5f7f5f73de8de48099d452daca`. Q and R use different models and runtimes, so their effects are not pooled as a combined compression experiment.

## Contents

1. [Background](#background)
2. [Hypotheses](#hypotheses)
3. [Theory](#theory)
4. [Methods](#methods)
5. [Experiments](#experiments)
6. [Results](#results)
7. [Analysis](#analysis)
8. [Conclusions](#conclusions)
9. [References](#references)

<a id="background"></a>
## 1. Background

A model changed in memory still needs a storage format, an actual execution path, a structure loader and result checks before another process can use it. [Case 002](../002-bf16-fp8-document-extraction/README.md) compared official model distributions; [Case 007](../007-interaction-aware-mlp-pruning/README.md) physically reduced one MLP's matrices. This case connects those kinds of changes to stored models and fresh-process execution.

**Q investigates numerical representation and runtime integration. R investigates reconstruction within a fixed smaller structure.** File size, actual low-precision kernel use, resident parameter memory, request latency and gold scores are different measured boundaries. Inspect the [artifact checker/loader](../../tools/modelpack/artifact.py), [GPTQ conversion](../../tools/modelpack/quantized.py), [fixed-structure fitting](../../tools/modelpack/r_study.py) and [scalar auditor](../../tools/modelpack/audit.py).

<a id="hypotheses"></a>
## 2. Hypotheses

These questions explain the [original frozen design](configs/build_freeze.json). They do not introduce acceptance criteria after observing results or claim external preregistration. Engineering conditions and exploratory comparisons are kept distinct.

| Question / kind | Original record and check | Observation |
|---|---|---|
| Can a fresh process read the Q artifact? / engineering condition | `reload`, `Q.recipe`; complete keys/shapes/checksums, new path and process | Matching smoke scores and short decode in two fresh processes |
| Does Q reduce stored and resident representation? / planned comparison | `Q.recipe`, `timing.allocation`; file and parameter bytes | Both decreased |
| Does R reconstruction transfer to unseen prompts? / reconstruction comparison | `R.teacher`, `R.eta_rule`; error against the same-input BF16 teacher | Squared output error decreased in all three structures |
| Does repair preserve R structure? / engineering condition | Fixed `R.removed`, standalone export; shapes and parameter counts | Only the down weight changed; size stayed the same |
| How do gold scores and requests change? / exploratory measurement | `data.primary_model_metric`, `timing`; paired NLL, correctness transitions, wall time | Mixed score changes; request speedup not established |

Q's NLL/mass decomposition, constant code predictions, generator cues and last-position-only analysis are **post-hoc diagnostics of the same records**. They do not replace the original primary metric or decision.

<a id="theory"></a>
## 3. Theory

### 3.1 W4A16: a smaller weight representation

**W4** means four-bit storage for targeted weights; **A16** uses BF16 activations in this execution. Each 128-weight group has a scale, and packed integers, scales and original shapes are stored together. Embeddings, normalization and the tied output head remain native BF16. Neither total model bytes nor VRAM must therefore become exactly one quarter of BF16.

GPTQ [1] is an existing quantization method that uses calibration information. This case executes one LLM Compressor recipe. It does not substitute a round-to-nearest simulation for kernel evidence: recorded vLLM execution loads compressed-tensors storage and uses Marlin [3].

### 3.2 R: fit output weights while keeping the deleted structure fixed

An **MLP** creates intermediate features for each token, then projects them back to the model's hidden width. R's SwiGLU computes `SiLU(gate(x)) × up(x)` and applies the `down` matrix. Its original 3,072 intermediate channels form 16 contiguous groups of 192 channels. A group is a fixed matrix-index bundle, not a semantic category of documents.

In Case 007, **INDEPENDENT** selected deletions from individual group contributions; **PAIRWISE** also used signed cross terms when groups are removed together. R reuses their selected sets without running another selector. **I25** is the independent 25% deletion, **P25** the pairwise 25% deletion, and **S50** the shared 50% structure selected by both methods. The suffix `-R` means that structure's down projection has been repaired. **R-B** is the original unpruned BF16 baseline.

For the same MLP input, the native BF16 teacher output is $Y\in\mathbb{R}^{N\times d}$. The activation $H\in\mathbb{R}^{N\times m}$ comes from the **actual smaller** gate/up path; it is not assumed identical to slicing a full-width activation. The uncorrected down weight $W_0$ and fitted $W$ both have shape $d\times m$.

$$\min_W \frac{1}{N}\lVert HW^T-Y\rVert_F^2+\lambda\lVert W-W_0\rVert_F^2.$$

$$G=H^TH/N,\quad C=Y^TH/N,\quad \lambda=\eta\,\mathrm{tr}(G)/m,$$

$$W(G+\lambda I)=C+\lambda W_0.$$

The code calls the cross statistic `b`; this report uses $C$ to avoid confusion with baseline B. The [solver](../../tools/modelpack/numerics.py), `fit_ridge`, uses CPU FP64 Cholesky and two solves, without an explicit inverse. The fixed relative normal-equation residual limit is $10^{-9}$. DEV selected one common $\eta=0.01$; final weights were stored and evaluated in BF16. This is data-dependent fitting without backpropagation, not an absence of calibration or learning information. SparseGPT [2] is related work, rather than the same fixed dense ridge method.

### 3.3 Recovery and gold scores measure different things

Recovery is `1 − Σ SSE(repaired) / Σ SSE(uncorrected)`: a **ratio of summed squared errors**, not an average of prompt recovery ratios or task accuracy. Each of 192 prompts uses the same 32 positions. Bootstrap draws resample paired prompts and recompute both sums.

Let the independently computed correct token be $y$, the four-label set be $S$, and its full-vocabulary probability mass be $m=\sum_{j\in S}p_j$:

$$\mathrm{NLL}_{full}=-\log p_y=\underbrace{-\log(p_y/m)}_{\mathrm{NLL}_{choice}}+\underbrace{(-\log m)}_{\mathrm{NLL}_{mass}}.$$

Lower negative log-likelihood (NLL) gives the correct answer a better probability score. Changes are **candidate minus the baseline in the same track**; positive is worse. `mean(-log m)` differs from `-log(mean m)`. KL(B ‖ candidate) measures baseline-distribution difference, without deciding whether B is correct. A **regression** loses a baseline-correct answer; a **gain** corrects a baseline-wrong answer.

<a id="methods"></a>
## 4. Methods

**Q: convert → validate storage → copy → fresh runtime.** GPTQ targeted 252 Linear modules, excluding the output head. Complete packed keys, scales and shapes were checked; `qartifact.finalize` added the standalone manifest before a separate-directory copy. Two fresh vLLM processes compared prompts and short decode outputs. The fused runtime's 144 MarlinLinearKernel routes count a different unit from the 252 conversion targets. Before/after parameter inventories and a separate profiler record connect storage compression to actual execution. [Q runtime code](../../tools/modelpack/q_runtime.py) · [Recorded runtime](results/raw/Q/Q-W4-runtime.json).

**R: fixed deletion → fit → BF16 storage → structure loading.** Layer 13 gate/up rows and down columns use the same retained indices. The boundary is the MLP input after post-attention layer normalization. Repair is stored in one smaller down weight. A config-based meta skeleton receives the layer-specific width, strictly assigns safetensors, restores tied embeddings and derives rotary buffers. The standalone loader does not reopen the original checkpoint. [Artifact code](../../tools/modelpack/artifact.py) · [Reload records](results/raw/R/reload.json).

Input mutation, shape mismatch, nonfinite outputs and absent full-output evidence block progress. Checks cover full block/norm/MLP/final-vocabulary outputs, rather than only the four labels. Primary prompt inputs contain no gold continuation or future answer. Token hashes and gold labels pair records. Public CPU checks recalculate option logits, retained normalizers and norm statistics. Independently reproducing full KL, hidden vectors or kernels requires excluded data and the original environment.

[Reproduction](REPRODUCTION.md) separates current-publication CPU checks, historical-snapshot restoration and future GPU reproduction. It connects Q's `build → finalize → copy → inspect → smoke evaluate` and R's per-arm copy paths. No GPU command was executed for this publication preparation.

<a id="experiments"></a>
## 5. Experiments

| Condition | Q | R |
|---|---|---|
| Model | Qwen3-4B-Instruct-2507 | Qwen3-0.6B |
| Fixed revision | `cdbee75f17c01a7cc42f958dc650907174af0554` | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Change | GPTQ W4, group 128, symmetric, block 128, dampening 0.01, no actorder | Zero-based layer 13 MLP; hidden 1024, intermediate 3072; no bias |
| Runtime | vLLM 0.29.0 / Marlin / BF16 activations and KV | Transformers 5.17.0 / BF16 |
| Smoke / calibration / DEV / held-out | 6 / 256 / 64 / 192 | 6 / 128 / 48 / 192 |
| Actual prompt length / cap | 98–130 / 1024 tokens | 103–134 / 512 tokens |

Inputs are English synthetic retrieval, two-value comparison and bounded integer programs. Each evaluation task contains 64 scenarios; all are reported. IDs, facts and token hashes were checked for split overlap, while templates are shared. Prompts were neither filled to the cap nor truncated through required evidence. [Q manifest](inputs/Q/manifest.json) · [R manifest](inputs/R/manifest.json).

Q's build environment uses LLM Compressor 0.13.0, compressed-tensors 0.18.0 and Transformers 5.14.1. Serving uses compressed-tensors 0.17.0 and Transformers 5.17.0, with Torch 2.13.0 and safetensors 0.8.0. Current web-example versions were not substituted. [Environment and source hashes](provenance/environment.json).

R removes I25 `[8,12,13,14]`, P25 `[8,13,14,15]`, or S50 `[7,8,9,11,12,13,14,15]`. The 25% structures retain 2304 channels; 50% retains 1536. Uncorrected and repaired versions plus R-B form seven arms. Calibration uses 128 prompts × 32 fixed positions (4096 vectors). Forty-eight DEV prompts select the **common eta** that minimizes mean relative squared error, averaged equally over the three structures, among five frozen candidates. Exact ties choose larger eta. [Q](configs/Q_evaluation_freeze.json) and [R](configs/R_evaluation_freeze.json) freezes precede held-out evaluation.

The primary model estimand is the equal-task-weighted paired change in full-vocabulary gold NLL. Pointwise 95% intervals use 5,000 paired prompt bootstrap draws within task, seed 808191. Heads, positions and arms do not increase the independent sample count. Each track has 192 scenarios; different models are not pooled into a 384-item risk denominator.

Timing uses 12 fixed inputs per track, three process rounds, five warmups and five paired blocks of three calls. Resampling preserves process/task/input/block dependence, with timing seed 808192. Q uses eager execution, batch 1 and fixed 1 GiB BF16 KV, including scheduling, sampling and readback in its blocking request. R measures pretokenized prefill plus argmax or a fixed eight-token request. Q's one-token request is not pure prefill. Ignoring EOS during fixed-work timing controls length; it is not a generation-quality result.

<a id="results"></a>
## 6. Results

### 6.1 Artifacts and executed paths

Q passed complete packed-artifact checks and two-process reload. R records two-process reloads for each of seven standalone artifacts. Public tensor manifests show identical shapes and parameter counts before/after repair, with only `model.layers.13.mlp.down_proj.weight` changing hash. Large model files remain private artifacts; the public ZIP contains code, recipes, inputs and measurements. [Q reload](results/raw/Q/reload.json) · [R export](results/raw/R/export.json).

### 6.2 Q size and gold scores

<!-- table:q-size -->
| Measured boundary | Q-BF16 | Q-W4 |
| --- | --- | --- |
| Safetensors (bytes) | 8,044,982,000 | 2,651,839,568 |
| Named parameters (bytes) | 8,044,936,192 | 2,651,735,728 |
| Allocator peak allocated (GiB) | 10.4868 | 5.3971 |
<!-- /table:q-size -->

Weight-file reduction is 67.0373%. Allocator peak includes the fixed KV allocation and initialization/execution memory; it is neither whole-device peak nor weight bytes alone.

<!-- table:q-score -->
| Metric | Q-BF16 | Q-W4 |
| --- | --- | --- |
| Correct / scenarios | 137/192 | 137/192 |
| Δ full NLL (nats), 95% CI | 0 | -0.591295 [-0.723442, -0.470060] |
| Δ choice NLL (nats), 95% CI | 0 | +0.379465 [+0.248205, +0.497079] |
| Mean label mass | 0.666963 | 0.673368 |
| Regressions / baseline correct | — | 1/137 |
| Gains / baseline wrong | — | 1/55 |
<!-- /table:q-score -->

Equal correct counts include one regression and one gain. The opposing changes in four-choice and full-vocabulary NLL are examined in section 7.

![Q full-vocabulary gold NLL change with a 95% interval](figures/q_scores.png)

The original primary-score figure: candidate minus baseline, negative means lower gold NLL. It does not establish task-wide accuracy improvement.

### 6.3 R reconstruction in the same smaller structure

<!-- table:r-local -->
| Structure | Mean relative error: before → after | Error-energy recovery [95% CI] |
| --- | --- | --- |
| I25 | 39.6698% → 8.1835% | 95.4019% [95.2529, 95.5491]% |
| P25 | 39.4471% → 8.4832% | 94.9609% [94.7869, 95.1338]% |
| S50 | 58.8122% → 13.7042% | 94.0924% [93.9041, 94.2807]% |
<!-- /table:r-local -->

All three structures reduced pooled local error on 192/192 prompts at the fixed 32 positions. Mean relative norm error and squared-error-energy recovery are different metrics.

![R squared-output-error energy recovery for three structures](figures/r_recovery.png)

Recovery from summed prompt error energies, with pointwise 95% intervals. This is the original 32-position metric, without last-position-only post-hoc selection.

<!-- table:r-score -->
| Arm / correct | Δ full NLL (nats), 95% CI | Mean KL(B ‖ candidate) |
| --- | --- | --- |
| R-B: 124/192 | +0.000000 [+0.000000, +0.000000] | 0.000000 |
| I25: 128/192 | -0.583823 [-0.660297, -0.505045] | 0.122761 |
| I25-R: 124/192 | -0.058966 [-0.082720, -0.034713] | 0.008785 |
| P25: 130/192 | +0.774289 [+0.727695, +0.820419] | 0.182441 |
| P25-R: 123/192 | -0.134708 [-0.155610, -0.113348] | 0.019861 |
| S50: 107/192 | +2.267479 [+2.169380, +2.371095] | 0.966793 |
| S50-R: 122/192 | -0.139956 [-0.173521, -0.106027] | 0.016958 |
<!-- /table:r-score -->

KL is an average of retained scalars computed from full logits. Closeness to R-B and correctness are separate. The following compares **repaired minus uncorrected** gold NLL within each smaller structure.

<!-- table:r-contrast -->
| Repaired − uncorrected | Δ full NLL (nats), 95% CI |
| --- | --- |
| I25-R − I25 | +0.524857 [+0.450223, +0.597687] |
| P25-R − P25 | -0.908996 [-0.952994, -0.865879] |
| S50-R − S50 | -2.407435 [-2.510137, -2.308216] |
<!-- /table:r-contrast -->

### 6.4 Request time

<!-- table:q-timing -->
| Request | BF16 / W4 mean (ms) | BF16/W4 ratio [95% CI] |
| --- | --- | --- |
| fixed_8_token_request | 164.711 / 150.969 | 1.0910 [0.8790, 1.2899] |
| one_token_request | 21.638 / 23.329 | 0.9275 [0.8474, 1.0095] |
<!-- /table:q-timing -->

<!-- table:r-timing -->
| Structure / boundary | Before/after time ratio [95% CI] |
| --- | --- |
| I25 / fixed_8_token_request | 1.0295 [0.9727, 1.0905] |
| I25 / prefill_and_argmax | 0.9678 [0.8668, 1.0718] |
| P25 / fixed_8_token_request | 1.0088 [0.9609, 1.0598] |
| P25 / prefill_and_argmax | 1.0450 [0.8942, 1.2265] |
| S50 / fixed_8_token_request | 0.9981 [0.9631, 1.0328] |
| S50 / prefill_and_argmax | 0.9313 [0.8329, 1.0374] |
<!-- /table:r-timing -->

R ratios are uncorrected time divided by repaired time. Every reported 95% interval includes 1. Identical shapes and no added matrix multiplication do not prove zero repair cost or latency equivalence. [Q timing](results/raw/Q/timing.json) · [R timing](results/raw/R/timing.json) · [Same-size contrasts](results/derived/additional_tables.json).

### 6.5 Reading the verification history

| Scope | Original / supplied user review | This publication preparation |
|---|---|---|
| CPU unit tests | Original: 19 PASS; review: 16 PASS, 3 not run without Transformers | Historical snapshot: 19 PASS, 0 skipped, including tiny CPU model fixtures |
| Scalars, intervals and timing | Original and review record PASS | Rechecked Q 384 / R 1344 arm records, Q 720 / R 2520 timing records |
| Derived files | Original regeneration recorded | Byte-identical summary, additional/local JSON, three PNGs and HTML |
| Large model / private H/Y / GPU / browser | Historical execution retained in validation | No GPU or private-vector rerun; real iPad NOT_TESTED |

384/1344 are input×arm records; 720/2520 are timing block records, not independent documents. Current-publication checks and historical-snapshot checks have [separate entry points](publication/README.md).

<a id="analysis"></a>
## 7. Analysis

### 7.1 Why the Q NLL views differ

**Post-hoc, same-input recalculation.** Original full NLL measures the correct token within the entire vocabulary. Choice NLL measures discrimination inside the four options. The mean change decomposes as `−0.591295 = +0.379465 − 0.970760`.

<!-- table:q-decomposition -->
| Task | Correct: BF16 → W4 | Δ full / choice / mass NLL (nats) |
| --- | --- | --- |
| retrieval | 63/64 → 64/64 | -0.125005 / -0.125005 / -0.000000 |
| comparison | 58/64 → 57/64 | -0.058983 / -0.058961 / -0.000022 |
| code | 16/64 → 16/64 | -1.589898 / +1.322361 / -2.912258 |
| all | 137/192 → 137/192 | -0.591295 / +0.379465 / -0.970760 |
<!-- /table:q-decomposition -->

On code, arithmetic mean label mass rises from 0.001172 to 0.020368 (0.1172% to 2.0368%). Full NLL benefits substantially from probability mass moving to the allowed-label set, while conditional gold-choice NLL worsens. The identity requires mean log-mass, rather than taking the logarithm after averaging mass. [Recalculation](publication/posthoc/recalculate.py) · [Post-hoc records](publication/posthoc/metrics.json).

![Q changes in full-vocabulary and conditional-choice NLL](figures/q_score_boundaries.png)

Two scores on the same 192 prompts. Positive is a worse gold probability score. This figure does not isolate causes of code performance.

### 7.2 Constant code predictions and generator cues

<!-- table:code-decisions -->
| Track / arm | Four-choice prediction / correct | Full argmax in labels |
| --- | --- | --- |
| Q / Q-BF16 | D: 64; 16/64 | 0/64 |
| Q / Q-W4 | D: 64; 16/64 | 0/64 |
| R / R-B | A: 64; 16/64 | 0/64 |
| R / I25 | A: 64; 16/64 | 0/64 |
| R / I25-R | A: 64; 16/64 | 0/64 |
| R / P25 | A: 64; 16/64 | 0/64 |
| R / P25-R | A: 64; 16/64 | 0/64 |
| R / S50 | A: 64; 16/64 | 0/64 |
| R / S50-R | A: 64; 16/64 | 0/64 |
<!-- /table:code-decisions -->

Q always chooses D among the four code options; every R arm always chooses A. Balanced gold positions give these constant choices 16/64 correct. In Q, the full-vocabulary argmax is never an allowed label on any of the 64 code prompts, in either arm. Forced-choice correctness here is limited evidence about code execution ability or free-generation instruction following.

The [generator](../../tools/modelpack/data.py) jointly uses `count = 2 + index % 4` and `gold = index % 4`: loop counts 2/3/4/5 correspond to A/B/C/D. Options are `answer, answer+1, answer+3, answer+7`, so the correct option is always the smallest number. Both cues were checked against all Q/R code splits. The study does not establish that models used those cues, that they caused the ridge recovery or NLL changes, or that there was cross-split data leakage. A subsequent data version could decouple labels from difficulty and place distractors on both sides of the answer. Current inputs and results remain unchanged.

### 7.3 Where R reconstruction transferred

<!-- table:split-recovery -->
| Structure | Calibration / DEV / held-out (%) | Scenarios |
| --- | --- | --- |
| I25 | 98.9409 / 95.6977 / 95.4019 | 128 / 48 / 192 |
| P25 | 98.8563 / 95.4431 / 94.9609 | 128 / 48 / 192 |
| S50 | 97.9732 / 94.6986 / 94.0924 | 128 / 48 / 192 |
<!-- /table:split-recovery -->

All splits share the same template families, with actual contexts of 103–134 tokens. This is not a full-512-token experiment or cross-domain generalization result. The 25%/50% budgets remove **one layer's intermediate channels**. Whole-model parameter-byte reductions are about 0.396%/0.792%; repair does not change those sizes. [Parameter inventory](results/raw/R/export.json).

All repaired structures have smaller baseline KL, while I25 correctness falls from 128 to 124, P25 from 130 to 123, and S50 rises from 107 to 122. Repaired I25 also has worse full NLL than its uncorrected version. Preserving native computation and improving gold scores are distinct objectives. No isolating experiment identifies a causal downstream effect of internal error direction. The review's larger last-position-only recovery diagnostic does not replace the original 32-position result.

<a id="conclusions"></a>
## 8. Conclusions

Q connects GPTQ W4A16 conversion, packed-artifact validation and fresh vLLM execution, reducing weight-file and recorded resident-parameter size. R fits only a fixed smaller MLP's down projection and reloads the per-layer structure, recovering 94.1–95.4% of squared output error on short synthetic held-out inputs. Neither track established a request-speed advantage, and gold-score changes were mixed. Study execution remains `COMPLETED`, deployment remains `NOT_ASSESSED`, and the same-record follow-up is not fresh confirmation.

| Track | Implemented and retained output |
|---|---|
| Q | Conversion/serialization/Marlin execution code with storage and runtime evidence |
| R | Fixed-structure ridge/strict loader with reconstruction, gold-score and cost comparisons |

Inspect [Q raw records](results/raw/Q/model_records.json), [R raw records](results/raw/R/model_records.json), [CPU reproduction](REPRODUCTION.md) and the [recorded-results explorer (download and open locally)](demo/index.html).

<a id="references"></a>
## 9. References

1. Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323v2), 2023 v2 (first version 2022). Upstream Q quantization method.
2. Elias Frantar, Dan Alistarh. [SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot](https://proceedings.mlr.press/v202/frantar23a.html), PMLR 202, 2023. Related reconstruction work, not a direct reproduction of this fixed dense ridge calculation.
3. vLLM / LLM Compressor projects. [W4A16 implementation guide](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a16/) and [vLLM execution guide](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/int4/). Actual [recorded environment](provenance/environment.json): LLM Compressor 0.13.0, compressed-tensors build 0.18.0/runtime 0.17.0, vLLM 0.29.0. Current links identify implementation/reference locations; they do not update the historical APIs.
4. Qwen team. [Fixed Qwen3-4B-Instruct-2507 snapshot](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554) and [fixed Qwen3-0.6B snapshot](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca). Models, tokenizers and configs.
5. Hugging Face Transformers / safetensors and PyTorch projects. [Installed source hashes and versions](provenance/environment.json): runtime Transformers 5.17.0, Torch 2.13.0, safetensors 0.8.0. Native model, serialization and matrix operations.
6. DIOVA [Case 006](../006-attention-decision-stability/README.md), [Case 007](../007-interaction-aware-mlp-pruning/README.md) and [reuse ledger](provenance/reuse.json). Score/full-validity conventions and fixed grouped structures. HOPE is background through Case 007's separate MoE context, not a new method applied to Q/R.

**Contribution and attribution.** This project implements comparison execution, artifact validation, the per-layer loader, fixed-structure ridge fitting and reanalysis. Qwen, GPTQ and execution-kernel authors retain upstream credit. Codex assisted implementation, debugging, execution, analysis and documentation. [NOTICE](NOTICE.md) · [License](LICENSE).
