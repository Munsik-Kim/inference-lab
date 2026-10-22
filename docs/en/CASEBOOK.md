# Questions about inference and model changes

English | [한국어](../ko/CASEBOOK.md) · [Home](../../README.md)

[New here?](START_HERE.md) · [Plain-language glossary](GLOSSARY.md)

These cases explain a practical problem, what I built to investigate it, and what the results do and do not show. They provide code evidence for local AI execution, model optimization, quality regression testing and tool-automation evaluation.

Start with **A: [save/restart a smaller model](#case-008) → [measure actual requests](#case-009)**, **B: [compare your saved answers with diova-compare](../../packages/diova-compare/examples/compare/README.md)**, or **C: [verify workflow completion and execution costs](#case-017)**. B is an independent comparison tool, not a model experiment. [Case014](#case-014) is separate numerical research; [Case007](#case-007) and [Case006](#case-006) show earlier model-slicing and answer-comparison work. Each experiment has its own model, inputs and conditions.

For reading measurements: BF16 is a 16-bit number format; INT4 uses four-bit integers. Learned model values are weights; inputs to a calculation are activations. Saved model files are checkpoints/artifacts. DEV means existing inputs used while developing settings, and held-out means inputs reserved from those decisions. A local error compares one calculation, while accuracy counts correct task answers. [The glossary](GLOSSARY.md) explains the remaining terms.

Each case follows **objective → dataset → assumptions and theory → experiment design → validation → results → interpretation → conclusion**. Choose a case, then follow its original evidence links.

- [001 — Can a kernel-selection change unblock execution?](#case-001)
- [002 — What does an official FP8 checkpoint change?](#case-002)
- [003 — How does the reference change a numerical conclusion?](#case-003)
- [004 — Is the complete attention call faster?](#case-004)
- [005 — Can a different setting improve the trade-off?](#case-005)
- [006 — When do scores and individual choices diverge?](#case-006)
- [007 — Does joint importance select a better smaller MLP?](#case-007)
- [008 — Build, reconstruct and reload](#case-008)
- [009 — What changes with longer decode and concurrent requests?](#case-009)
- [010 — Low-bit state storage, restart and readout horizon](#case-010)
- [011 — What changes when only the readout changes?](#case-011)
- [012 — Where should an image-generation budget go?](#case-012)
- [013 — Does lower weight error improve real answers?](#case-013)

- [014 — Improving quality at the same INT4 storage budget](#case-014)

- [015 — Same Tools, Different Interfaces](#case-015)
- [016 — NVFP4 KV storage/capacity](#case-016)

<a id="case-001"></a>
## 001 — Can a kernel-selection change unblock execution?

<!-- claims: c001-runtime -->

[Objective](#case-001-objective) · [Dataset](#case-001-data) · [Assumptions and theory](#case-001-theory) · [Experiment design](#case-001-design) · [Validation](#case-001-validation) · [Results](#case-001-results) · [Interpretation](#case-001-interpretation) · [Conclusion](#case-001-conclusion)

<a id="case-001-objective"></a>
### 1. Objective

Built a before/after runner to find why an eight-bit integer (INT8) model fails on RTX 5080 and check an existing library fix. A selection guard is a condition that chooses which GPU operation to use. The question is whether execution works, before measuring speed or quality.

<a id="case-001-data"></a>
### 2. Dataset

The same RedHatAI Qwen2.5-0.5B saved model (checkpoint), with eight-bit weights and calculation inputs (W8A8), receives a short question about France’s capital. This is an initialization and generation check, not a large quality dataset.

<a id="case-001-theory"></a>
### 3. Assumptions and theory

Selecting an unsupported kernel first can fail even when a usable fallback exists. The tested upstream guard checks SM120 support so the Triton INT8 path can be selected.

<a id="case-001-design"></a>
### 4. Experiment design

Only the Python runtime guard from hclsys’s PR was applied to the official vLLM 0.29.0 wheel. The before/after runs used the same PC, environment and model. The guard and Triton kernel are upstream implementations.

<a id="case-001-validation"></a>
### 5. Validation

The checks recorded exit codes, generated text, CUDA weight dtype and loaded-module selection. Environment locks and execution evidence identify the exact conditions.

<a id="case-001-results"></a>
### 6. Results

Initialization exited with code 1 before the change. Generation exited with code 0 afterward and returned “The capital of France is Paris.” The 96 INT8 modules selected `TritonInt8ScaledMMLinearKernel`.

<a id="case-001-interpretation"></a>
### 7. Interpretation

The count is loaded modules, not 96 dynamic GPU kernel invocations. One answer does not measure throughput, general task quality or the full upstream PR build.

<a id="case-001-conclusion"></a>
### 8. Conclusion

The existing guard resolved this compatibility path in the recorded environment. The project contributes the before/after reproducer and execution evidence, not a new INT8 kernel.

[Recorded result and commands](../../cases/001-sm120-int8-fallback/README.md) · [Original execution evidence](../../cases/001-sm120-int8-fallback/evidence/original.json) · [Recorded upstream validation comment](https://github.com/vllm-project/vllm/pull/54316#issuecomment-5614091211)

<a id="case-002"></a>
## 002 — What does an official FP8 checkpoint change?

<!-- claims: c002-quality c002-cost -->

[Objective](#case-002-objective) · [Dataset](#case-002-data) · [Assumptions and theory](#case-002-theory) · [Experiment design](#case-002-design) · [Validation](#case-002-validation) · [Results](#case-002-results) · [Interpretation](#case-002-interpretation) · [Conclusion](#case-002-conclusion)

<a id="case-002-objective"></a>
### 1. Objective

Built a request-and-scoring runner to compare extracting information from Korean documents, GPU memory and answer time. It compares the official model stored in 16-bit BF16 with its eight-bit floating-point (FP8) counterpart.

<a id="case-002-data"></a>
### 2. Dataset

One hundred synthetic Korean documents contain owner, task, date and amount facts with confirmed updates. Gold is computed first. Three rounds reuse the same 100 documents; they are not 300 independent quality samples.

<a id="case-002-theory"></a>
### 3. Assumptions and theory

Valid JSON and four correct fields are different requirements. Resource savings from smaller weight formats need to be read alongside task performance.

<a id="case-002-design"></a>
### 4. Experiment design

The official Qwen3-4B-Instruct-2507 BF16 and FP8 checkpoints ran on the same tasks. Both reserved 1 GiB in BF16 for previously calculated key/value information (KV cache). A client recorded answers as they arrived (streaming).

<a id="case-002-validation"></a>
### 5. Validation

All four fields had to match for a document to count as correct. The analysis paired outputs by document to calculate an accuracy difference and interval; measurements included request time and whole-device memory.

<a id="case-002-results"></a>
### 6. Results

The first quality round scored 31/100 for BF16 and 32/100 for FP8. FP8−BF16 was +1 percentage point, with a 95% interval of −2 to +5 points. Observed peak memory was about 11.62 GiB and 8.21 GiB.

<a id="case-002-interpretation"></a>
### 7. Interpretation

FP8 requests generally finished earlier, but output lengths differed; this is not a pure decode-speed comparison. Device memory includes background allocations. Extraction accuracy was low even when JSON formatting was valid.

<a id="case-002-conclusion"></a>
### 8. Conclusion

The paired study records resource use and extraction behavior of these official checkpoints. One additional correct document does not establish improvement or equivalence. The work is an evaluation, not a new quantizer or a completed extraction service.

[Results](../../cases/002-bf16-fp8-document-extraction/README.md) · [Aggregate evidence](../../cases/002-bf16-fp8-document-extraction/results/aggregate.json) · [Methods and reproduction](../../cases/002-bf16-fp8-document-extraction/METHODS.md)

<a id="case-003"></a>
## 003 — How does the reference change a numerical conclusion?

**What I built:** an independent numerical implementation and comparison tools that separate the published method from choices in its comparison reference.

<!-- claims: c003-reference -->

[Objective](#case-003-objective) · [Dataset](#case-003-data) · [Assumptions and theory](#case-003-theory) · [Experiment design](#case-003-design) · [Validation](#case-003-validation) · [Results](#case-003-results) · [Interpretation](#case-003-interpretation) · [Conclusion](#case-003-conclusion)

<a id="case-003-objective"></a>
### 1. Objective

Audit EFQ-Softmax approximation error through an independent equation implementation, separating code mapping from the comparison scale.

<a id="case-003-data"></a>
### 2. Dataset

The study used actual Qwen3-0.6B attention traces at lengths 512, 2048 and 4096. At each length, 16 documents × 3 layers × 4 heads yield 192 head units. Heads within documents are dependent observations.

<a id="case-003-theory"></a>
### 3. Assumptions and theory

An approximation may stay close to a same-scale baseline while another scale rule yields lower error. Same-scale comparisons test mapping; the headroom comparison also changes the reference’s scale choice.

<a id="case-003-design"></a>
### 4. Experiment design

The paper’s equations were implemented with the same attention inputs (query/key/value, Q/K/V), rules for which positions may be used (masks), and 32-bit arithmetic (FP32). Separate comparisons covered EFQ selected on DEV, rounding to the nearest permitted value at EFQ’s scaling factor, and nearest rounding at a headroom scale, which is a different rule for choosing that factor.

<a id="case-003-validation"></a>
### 5. Validation

Checks covered the equations, trace capture and independent recalculation of the retained decomposition. The original screen uses same-scale nearest; its denominator is not replaced post-hoc by headroom nearest.

<a id="case-003-results"></a>
### 6. Results

EFQ-Mean median relative output error ranged from 11.49% to 14.49%. The development choice matched EFQ-MMLU and was similar in aggregate to same-scale nearest. Calibrated EFQ median error was 1.133×, 1.179× and 1.148× the headroom-nearest value across the three lengths.

<a id="case-003-interpretation"></a>
### 7. Interpretation

Headroom nearest had lower median and p95 error at every length. Passing a same-scale comparison and outperforming a stronger differently scaled baseline are separate questions.

<a id="case-003-conclusion"></a>
### 8. Conclusion

The project implements and audits Han and colleagues’ EFQ method and its reference-dependent interpretation. This FP32 numerical simulation does not measure packed-FP4 hardware speed or downstream model quality.

[Findings](../../cases/003-efq-softmax-numerical-audit/README.md) · [Scale and implementation definitions](../../cases/003-efq-softmax-numerical-audit/METHODS.md) · [Independent decomposition](../../cases/003-efq-softmax-numerical-audit/results/comparison_decomposition.json)

<a id="case-004"></a>
## 004 — Is the complete attention call faster?

<!-- claims: c004-cost -->

[Objective](#case-004-objective) · [Dataset](#case-004-data) · [Assumptions and theory](#case-004-theory) · [Experiment design](#case-004-design) · [Validation](#case-004-validation) · [Results](#case-004-results) · [Interpretation](#case-004-interpretation) · [Conclusion](#case-004-conclusion)

<a id="case-004-objective"></a>
### 1. Objective

Check whether low-precision attention remains faster after preparation and conversion are included, while measuring local output error.

<a id="case-004-data"></a>
### 2. Dataset

The study used 16 document inputs captured at Qwen3-0.6B layer 13 and lengths 512, 2048 and 4096. The real Qwen H16/H8 geometry is separate from the synthetic H8/H8 fixture.

<a id="case-004-theory"></a>
### 3. Assumptions and theory

Preparing BF16 Q/K/V for a low-precision call is part of the cost. On short inputs it can outweigh kernel savings. Local numerical error and task accuracy are different measurements.

<a id="case-004-design"></a>
### 4. Experiment design

The comparison used the official SageAttention operation with INT8 attention-score multiplication (QK) and FP8 information combination (PV). Its reference was the BF16 scaled dot-product attention (SDPA) path selected on DEV and checked to combine operations in a fused implementation. The boundary starts with GPU BF16 inputs and ends with BF16 outputs, including smoothing, quantization, conversion and wrappers.

<a id="case-004-validation"></a>
### 5. Validation

Checks covered execution routes, validity and paired call timings. Calls process every query; error uses 32 fixed queries per head with all causally valid keys at those positions.

<a id="case-004-results"></a>
### 6. Results

Complete-operator speedups at lengths 512, 2048 and 4096 were 0.373×, 1.465× and 2.100×. All three real-Qwen settings exceeded the fixed local-error limits of median ≤1% and p95 ≤3%.

<a id="case-004-interpretation"></a>
### 7. Interpretation

Below 1 means slower. Longer calls showed speed gains alongside error-screen rejection. Heads and repeated lengths do not increase independent document count, and operator time is not whole-model time.

<a id="case-004-conclusion"></a>
### 8. Conclusion

The study establishes that the measured boundary changes the speed conclusion on real model inputs. The project contributes adapters, route checks and complete-call measurements; SageAttention supplies the kernels.

[Design and results](../../cases/004-low-precision-attention-break-even/README.md) · [Cost and error interpretation](../../cases/004-low-precision-attention-break-even/ANALYSIS.md) · [Recorded selection table](../../cases/004-low-precision-attention-break-even/results/selection_table.json)

<a id="case-005"></a>
## 005 — Can a different setting improve the trade-off?

<!-- claims: c005-stop c005-tradeoff c005-validity -->

[Objective](#case-005-objective) · [Dataset](#case-005-data) · [Assumptions and theory](#case-005-theory) · [Experiment design](#case-005-design) · [Validation](#case-005-validation) · [Results](#case-005-results) · [Interpretation](#case-005-interpretation) · [Conclusion](#case-005-conclusion)

<a id="case-005-objective"></a>
### 1. Objective

Change a limited set of precision settings around the Case004 path to ask whether local error can fall while retaining complete-call speed.

<a id="case-005-data"></a>
### 2. Dataset

Development uses eight synthetic documents at Qwen3-0.6B layer 13, causal L4096. The development stop means no fresh confirmation-set results were collected.

<a id="case-005-theory"></a>
### 3. Assumptions and theory

Precision changes can also change conversion, accumulation and scaling cost. The fixed entry rule requires median error ≤1%, p95 ≤3%, valid full outputs and complete speedup ≥1.50× together.

<a id="case-005-design"></a>
### 4. Experiment design

Variants V1–V4 around A_PUBLIC used explicit entry points. Effective dispatch was audited and small validity fixtures ran before evaluation on real development inputs.

<a id="case-005-validation"></a>
### 5. Validation

The checks covered nonfinite outputs, sampled-query error and complete-call time. V3 was excluded when a small fixture’s centered V range reached zero and the tested path produced NaN/Inf outputs.

<a id="case-005-results"></a>
### 6. Results

None of the three valid changed settings met every condition. V4 lowered median error from A_PUBLIC’s 2.294% to 1.958%, while speedup fell from 2.110× to 1.338×. The decision is **STOP_DEV_SCREEN**: stop at the setting-development screen because no tested candidate met every requirement.

<a id="case-005-interpretation"></a>
### 7. Interpretation

V4 is an observed fidelity–cost trade-off. V1/V2 retained speed with little aggregate error reduction; accumulation and V-range policy changed together, so component causality is not isolated. V3 identifies a specific validity boundary in the pinned implementation.

<a id="case-005-conclusion"></a>
### 8. Conclusion

The completed development screen stopped without a finalist. Post-hoc threshold sensitivity remains interpretation, not a new pass under relaxed rules. Local error does not judge all low-precision attention or downstream model quality.

[Development result](../../cases/005-attention-precision-pareto/README.md) · [Recorded decision and measurements](../../cases/005-attention-precision-pareto/results/dev_summary.json) · [Separate post-hoc review](../../cases/005-attention-precision-pareto/POSTHOC_THRESHOLD_REVIEW.md)

<a id="case-006"></a>
## 006 — When do scores and individual choices diverge?

**What I built:** a way to replace one information-combining operation (attention) and inspect which individual answers, probabilities and ties change on matching questions.

<!-- claims: c006-native c006-readout c006-limits c006-timing c006-implementation -->

[Objective](#case-006-objective) · [Dataset](#case-006-data) · [Assumptions and theory](#case-006-theory) · [Experiment design](#case-006-design) · [Validation](#case-006-validation) · [Results](#case-006-results) · [Interpretation](#case-006-interpretation) · [Conclusion](#case-006-conclusion)

<a id="case-006-objective"></a>
### 1. Objective

The question was whether similar accuracy totals hide different correct items or different probabilities for gold answers. The same model and questions were compared while changing one attention path.

B is the model’s original BF16 path. A_PUBLIC and V4 are attention settings, not different models. A_PUBLIC carries forward the public reference setting used in Case 005; V4 retains that case’s fourth variant name. V4 does not mean FP4 or four-bit precision.

BF16 and FP16 are different 16-bit floating-point formats; FP8 is an eight-bit floating-point format and INT8 is an eight-bit integer format. Their representable values and spacing differ, affecting arithmetic and conversion cost. Model weights stayed BF16; the attention multiplication paths below changed.

**What changes in each setting?**

**Original arithmetic · B**

- Attention scores · QK product: Native BF16 SDPA
- Value combination · PV product: Native BF16 SDPA

**FP8 value-combination path · A_PUBLIC**

- Attention scores · QK product: INT8 Q/K path
- Value combination · PV product: FP8 PV; recorded accumulation option fp32+fp16

**FP16 value-combination path · V4**

- Attention scores · QK product: INT8 Q/K path
- Value combination · PV product: FP16 PV; recorded accumulation option fp32

Accumulation adds the multiplication results. Formats describe path operands, not a claim that every intermediate uses that bit width. Both low-precision paths return BF16 attention outputs.

[Method and calculation source](../../cases/006-attention-decision-stability/METHODS.md)

<a id="case-006-data"></a>
### 2. Dataset

These are synthetic English four-option tasks: RETRIEVAL looks up facts, COMPARISON compares facts or relations, and CODE predicts restricted short-program outputs. Gold is computed before inference without a model judge.

Primary prompts contain 4,096 tokens; only irrelevant filler adjusts length. Standard inputs are not filtered by B’s correctness or scores. Stress inputs are selected from a separate pool by B’s winner gap, so their rates must remain separate.

Data | Prompts | Purpose
--- | --- | ---
Development / separate smoke | 96 / 6 | Check presentation, intervention and one-token labels
Standard evaluation | 192 (64 per task) | Unfiltered primary comparison
Selected stress · boundary_pool | 46 selected from a separate pool of 192 | B-score-conditioned comparison
Length / generation / timing support | Same 48 / 24 / 12 IDs | Separate measurements, not extra independent samples

<a id="case-006-theory"></a>
### 3. Assumptions and theory

Attention has two stages. Q (query) is the learned vector used to look for relevant information at the current position; K (key) is compared with it at each position; V (value) contains the information to combine. These are numerical model representations.

① QK product: dot products between the current Q and permitted K vectors produce attention scores. A causal mask excludes future positions. Scaling and softmax convert the scores to weights P that sum to one.

② PV product: multiply and sum V vectors according to weights P to produce the attention output. A_PUBLIC and V4 both use an INT8 QK path; their value-combination paths use FP8 and FP16 respectively.

The adapter called SageAttention’s explicit FP8-PV / FP16-PV functions. Both use K centering (smooth_k=True), no V centering (smooth_v=False), and per_warp Q/K quantization. A_PUBLIC has a recorded V scale maximum of 2.25; V4’s required BF16-to-FP16 V conversion stays inside the call. A scale maps values into the lower-precision representable range.

Operand format, accumulation, scaling and conversion change as a bundle. Differences cannot be attributed solely to FP8 versus FP16. How local perturbations affect final answers also depends on the remaining model computation.

NLL is the negative log probability of the gold choice; lower is better. ΔNLL=candidate−B, so positive is worse. A flip changes B’s choice; a regression loses a correct B answer. Gains correct a wrong B answer, and wrong-to-wrong changes are counted separately.

Top ties use exact stored equality at the maximum option score, not display rounding or a small tolerance. The original deterministic tie-breaking rule is retained.

[Method and calculation source](../../cases/006-attention-decision-stability/METHODS.md)

<a id="case-006-design"></a>
### 4. Experiment design

Only zero-based layer 13’s prompt-prefill attention in Qwen3-0.6B was replaced. Other layers and subsequent decode used BF16. All arms received the same tokens, weights and masks; no future answer tokens entered primary inputs.

The post-hoc readout diagnostic reuses the same 192 standard and 46 selected stress scenarios. Each arm’s final hidden state and represented BF16 weights are evaluated through a separate FP32 head. Candidate and B are compared within the same readout. This is neither a new independent evaluation nor a model-improvement setting.

[Method and calculation source](../../cases/006-attention-decision-stability/METHODS.md)

<a id="case-006-validation"></a>
### 5. Validation

Checks establish native-versus-B agreement, layer-13-only routing, identical Q/K/V, input immutability and restoration on exception. Validity scans cover attention, all decoder blocks, final norm and full-vocabulary output, not only four option scores.

Every answer label is verified as one token in its actual continuation context. The task-balanced mean paired NLL change uses 5,000 prompt resamples for pointwise 95% intervals.

Readout controls include widening native outputs without recomputation, rounding FP32 outputs back through the original dtype, and a separate CPU FP64 calculation of four option dot products. Public option-record reanalysis and excluded full-vector checks have different audit scopes.

<a id="case-006-results"></a>
### 6. Results

B/A_PUBLIC/V4 had 104/106/108 correct choices on the standard 192. A_PUBLIC changed 5 of 192 choices: two gains and three wrong-to-wrong changes. V4 changed 8 of 192: four gains and four wrong-to-wrong changes. Both had zero standard-set regressions. In the separate selected stress set of 46, A_PUBLIC lost two correct B answers and V4 lost one.

The task-balanced paired gold-choice NLL change is candidate minus BF16; positive is worse. Both mean-change 95% intervals included zero. Original whole-model prefill speedups were 1.0041× for A_PUBLIC and 1.0014× for V4.

**Post-hoc readout diagnostic:** reuse the same 192 + 46 scenarios. Under FP32 readout, each candidate differed from B on 3 of 192 standard choices. Both disappearing and new flips were observed; the diagnostic does not replace the native study.

<a id="case-006-interpretation"></a>
### 7. Interpretation

Every original standard-set flip involved an exact top tie in B or the candidate. Tie-breaking is part of native behavior. Some flips vanished and new ones appeared under FP32 readout, so the diagnostic cannot be described simply as removing errors.

Zero observed standard-set regressions does not establish general quality preservation. The separate stress set had regressions, and the standard NLL intervals are not an equivalence test. B scored 15/64 on code; strict structured generation was 0/24 for every arm.

The shadow projection reads the same states through a different arithmetic path. It does not isolate final rounding as the sole cause of every difference.

<a id="case-006-conclusion"></a>
### 8. Conclusion

The completed comparison traces a one-layer numerical intervention through gold probabilities, individual choices, ties and cost. It distinguishes disagreement from losing or gaining correct answers and records sensitivity to output-head arithmetic.

The original study is COMPLETED_CONTROLLED_STUDY; the supplement is POST_HOC_SAME_INPUT_READOUT_DIAGNOSTIC; deployment is NOT_ASSESSED. These synthetic one-layer prefill results do not constitute a full-model low-precision or service-quality verdict.

**Interactive results:** [Case 006](https://munsik-kim.github.io/inference-lab/en/case006.html)

[Original results and source navigation](../../cases/006-attention-decision-stability/README.md) · [Separate readout methods and evidence](../../cases/006-attention-decision-stability/supplemental/readout-ties-v1/README.md) · [Open the recorded explorers](GETTING_STARTED.md#offline-explorers)

<a id="case-007"></a>
## 007 — Does joint importance select a better smaller MLP?

**What I built:** tools to choose groups inside one feed-forward block (MLP), cut the actual weight matrices, and check both computation and answer changes.

<!-- claims: c007-transfer c007-quality c007-random -->

[Objective](#case-007-objective) · [Dataset](#case-007-data) · [Assumptions and theory](#case-007-theory) · [Experiment design](#case-007-design) · [Validation](#case-007-validation) · [Results](#case-007-results) · [Interpretation](#case-007-interpretation) · [Conclusion](#case-007-conclusion)

<a id="case-007-objective"></a>
### 1. Objective

The aim was to remove part of an MLP while keeping its output close to the original. The comparison evaluates removal groups separately or accounts for their joint effect.

A channel is one coordinate of the intermediate vector computed for a token. This MLP computes 3,072 intermediate values from 1,024 input values, then combines them into 1,024 output values. A channel is not an input document or a word.

A group is a fixed deletion unit: 192 channels in index order. Group 0 contains channels 0–191; group 1 contains 192–383, continuing through group 15. These are not learned clusters of semantically similar channels. An “individual group” means one of these 16 blocks.

B is the unpruned BF16 model. Deleting a group removes its gate/up rows and down columns together. Four groups remove 768 channels and leave 2,304. That does not remove 25% of the whole model.

**What do the two methods select?**

**Individual-score selection · INDEPENDENT**

- What it computes: For each group, average the squared output change from removing that group alone over calibration inputs.
- How it selects: Choose the four or eight lowest-scoring groups. The selection score omits cancellation or reinforcement between groups.

**Interaction-aware selection · PAIRWISE**

- What it computes: Include individual scores and pairwise terms measuring how group output changes cancel or reinforce one another.
- How it selects: Evaluate complete four- or eight-group removal combinations and choose the minimum predicted joint squared error.

INDEPENDENT does not establish statistical independence. “Pair” refers to terms in the error calculation, not a rule to remove only two groups. Both methods used the same 96 calibration inputs.

[Method and calculation source](../../cases/007-interaction-aware-mlp-pruning/src/core.py)

<a id="case-007-data"></a>
### 2. Dataset

Three synthetic English task families provide exact answers: RETRIEVAL looks up a fact by key; COMPARISON compares or calculates values; CODE predicts a restricted short program’s result. Generator rules and exact computation establish gold before model scoring.

Every prompt contains 512 tokens with its question and required facts preserved. Local error uses 32 fixed positions per prompt. Positions and repeated configurations do not add independent inputs.

Data | Prompts | Purpose
--- | --- | ---
CALIBRATION | 96 (32 per task) | Select removal sets
DEVELOPMENT | 48 (16 per task) | Check implementation and numerical agreement
HELD_OUT | 192 (64 per task) | Compare after selection is frozen
Separate smoke | 6 | Basic execution checks

<a id="case-007-theory"></a>
### 3. Assumptions and theory

Why can cancellation matter? Suppose one group adds +3 and another adds −3 along an output direction. Their sum is zero: removing both leaves that direction unchanged. If each instead adds +3, removing both changes it by a magnitude of six.

The values below are an invented one-dimensional teaching example, not experiment measurements. The actual study computed each group’s 1,024-dimensional output contribution at fixed token positions on calibration inputs. Group contributions were computed in FP32 from the represented BF16 inputs and weights. The physically smaller models were evaluated in BF16 after selection.

Writing each contribution as a vector cᵢ, the deletion change is their sum. Its squared size equals the individual squared sizes plus twice all signed pairwise inner products. INDEPENDENT uses the first part; PAIRWISE includes both.

Q is the 16×16 matrix of individual sizes and inner products. For a vector z marking removed groups with ones, PAIRWISE uses zᵀQz. This describes output change at the same MLP input; it does not directly optimize gold probabilities or whole-model quality.

**Teaching example · not measured evidence**

Assumed contributions | Sum of individual scores | Squared joint change
--- | --- | ---
+3, −3 | 3² + (−3)² = 18 | (3 − 3)² = 0
+3, +3 | 3² + 3² = 18 | (3 + 3)² = 36

Equal individual-score sums can hide different joint changes. This example is not evidence that PAIRWISE always wins on real inputs.

[Method and calculation source](../../cases/007-interaction-aware-mlp-pruning/src/core.py)

<a id="case-007-design"></a>
### 4. Experiment design

Only zero-based layer 13 of Qwen3-0.6B changes. Its original intermediate width of 3,072 is partitioned into 16 contiguous groups of 192 channels. Remove four groups (25%) or eight (50%). Other layers and attention keep their native BF16 path; there is no retraining.

The runner enumerated all 1,820 / 12,870 combinations at the two budgets using calibration statistics, then froze selection. It sliced gate/up rows and down columns into actual widths of 2,304 / 1,536. A full-size masked computation served only as an equivalence reference.

Twenty frozen random removal sets per budget provide a local-error reference on the same inputs. Their model-level answer quality and request times were not measured.

<a id="case-007-validation"></a>
### 5. Validation

Development checks cover the contribution-sum identity, row/column mapping, sliced-versus-masked numerical agreement, and module restoration after exceptions. Evaluation checks full model outputs for NaN/Inf and verifies input and selection identities.

The 95% intervals use 5,000 paired, within-task prompt resamples. Local error, gold-choice probabilities, full-vocabulary distribution differences and time are separate measurements. Public scalar checks recalculate retained values; they cannot reconstruct excluded hidden/logit vectors or GPU execution.

<a id="case-007-results"></a>
### 6. Results

At 25%, mean local relative error was 29.8291% for INDEPENDENT and 29.6288% for PAIRWISE. The paired PAIRWISE−INDEPENDENT change was −0.200277 percentage points, with 95% interval [−0.284779, −0.114472]; 141/192 prompts had lower error. At 50%, removal sets, weights and outputs were identical. Both selections had lower mean local error than every one of the 20 frozen random sets per budget.

At 25%, INDEPENDENT changed 48/192 choices and PAIRWISE 12/192; PAIRWISE also had lower full-vocabulary KL(B ∥ candidate). But the post-hoc direct PAIRWISE−INDEPENDENT gold NLL difference was +0.069860 nats, with 95% interval [0.025925, 0.111997], worse for PAIRWISE. Local MLP speedups were 1.176–1.412×; all whole-model prefill speedup intervals included 1.

<a id="case-007-interpretation"></a>
### 7. Interpretation

At 25%, PAIRWISE slightly reduced local error and stayed closer to B’s vocabulary distribution and decisions. INDEPENDENT nevertheless had better gold-choice NLL, the probability loss for the independently computed answer. Preserving B and improving gold-based scores are different objectives.

At 50%, both methods produced the same groups, weights and outputs. The [0, 0] difference interval is not independent evidence of quality equivalence. The random reference describes the 20 frozen sets; it does not establish optimality over all removal sets.

B scored 63/64 on retrieval, 15/64 on comparison and 16/64 on code. Read the answer changes within these weak-baseline strata and synthetic forced-choice tasks.

<a id="case-007-conclusion"></a>
### 8. Conclusion

The completed comparison connects group selection, physical matrix slicing, held-out inputs, answer scores and cost. A small local improvement was observed at 25%; the two selectors produced the same structure at 50%.

The frozen rule requires the 95% upper bound of PAIRWISE−INDEPENDENT local-error change to be below zero at both budgets for positive transfer. The result remains COMPLETED_NO_CLEAR_TRANSFER; deployment is NOT_ASSESSED. The scope is one model, one layer and no retraining.

**Interactive results:** [Case 007](https://munsik-kim.github.io/inference-lab/en/case007.html)

[Reviewed results and sources](../../cases/007-interaction-aware-mlp-pruning/README.md) · [Methods and frozen scope](../../cases/007-interaction-aware-mlp-pruning/METHODS.md) · [Package and explorer](GETTING_STARTED.md#case007-explorer)

<a id="case-008"></a>
## 008 — Build, reconstruct and reload

<!-- claims: c008-tracks -->

<a id="case-008-objective"></a>
### Objective

Built tools to save a changed model and run it in a newly started program. Q connects the existing GPTQ compression method to Qwen 4B storage and execution. R fits the output matrix (down projection) of an already smaller Qwen 0.6B feed-forward block (MLP). They use separate models and execution software.

<a id="case-008-data"></a>
### Dataset

English questions created for retrieval, comparison and bounded integer-code tasks. Q uses 256 inputs for conversion statistics (calibration) / 64 for setting development (DEV) / 192 reserved for evaluation (held-out); R uses 128 / 48 / 192, plus six basic execution checks (smoke inputs) each. Actual prompt lengths are Q 98–130 and R 103–134 tokens.

<a id="case-008-theory"></a>
### Assumptions and theory

W4A16 targets weight storage at four bits with native BF16 activations. R keeps Case 007 removal sets fixed and solves a ridge fit toward the native BF16 MLP output. I25/P25 are independent/pairwise 25% structures; S50 is their common 50% structure. The suffix -R means repaired down weights, with the same shape.

<a id="case-008-design"></a>
### Experiment design

Build, validate tensor/scale inventories, copy to another directory and reload in fresh processes. R selects one common eta on DEV and saves BF16 weights. Held-out prompts never select groups or eta.

<a id="case-008-validation"></a>
### Validation

Retained full-output checks, strict artifact loading and paired input hashes connect the measurements. CPU tools recalculate scores, norm-based recovery and intervals. Historical restoration preserves the original 105-file bundle while the publication verifier checks edited reports.

<a id="case-008-results"></a>
### Results

Q safetensors decrease from 8,044,982,000 to 2,651,839,568 bytes; both arms score 137/192 with one regression and one gain. R recovers 94.1–95.4% of squared output error at 32 fixed positions in each of 192 held-out prompts. Request-speed intervals include 1.

<a id="case-008-interpretation"></a>
### Interpretation

Q full NLL improves while conditional-choice NLL worsens. Same-record task analysis separates label mass from option discrimination. Both tracks use constant choices on the code task. R moves closer to the native distribution, while gold-score changes are mixed.

<a id="case-008-conclusion"></a>
### Conclusion

Implemented checkpoint conversion/serialization/runtime integration and same-structure output reconstruction. Execution is COMPLETED; deployment is NOT_ASSESSED. The structured report carries definitions, source tables, code-task cues and reproduction commands.

[Structured report](../../cases/008-build-reconstruct-reload/REPORT.md) · [Reproduction / source](../../cases/008-build-reconstruct-reload/REPRODUCTION.md) · [Measured summary](../../cases/008-build-reconstruct-reload/results/derived/summary.json)



<a id="case-009"></a>
## 009 — Longer decode and concurrent requests

Built a request-measurement runner for the existing Qwen 4B model files with BF16 or four-bit weights (W4). It compares prepared GPU execution sequences (graph) and operations dispatched as they occur (eager), with longer generated answers and simultaneous requests. All three server rounds remain available. Official quality tasks were run separately: a calculated score and a normal process exit are checked separately.

[Methods and results](../../cases/009-q-serving-quality/REPORT.md) · [CPU audits](../../cases/009-q-serving-quality/REPRODUCTION.md)

<a id="case-010"></a>
## 010 — Finite-precision recurrent memory: storage, restart and readout horizon

<!-- claims: c010-state-bytes -->
**Objective and implementation:** store the memory updated with each input (recurrent state) in fewer bits and restore its failures, random-generator state and current position together. The tools compare carrying storage-error corrections to the next step (residual transport), storing components at different precision (mixed precision), length until the first wrong answer and exact storage bytes.

**Data and design:** the same three S3 group-tracking checkpoints serve both stages. Stage A (v1) explores storage formats and strengthens comparators using 512 sequences per checkpoint. Stage B (v2) revises failure persistence and evaluates five arms on 1,024 fresh sequences per checkpoint. The two groups of sequences and the 546/630/195 families of uncertainty comparisons stay separate.

**Validation and results:** v2 is the current implementation. Serialized state falls from Native 12,305 to INT8 3,137 bytes per stream (about 74.5%), with each checkpoint's mean continuous-correct-length point difference under one token. Fresh-process byte/prediction checks distinguish synthetic contracts from historical measured-input diagnostics.

**Interpretation and conclusion:** Rank2 does not consistently beat Mixed5/6 within the N128 storage cap; these are the recorded method and budget names. Mean correct-prefix length, a horizon estimated from observed success rates, and the tested-grid length supported by uncertainty bounds measure different things. Saving a failure improves restart behavior; whether memory lasts longer is evaluated separately.

[Unified report](../../cases/010-ckda-finite-precision-memory-horizon/REPORT.md) · [Current implementation](../../cases/010-ckda-finite-precision-memory-horizon/versions/v2/source/online_v2.py) · [CPU audits](../../cases/010-ckda-finite-precision-memory-horizon/REPRODUCTION.md) · [Snapshot map](../../cases/010-ckda-finite-precision-memory-horizon/VERSION_MAP.md)


## Terms used here

- **Inference:** running an already trained model to produce scores or outputs.
- **Quantization; BF16/FP8/INT8:** representing values with particular limited-precision formats; weights and individual attention operands need not use the same format.
- **Attention:** combining value vectors using scores computed from queries and keys.
- **Prefill / decode:** processing the observed prompt / producing subsequent tokens with a cache.
- **Complete operator cost:** the measured adapter call, including its required preparation and output conversion.
- **Local output error:** a numerical difference at one operator boundary, not a task-accuracy loss.
- **Gold:** an independently established target answer, not the BF16 model's prediction.
- **Paired comparison:** matching the same input across configurations or rounds before calculating differences.
- **Top-score tie:** exact equality at the maximum allowed-option score under the recorded arithmetic.
- **Post-hoc diagnostic:** analysis motivated after results were known, labelled separately from the original fixed evaluation.
- **Readout:** the output projection converting a final hidden state into vocabulary scores.

The follow-up using Case010 state storage is [Case011 readout adaptation](#case-011).

<a id="case-011"></a>
## 011 — Same State, Different Readout

**Objective and implementation:** fit, save and restore only the last answer-reading layer’s two arrays (weight/bias), keeping how the state evolves and is stored unchanged. Three readouts within each storage mode read the same evolving feature; their answers do not affect the next state.

**Data and method:** three existing CKDA checkpoints, Native/INT8 storage, and 16,384 Native feature rows for each refit. SHORT fits positions 1–32; MIXED covers 1–256. Selection uses correct-answer probability loss (cross-entropy, CE; lower is better) on DEV, followed by the same 1,024 fresh sequences through 2,048 symbolic steps.

**Checks and findings:** the tool checks a two-tensor, 1,158-value final layer with 0 additional recurrent-state bytes. INT8 MIXED changed mean uninterrupted correct length by −9.89/−10.40/−13.49 tokens on seeds 0/1/2, while correct-answer probability loss (gold CE) at positions beyond those used for fitting decreased in all three. SHORT exported unchanged weights; two selected MIXED fits reached the 200-iteration cap.

**Interpretation:** better gold probability scores did not delay the first error. Same-record error shifts and storage interactions are labelled post-hoc analyses. This bounded fitting result does not establish what every possible decoder could recover.

[Overview](https://munsik-kim.github.io/inference-lab/en/case011.html#overview) · [Report](../../cases/011-frozen-state-readout-adaptation/REPORT.md) · [Intervention code](../../cases/011-frozen-state-readout-adaptation/source/adapter.py) · [CPU reanalysis](../../cases/011-frozen-state-readout-adaptation/REPRODUCTION.md)

<a id="case-012"></a>
## 012 — Where should an image-generation budget go?

**Objective and implementation:** a runner divides time between successive image updates (steps) and repeated core computation within each update (loops). It saves images from identical initial noise and compares time, memory and count/color/left-right requirements.

**Data and results:** one Looped-DiT B/32 checkpoint generated 192 MAIN images from 16 prompts and four initial noises each. Loops 1 / Steps 89, Loops 2 / Steps 66 and Loops 4 / Steps 50 had request medians of 4.534/4.385/4.762 seconds, including text encoding and CPU image conversion. The exact time-matching target was missed.

**Completed human checklist:** all 192 images have answers: 100 direct checks and 92 explicit visual-group reuses. L1/L2/L4 pass counts are 52/64, 51/64 and 53/64. L4 gains four passes and loses three versus L1. These are descriptive assigned-label counts without a confidence interval; [answers, origins and CPU calculation](../../presentation/evaluations/case012-human-v1/README.md) are available.

**Preserved AI assessment:** all 192 MAIN images were judged with the same checklist. Every item is met in L1 51/64, L2 53/64 and L4 53/64 images. L4−L1 is +3.12pp, with a 95% prompt-cluster bootstrap interval [-3.12, +10.94]pp including zero. L4 alone passes 4 pairs and L1 alone passes 2. One AI rater, zero humans; 7 uncertain items count as failures. Prior example exposure and time mismatch are disclosed in the [assessment supplement](../../cases/012-looped-dit-inference-budget/publication/assessment_v1/README.md).

[Plain-language overview](../../cases/012-looped-dit-inference-budget/README.md) · [Compare every image](../../cases/012-looped-dit-inference-budget/publication/IMAGES.md) · [Adapter code](../../cases/012-looped-dit-inference-budget/source/adapter.py) · [CPU audit](../../cases/012-looped-dit-inference-budget/REPRODUCTION.md) · [Full report](../../cases/012-looped-dit-inference-budget/REPORT.md)

[Interactive human checklist](https://munsik-kim.github.io/inference-lab/en/case012-annotate.html) · [Measured results](https://munsik-kim.github.io/inference-lab/en/case012.html). Your partial labels can be downloaded as JSON and restored later; they remain separate from the AI scores and are not automatically uploaded.

<a id="case-013"></a>
## 013 — Does lower weight error improve real answers?

**Built:** convert weights to signed four-bit integers (INT4) in groups of 64, store two four-bit values (nibbles) per byte, and reload them with row/column scaling factors preserved. The model adapter and evaluation tools compare matching text, long-input answers and local-tool actions, then recheck saved quality results on CPU. RTN simply rounds to the nearest permitted weight value. D16 uses scaled INT4 weights with BF16 calculation inputs; D4 uses the same weights and also quantizes inputs to INT4.

**Observed:** D16 lowered average squared weight error (MSE) against RTN in both models. Text-prediction score PPL (perplexity) is better when lower on the same text. On the recorded WikiText subset, 0.6B PPL was 27.576→24.445; 4B PPL was 10.810→10.894. In eight existing 4B tasks used during setting development (DEV), where the model had to choose and execute tools itself, BF16/RTN/D16/D4 completed 8/8, 8/8, 6/8, 0/8. The four-task 16K follow-up reuses existing tasks. Full packed-model numeric error exceeded the frozen gate; serving benefits remain unestablished.

**Scope:** an implementation variant based on the published equations, using two fixed Qwen versions on RTX 5080. The quality path restores compressed values to BF16 for matrix calculations (reference computation); it does not measure packed execution speed. Evaluation reserved from development (formal TEST), same-condition GPTQ request comparison and direct packed four-bit-weight/four-bit-input execution (W4A4) were not completed. Original outputs and failure-check records are retained.

[Overview](../../cases/013-dualquant-reference-quality/README.md) · [Solver/adapter](../../cases/013-dualquant-reference-quality/source/paper_int4.py) · [CPU checks](../../cases/013-dualquant-reference-quality/REPRODUCTION.md) · [Report](../../cases/013-dualquant-reference-quality/REPORT.md)


### Follow-up — Tool-call contracts and recovery

Kept the quantized weights unchanged and separately compared call-error explanations and restricting generated tool names to the available list (enum decoding). On the same eight existing DEV tasks, D16 changed 6/8→8/8 and D4 changed 0/8→2/8. Name-limited conditions made zero calls to unregistered tools; six other D4 tasks failed. The same final JSON answer can receive a different correctness judgment if the actual file changes and checks differ. Performance on new tasks and speed from packed request serving remain unestablished.

[Overview and CPU commands](../../research/agent_tool_contract_repair_20261014/README.md) · [Follow-up report](../../research/agent_tool_contract_repair_20261014/records/REPORT.md) · [Earlier diagnostic](../../research/agent_stability_20261013/README.md)

<!-- case014-entry -->
<a id="case-014"></a>
## 014 — Improving Model Quality at the Same INT4 Storage Budget

This separate research study asks whether correcting compression error improves text prediction while preserving the converted matrices’ INT4 storage. Its stages are P0: check the equations and storage code; P1: a choice made on calibration data (CAL) failed to transfer to reserved data; P1b: select on separate data; P2: collect intermediate values altered by earlier layer compression at the next layer, fit corrections, and compare the full model.

Qwen3-0.6B · 196 learned matrix transformations (projections) · PG19 PPL 57.600→50.156 (12.92% lower). All 64/64 excerpts have lower correct-token probability loss (NLL). For ArXiv, the QUARTER setting has a worse point estimate with an uncertainty interval that includes zero; HALF has higher NLL. Quality was checked after restoring values to BF16. Packed execution speed and GPU-memory (VRAM) gains are unevaluated; equal projection bytes do not mean equal total model-file size.

[Pages](https://munsik-kim.github.io/inference-lab/en/case014.html#overview) · [Report](../../cases/014-quantization-compensation/REPORT.md) · [Code](../../cases/014-quantization-compensation/followups/p2_core/source/run_p2.py) · [CPU](../../cases/014-quantization-compensation/REPRODUCE.md) · [Stage map](../../cases/014-quantization-compensation/STAGE_MAP.md)

<!-- case015-entry -->
<a id="case-015"></a>
## 015 — Same Tools, Different Interfaces

Check whether an AI finishes the same task after a tool rename. The example is **read item count and price per item → calculate the purchase total → save it → validate the record**. Functions and input formats stay fixed; only names change, such as `read_order` to `readOrder`. Across 16 problems, BF16 was **15/16→16/16**, RTN **16/16→16/16**, and D16 **12/16→12/16**. The [task and terminology guide](../../cases/015-tool-interface-sensitivity/README.md) explains orders as purchase records, unit price as price per item, and how to read the results. The 96 runs of the earlier P1 study and the new A1 application are counted separately.

[Pages](https://munsik-kim.github.io/inference-lab/en/case015.html#overview) · [Report](../../cases/015-tool-interface-sensitivity/REPORT.md) · [CPU example](../../cases/015-tool-interface-sensitivity/REPRODUCE.md) · [Runner](../../cases/015-tool-interface-sensitivity/followups/order-api-regression-v1/src/order_regression/runner.py)

<!-- case016-entry -->
<a id="case-016"></a>
## Case016 — Native KV capacity and request costs at the same memory budget

Built native NVFP4 KV storage/read checks and a fixed-pool comparison tool on RTX5080. Token capacity was 3.56× at approximately4GiB; F4/B16 request-throughput ratios at16K concurrency2/4 were 1.21/1.35×. Other conditions were not uniformly faster; original agent-baseline failures remain preserved.

[Report](../../cases/016-kv-agent-bench/REPORT.md) · [CPU commands](../../cases/016-kv-agent-bench/REPRODUCE.md) · [Implementation](../../cases/016-kv-agent-bench/followups/nvfp4-technical-r1/src/nvfp4_technical/runner.py)

<!-- case017-entry -->
<a id="case-017"></a>
## 017 — Agent Contract–Cost Bench

We built a benchmark that checks how tool call policies change completion of saved state, observed validation and final answers. It uses owned synthetic records for product totals, inventory transfers and document updates.

With BF16 weights and native NVFP4 KV, grammar changes completion 7/24→15/24 and correct saved states 7/24→22/24 on 24 new synthetic tasks: 8 recoveries and 0 losses. Added execution costs and remaining failures are reported alongside the gain. More tasks completed, while total execution costs also increased. Dividing all 24 runs, including failures, by completions gives approximately 3.62% more input tokens, 12.36% more output tokens and 11.27% more episode-wall time per completion.

Protocol deviation: BF16 KV with free output completed 5/8 DEV tasks, below the original 6/8 expansion criterion. The local plan and execution gate omitted this success threshold and proceeded to P2 after checking completion, freezing and sealing. No separate waiver approval was found. We disclose this deviation and present P2 as an exploratory measurement on new synthetic tasks.

[Report](../../cases/017-agent-contract-cost-bench/REPORT.md) · [CPU](../../cases/017-agent-contract-cost-bench/REPRODUCE.md) · [Runner](../../cases/017-agent-contract-cost-bench/src/agent_contract_cost/runner.py) · [Deviation](../../cases/017-agent-contract-cost-bench/PROTOCOL_DEVIATION.md)

<!-- case018-entry -->
<a id="case-018"></a>
## Case018 — State Recovery and Final-Report Consistency

Built an evaluator for wrong-target, duplicate and partial inventory recovery, checking saved state and final quantities. On24 new synthetic tasks, both policies scored Q24/W0 with BF16 KV and Q19/W5 with NVFP4 KV. The freshness guard blocked ineligible finals without improving repair or correct reporting. Quantity re-observation is a separate posthoc observational analysis.

[Report](../../cases/018-agent-state-recovery/REPORT.md) · [CPU replay](../../cases/018-agent-state-recovery/REPRODUCE.md)
