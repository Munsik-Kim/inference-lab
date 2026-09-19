# Seven questions about local inference

English | [한국어](../ko/CASEBOOK.md) · [Home](../../README.md)

[New here?](START_HERE.md) · [Plain-language glossary](GLOSSARY.md)

The cases build tools for execution diagnosis, cost measurement, numerical checks and model-change evaluation. Start with [Case007](#case-007) for matrix slicing or [Case006](#case-006) for answer comparisons, then follow the earlier investigations. Each uses its own model, inputs and protocol.

<a id="case-001"></a>
## 001 — Can a kernel-selection change unblock execution?

<!-- claims: c001-runtime -->
A model cannot benefit from a supported fallback if the engine selects an implementation that fails first. On RTX 5080 / SM120, the original vLLM 0.29.0 W8A8 selection accepted CUTLASS despite that dispatcher's unsupported INT8 path.

The reproducer applied the exact Python runtime guard from hclsys's PR to the official wheel, then ran the same RedHatAI Qwen2.5-0.5B W8A8 checkpoint. Before the change, initialization exited with code 1. Afterward, generation exited with code 0 and returned “The capital of France is Paris.” Inspection found 96 INT8 modules selecting `TritonInt8ScaledMMLinearKernel`, with INT8 weights on CUDA. The count describes loaded modules, not 96 dynamic GPU kernel calls.

The before/after harness retains environment locks and execution evidence. The guard and Triton implementation are upstream work. A replay reused the existing environment and checkpoint on the same PC; it was not another person's fresh installation. One short generation verifies this compatibility path, not throughput, general task quality or a full PR build.

[Recorded result and commands](../../cases/001-sm120-int8-fallback/README.md) · [Original execution evidence](../../cases/001-sm120-int8-fallback/evidence/original.json) · [Recorded upstream validation comment](https://github.com/vllm-project/vllm/pull/54316#issuecomment-5614091211)

<a id="case-002"></a>
## 002 — What does an official FP8 checkpoint change?

<!-- claims: c002-quality c002-cost -->
A smaller deployment artifact is useful only in the context of both resource use and task performance. This comparison used official BF16 and FP8 Qwen3-4B-Instruct-2507 releases on synthetic Korean document extraction.

Structured facts and confirmed updates determined the correct answer (gold) before inference. A document was correct only if owner, task, date and amount all matched. On the first quality round, BF16 scored 31/100 and FP8 32/100. The paired FP8-minus-BF16 difference was +1 percentage point, with a 95% interval of −2 to +5 points. That does not establish improvement or equivalence.

With the same 1 GiB BF16 KV-cache budget, recorded whole-device sampled peaks were about 11.62 GiB and 8.21 GiB. FP8 requests generally completed sooner, but output lengths differed; this is not isolated decode speed. Device totals include background allocations.

The project built fact-derived data, strict scoring, a streaming client and paired measurement/analysis. Three timing rounds reuse the same 100 documents, not 300 independent quality examples. Both models' extraction accuracy was low, despite valid JSON. The comparison covers those official checkpoint and runtime bundles.

[Results](../../cases/002-bf16-fp8-document-extraction/README.md) · [Aggregate evidence](../../cases/002-bf16-fp8-document-extraction/results/aggregate.json) · [Methods and reproduction](../../cases/002-bf16-fp8-document-extraction/METHODS.md)

<a id="case-003"></a>
## 003 — How does the reference change a numerical conclusion?

<!-- claims: c003-reference -->
An approximation may stay close to a baseline using the same scale, while a baseline using another scale rule achieves lower error. This audit separates code mapping from the choice of comparison scale on real Qwen3-0.6B attention traces.

An independent equation implementation kept Q/K/V, masking and FP32 arithmetic common across methods. EFQ-Mean's median relative output error ranged from 11.49% to 14.49% across the three lengths. The development-selected setting matched published EFQ-MMLU and stayed near nearest rounding under the same EFQ scale in aggregate. Headroom-scale nearest had lower median and p95 error at every length. Calibrated EFQ's median error was 1.133×, 1.179× and 1.148× that headroom reference at lengths 512, 2048 and 4096.

The project implemented and checked the equations, validated trace capture and independently recalculated evidence. The method is Han and coauthors' work. Each length's 192 head units come from 16 documents, three layers and four heads; they are dependent measurements. The screen's denominator was same-scale nearest, not the stronger headroom reference. This FP32 numerical simulation measured neither packed-FP4 hardware speed nor downstream model quality.

[Findings](../../cases/003-efq-softmax-numerical-audit/README.md) · [Scale and implementation definitions](../../cases/003-efq-softmax-numerical-audit/METHODS.md) · [Independent decomposition](../../cases/003-efq-softmax-numerical-audit/results/comparison_decomposition.json)

<a id="case-004"></a>
## 004 — Is the complete attention call faster?

<!-- claims: c004-cost -->
Kernel-only timing can hide the cost of preparing low-precision operands. The comparison starts with GPU-resident BF16 Q/K/V and ends with BF16 output, including smoothing, quantization, conversions and the wrapper.

Official SageAttention INT8-QK/FP8-PV was compared with a development-selected, verified fused BF16 SDPA path. On real Qwen3-0.6B layer-13 inputs, complete-operator speedups were 0.373×, 1.465× and 2.100× at lengths 512, 2048 and 4096. A ratio below one means slower. All three lengths exceeded the fixed local-error screen: median ≤1% and p95 ≤3%. Longer calls could be faster without qualifying under the combined rule.

The project built adapters, runtime checks, paired complete-cost timing and sampled-query error analysis. It measured full-query calls, but fidelity used 32 query positions per head and all their causal-valid keys. Sixteen documents supply dependent head/length observations. The separate synthetic H8/H8 sweep is not the Qwen H16/H8 geometry. These are operator timings, not whole-model latency. SageAttention supplies the kernels, and the error screen is a local engineering choice, not a downstream accuracy guarantee.

[Design and results](../../cases/004-low-precision-attention-break-even/README.md) · [Cost and error interpretation](../../cases/004-low-precision-attention-break-even/ANALYSIS.md) · [Recorded selection table](../../cases/004-low-precision-attention-break-even/results/selection_table.json)

<a id="case-005"></a>
## 005 — Can a different setting improve the trade-off?

<!-- claims: c005-stop c005-tradeoff c005-validity -->
After Case004, the next bounded question was whether a small set of precision configurations could lower local error while retaining complete-call speed. The development test used eight synthetic documents at Qwen3-0.6B layer 13, causal L4096.

Three changed settings completed the real-input comparison; V3 was excluded by a small validity fixture. No changed setting met median error ≤1%, p95 ≤3% and speedup ≥1.50× together. The result remains **STOP_DEV_SCREEN**: no finalist and no fresh confirmation.

V4 reduced median error from A_PUBLIC's 2.294% to 1.958%, while complete-call speedup fell from 2.110× to 1.338×. It is an observed fidelity–cost trade-off outside the original acceptance region. V1/V2 retained speed with little aggregate error reduction; accumulation and V-range policy changed as bundles, not isolated causes. V3's smoothing path encountered zero centered ranges and nonfinite outputs in the pinned implementation's small fixture.

The project audited effective dispatch, implemented explicit adapters and retained the stop and failure trail. Subsequent threshold sensitivity and paired-unit descriptions are post-hoc interpretation, not new passing results. Local error is not task accuracy, and this finite candidate list cannot reject low-precision attention generally.

[Development result](../../cases/005-attention-precision-pareto/README.md) · [Recorded decision and measurements](../../cases/005-attention-precision-pareto/results/dev_summary.json) · [Separate post-hoc review](../../cases/005-attention-precision-pareto/POSTHOC_THRESHOLD_REVIEW.md)

<a id="case-006"></a>
## 006 — When do scores and individual choices diverge?

<!-- claims: c006-native c006-readout c006-limits c006-timing c006-implementation -->
Similar aggregate accuracy can hide gains, regressions and changes between wrong answers. This study compares each candidate with BF16 and an independently computed correct answer (gold). A regression means BF16 answered correctly and the candidate did not. B is the original BF16 baseline. A_PUBLIC and V4 name two low-precision settings applied to one attention operation in the same model; model weights remain BF16. A_PUBLIC uses INT8 QK / FP8 PV; V4 uses INT8 QK / FP16 PV.

A scoped intervention replaces only zero-based layer 13's square prompt-prefill attention in BF16 Qwen3-0.6B. Other layers and decode stay BF16. Exact input IDs, full-output validity checks and a fixed one-token option interface constrain the comparison. On 192 standard scenarios, B/A_PUBLIC/V4 were correct on 104/106/108. A_PUBLIC changed 5 of 192 choices: two gains and three different wrong answers. V4 changed 8 of 192: four gains and four different wrong answers. Neither regressed in that set; the separate BF16-conditioned 46-scenario stress set contained two A_PUBLIC regressions and one V4 regression. For both settings, the 95% interval for the task-balanced mean paired change in gold-choice negative log-likelihood (NLL; candidate minus BF16) included zero. A positive change means a worse probability score for the correct choice; this does not establish equivalence.

Readout maps the final hidden state to vocabulary scores. The original output computation is the native view; the separate diagnostic computation is the shadow view. Every native standard flip involved an exact top-score tie in B or its candidate. A post-hoc FP32 output projection on the same hidden states and represented BF16 weights produced 3 of 192 flips per candidate, including newly appearing flips. Candidates are compared with B within each readout. The same 192 + 46 scenarios were reused. Tie handling is part of native behavior, and this arithmetic sensitivity check does not replace the original result or establish a better model.

The scoped adapter, paired scorer and tie-aware explorer connect each answer change to its input and output scores. BF16 code accuracy was 15/64; strict structured generation was 0/24 in every arm. Complete model-prefill speedups were only 1.0041× and 1.0014×, not Case004's operator ratios. **COMPLETED_CONTROLLED_STUDY** describes completed evidence; **NOT_ASSESSED** describes deployment.

[Original results and source navigation](../../cases/006-attention-decision-stability/README.md) · [Separate readout methods and evidence](../../cases/006-attention-decision-stability/supplemental/readout-ties-v1/README.md) · [Open the recorded explorers](GETTING_STARTED.md#offline-explorers)

<a id="case-007"></a>
## 007 — Does joint importance select a better smaller MLP?

<!-- claims: c007-transfer c007-quality c007-random -->
Deleting groups together can produce a different error from adding their separate importance scores. Case007 decomposes the SwiGLU feed-forward block at Qwen3-0.6B layer 13 into 16 contiguous channel groups. INDEPENDENT uses individual importance; PAIRWISE includes signed cross-group terms. It selects on 96 calibration prompts, validates on 48 development prompts, and physically slices gate/up rows and down columns before evaluation on 192 held-out prompts. The deletion budgets are 4/16 (25%) and 8/16 (50%); prompts have 512 tokens, with local error evaluated at 32 fixed positions.

At 25%, mean local relative error fell from 29.8291% to 29.6288%. The paired change was −0.200277 percentage points, with a 95% interval [−0.284779, −0.114472]; 141/192 prompts had lower error. At 50%, both selectors produced the same removal set, weights and outputs. The overall frozen two-budget decision remains **COMPLETED_NO_CLEAR_TRANSFER**. Both selectors had lower mean held-out local error than each of the 20 frozen random sets per budget. This is a descriptive local reference, not a claim of global optimality or random-model quality.

At 25%, PAIRWISE preserved more choices of the unpruned BF16 baseline B: 12/192 changes versus INDEPENDENT's 48/192, and lower full-vocabulary KL(B ∥ candidate). But gold-choice NLL, a probability loss for the independently computed correct answer, was better for INDEPENDENT. The separately labelled post-hoc PAIRWISE−INDEPENDENT NLL difference was +0.069860 nats [0.025925, 0.111997]; positive is worse. Baseline fidelity and gold quality are different objectives.

The work connects group accounting, exhaustive selection, real matrix slicing and held-out checks. Local MLP speedups were 1.176–1.412×; every whole-model prefill speedup interval included 1. One model/layer, synthetic English tasks and no retraining bound the result; deployment is **NOT_ASSESSED**. HOPE motivates interaction-aware deletion in a separate MoE setting; the dense-group objective follows directly from this MLP decomposition.

[Reviewed results and sources](../../cases/007-interaction-aware-mlp-pruning/README.md) · [Methods and frozen scope](../../cases/007-interaction-aware-mlp-pruning/METHODS.md) · [Package and explorer](GETTING_STARTED.md#case007-explorer)

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
