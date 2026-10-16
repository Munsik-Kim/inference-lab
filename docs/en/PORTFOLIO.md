# Tools, implementation and technical collaboration

English | [한국어](../ko/PORTFOLIO.md) · [Home](../../README.md)

[New here?](START_HERE.md) · [Plain-language glossary](GLOSSARY.md)

[Practical model guides](LLM_GUIDE.md) connect verified Qwen3-4B execution paths with quality, failures and mitigation. English summaries link to the complete Korean detail.

## A 30-second introduction

<!-- claims: c008-tracks c008-artifact-implementation c007-implementation -->
DIOVA modifies and runs public models under a limited GPU budget, checking efficiency alongside answers and actual tool-task results. Its tools connect smaller stored artifacts with fresh-process execution and item-level losses and gains.

[Featured work and runnable tools](https://munsik-kim.github.io/inference-lab/en/index.html#featured-stories) · [Compare your own records](../../packages/diova-compare/examples/compare/README.md)

## Featured A · Smaller artifacts through actual requests

Connect Case008 Q's **GPTQ conversion → artifact inspection → fresh-process reload** to Case009's **request cost and quality comparison**. [Conversion](../../tools/modelpack/quantized.py) · [Artifact checks](../../tools/modelpack/qartifact.py) · [Serving runner](../../cases/009-q-serving-quality/scripts/serving.py).

**8.045 GB → 2.652 GB, about 67.0% smaller** describes Case008 Q's Qwen3-4B weight files. Case009 is a separate serving experiment. [Case008 Q](https://munsik-kim.github.io/inference-lab/en/case008.html#track-q) → [Case009](https://munsik-kim.github.io/inference-lab/en/case009.html#serving). Case008 R uses a separate pruned Qwen3-0.6B model.

**Technical judgment — current design:** file size alone cannot expose reload or runtime-path failures, so the implementation checks a new process too. BF16/W4 are paired within graph and within eager modes to separate launch policy from weight changes. Evidence: [boundary tests](../../cases/008-build-reconstruct-reload/tests/test_boundaries.py) and [request protocol](../../cases/009-q-serving-quality/configs/serving_protocol.json). Other-device gains and abnormal Case009 quality-process exits remain unresolved. This interprets the current design without inventing the author's historical intent.

## Featured B · diova-compare on matched questions

Validate shared input/gold contracts, separate lost answers, gains and different wrong answers, and retain source exit status when supplied. [Runnable synthetic example](../../packages/diova-compare/examples/compare/README.md) · [Schema and CLI](../../packages/diova-compare/README.md) · [Core](../../packages/diova-compare/src/diova_compare/core.py) · [Tests](../../packages/diova-compare/tests/test_compare.py).

**Technical judgment — current design:** paired losses/gains expose what mean accuracy alone hides; both synthetic artifacts score 2/4 while each loses or gains an answer. Basic comparison needs no logits. Callers supply free-form correctness and provenance identifiers; these checks do not authenticate source data. Before a new task, confirm its grading and pairing rules.

## Featured C · Verify actual tool-task completion

A final success JSON could hide absent patch/validation events. Tested registered-name constraints and error feedback against executed events and changed fixture state. [Guard](../../research/agent_tool_contract_repair_20261014/guard.py) · [Event analysis](../../research/agent_tool_contract_repair_20261014/analyze.py) · [Tests](../../research/agent_tool_contract_repair_20261014/test_guard.py) · [Report](../../research/agent_tool_contract_repair_20261014/records/REPORT.md).

**Same eight previously observed DEV tasks · small sample · dequantized BF16 reference computation:** D16 changed 6/8 → 8/8, D4 0/8 → 2/8, leaving six D4 failures. New-input generalization and packed low-bit acceleration remain unverified.

**Technical judgment — current design:** final-JSON-only grading misses absent actions. Name constraints, feedback and their combination were separated; enum and both tied, so added feedback benefit was not established. New tasks and remaining D4 failures are the next decision boundary.

## Research and numerical engineering

[Case014](https://munsik-kim.github.io/inference-lab/en/case014.html#overview) tests quality compensation at identical INT4 projection bytes, separating the recorded PG19 gain from ArXiv limits. These results do not establish accuracy, speed, VRAM savings or identical whole-checkpoint size.

[All cases](CASEBOOK.md) retain Case006/007, Case008 R and Case010–012, including negative results. The code tour below provides deeper implementation evidence.

## Structure-aware loading and paired evaluation

![Actual local Case007 comparison screen, showing one fixed code input](../../presentation/screenshots/case007-comparison.png)

*Local Edge screenshot: the first fixed CODE input, 25% deletion. [Case006 readout screen](../../presentation/screenshots/case006-readout.png) shows the separately labelled same-input diagnostic. These are selected views for navigation, not aggregate performance summaries.*

R connects a ridge fit in the smaller down projection with structure-aware checkpoint storage and reload: per-layer metadata → meta skeleton → strict assignment → tied-weight and buffer restoration → fresh-process execution without the original checkpoint. Current support covers one Qwen3 MLP layer. Local SSE and gold scores remain in the performance section and original reports.

[Contribution table](../../README.md) · [Related work and implementation](../related-work/README.md) · [CPU paired comparison CLI](../../packages/diova-compare/README.md) · [Implementation interview notes (Korean)](../interview-notes.ko.md)

Case009 adds matched graph/eager serving measurements and official quality-task records using the existing Qwen 4B artifacts. The [complete curves and quality report](../../cases/009-q-serving-quality/REPORT.md) retain runtime-exit failures alongside calculated metrics. The new CLI consumes these scalars and historical records through the same versioned contract.

## Working capabilities and their evidence

- **Runtime diagnosis:** a before/after runner isolates an upstream kernel-selection guard on SM120. [Case001 runner](../../cases/001-sm120-int8-fallback/run.py).
- **Cost measurement:** attention adapters include smoothing, quantization and conversion in complete-call time. [Case004 adapters](../../cases/004-low-precision-attention-break-even/src/backends.py).
- **Numerical analysis:** reference comparisons separate EFQ mapping and scale choice, while a fixed precision screen records its stopping decision. [Case003 analysis](../../cases/003-efq-softmax-numerical-audit/ANALYSIS.md) · [Case005 analysis](../../cases/005-attention-precision-pareto/ANALYSIS.md).
- **Structural model changes:** matched row/column selection produces a smaller dense MLP, checked against a masked computation. [Case007 surgery](../../cases/007-interaction-aware-mlp-pruning/src/surgery.py).
- **Input-level evaluation:** gold-grounded transitions, score shifts and exact ties are inspectable for each scenario. [Case006 analysis](../../cases/006-attention-decision-stability/ANALYSIS.md) · [readout scalar checker](../../cases/006-attention-decision-stability/supplemental/readout-ties-v1/scripts/verify_scalar.py).
- **Analysis delivery:** source hashes connect display records to preserved evidence, and a small selector replay exposes the actual calibration objective. [Showcase builder](../../tools/showcase/build.py) · [CPU replay](../../tools/showcase/replay_selection.py).

<a id="code-tour"></a>
## Code tour

**Start with artifact creation and reload.** [`save_dense()`](https://github.com/Munsik-Kim/inference-lab/blob/9bc8b8fced8dfd5147f9bfdc67564eda04670810/tools/modelpack/artifact.py#L32) saves tensor and structure manifests; [`load_dense()`](https://github.com/Munsik-Kim/inference-lab/blob/9bc8b8fced8dfd5147f9bfdc67564eda04670810/tools/modelpack/artifact.py#L109) builds on meta, adjusts per-layer widths, assigns weights strictly and restores tied weights and rotary buffers. [`fit_ridge()`](https://github.com/Munsik-Kim/inference-lab/blob/9bc8b8fced8dfd5147f9bfdc67564eda04670810/tools/modelpack/numerics.py#L13) solves the CPU FP64 reconstruction problem. [Original storage tests](../../cases/008-build-reconstruct-reload/tests/test_core.py) reject missing tensors and incompatible structures. [The standalone CPU demo](../../tools/modelpack_demo/roundtrip.py) runs this loader in another process.

<!-- claims: c006-implementation -->
1. **Control a native attention call.** Start at [`ScopedAttention`](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/006-attention-decision-stability/src/intervention.py#L30). `route` permits the layer-13 square prefill intervention; `__exit__` restores the prior registry entry. Read the [restoration-on-exception test](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/006-attention-decision-stability/tests/test_core.py#L193).
2. **Make the matrix smaller.** [`sliced`](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/007-interaction-aware-mlp-pruning/src/surgery.py#L6) maps retained groups to gate/up rows and down columns. `masked` supplies the reference; `replace_mlp` restores the original module in `finally`. The [shape/bias/equivalence test](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/007-interaction-aware-mlp-pruning/tests/test_core.py#L62) covers that contract.
3. **Reject incomplete evidence.** [`assess`](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/006-attention-decision-stability/src/validity.py#L10) requires full-output and routing checks. Pair joins reject missing, duplicate or mismatched inputs in the [original tests](https://github.com/Munsik-Kim/inference-lab/blob/bc1da81a82bff2e2827c5b4cabc9082cb6db5484/cases/006-attention-decision-stability/tests/test_core.py#L249). The new [display checker](../../tools/showcase/check.py) compares the extracted display data with these preserved sources.

## A 60–90 second demonstration script

*Run the synthetic CPU example while presenting. This document is a script, not a recording.*

- **0–15s:** explain how equal accuracy can hide a lost correct answer.
- **15–40s:** validate both [fixture files and compare them](../../packages/diova-compare/examples/compare/README.md); label this as stored synthetic answers on CPU.
- **40–65s:** open the generated report.md and pairs.csv; inspect sum-01's loss and sum-02's gain.
- **65–90s:** open the [core](../../packages/diova-compare/src/diova_compare/core.py) and [transition/rejection/failure-preservation tests](../../packages/diova-compare/tests/test_compare.py).

Historical GPU records are a separate route: [Case008 Q](https://munsik-kim.github.io/inference-lab/en/case008.html#track-q) → [Case009 quality](https://munsik-kim.github.io/inference-lab/en/case009.html#quality). The random tiny-model [CPU save/reload demo](GETTING_STARTED.md#tiny-model-demo) remains another implementation example.

## Three introduction lines with source links

<!-- claims: c004-cost c005-stop -->
- Built a group-selection and matrix-slicing pipeline with masked-versus-sliced and held-out comparisons. [Case007 methods](../../cases/007-interaction-aware-mlp-pruning/METHODS.md).
- Implemented a single-layer Qwen intervention and paired analysis of answer probabilities, regressions, gains and ties. [Case006 methods](../../cases/006-attention-decision-stability/METHODS.md).
- Measured complete attention cost and retained **STOP_DEV_SCREEN** alongside the observed precision–cost trade-off. [Case004 cost](../../cases/004-low-precision-attention-break-even/README.md) · [Case005 decision](../../cases/005-attention-precision-pareto/README.md).

## Evaluating the tools

<!-- claims: c007-transfer c007-quality c006-native c006-readout -->
Case008 R reduced deletion-induced local squared output error by **94.1–95.4%** in one Qwen 0.6B MLP on 192 short synthetic held-out prompts. Gold-score changes differed by probability metric and repaired structure. Its measured request-time intervals included 1. Inspect the [Q quality and runtime comparison](https://munsik-kim.github.io/inference-lab/en/case008.html#q-evaluation) and the [R comparison](https://munsik-kim.github.io/inference-lab/en/case008.html#r-evaluation).

At 25% MLP deletion, PAIRWISE slightly lowered local error and preserved more baseline choices, but INDEPENDENT had better gold NLL. At 50%, the selected modules were identical; the frozen status is **COMPLETED_NO_CLEAR_TRANSFER**. In Case006, A_PUBLIC changed 5 of 192 standard choices and V4 changed 8 of 192. A same-input readout follow-up examines ties and the final vocabulary projection. These examples make both implementation choices and competing evaluation objectives visible. [Casebook](CASEBOOK.md) · [Open the local screen](GETTING_STARTED.md#local-showcase).

<a id="contribution-and-reuse"></a>
## Contribution and reuse

<!-- claims: attribution -->
The project's work is comparison execution, scoped adapters, matrix surgery, validity checks, numerical/behavioral analysis and evidence viewers. Qwen and Red Hat AI provide checkpoints; Transformers, PyTorch and vLLM provide model/runtime components; SageAttention provides low-precision kernels. hclsys authored the Case001 guard. Han and coauthors developed EFQ-Softmax. HOPE motivates interaction-aware deletion in a separate MoE expert-pruning setting; the dense-group objective follows directly from the chosen MLP decomposition.

GPTQ is the upstream quantization method; LLM Compressor and compressed-tensors implement conversion/storage, and safetensors provides the weight format. Case008 adds artifact validation, a layer-aware loader and fixed-structure ridge fitting. [Case008 attribution](../../cases/008-build-reconstruct-reload/NOTICE.md).

OpenAI Codex assisted implementation, local execution, tests, analysis and writing. [Case001](../../cases/001-sm120-int8-fallback/NOTICE.md), [Case003](../../cases/003-efq-softmax-numerical-audit/NOTICE.md), [Case006](../../cases/006-attention-decision-stability/NOTICE.md) and [Case007 notices](../../cases/007-interaction-aware-mlp-pruning/NOTICE.md) credit reused work. Project code uses [Apache-2.0](../../LICENSE); model/upstream terms remain applicable.

## Technical collaboration

The public implementations demonstrate capabilities for these tasks:

- Compare local quantization/execution settings and diagnose bottlenecks.
- Evaluate answer and tool-action regressions after model changes.
- Turn measurement, recording and reanalysis into reusable tools.


The code supports discussions about **model-change evaluation**, **local inference diagnosis** and **reproducible analysis tools and result screens**. A useful starting point is a model boundary, a public input ID and a metric to compare. [Repository issues](https://github.com/Munsik-Kim/inference-lab/issues) provide a public technical-question channel; keep credentials and private customer data out of public reports.

The evidence covers specific models and one device, with synthetic studies and the separate official tasks in Case009. Deployment is **NOT_ASSESSED**. CPU audits check retained scalars; excluded full vectors and independent GPU replication have a separate verification scope in the [reproduction guide](GETTING_STARTED.md).


## Case010 — State storage and failure persistence

Connected actual state packing, residual transport, atomic row updates and failure-aware serialization. Follow the [v2 implementation](../../cases/010-ckda-finite-precision-memory-horizon/versions/v2/source/online_v2.py), [fresh-process synthetic tests](../../cases/010-ckda-finite-precision-memory-horizon/versions/v2/tests/test_online_v2.py) and [byte-budget/readout comparison](../../cases/010-ckda-finite-precision-memory-horizon/REPORT.md). [CPU guide](GETTING_STARTED.md#case010-cpu). The [notice](../../cases/010-ckda-finite-precision-memory-horizon/NOTICE.md) separates ComplexKDA transitions and training from the project’s codec and evaluation tools.

## Case011 — Replace the readout on the same state

Three readouts share a cache, while two-tensor patches validate shape and base-checkpoint identity before fresh-process application. Read the [feature boundary](../../cases/011-frozen-state-readout-adaptation/source/adapter.py), [patch loader](../../cases/011-frozen-state-readout-adaptation/source/head_patch.py), [contract tests](../../cases/011-frozen-state-readout-adaptation/tests/test_fitting_patch.py) and [independent auditor](../../cases/011-frozen-state-readout-adaptation/analysis/audit.py) in that order.

Additional recurrent-state storage is 0 B. Gold CE improved beyond fitting support, while INT8 mean uninterrupted correct length fell on all three checkpoints. [Walkthrough](https://munsik-kim.github.io/inference-lab/en/case011.html) · [Exact results and solver status](../../cases/011-frozen-state-readout-adaptation/REPORT.md) · [CPU checks](GETTING_STARTED.md#case011-cpu). [Attribution](../../cases/011-frozen-state-readout-adaptation/NOTICE.md) distinguishes PyTorch/NumPy and ComplexKDA from the project integration.


## Case 013 — Scale-preserving INT4 and quality readback

[INT4 solver and adapter](../../cases/013-dualquant-reference-quality/source/paper_int4.py) · [Packed serialization](../../cases/013-dualquant-reference-quality/source/reference_int4.py) · [Recorded scalar audit](../../cases/013-dualquant-reference-quality/analysis/audit.py)

<!-- case014-entry -->
## Case014 — Own-trajectory compensation and full-model quality

Implemented capture, FP64 full-input fit, INT4 codec/independent reload, three-term decomposition and paired document audit, with PG19 quality gains and ArXiv limits.

[Implementation](../../cases/014-quantization-compensation/PORTFOLIO.md) · [Code](../../cases/014-quantization-compensation/followups/p2_core/source/run_p2.py) · [Tests](../../cases/014-quantization-compensation/followups/p2_core/tests/test_p2.py) · [CPU reproduction](../../cases/014-quantization-compensation/REPRODUCE.md)
