# Engineering contribution and evidence

English | [한국어](../ko/PORTFOLIO.md) · [Home](../../README.md)

## A 30-second introduction

<!-- claims: c006-native c006-readout -->
Inference Lab investigates why successful execution, faster operators and stable model answers need different tests. It implements controlled comparisons, tasks with independently computed correct answers (gold), runtime and validity checks, and CPU-auditable reporting around existing inference components. B is the original BF16 baseline. A_PUBLIC and V4 name two low-precision settings applied to one attention operation in the same model; model weights remain BF16. In its representative Qwen study, A_PUBLIC changed 5 of 192 standard choices and V4 changed 8 of 192, including gains and different wrong answers. Readout maps the final hidden state to vocabulary scores. A same-input follow-up separates the original (native) output computation from a diagnostic (shadow) computation. It showed that native ties matter to that description without replacing the native result. The [casebook](CASEBOOK.md#case-006) and [offline evidence guide](GETTING_STARTED.md#offline-explorers) connect the observations to exact items. These are narrow synthetic studies, not a model-quality guarantee.

## Capabilities tied to working artifacts

- **Diagnose an execution path.** Case001 separates a selector accepting an unsupported CUTLASS implementation from the existing Triton fallback. [Before/after runner](../../cases/001-sm120-int8-fallback/run.py) and [recorded validation](../../cases/001-sm120-int8-fallback/README.md) distinguish the project's reproducer from the upstream fix.
- **Define a fair cost boundary.** Case004 measures required preprocessing and output conversion inside a complete operator call, with kernel-only timing labelled separately. [Methods](../../cases/004-low-precision-attention-break-even/METHODS.md) and [backend adapters](../../cases/004-low-precision-attention-break-even/src/backends.py) expose that contract.
- **Compare numerical approximations without moving the gate.** Case003 separates mapping and scale effects; Case005 preserves a failed development gate while showing a real trade-off. [Numerical audit](../../cases/003-efq-softmax-numerical-audit/ANALYSIS.md), [development analysis](../../cases/005-attention-precision-pareto/ANALYSIS.md) and [separate post-hoc review](../../cases/005-attention-precision-pareto/POSTHOC_THRESHOLD_REVIEW.md) retain their different evidence scopes.
- **Control intervention inside a native model.** Case006 intercepts a single prefill layer, restores the original path and validates full outputs rather than sampled logits alone. [Scoped intervention](../../cases/006-attention-decision-stability/src/intervention.py) and [validity checks](../../cases/006-attention-decision-stability/src/validity.py) make the boundary inspectable.
- **Investigate score interpretation.** The readout supplement separates exact ties, gold transitions and candidate-versus-B comparisons within each readout condition; native and FP32 remain separate views. [Methods](../../cases/006-attention-decision-stability/supplemental/readout-ties-v1/METHODS.md) and [scalar checker](../../cases/006-attention-decision-stability/supplemental/readout-ties-v1/scripts/verify_scalar.py) keep this post-hoc diagnosis distinct from fresh evaluation.
- **Deliver inspectable evidence.** Exact inputs, source identities, paired records and local HTML connect aggregate tables to individual observations. [Input manifest](../../cases/006-attention-decision-stability/configs/eval_manifest_freeze_v1.json), [independent scalar calculation](../../cases/006-attention-decision-stability/scripts/audit_study.py) and [explorer instructions](GETTING_STARTED.md#offline-explorers) show the route.

## Structural compression: a second implementation example

<!-- claims: c007-transfer c007-quality -->
Case007 converts selected channel groups into a genuinely smaller dense MLP, using scoped replacement and masked-versus-sliced checks. At 25%, PAIRWISE's held-out mean local error was 29.6288%, versus 29.8291% for INDEPENDENT. It stayed closer to the unpruned BF16 baseline but did not give the best gold-choice NLL. At 50% the selectors produced identical modules, so the frozen overall result remains **COMPLETED_NO_CLEAR_TRANSFER**; whole-model prefill acceleration was not established. This demonstrates selection, structural model changes and transfer evaluation without equating baseline preservation with task quality. [Casebook](CASEBOOK.md#case-007) · [Model surgery source](../../cases/007-interaction-aware-mlp-pruning/src/surgery.py) · [Recorded explorer](GETTING_STARTED.md#case007-explorer).

Qwen/Transformers supply the model; HOPE provides conceptual motivation in a separate MoE expert-pruning setting. The group's signed contribution identity is elementary, not a new pruning algorithm or a HOPE reproduction. [Case007 attribution and reused work](../../cases/007-interaction-aware-mlp-pruning/NOTICE.md). The same Codex-assistance and scalar-versus-GPU verification distinctions below apply.

<a id="contribution-and-reuse"></a>
## Contribution and reuse

<!-- claims: attribution -->
The project's contribution is the harness, experiment controls, evidence collection, numerical/behavioral analysis and presentation. Qwen and Red Hat AI supply checkpoints. vLLM supplies the serving engine; PyTorch and Transformers supply numerical operators and model integration; SageAttention supplies the low-precision kernels. hclsys authored the Case001 guard. Han and coauthors developed EFQ-Softmax. These are not newly invented kernels or quantizers in this repository.

[Case001 attribution](../../cases/001-sm120-int8-fallback/NOTICE.md), [Case003 attribution](../../cases/003-efq-softmax-numerical-audit/NOTICE.md) and [Case006 attribution](../../cases/006-attention-decision-stability/NOTICE.md) identify original work and applicable licenses. The repository [Apache-2.0 license](../../LICENSE) does not replace upstream model terms. OpenAI Codex assisted implementation, local execution, tests, analysis and documentation. Separate CPU calculation paths establish consistency of retained records, not third-party GPU replication or an independent human review.

## Where this work is relevant

**Research and evaluation engineering:** specifying comparable conditions, maintaining independent gold, recording failures, checking uncertainty and distinguishing what a result measures.

**Inference diagnostics and analysis-tool collaboration:** adapters around existing runtimes, provenance-aware comparison reports, CPU reanalysis and an offline evidence viewer. These are collaboration topics supported by the recorded implementations. Findings remain scoped to the tested models, device and tasks; deployment is not assessed.

## Three evidence-backed introduction lines

<!-- claims: c006-implementation c004-cost c005-stop -->
- Built a controlled Qwen3-0.6B layer-13 prefill comparison that distinguishes baseline disagreement from gold-grounded gains and regressions. [Scope and implementation](../../cases/006-attention-decision-stability/METHODS.md).
- Compared complete low-precision attention cost with fused BF16 and kept local-error rejection visible even at faster measured lengths. [Operator results](../../cases/004-low-precision-attention-break-even/README.md).
- Preserved Case005's **STOP_DEV_SCREEN** while exposing the measured precision–cost trade-off and separate post-hoc interpretation. [Development decision](../../cases/005-attention-precision-pareto/README.md).

## Demonstrate the work in three steps

1. Read the standard result table and its independent scenario denominator. Name the intervention before showing any speed number.
2. In the original explorer, inspect one item by ID: gold, baseline/candidate scores and outcome type. Then switch to selected stress and explain the selection rule.
3. In the separate supplement, compare native ties and shadow readout for the same ID. Return to [limitations](../../cases/006-attention-decision-stability/LIMITATIONS.md): weak code utility, failed structured generation, small model-prefill effect and no deployment assessment remain part of the result.

The [getting-started guide](GETTING_STARTED.md) explains local viewing and CPU verification. No hosted inference service is part of this demonstration.
