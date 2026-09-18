# Inference Lab

English | [한국어](README.ko.md)

Inference Lab compares local LLM execution, the full cost of low-precision operations, and changes in model answers on an RTX 5080 16GB. It combines reproducible case studies with tools for recalculating results and inspecting individual evidence. The contribution is an evaluation system around existing models and kernels: controlled comparisons, retained failures, and explanations tied to measurements. Reading the reports and exploring the recorded results requires no GPU.

[Casebook](docs/en/CASEBOOK.md) · [Explore the results](docs/en/GETTING_STARTED.md) · [Implementation and contribution](docs/en/PORTFOLIO.md)

## Why these comparisons matter

A faster kernel, a faster call including data conversion, a faster model, and a correct answer are different outcomes. A useful comparison must identify which one it measures. These seven cases move from execution compatibility to operator costs and model behavior, using different models, inputs and protocols. They form a reading path, not one continuous performance curve. Each case keeps its own settings, evidence and limits.

## Start with Case006: when similar scores hide different answers

<!-- claims: c006-native c006-readout c006-limits -->
Case006 asks whether changing one attention operation can alter individual decisions even when aggregate accuracy looks similar. A scoped adapter replaced only zero-based layer 13's prompt-prefill attention in Qwen3-0.6B. All other layers and subsequent token decoding stayed BF16. B is the original BF16 baseline. A_PUBLIC and V4 name two low-precision settings applied to one attention operation in the same model; model weights remain BF16. Synthetic retrieval, comparison and code questions had independently computed correct answers (gold); the BF16 model was a comparator, not the answer key.

On 192 standard scenarios, A_PUBLIC changed **5 of 192** choices relative to BF16, and V4 changed **8 of 192**. The changes included gains and different wrong answers, with no standard-set regressions (BF16 correct, candidate wrong). A separately selected 46-scenario stress set did contain regressions. For both settings, the 95% interval for the task-balanced mean paired change in gold-choice negative log-likelihood (NLL; candidate minus BF16) included zero. A positive change means a worse probability score for the correct choice; this does not establish equivalence. [Original results and scope](cases/006-attention-decision-stability/README.md).

![Paired changes in gold-answer scores on the standard set](cases/006-attention-decision-stability/figures/01_score_changes.png)

*Gold-choice NLL changes are candidate minus BF16; positive values mean less probability on the correct choice. Each dot is a scenario–candidate pair; the intervals reported above describe the task-balanced mean paired change. Score changes and answer correctness are distinct.*

Readout is the final computation mapping a model hidden state to vocabulary scores. The post-hoc supplement distinguishes the original (native) output computation from a separate diagnostic (shadow) output computation. It found that every native standard-set choice change involved a top-score tie in the baseline or candidate. Evaluating the same final hidden states and represented BF16 weights through a separate FP32 output head produced **3 of 192** changes for each candidate, including new changes. Each candidate was compared with B under the same readout. It reused the same 192 standard and 46 stress scenarios; it is not fresh confirmation or an improved-model claim. [Separate readout diagnostic](cases/006-attention-decision-stability/supplemental/readout-ties-v1/README.md).

The evidence viewer connects exact questions, gold labels, option scores and outcomes. The limits remain visible: BF16 answered only **15/64** code questions correctly, and all arms had **0/24** strictly valid structured outputs in the separate generation diagnostic. This is a completed controlled study, with deployment **NOT_ASSESSED**.

## Choose a reading path

1. **Understand the whole project:** [Casebook](docs/en/CASEBOOK.md), seven questions and what each comparison established.
2. **Inspect the engineering contribution:** [Portfolio guide](docs/en/PORTFOLIO.md), linked to implementation, analysis and attribution.
3. **Check results yourself:** [Getting started](docs/en/GETTING_STARTED.md), from the current ZIP and offline explorers to CPU reanalysis and separate GPU reproduction.

## Seven questions, seven bounded results

| Case and question | What was observed |
|---|---|
| [001 — Can the model start?](docs/en/CASEBOOK.md#case-001) | An upstream selection guard enabled generation where the original W8A8 path failed on SM120. |
| [002 — What changes with an official FP8 model?](docs/en/CASEBOOK.md#case-002) | Memory and request time decreased, but document extraction accuracy was low for both releases. |
| [003 — How faithful is a softmax approximation?](docs/en/CASEBOOK.md#case-003) | The interpretation changed with the scale-matched versus headroom-scale reference. |
| [004 — Does the whole operator save time?](docs/en/CASEBOOK.md#case-004) | Some longer Qwen calls were faster; none of the three lengths passed the fixed local error screen. |
| [005 — Can settings improve the trade-off?](docs/en/CASEBOOK.md#case-005) | Lower-error FP16 PV cost more time; no changed setting qualified, so fresh confirmation did not run. |
| [006 — Do individual answers change?](docs/en/CASEBOOK.md#case-006) | Paired decisions exposed changes that aggregate accuracy alone did not describe. |
| [007 — Which MLP groups should be removed?](docs/en/CASEBOOK.md#case-007) | Pairwise selection slightly reduced local error at 25%; 50% selections were identical, and closer baseline fidelity did not give the best gold NLL. |

<!-- claims: c007-transfer c007-quality -->
Case007 adds actual structural compression: it slices one MLP, the model's feed-forward block, at fixed deletion budgets. Its result remains **COMPLETED_NO_CLEAR_TRANSFER** under the frozen two-budget rule; whole-model prefill acceleration was not established. [English study](cases/007-interaction-aware-mlp-pruning/README.md) · [Download and inspect](docs/en/GETTING_STARTED.md#case007-explorer).

## Contribution, reuse and scope

The project implements comparison harnesses, input/gold construction, runtime checks, scalar audits and recorded-evidence viewers. Qwen and Red Hat AI supply checkpoints; vLLM, PyTorch, Transformers and SageAttention supply execution components. The Case001 guard is hclsys's work, and EFQ-Softmax is Han and coauthors' method. The [portfolio attribution](docs/en/PORTFOLIO.md#contribution-and-reuse) links the original notices and separates those contributions.

OpenAI Codex assisted code, local checks, analysis and writing. Separate calculation paths check consistency; they do not constitute third-party GPU replication. These are device- and task-scoped investigations, including negative results, not a new quantizer, general SDK or deployed product. Project materials use [Apache-2.0](LICENSE); case notices retain upstream attribution and applicable licenses.
