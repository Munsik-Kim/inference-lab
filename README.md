# Inference Lab

English | [한국어](README.ko.md)

Inference Lab connects model compression, numerical diagnostics and answer comparisons to recorded measurements on an RTX 5080. It includes tools that slice a smaller Qwen MLP, compare model outputs input by input, recalculate selection decisions on CPU, and explore results in a browser.

[Explore results](docs/en/GETTING_STARTED.md#local-showcase) · [Inspect the implementation](docs/en/PORTFOLIO.md#code-tour) · [Replay a selection](docs/en/GETTING_STARTED.md#cpu-selector)

## Case 007: select channels, build a smaller MLP

<!-- claims: c007-transfer c007-quality -->
The compression pipeline measures 16 channel groups in one Qwen3-0.6B MLP, chooses groups to delete, then slices the gate/up rows and down columns. INDEPENDENT scores groups separately; PAIRWISE includes their signed interactions. Both use the same deletion budget and calibration inputs, followed by 192 held-out prompts.

At 25% deletion, PAIRWISE lowered mean local reconstruction error from 29.8291% to 29.6288%. At 50%, both selectors produced the same smaller module. The frozen two-budget result is **COMPLETED_NO_CLEAR_TRANSFER**. PAIRWISE stayed closer to baseline choices at 25%, while INDEPENDENT gave better gold-choice negative log-likelihood (NLL), a probability loss for the independently calculated correct answer. Lower NLL is better.

The smaller **one-layer MLP** ran 1.176–1.412× faster locally; every whole-model prefill speedup interval included 1. The [comparison screen](docs/en/GETTING_STARTED.md#local-showcase) connects each input to removed groups, local error and answer transitions. [Study and measurements](cases/007-interaction-aware-mlp-pruning/README.md) · [Matrix surgery](cases/007-interaction-aware-mlp-pruning/src/surgery.py).

## Case 006: follow scores and choices through an intervention

<!-- claims: c006-native c006-readout c006-limits -->
A scoped adapter changes only layer 13's prompt-prefill attention. B is the original BF16 baseline; A_PUBLIC and V4 are two low-precision attention settings in the same BF16-weight model. The viewer compares correct-answer probabilities, choices and regressions—answers B got right that a candidate lost.

On 192 standard scenarios, A_PUBLIC changed **5 of 192** choices and V4 changed **8 of 192**. There were no regressions in that set; the separate 46-item selected stress set recorded two and one, respectively. Both task-balanced mean paired NLL-change 95% intervals included zero (candidate minus B; positive is worse). This does not establish equivalence.

A post-hoc readout diagnostic recomputed the final vocabulary scores from the same hidden states and represented BF16 weights. Every native standard flip involved an exact top-score tie in B or its candidate. The separate FP32 head produced **3 of 192** flips per candidate, including new flips. The screen keeps these reused-input views separate. B's code score was **15/64**; strict structured generation was **0/24** for each arm. [Native study](cases/006-attention-decision-stability/README.md) · [Readout diagnostic](cases/006-attention-decision-stability/supplemental/readout-ties-v1/README.md).

## Seven connected questions

| Case | Tool and observation |
|---|---|
| [001 — Can the model execute?](docs/en/CASEBOOK.md#case-001) | A before/after runner verified an upstream SM120 selection guard. |
| [002 — What changes with FP8?](docs/en/CASEBOOK.md#case-002) | Paired extraction tests measured resource savings and low task accuracy. |
| [003 — Which numerical reference?](docs/en/CASEBOOK.md#case-003) | An equation audit separated code mapping from scale choice. |
| [004 — What does the full call cost?](docs/en/CASEBOOK.md#case-004) | Complete attention timing included operand preparation and conversion. |
| [005 — Which precision–cost trade-off?](docs/en/CASEBOOK.md#case-005) | A bounded screen retained measured trade-offs and STOP_DEV_SCREEN. |
| [006 — Which individual answers change?](docs/en/CASEBOOK.md#case-006) | Paired scores and ties expose changes behind aggregate accuracy. |
| [007 — Which MLP groups to remove?](docs/en/CASEBOOK.md#case-007) | Selection, physical slicing and held-out checks form one compression study. |

These are related engineering questions with different protocols, not a single performance curve.

## Use the tools and inspect the contribution

The [getting-started guide](docs/en/GETTING_STARTED.md) covers the bilingual local showcase, preserved English explorers, a small CPU selection replay and full scalar audits. Viewing results needs no GPU. The [portfolio guide](docs/en/PORTFOLIO.md) offers a code tour and a 60–90 second demonstration script.

The project implements comparison harnesses, model surgery, validity checks, analysis and evidence viewers around Qwen, Transformers, PyTorch, vLLM and SageAttention. [Attribution](docs/en/PORTFOLIO.md#contribution-and-reuse) credits the upstream methods and kernels. OpenAI Codex assisted implementation, execution, analysis and documentation. Project code uses [Apache-2.0](LICENSE); upstream terms remain applicable.

This work supports technical discussions about model-change evaluation, local inference diagnosis and reproducible analysis tools. [Public technical questions](https://github.com/Munsik-Kim/inference-lab/issues) can refer to a case and public input ID. Results are scoped to the tested device and synthetic tasks; deployment remains **NOT_ASSESSED**.
