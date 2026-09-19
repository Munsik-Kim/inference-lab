# Inference Lab

English | [한국어](README.ko.md)

Inference Lab tests what changes when we try to make AI models lighter or faster: how they run, how long their calculations take, and which answers they choose. Local experiments on an RTX 5080 pair tools for changing and comparing models with recorded results you can inspect. Each case connects a summary to individual examples, code and reproduction instructions.

[New here?](docs/en/START_HERE.md) · [Explore recorded results](https://munsik-kim.github.io/inference-lab/en/index.html) · [See what was built](docs/en/PORTFOLIO.md)

## Three questions

1. **Does it run?** Check whether the model follows a working execution path on the tested GPU and software.
2. **Is it actually faster?** Measure the core operation, the call including data preparation, and the whole model separately.
3. **What happens to the answers?** Compare the probability assigned to the correct answer and the choice made for each question, alongside overall accuracy.

## How the questions developed

Execution (001) → smaller number formats (002–003) → operation costs (004–005) → model answers (006) → smaller model structure (007).

![Reading route: run, change, measure, compare, verify](docs/assets/inference-reading-path.svg)

*This is a conceptual reading route. The cases have different models, inputs and protocols; the arrows are not measured results from one continuous experiment.*

## Two examples to explore

### Case 007 — Which parts of a calculation can we remove?

<!-- claims: c007-transfer c007-quality -->
An MLP is a large block that transforms information inside the model. The compression tool divides one block into 16 channel groups and compares selecting groups individually with selecting them using their interactions. It then cuts the `gate_proj`, `up_proj` and `down_proj` matrices to build a smaller block.

With 25% of groups removed, interaction-aware selection kept the block slightly closer to its original output on 192 inputs unused for selection. At 50%, both methods chose the same structure. The smaller **one-layer MLP** ran 1.176–1.412× faster; whole-model timings did not establish a speedup. Staying closer to the original model also differed from giving the correct answer the best probability score.

Technical terms: Qwen3-0.6B · SwiGLU · INDEPENDENT / PAIRWISE · structured pruning.

[Results and exact metrics](docs/en/CASEBOOK.md#case-007) · [Matrix-slicing code](cases/007-interaction-aware-mlp-pruning/src/surgery.py) · [Open the explorer](https://munsik-kim.github.io/inference-lab/en/case007.html)

### Case 006 — Can similar accuracy hide different answers?

<!-- claims: c006-native c006-readout c006-limits -->
A comparison tool replaces one attention operation—the calculation that combines information from the input—with lower-precision arithmetic. It tracks the choice among four answers to the same question before and after the change. It shows newly correct answers, lost correct answers and changes between wrong answers.

Two low-precision settings changed **5 of 192** and **8 of 192** standard-set choices. Neither lost a previously correct answer in that set; a separate selected stress set did. A follow-up on the same inputs examined equal top scores and the final calculation that turns internal states into word scores. These records describe one layer of one model on synthetic questions.

Technical terms: BF16 baseline · A_PUBLIC / V4 · decision flip · readout sensitivity.

[Results, limitations and readout diagnostic](docs/en/CASEBOOK.md#case-006) · [Scoped intervention code](cases/006-attention-decision-stability/src/intervention.py) · [Open the explorers](https://munsik-kim.github.io/inference-lab/en/case006.html)

## Seven questions, with working tools and recorded outcomes

| Question | What the case provides |
|---|---|
| [001 — What if a model will not run on the GPU?](docs/en/CASEBOOK.md#case-001) | A before/after check of an upstream execution-path fix. Terms: SM120, vLLM, W8A8. |
| [002 — Do fewer bits save memory and time?](docs/en/CASEBOOK.md#case-002) | Official model comparisons on document extraction: lower resource use, but low task accuracy. Terms: BF16, FP8. |
| [003 — How close is a simpler calculation?](docs/en/CASEBOOK.md#case-003) | Numerical error checks with different comparison scales. Terms: EFQ-Softmax, FP32 simulation. |
| [004 — Does a faster core operation make a cheaper call?](docs/en/CASEBOOK.md#case-004) | Attention timing with preparation included; longer inputs gained speed while output error exceeded the fixed limits. Terms: kernel, complete-call cost. |
| [005 — Can a setting balance speed and error?](docs/en/CASEBOOK.md#case-005) | Measured trade-offs; no tested changed setting met both requirements, so fresh confirmation stopped. Terms: precision, local error. |
| [006 — Which individual answers change?](docs/en/CASEBOOK.md#case-006) | Paired answer scores, choices and equal-top-score diagnostics. Terms: NLL, top-score tie. |
| [007 — Which groups should a smaller block keep?](docs/en/CASEBOOK.md#case-007) | Group selection, actual matrix slicing and evaluation on unused inputs. Terms: MLP, pruning. |

## Go deeper

Read the five-minute [introduction](docs/en/START_HERE.md) or look up a term in the [glossary](docs/en/GLOSSARY.md). The [Casebook](docs/en/CASEBOOK.md) leads to methods, source and exact results. [Getting started](docs/en/GETTING_STARTED.md) explains ZIP downloads, local HTML and CPU checks. The explorers show saved measurements and need no GPU; GitHub's HTML source view displays code, not the running screen.

## Contribution and sources

The project builds comparison runners, model-slicing adapters, numerical checks and evidence viewers around public models and execution libraries. The [portfolio](docs/en/PORTFOLIO.md#contribution-and-reuse) credits Qwen, Transformers, PyTorch, vLLM, SageAttention and the upstream methods and fixes. OpenAI Codex assisted implementation, execution, analysis and writing. [Apache-2.0](LICENSE) applies to project materials; case notices retain upstream terms. These studies support work on model-change evaluation and inference diagnostics; deployment suitability remains unevaluated.
