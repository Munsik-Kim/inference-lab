# DIOVA

English | [한국어](README.ko.md)

**D**eep-learning **I**nference **O**ptimization, **V**alidation & **A**nalysis

DIOVA connects model compression with inference measurement and output analysis. The project includes PyTorch model adapters, RTX 5080 measurements, and tools for exploring individual results.

**Python · PyTorch · Transformers · vLLM · NumPy**

[Capabilities](#capabilities) · [Tech stack](#tech-stack) · [Projects](#projects)

<a id="capabilities"></a>
## Capabilities

### Model modification and structural compression

Replace selected model operations and build smaller MLP modules by slicing aligned weight matrices. Check the computation and restore the original module.

Python · PyTorch · Transformers. [Attention adapter](cases/006-attention-decision-stability/src/intervention.py) · [Matrix slicing and tests](cases/007-interaction-aware-mlp-pruning/tests/test_core.py)

### GPU inference measurement and output evaluation

Compare latency, GPU memory and answer scores on matched inputs. Measure individual operations and complete model calls at separate boundaries.

vLLM · CUDA runtime · NVML · NumPy. [Request and memory runner](cases/002-bf16-fp8-document-extraction/scripts/run.py) · [Timing boundaries](cases/004-low-precision-attention-break-even/src/timing.py)

### Reproducible analysis and result explorers

Build Python tools for replaying recorded calculations and web interfaces for inspecting individual results. Automated checks link the displays to their sources.

Python · JavaScript · GitHub Actions · GitHub Pages. [CPU selector replay](tools/showcase/replay_selection.py) · [Display checks and tests](tests/showcase/test_showcase.py)

<a id="tech-stack"></a>
## Tech stack in use

| Work | Technologies and implementation |
|---|---|
| Model implementation | **Python / PyTorch / Transformers** — Inspect model objects; replace operations; slice matrices; check shapes and dtypes. |
| GPU inference integration | **vLLM / PyTorch SDPA / SageAttention / CUDA runtime / NVML** — Connect public backends, run a local server and record execution routes and resource use. |
| Numerical analysis and selection | **NumPy / Matplotlib / combinatorial search / paired statistics** — Build contribution matrices, select groups and analyze errors, scores and uncertainty. |
| Reproduction and checks | **Git / unittest / SHA256 / GitHub Actions** — Check preserved files and input pairing; recalculate scalars; run documentation and display CI. |
| Result interfaces | **HTML / CSS / JavaScript / GitHub Pages** — Build bilingual static explorers with filters, item links and source navigation. |

[Read the implementation and related tests](docs/en/PORTFOLIO.md#code-tour). CUDA supports execution and measurement; public kernels supply the replaced operations.

<a id="projects"></a>
## Selected projects

### Case 007 — From channel selection to a smaller model block

<!-- claims: c007-transfer c007-quality -->
An MLP is a block that transforms information inside the model. The selector and weight-slicing adapter turn chosen channel groups into a physically smaller module. Built with **PyTorch · NumPy · structured pruning**.

At 25% removal of one MLP's groups, interaction-aware selection slightly reduced local error on 192 held-out inputs; both methods chose the same structure at 50%. The smaller **one-layer MLP** ran 1.176–1.412× faster, while whole-model prefill—the initial processing of a prompt—did not establish a speedup. Closer preservation of the original model differed from assigning the best probability to the correct answer.

[Explore results](https://munsik-kim.github.io/inference-lab/en/case007.html) · [Matrix-slicing code](cases/007-interaction-aware-mlp-pruning/src/surgery.py) · [Detailed study](docs/en/CASEBOOK.md#case-007)

### Case 006 — Model interventions and answer-by-answer comparisons

<!-- claims: c006-native c006-readout c006-limits -->
A one-layer attention adapter and paired evaluation tools compare answer probabilities, choice transitions and exact top-score ties on the same question. Built with **PyTorch · Transformers · SageAttention**.

The two settings changed 5 of 192 and 8 of 192 standard-set choices. Neither lost a previously correct answer in that set; a separate selected stress set did. A post-hoc diagnostic on the same inputs examined the precision of the final vocabulary-score calculation. These records cover one layer of one model on synthetic questions.

[Explore results](https://munsik-kim.github.io/inference-lab/en/case006.html) · [Attention adapter](cases/006-attention-decision-stability/src/intervention.py) · [Detailed study and diagnostic](docs/en/CASEBOOK.md#case-006)

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

DIOVA stands for **Deep-learning Inference Optimization, Validation & Analysis**.

The project builds comparison runners, model-slicing adapters, numerical checks and evidence viewers around public models and execution libraries. The [portfolio](docs/en/PORTFOLIO.md#contribution-and-reuse) credits Qwen, Transformers, PyTorch, vLLM, SageAttention and the upstream methods and fixes. OpenAI Codex assisted implementation, execution, analysis and writing. [Apache-2.0](LICENSE) applies to project materials; case notices retain upstream terms. These studies support work on model-change evaluation and inference diagnostics; deployment suitability remains unevaluated.
