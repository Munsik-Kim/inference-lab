**Munsik-Kim | DIOVA**

English | [한국어](README.ko.md)

**D**eep-learning **I**nference **O**ptimization, **V**alidation & **A**nalysis

# From model compression to working checkpoints.

Model conversion, structure-aware checkpoint loaders, and GPU performance and output comparison tools.

Built checkpoint checks, weight reconstruction, physical matrix slicing and paired evaluation around public Qwen models. Follow the code and runnable examples, then inspect the measured results.

**PyTorch · Transformers · LLM Compressor · safetensors · vLLM · NumPy**

[Model build tools](docs/en/PORTFOLIO.md#code-tour) · [Run the CPU demo](docs/en/GETTING_STARTED.md#tiny-model-demo) · [Explore projects](#projects)


## Project implementation and upstream components

| Project implementation | Upstream components | Code and tests |
|---|---|---|
| Artifact validation, partial-output protection and strict reload | PyTorch · Transformers · safetensors | [code](tools/modelpack/artifact.py) · [tests](cases/008-build-reconstruct-reload/tests/test_core.py) |
| Q conversion, copying and fresh-runtime checks | GPTQ · LLM Compressor · compressed-tensors · vLLM/Marlin | [code](tools/modelpack/quantized.py) · [tests](cases/008-build-reconstruct-reload/tests/test_boundaries.py) |
| Fixed-structure ridge fitting and correction in smaller weights | NumPy linear algebra · channel-reconstruction research | [code](tools/modelpack/numerics.py) · [tests](cases/008-build-reconstruct-reload/tests/test_core.py) |
| Matched-input measurement and paired result reports | vLLM benchmark · lm-evaluation-harness · comparison metrics | [code](packages/diova-compare/src/diova_compare/core.py) · [tests](packages/diova-compare/tests/test_compare.py) |

[Related work and implementation boundaries](docs/related-work/README.md) · [CPU comparison package](packages/diova-compare/README.md)

<a id="capabilities"></a>
## Capabilities

### Compress, save and reload models

Build GPTQ checkpoints and connect them to the inference engine. Save and restore models with layer-specific MLP widths, and fit the output weights of a smaller module.

LLM Compressor · PyTorch · Transformers · safetensors. [Checkpoint loader](tools/modelpack/artifact.py) · [Save–reload tests](cases/008-build-reconstruct-reload/tests/test_core.py)

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
| Quantization and compressed storage | **LLM Compressor / GPTQ / compressed-tensors** — Convert Linear weights, inspect packed tensors and record the conversion recipe. |
| Model structure and serialization | **PyTorch / Transformers / safetensors** — Slice matrices, save layer widths and reload strict tensor shapes and tied weights. |
| Numerical reconstruction and selection | **NumPy / CPU FP64 / ridge regression / Cholesky** — Fit a fixed smaller MLP’s output weights; compare group selections and paired scores. |
| GPU execution and checks | **vLLM / Marlin / PyTorch SDPA / SageAttention / CUDA runtime / NVML** — Verify actual runtime routes; measure memory and complete-call costs on matching inputs. |
| Reanalysis and result interfaces | **Python / unittest / GitHub Actions / HTML / CSS / JavaScript** — Check scalars and artifacts, run tiny CPU fixtures and build bilingual result screens. |

[Read the implementation and related tests](docs/en/PORTFOLIO.md#code-tour). CUDA supports execution and measurement; public kernels supply the replaced operations.

<a id="projects"></a>
## Selected projects

### Case 008 — Build, reconstruct, reload

<!-- claims: c008-tracks c008-artifact-implementation -->
Two toolchains: build a quantized Qwen 4B checkpoint, or reconstruct the outputs of a smaller Qwen 0.6B MLP. Each has its own model, measurements and result screen.

#### Q · GPTQ conversion, storage and fresh-runtime execution

Built a pipeline that converts Qwen weights to GPTQ W4A16, validates the packed checkpoint, and runs it in fresh vLLM processes.

**8.045 GB → 2.652 GB** · **67.0% smaller weight files** than native BF16 — Weight files · Original BF16 → GPTQ W4A16 · Decimal GB.

Qwen3-4B-Instruct-2507 · GPTQ W4A16. **LLM Compressor · compressed-tensors · vLLM**.

[Conversion code](tools/modelpack/quantized.py) · [Build and reload](https://munsik-kim.github.io/inference-lab/en/case008.html#track-q) · [Quality and runtime evaluation](https://munsik-kim.github.io/inference-lab/en/case008.html#q-evaluation)

#### R · Structure-aware reload and fixed-weight reconstruction

Stores a ridge fit in the fixed smaller down projection. The loader uses per-layer shape metadata and a meta skeleton, strictly assigns saved tensors, restores tied weights and derived buffers, and reloads in a fresh process without the original checkpoint.

Qwen3-0.6B · One MLP layer · BF16 · Reconstruction evaluated on 192 short synthetic held-out prompts · **PyTorch · NumPy · safetensors**.

[Structure-aware loader](tools/modelpack/artifact.py) · [Reconstruction method](https://munsik-kim.github.io/inference-lab/en/case008.html#track-r) · [Quality and runtime evaluation](https://munsik-kim.github.io/inference-lab/en/case008.html#r-evaluation)

[Methods and results](cases/008-build-reconstruct-reload/REPORT.md) · [Run the CPU demo](docs/en/GETTING_STARTED.md#tiny-model-demo) · [Code, recipes and measurements ZIP](downloads/case008_build_reconstruct_reload_reviewed_publication_v2.zip)

### Case 007 — From channel selection to smaller weight matrices

<!-- claims: c007-implementation -->
Built selectors for individual and pairwise channel-group contributions, then turned their selections into physically smaller PyTorch MLPs by slicing the aligned gate, up and down projections. The smaller module is checked against the masked original computation.

**Select groups → Slice matrices → Verify computation**

Qwen3-0.6B · One MLP layer · 16 channel groups. **PyTorch · NumPy · structured pruning**.

[Run the selector](docs/en/GETTING_STARTED.md#cpu-selector) · [Model-slicing code](cases/007-interaction-aware-mlp-pruning/src/surgery.py) · [Methods and results](https://munsik-kim.github.io/inference-lab/en/case007.html#design)

### Case 006 — Trace model changes question by question

<!-- claims: c006-implementation c006-readout -->
Built a scoped attention adapter and paired evaluation tools to compare answer probabilities, choices and exact ties on the same questions. The tools expose item-level changes and support a separate diagnostic of the final output projection’s numerical precision.

**Lost correct answers · New correct answers · Wrong-to-wrong changes**

Qwen3-0.6B · One attention layer · Fixed synthetic questions. **PyTorch · Transformers · SageAttention**.

[Explore comparisons](https://munsik-kim.github.io/inference-lab/en/case006.html#results) · [Attention adapter](cases/006-attention-decision-stability/src/intervention.py) · [Methods and results](https://munsik-kim.github.io/inference-lab/en/case006.html#design)


## New measurement — longer decode and client concurrency

Case009 reuses the existing Qwen3-4B BF16/W4 checkpoints at 256 generated tokens and client concurrency 1/4/16/32. The complete graph grid and matched eager comparison retain all three server rounds. Official task calculations are separate; their native process-exit failures remain explicitly labelled.

![Complete 128-input / 256-output TPOT and throughput curves on RTX 5080](cases/009-q-serving-quality/figures/serving-L128.png)

[Both input lengths and quality table](cases/009-q-serving-quality/REPORT.md) · [CPU reanalysis](cases/009-q-serving-quality/REPRODUCTION.md) · [Installed paired CLI](packages/diova-compare/README.md)

## Questions and recorded tools


| Question | What the case provides |
|---|---|
| [001 — What if a model will not run on the GPU?](docs/en/CASEBOOK.md#case-001) | A before/after check of an upstream execution-path fix. Terms: SM120, vLLM, W8A8. |
| [002 — Do fewer bits save memory and time?](docs/en/CASEBOOK.md#case-002) | Official model comparisons on document extraction: lower resource use, but low task accuracy. Terms: BF16, FP8. |
| [003 — How close is a simpler calculation?](docs/en/CASEBOOK.md#case-003) | Numerical error checks with different comparison scales. Terms: EFQ-Softmax, FP32 simulation. |
| [004 — Does a faster core operation make a cheaper call?](docs/en/CASEBOOK.md#case-004) | Attention timing with preparation included; longer inputs gained speed while output error exceeded the fixed limits. Terms: kernel, complete-call cost. |
| [005 — Can a setting balance speed and error?](docs/en/CASEBOOK.md#case-005) | Measured trade-offs; no tested changed setting met both requirements, so fresh confirmation stopped. Terms: precision, local error. |
| [006 — Which individual answers change?](docs/en/CASEBOOK.md#case-006) | Paired answer scores, choices and equal-top-score diagnostics. Terms: NLL, top-score tie. |
| [007 — Which groups should a smaller block keep?](docs/en/CASEBOOK.md#case-007) | Group selection, actual matrix slicing and evaluation on unused inputs. Terms: MLP, pruning. |
| [008 — Can a changed model be saved and reloaded?](docs/en/CASEBOOK.md#case-008) | GPTQ checkpoint integration and fixed-MLP ridge reconstruction; smaller Q files and lower R local error, with mixed gold scores. |
| [009 — How do longer decode and concurrency change cost?](cases/009-q-serving-quality/README.md) | Matched graph/eager serving curves and official quality calculations with recorded runtime-exit failures. |

## Go deeper

Read the five-minute [introduction](docs/en/START_HERE.md) or look up a term in the [glossary](docs/en/GLOSSARY.md). The [Casebook](docs/en/CASEBOOK.md) leads to methods, source and exact results. [Getting started](docs/en/GETTING_STARTED.md) explains ZIP downloads, local HTML and CPU checks. The explorers show saved measurements and need no GPU; GitHub's HTML source view displays code, not the running screen.

## Contribution and sources

DIOVA stands for **Deep-learning Inference Optimization, Validation & Analysis**.

The project builds comparison runners, model-slicing adapters, numerical checks and evidence viewers around public models and execution libraries. The [portfolio](docs/en/PORTFOLIO.md#contribution-and-reuse) credits Qwen, Transformers, PyTorch, vLLM, SageAttention and the upstream methods and fixes. OpenAI Codex assisted implementation, execution, analysis and writing. [Apache-2.0](LICENSE) applies to project materials; case notices retain upstream terms. These studies support work on model-change evaluation and inference diagnostics; deployment suitability remains unevaluated.

[Corrected CPU wheel 0.1.1](downloads/diova_compare-0.1.1-py3-none-any.whl) · [Case009 reviewed archive](downloads/case009_serving_quality_reviewed_publication_v2.zip) · [Metadata](downloads/case009_serving_quality_reviewed_publication_v2.json)
