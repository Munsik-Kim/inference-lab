**Munsik-Kim | DIOVA**

English | [한국어](README.ko.md)

# Store smaller models, run them, and check the answers that changed.

I built tools to run public AI models on a limited GPU and check changed answers and actual task results. The public implementations and measurements relate to **local model execution and optimization, quality checks after model changes, and tool-automation evaluation**.

[Portfolio and model guides (Pages)](https://munsik-kim.github.io/inference-lab/en/index.html) · [Try it: compare your saved results](packages/diova-compare/examples/compare/README.md)

## Featured work A · B · C

<!-- claims: c008-tracks c008-artifact-implementation -->
| Problem | Built here | Evidence / try it |
|---|---|---|
| A. Store a smaller model and run actual requests | Connected an existing compression method (GPTQ) to checks of saved model files, restart in a new process, and request comparison under matching conditions | [Case008 Q](https://munsik-kim.github.io/inference-lab/en/case008.html#track-q) → [Case009](https://munsik-kim.github.io/inference-lab/en/case009.html#serving) |
| B. Compare changed answers on the same questions | Built diova-compare, an independent CPU tool: check that questions and answer keys match, list lost/new correct answers and other changes, and retain original run failures | [CPU example](packages/diova-compare/examples/compare/README.md) · [Code and tests](packages/diova-compare/README.md#implementation-and-checks) |
| C. Verify actual tool-task completion | Limited calls to available tool names, explained call errors, and checked records of actual file changes and validation | [Case013 follow-up implementation and report](research/agent_tool_contract_repair_20261014/README.md) |

## Recorded storage and runtime memory

Qwen3-4B-Instruct-2507 · RTX 5080 16GB · original BF16 → GPTQ W4A16. These rows have different measurement boundaries and execution conditions.

| Existing experiment / measurement | Original → compressed model | Meaning / conditions |
|---|---|---|
| [Case008 Q · weight files](https://munsik-kim.github.io/inference-lab/en/case008.html#q-evaluation) | **8.045 GB → 2.652 GB** | **67.0% smaller files**, decimal GB |
| [Case008 Q · peak PyTorch GPU allocation](https://munsik-kim.github.io/inference-lab/en/qwen3-4b.html#storage) | **10.4868 → 5.3971 GiB** | About 48.5% lower; eager, one request, 1 GiB cache; includes initialization and execution |
| [Case009 · highest whole-GPU memory samples](https://munsik-kim.github.io/inference-lab/en/qwen3-4b.html#device-memory) | **14.08–14.51 → 9.08–9.52 GiB** | **4.98–5.06 GiB / 34.4–35.8% lower** within matching rounds; graph, 4 GiB cache, three rounds, one-second samples |

Case009 samples include startup, warmup and desktop use. Read its lost correct answers and abnormal evaluation-process exits in the [model guide](docs/en/LLM_GUIDE.md). Case008 R is a separate study reducing and restoring one block of Qwen3-0.6B.

## Use or reproduce

- Compare saved answers from your models: [prepare two files → validate → compare → reports](packages/diova-compare/examples/compare/README.md). Uses Python 3.10+ and CPU; you supply correctness judgments for free-form answers.
- Run a local model: [model guides](docs/en/LLM_GUIDE.md). Current detailed coverage is Qwen3-4B, with English summaries and Korean detail; it does not validate every 16GB GPU or task.
- Audit existing records: [reproduction](docs/en/GETTING_STARTED.md). CPU recalculation and original GPU runs are separate paths.

<a id="capabilities"></a>
<a id="tech-stack"></a>
## Implementation and technology evidence

The [implementation / upstream technology / code / tests table](docs/en/PORTFOLIO.md#implementation-map) shows the contributions; continue with the [code tour](docs/en/PORTFOLIO.md#code-tour). [Technical collaboration](docs/en/PORTFOLIO.md#technical-collaboration) describes work supported by the public implementations.

<a id="case-006--trace-model-changes-question-by-question"></a>
<a id="case-007--from-channel-selection-to-smaller-weight-matrices"></a>
<a id="case-008--build-reconstruct-reload"></a>
<a id="case-010--low-bit-state-storage-restart-and-memory-horizon"></a>
<a id="case-011--same-state-different-readout"></a>
<a id="case-012--image-generation-time-loops-vs-steps"></a>
<a id="case-013--does-lower-weight-error-improve-real-answers"></a>
<a id="case015--same-tools-different-interfaces"></a>
<a id="compress-save-and-reload-models"></a>
<a id="follow-up--tool-names-and-verified-actions"></a>
<a id="gpu-inference-measurement-and-output-evaluation"></a>
<a id="implementation-details-and-research-records"></a>
<a id="new-measurement--longer-decode-and-client-concurrency"></a>
<a id="project-implementation-and-upstream-components"></a>
<a id="q--gptq-conversion-storage-and-fresh-runtime-execution"></a>
<a id="r--structure-aware-reload-and-fixed-weight-reconstruction"></a>
<a id="reproducible-analysis-and-result-explorers"></a>
<a id="research-and-numerical-engineering--case014"></a>
<a id="tech-stack-in-use"></a>
<a id="projects"></a>
## Questions and recorded tools

<!-- claims: c007-implementation c006-implementation c006-readout c010-state-bytes -->
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
| [010 — Can low-bit state restart after failure?](docs/en/CASEBOOK.md#case-010) | Stored state for one stream: 12,305 → 3,137 bytes, 74.5% smaller. State bytes are separate from total RAM or VRAM. |
| [011 — What changes when only the readout changes?](docs/en/CASEBOOK.md#case-011) | Two-tensor adaptation and reload; paired gold-score and first-error comparisons on the same state. |
| [012 — Where should image-generation time go?](docs/en/CASEBOOK.md#case-012) | Same-noise generation-step/internal-repeat runner and timing. Assessment of requested count, color and left/right conditions in all 192 images, with paired constraint gains and losses. |
| [013 — Does lower weight error improve real answers?](docs/en/CASEBOOK.md#case-013) | INT4 scale-preserving conversion/reload and matched full-model, long-input and tool-quality records; packed gate remains blocked. |
| [014 — Can quality improve at the same 4-bit storage?](docs/en/CASEBOOK.md#case-014) | PG19 text-quality gains and ArXiv limits. Quality was evaluated after BF16 restoration; VRAM savings were not measured. |
| [015 — Can a renamed tool API still complete an order workflow?](https://munsik-kim.github.io/inference-lab/en/case015.html#overview) | Compare lookup, calculation, update and validation. Across 16 A1 scenarios: BF16 15/16→16/16, RTN 16/16→16/16, D16 12/16→12/16. Original P1 96 and A1 96 keep separate denominators; quality uses INT4 restored to BF16. [Code and CPU example](cases/015-tool-interface-sensitivity/REPRODUCE.md). |

## Go deeper

[Start here](docs/en/START_HERE.md) · [Glossary](docs/en/GLOSSARY.md) · [All cases and sources](docs/en/CASEBOOK.md) · [Related work and implementation boundaries](docs/related-work/README.md)

## Contribution and sources

DIOVA stands for **Deep-learning Inference Optimization, Validation & Analysis**. The project connects comparison runners, model-changing tools, checks and result screens to public models and libraries. The [contribution and source notes](docs/en/PORTFOLIO.md#contribution-and-reuse) identify authors and reused work. OpenAI Codex assisted implementation, execution, analysis and writing. [Apache-2.0](LICENSE) applies to project materials; case notices retain upstream terms. Production deployment suitability has not been assessed.
