**Munsik-Kim | DIOVA**

English | [한국어](README.ko.md)

# Store smaller models, run them, and check the answers that changed.

A smaller AI model can fit on a personal GPU yet answer differently or fail to finish a task. In DIOVA, I built Python tools that check all three: saving and running smaller models, comparing answers to the same questions, and verifying that an AI actually made and checked the requested change. The recorded runs use public Qwen models and an RTX 5080 16GB.

This work relates to **local AI execution, inference engineering** (making trained models run), **model optimization, quality regression testing** (finding answers that became wrong after a change), and **tool-automation evaluation**. The code and measurements show what I built and the conditions in which I checked it.

Work you can discuss with me includes diagnosing why a local model will not run, comparing model settings for memory/time/answer quality, building checks for changed answers or missing tool actions, and turning saved results into reusable reports. [Capabilities and source code](docs/en/PORTFOLIO.md) connect these tasks to implementations; production deployment has not been assessed.

[Portfolio and model guides (Pages)](https://munsik-kim.github.io/inference-lab/en/index.html) · [Try it: compare your saved results](packages/diova-compare/examples/compare/README.md)

## Featured work A · B · C

| Problem | Built here | Evidence / try it |
|---|---|---|
| A. Store a smaller model and run actual requests | Connected an existing compression method (GPTQ) to checks of saved model files, restart in a new process, and request comparison under matching conditions | [Case008 Q](https://munsik-kim.github.io/inference-lab/en/case008.html#track-q) → [Case009](https://munsik-kim.github.io/inference-lab/en/case009.html#serving) |
| B. Compare changed answers on the same questions | Built diova-compare, an independent CPU tool: check that questions and answer keys match, list lost/new correct answers and other changes, and retain original run failures | [CPU example](packages/diova-compare/examples/compare/README.md) · [Code and tests](packages/diova-compare/README.md#implementation-and-checks) |
| C. Verify actual tool-task completion | Limited calls to available tool names, explained call errors, and checked records of actual file changes and validation | [Case013 follow-up implementation and report](research/agent_tool_contract_repair_20261014/README.md) |

The current model guide covers Qwen3-4B-Instruct-2507 on an RTX 5080 16GB; full detail is Korean with English summaries. Separate studies use the smaller Qwen3-0.6B model to change its structure or compensate for compression error. The comparison command-line tool (CLI) runs on CPU with Python 3.10+; for freely generated answers, you supply the correctness judgments. It does not run or automatically grade a model.

[Install and map your records](packages/diova-compare/examples/compare/README.md) · [Model guides](docs/en/LLM_GUIDE.md) · [Capabilities and code](docs/en/PORTFOLIO.md) · [Reproduction](docs/en/GETTING_STARTED.md)

## Project implementation and upstream components

| Project implementation | Upstream components | Code and tests |
|---|---|---|
| Check saved model files (artifacts), prevent incomplete saves, and reject incompatible reloads | PyTorch · Transformers · safetensors | [code](tools/modelpack/artifact.py) · [tests](cases/008-build-reconstruct-reload/tests/test_core.py) |
| Connect Q model conversion, copying and checks in a newly started inference process | GPTQ · LLM Compressor · compressed-tensors · vLLM/Marlin | [code](tools/modelpack/quantized.py) · [tests](cases/008-build-reconstruct-reload/tests/test_boundaries.py) |
| Fit output weights to reduce a smaller block’s error (ridge regression) and save the correction | NumPy linear algebra · channel-reconstruction research | [code](tools/modelpack/numerics.py) · [tests](cases/008-build-reconstruct-reload/tests/test_core.py) |
| Measure matching inputs and report their individual before/after changes (paired comparison) | vLLM benchmark · lm-evaluation-harness · comparison metrics | [code](packages/diova-compare/src/diova_compare/core.py) · [tests](packages/diova-compare/tests/test_compare.py) |
| Save a model’s changing memory in fewer bits, preserve failures, and resume in a new process | NumPy · PyTorch · ComplexKDA recurrence | [implementation](cases/010-ckda-finite-precision-memory-horizon/versions/v2/source/online_v2.py) · [synthetic restart tests](cases/010-ckda-finite-precision-memory-horizon/versions/v2/tests/test_online_v2.py) |
| Change only the answer-reading layer on the same stored state; save/reapply its two arrays | ComplexKDA · PyTorch L-BFGS · NumPy | [adapter](cases/011-frozen-state-readout-adaptation/source/adapter.py) · [patch tests](cases/011-frozen-state-readout-adaptation/tests/test_fitting_patch.py) |
| Generate from the same initial noise; compare steps, internal repeats, time and requested count/color/left-right conditions | Looped-DiT · FLAN-T5 · PyTorch · Pillow | [adapter](cases/012-looped-dit-inference-budget/source/adapter.py) · [CPU tests](cases/012-looped-dit-inference-budget/tests/test_analysis.py) |
| Convert weights to four-bit integers (INT4) with their scaling factors; recheck recorded quality results | DualQuant public equations · PyTorch · NumPy | [solver](cases/013-dualquant-reference-quality/source/paper_int4.py) · [CPU audit](cases/013-dualquant-reference-quality/analysis/audit.py) |
| Collect intermediate values altered by earlier layer compression at the next layer, fit corrections, save/reload INT4 and check whole-model text quality | Guo compensation · PyTorch · Transformers · NumPy | [code](cases/014-quantization-compensation/followups/p2_core/source/run_p2.py) · [tests](cases/014-quantization-compensation/followups/p2_core/tests/test_p2.py) |
| API rename, received-result and state regression testing | Qwen · PyTorch · Transformers · Python stdlib | [runner](cases/015-tool-interface-sensitivity/followups/order-api-regression-v1/src/order_regression/runner.py) · [tests](cases/015-tool-interface-sensitivity/followups/order-api-regression-v1/tests/test_order_contracts.py) |

[Related work and implementation boundaries](docs/related-work/README.md) · [CPU comparison package](packages/diova-compare/README.md)

[Practical model guides for a 16GB GPU](docs/en/LLM_GUIDE.md) — Qwen3-4B key findings and Korean detail. [Read online](https://munsik-kim.github.io/inference-lab/en/llm-guide.html).

## Case015 — Same Tools, Different Interfaces

Compare order lookup, multiplication, update and validation across an API rename. On the new synthetic A1 workflow, BF16 changed **15/16→16/16**, RTN **16/16→16/16** and D16 **12/16→12/16**. Audit actual state separately from information received before generation. The original96 P1 episodes and96 A1 episodes keep separate denominators.

[Pages](https://munsik-kim.github.io/inference-lab/en/case015.html#overview) · [Report](cases/015-tool-interface-sensitivity/REPORT.md) · [CPU example](cases/015-tool-interface-sensitivity/REPRODUCE.md) · [Runner](cases/015-tool-interface-sensitivity/followups/order-api-regression-v1/src/order_regression/runner.py)

<a id="capabilities"></a>
## Capabilities

### Compress, save and reload models

Build saved model files (checkpoints) using the existing GPTQ compression method and connect them to software that runs model requests (an inference engine). Save and restore a Qwen3 model with one reduced feed-forward block (MLP), and fit that smaller block’s output weights to reduce its error.

LLM Compressor · PyTorch · Transformers · safetensors. [Checkpoint loader](tools/modelpack/artifact.py) · [Save–reload tests](cases/008-build-reconstruct-reload/tests/test_core.py)

### GPU inference measurement and output evaluation

Compare elapsed time (latency), GPU memory and answer scores on the same inputs. Measure a single operation and a complete model call separately, so a faster part is not mistaken for a faster answer.

vLLM · CUDA runtime · NVML · NumPy. [Request and memory runner](cases/002-bf16-fp8-document-extraction/scripts/run.py) · [Timing boundaries](cases/004-low-precision-attention-break-even/src/timing.py)

### Reproducible analysis and result explorers

Build Python tools that recalculate saved measurements and web screens that show what changed for each question. Automated checks compare the displayed values with their source files.

Python · JavaScript · GitHub Actions · GitHub Pages. [CPU selector replay](tools/showcase/replay_selection.py) · [Display checks and tests](tests/showcase/test_showcase.py)

<a id="tech-stack"></a>
## Tech stack in use

| Work | Technologies and implementation |
|---|---|
| Quantization and compressed storage | **LLM Compressor / GPTQ / compressed-tensors** — Convert learned matrix (Linear) weights, inspect arrays packed into fewer bits and record conversion settings (the recipe). |
| Model structure and serialization | **PyTorch / Transformers / safetensors** — Slice matrices, save layer sizes and restore checked array dimensions (tensor shapes) and shared weights. |
| Numerical reconstruction and selection | **NumPy / CPU FP64 / ridge regression / Cholesky** — Use 64-bit numerical fitting for a fixed smaller MLP’s output weights; compare removed groups and question-by-question scores. |
| GPU execution and checks | **vLLM / Marlin / PyTorch SDPA / SageAttention / CUDA runtime / NVML** — Verify actual runtime routes; measure memory and complete-call costs on matching inputs. |
| Reanalysis and result interfaces | **Python / unittest / GitHub Actions / HTML / CSS / JavaScript** — Check saved numerical values and model files, run small CPU examples and build bilingual result screens. |

[Read the implementation and related tests](docs/en/PORTFOLIO.md#code-tour). CUDA supports execution and measurement; public kernels supply the replaced operations.

<a id="projects"></a>
## Implementation details and research records

### Case 008 — Build, reconstruct, reload

<!-- claims: c008-tracks c008-artifact-implementation -->
Two toolchains: build a quantized Qwen 4B checkpoint, or reconstruct the outputs of a smaller Qwen 0.6B MLP. Each has its own model, measurements and result screen.

#### Q · GPTQ conversion, storage and fresh-runtime execution

Built a pipeline that converts Qwen weights to GPTQ W4A16, validates the packed checkpoint, and runs it in fresh vLLM processes.

**8.045 GB → 2.652 GB** · **67.0% smaller weight files** than native BF16 — Weight files · Original BF16 → GPTQ W4A16 · Decimal GB.

Qwen3-4B-Instruct-2507 · GPTQ W4A16. **LLM Compressor · compressed-tensors · vLLM**.

[Conversion code](tools/modelpack/quantized.py) · [Build and reload](https://munsik-kim.github.io/inference-lab/en/case008.html#track-q) · [Quality and runtime evaluation](https://munsik-kim.github.io/inference-lab/en/case008.html#q-evaluation)

#### R · Structure-aware reload and fixed-weight reconstruction

Fits a smaller block’s output matrix (down projection) toward the original output using ridge regression. The saved model records each layer’s size. The loader builds an empty structure (meta skeleton), checks and assigns all saved number arrays (tensors), restores shared weights and derived working arrays (buffers), and runs in a new process without the original checkpoint.

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

Case009 compares the existing Qwen3-4B models with 16-bit weights (BF16) or four-bit weights (W4), generating 256 text units (tokens) per request with up to 1/4/16/32 simultaneous requests. It keeps all three server rounds for both prepared GPU execution graphs (graph) and calls dispatched as they occur (eager). Official quality tasks were run separately; abnormal process exits remain labelled alongside their calculated scores.

![Complete 128-input / 256-output TPOT and throughput curves on RTX 5080](cases/009-q-serving-quality/figures/serving-L128.png)

[Both input lengths and quality table](cases/009-q-serving-quality/REPORT.md) · [CPU reanalysis](cases/009-q-serving-quality/REPRODUCTION.md) · [Installed paired CLI](packages/diova-compare/README.md)

## Case 010 — Low-bit state storage, restart and memory horizon

<!-- claims: c010-state-bytes -->
Built storage that packs the memory updated with each input (recurrent state) into fewer bits and resumes in a new process while preserving numerical failures and each input stream’s random-generator state. The study compares the length of continuously correct readout and complete-call cost under explicit storage caps.

**Saved state per input stream: 12,305 → 3,137 bytes, about 74.5% smaller** · Original 32-bit values (Native FP32) → 8-bit integers (INT8). On fresh inputs for the same three trained CKDA checkpoints, each point estimate of the mean continuous-correct-length difference was under one token. These are state bytes, not whole-process RAM/VRAM or a quality-equivalence result.

[Failure-aware implementation](cases/010-ckda-finite-precision-memory-horizon/versions/v2/source/online_v2.py) · [CPU audits](cases/010-ckda-finite-precision-memory-horizon/REPRODUCTION.md) · [Methods and results of both stages](cases/010-ckda-finite-precision-memory-horizon/REPORT.md)

## Case 011 — Same State, Different Readout

Built a tool that keeps the model's evolving state unchanged across readout comparisons and replaces only the final layer used to read an answer. The two adjusted number arrays (tensors) can be saved as a change file (patch) and applied in a new process.

**1,158 final-layer values · 0 extra recurrent-state bytes · Fresh-process patch reload**

Gold-label scores improved, but the average uninterrupted correct prefix became shorter. On the same state, better probability scores did not translate into a later first error.

CKDA symbolic state tracking · Three existing checkpoints · Final-layer supervised adaptation. The state evolves with each input; readouts within a storage mode share that trajectory.

[Overview](https://munsik-kim.github.io/inference-lab/en/case011.html#overview) · [Implementation](cases/011-frozen-state-readout-adaptation/source/adapter.py) · [Detailed results](https://munsik-kim.github.io/inference-lab/en/case011.html#results)

## Case 012 — Image-generation time: loops vs steps

Built a runner to divide image-generation time between successive updates and internal model repetition. It saves three images from identical initial noise and compares count, color and left/right requirements alongside measured time.

**Identical initial noise · Per-setting time and memory · Blinded annotation**

L1/L2/L4 mean one, two or four repetitions of the core calculation within each generation step.

The human checklist is complete for 192 MAIN images: 100 directly checked, 92 assigned through explicit visual-group answer reuse. All requested constraints passed in **52/64 at L1, 51/64 at L2 and 53/64 at L4**. L4 gained four passes and lost three against L1, for a net difference of one image. L2 had the shortest recorded median request time. These are descriptive counts with reused judgments; the original AI scores remain separately available.

[Plain-language overview](cases/012-looped-dit-inference-budget/README.md) · [Compare every image](cases/012-looped-dit-inference-budget/publication/IMAGES.md) · [Adapter code](cases/012-looped-dit-inference-budget/source/adapter.py) · [Methods and results](cases/012-looped-dit-inference-budget/REPORT.md)

[Completed human review](presentation/evaluations/case012-human-v1/README.md) · [Check the images yourself on the site](https://munsik-kim.github.io/inference-lab/en/case012-annotate.html) · [Result page](https://munsik-kim.github.io/inference-lab/en/case012.html). Download JSON at any point and import it later to resume. Browser records are not automatically uploaded; human labels are recalculated separately from AI scores.

## Case 013 — Does lower weight error improve real answers?

Built tools that store and reload weights as four-bit integers (INT4), preserve their scaling factors, and compare text scores, answers to long inputs and local tool actions on matching inputs. Recorded all four methods on Qwen 0.6B and 4B using an RTX 5080. D16 stores weights in INT4 and keeps calculation inputs in BF16; D4 uses the same weights and also quantizes those inputs to INT4. Both quality paths restore quantized values to BF16 for calculation (reference computation); these names do not denote two weight-compression levels.

D16 reduced weight error in both models. The next-token prediction score (perplexity, PPL; lower is better) improved in 0.6B but slightly worsened in 4B; the 4B free-tool pilot completed **6/8 versus RTN 8/8** tasks. RTN is the comparison that simply rounds weights to the nearest permitted value. The model using compressed GPU operations failed the preset numerical check, so faster request serving remains unestablished. These quality results use a small set of existing tasks used during setting development (DEV).

[Overview](cases/013-dualquant-reference-quality/README.md) · [Solver/adapter](cases/013-dualquant-reference-quality/source/paper_int4.py) · [CPU readback](cases/013-dualquant-reference-quality/REPRODUCTION.md) · [Full report](cases/013-dualquant-reference-quality/REPORT.md)

### Follow-up — Tool names and verified actions

Connected existing records to the CPU comparison tool, limited generated tool names to the available names (enum decoding), and explained call errors. On the same eight existing DEV tasks with unchanged weights, D16 changed from **6/8 to 8/8**, and D4 from **0/8 to 2/8**. Name-limited conditions had zero calls to unregistered tools; six other D4 failures remain. Success required recorded actions and checks, not just a final message claiming completion. Performance on new tasks has not been established.

[Overview](research/agent_tool_contract_repair_20261014/README.md) · [Full results and report](research/agent_tool_contract_repair_20261014/records/REPORT.md) · [Earlier diagnostic](research/agent_stability_20261013/README.md) · [Recorded comparison CLI](packages/diova-compare/README.md)

<!-- case014-entry -->
## Research and numerical engineering — Case014

Built a separate research tool that collects intermediate values at the next layer after earlier layers have been compressed, then fits corrections for their changed computation. It saves/reloads corrected INT4 weights and compares whole-model text prediction quality.

**PG19 text-prediction score (PPL) 12.92% lower · 64/64 excerpts with lower correct-token loss (NLL) · unchanged bytes for the converted matrices**

Qwen3-0.6B · 196 learned matrix transformations (Transformer projections) · 256-token excerpts from 64 fresh PG19 books. The same gain was not established on ArXiv text. Quality was checked after restoring values to BF16; speed and GPU-memory (VRAM) savings were not measured. Equal projection bytes do not establish equal total model-file size.

[Compare results](https://munsik-kim.github.io/inference-lab/en/case014.html#results) · [Implementation and reproduction](cases/014-quantization-compensation/REPRODUCE.md) · [Research journey](cases/014-quantization-compensation/STAGE_MAP.md)



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
| [010 — Can low-bit state restart consistently after failure?](docs/en/CASEBOOK.md#case-010) | Actual bit packing, failure-aware serialization, first-error survival and byte-budget comparisons. |
| [011 — What changes when only the readout changes?](docs/en/CASEBOOK.md#case-011) | Two-tensor adaptation and reload; paired gold-score and first-error comparisons on the same state. |
| [012 — Where should image-generation time go?](docs/en/CASEBOOK.md#case-012) | Same-noise generation-step/internal-repeat runner and timing. Assessment of requested count, color and left/right conditions in all 192 images, with paired constraint gains and losses. |
| [013 — Does lower weight error improve real answers?](docs/en/CASEBOOK.md#case-013) | INT4 scale-preserving conversion/reload and matched full-model, long-input and tool-quality records; packed gate remains blocked. |
| [014 — Can quality improve at the same INT4 storage budget?](docs/en/CASEBOOK.md#case-014) | Own-trajectory compensation/storage/reload/full-model quality, with PG19 gains and ArXiv limits. |

## Go deeper

Read the five-minute [introduction](docs/en/START_HERE.md) or look up a term in the [glossary](docs/en/GLOSSARY.md). The [Casebook](docs/en/CASEBOOK.md) leads to methods, source and exact results. [Getting started](docs/en/GETTING_STARTED.md) explains ZIP downloads, local HTML and CPU checks. The explorers show saved measurements and need no GPU; GitHub's HTML source view displays code, not the running screen.

## Contribution and sources

DIOVA stands for **Deep-learning Inference Optimization, Validation & Analysis**.

The project builds comparison runners, model-slicing adapters, numerical checks and evidence viewers around public models and execution libraries. The [portfolio](docs/en/PORTFOLIO.md#contribution-and-reuse) credits Qwen, Transformers, PyTorch, vLLM, SageAttention and the upstream methods and fixes. OpenAI Codex assisted implementation, execution, analysis and writing. [Apache-2.0](LICENSE) applies to project materials; case notices retain upstream terms. These studies support work on model-change evaluation and inference diagnostics; deployment suitability remains unevaluated.

[Corrected CPU wheel 0.1.1](downloads/diova_compare-0.1.1-py3-none-any.whl) · [Case009 reviewed archive](downloads/case009_serving_quality_reviewed_publication_v2.zip) · [Metadata](downloads/case009_serving_quality_reviewed_publication_v2.json)
