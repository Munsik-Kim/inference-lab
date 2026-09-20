# Tools, implementation and technical collaboration

English | [한국어](../ko/PORTFOLIO.md) · [Home](../../README.md)

[New here?](START_HERE.md) · [Plain-language glossary](GLOSSARY.md)

## A 30-second introduction

<!-- claims: c007-transfer c007-quality c006-native c006-readout c008-tracks c008-artifact-implementation -->
DIOVA builds tools to compress, save and reload models, then compare their outputs and execution costs. Explore the PyTorch implementations, RTX 5080 measurements and browser-based results.

Case008 connects checkpoint conversion, strict loading and fixed-structure ridge repair. Q weight files decreased about 67.0%; R reduced local squared error by 94.1–95.4% on short synthetic inputs. The models and metrics are separate, and gold scores were mixed. [Q/R screen](https://munsik-kim.github.io/inference-lab/en/case008.html) · [Tiny CPU example](GETTING_STARTED.md#tiny-model-demo).

At 25% MLP deletion, PAIRWISE slightly lowered local error and preserved more baseline choices, but INDEPENDENT had better gold NLL. At 50%, the selected modules were identical; the frozen status is **COMPLETED_NO_CLEAR_TRANSFER**. In Case006, A_PUBLIC changed 5 of 192 standard choices and V4 changed 8 of 192. A same-input readout follow-up examines ties and the final vocabulary projection. These examples make both implementation choices and competing evaluation objectives visible. [Casebook](CASEBOOK.md) · [Open the local screen](GETTING_STARTED.md#local-showcase).

![Actual local Case007 comparison screen, showing one fixed code input](../../presentation/screenshots/case007-comparison.png)

*Local Edge screenshot: the first fixed CODE input, 25% deletion. [Case006 readout screen](../../presentation/screenshots/case006-readout.png) shows the separately labelled same-input diagnostic. These are selected views for navigation, not aggregate performance summaries.*

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

*Script for a live walkthrough; no recorded video is supplied.*

“Open Case008 and select Q. The flow follows a BF16 source through GPTQ conversion and a new vLLM process. Compare file bytes and allocator memory: they describe different resources. The correct counts match, but the probability-score directions differ.

“Move to R. Compare I25 before and after output-weight repair: the channel count stays fixed while local error falls. Correct counts remain a separate result. Follow the CPU example link, run the tiny random-model command, and inspect summary.json. The builder exited before another process loaded a copied artifact. Shapes, complete outputs and shared weights must match. Then open the loader source and its failure tests. Case007 and Case006 provide the earlier group-selection and answer-level comparisons.”

## Three introduction lines with source links

<!-- claims: c004-cost c005-stop -->
- Built a group-selection and matrix-slicing pipeline with masked-versus-sliced and held-out comparisons. [Case007 methods](../../cases/007-interaction-aware-mlp-pruning/METHODS.md).
- Implemented a single-layer Qwen intervention and paired analysis of answer probabilities, regressions, gains and ties. [Case006 methods](../../cases/006-attention-decision-stability/METHODS.md).
- Measured complete attention cost and retained **STOP_DEV_SCREEN** alongside the observed precision–cost trade-off. [Case004 cost](../../cases/004-low-precision-attention-break-even/README.md) · [Case005 decision](../../cases/005-attention-precision-pareto/README.md).

<a id="contribution-and-reuse"></a>
## Contribution and reuse

<!-- claims: attribution -->
The project's work is comparison execution, scoped adapters, matrix surgery, validity checks, numerical/behavioral analysis and evidence viewers. Qwen and Red Hat AI provide checkpoints; Transformers, PyTorch and vLLM provide model/runtime components; SageAttention provides low-precision kernels. hclsys authored the Case001 guard. Han and coauthors developed EFQ-Softmax. HOPE motivates interaction-aware deletion in a separate MoE expert-pruning setting; the dense-group objective follows directly from the chosen MLP decomposition.

GPTQ is the upstream quantization method; LLM Compressor and compressed-tensors implement conversion/storage, and safetensors provides the weight format. Case008 adds artifact validation, a layer-aware loader and fixed-structure ridge fitting. [Case008 attribution](../../cases/008-build-reconstruct-reload/NOTICE.md).

OpenAI Codex assisted implementation, local execution, tests, analysis and writing. [Case001](../../cases/001-sm120-int8-fallback/NOTICE.md), [Case003](../../cases/003-efq-softmax-numerical-audit/NOTICE.md), [Case006](../../cases/006-attention-decision-stability/NOTICE.md) and [Case007 notices](../../cases/007-interaction-aware-mlp-pruning/NOTICE.md) credit reused work. Project code uses [Apache-2.0](../../LICENSE); model/upstream terms remain applicable.

## Technical collaboration

The code supports discussions about **model-change evaluation**, **local inference diagnosis** and **reproducible analysis tools and result screens**. A useful starting point is a model boundary, a public input ID and a metric to compare. [Repository issues](https://github.com/Munsik-Kim/inference-lab/issues) provide a public technical-question channel; keep credentials and private customer data out of public reports.

The evidence covers specific models, one device and synthetic tasks. Deployment is **NOT_ASSESSED**. CPU audits check retained scalars; excluded full vectors and independent GPU replication have a separate verification scope in the [reproduction guide](GETTING_STARTED.md).
