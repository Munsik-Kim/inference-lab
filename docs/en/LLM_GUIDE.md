# Using an LLM on a 16GB GPU

English summary | [한국어](../ko/LLM_GUIDE.md) · [Home](../../README.md)

If you want to run a model on your own PC, start by asking whether it fits in GPU memory, how long answers take, and whether it performs your task correctly. This guide uses saved Qwen measurements on an RTX 5080 to show how to check those questions. Open-weight models publish their learned number arrays (weights); their usage terms still apply.

For a recruiter or technical collaborator, the connected code demonstrates model-file conversion and restart checks, request measurement, and answer/tool-action evaluation. These relate to local AI execution, model optimization and quality regression testing. [Capabilities and source evidence](PORTFOLIO.md) explain the work that can be discussed.

The current detailed model guide covers **Qwen3-4B-Instruct-2507**. The [online model guide](https://munsik-kim.github.io/inference-lab/en/llm-guide.html) links to these pages, also available in the same local static build:

| Build path | Role |
|---|---|
| `ko/llm-guide.html` | Full guide introduction in Korean |
| `ko/qwen3-4b.html` | Model, converted and saved model files (artifacts), execution, quality, failures, attempted mitigations and code |
| `ko/measurements.html` | Measurement boundaries and evidence scope |
| `en/llm-guide.html`, `en/qwen3-4b.html`, `en/measurements.html` | Short English introductions and findings, linked to Korean detail; not full translations |

[Read the English summary online](https://munsik-kim.github.io/inference-lab/en/qwen3-4b.html), or [open the full Korean guide](https://munsik-kim.github.io/inference-lab/ko/qwen3-4b.html). [Build and open locally](../../presentation/README.md#practical-guides); reading the site requires no GPU or model download. Existing Case URLs, anchors and language navigation remain in place.

Cases008 Q and 009 checked model files converted with the existing GPTQ method and actual low-bit execution in vLLM. Case013 and its follow-up instead restore compressed values to a 16-bit number format (BF16) to check quality; this is reference computation, not a measurement of compressed GPU speed. D16 uses INT4 weights with BF16 inputs; D4 uses the same weights and also quantizes inputs to INT4. The failed full-model numerical check for packed execution, abnormal quality-process exits and failures on existing setting-development tasks (DEV) remain visible. Case008 R separately removes and repairs part of one Qwen3-0.6B block; it is not a step in Qwen3-4B compression.

See the [shared glossary](GLOSSARY.md), [CPU and reproduction instructions](GETTING_STARTED.md) and [original cases](CASEBOOK.md). Narrative is maintained in `presentation/guides/`; the builder inserts measured values from public records into both home stories and guide tables. The generated `data/practical-guide.json` and `build_manifest.json` retain exact values, field paths, the evidence commit and source hashes.

2026-10-15: guide implementation and reanalysis of existing public records. No new GPU experiment, quantization or benchmark run; reader evaluation was not performed.

<!-- case014-entry -->
## Separate0.6B compensation study

[Case014](../../cases/014-quantization-compensation/README.md) · [Pages](https://munsik-kim.github.io/inference-lab/en/case014.html#overview). This separate Qwen3-0.6B study adjusts weights to reduce compression error and checks text-prediction quality. Its PG19 gain and ArXiv limits do not contribute numbers to the Qwen3-4B tables above.
