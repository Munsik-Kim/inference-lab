# Using an LLM on a 16GB GPU

English summary | [한국어](../ko/LLM_GUIDE.md) · [Home](../../README.md)

A guide for developers running open-weight models on an RTX 5080 and teams considering model changes or automation. Fitting in memory, serving faster and doing a task correctly are separate questions.

The current detailed model guide covers **Qwen3-4B-Instruct-2507**. The [online model guide](https://munsik-kim.github.io/inference-lab/en/llm-guide.html) links to these pages, also available in the same local static build:

| Build path | Role |
|---|---|
| `ko/llm-guide.html` | Full guide introduction in Korean |
| `ko/qwen3-4b.html` | Model, artifacts, execution, quality, failures, mitigations and implementation |
| `ko/measurements.html` | Measurement boundaries and evidence scope |
| `en/llm-guide.html`, `en/qwen3-4b.html`, `en/measurements.html` | Short English introductions and findings, linked to Korean detail; not full translations |

[Read the English summary online](https://munsik-kim.github.io/inference-lab/en/qwen3-4b.html), or [open the full Korean guide](https://munsik-kim.github.io/inference-lab/ko/qwen3-4b.html). [Build and open locally](../../presentation/README.md#practical-guides); reading the site requires no GPU or model download. Existing Case URLs, anchors and language navigation remain in place.

Cases008 Q and 009 verified actual GPTQ/vLLM low-bit execution. Case013 and its follow-up use dequantized BF16 reference computation for quality diagnostics. Failed packed full-model correspondence, abnormal quality-process exits and remaining DEV failures are retained. Case008 R is separate Qwen3-0.6B pruning/reconstruction, not Qwen3-4B quantization repair.

See the [shared glossary](GLOSSARY.md), [CPU and reproduction instructions](GETTING_STARTED.md) and [original cases](CASEBOOK.md). Narrative is maintained in `presentation/guides/`; the builder inserts measured values from public records into both home stories and guide tables. The generated `data/practical-guide.json` and `build_manifest.json` retain exact values, field paths, the evidence commit and source hashes.

2026-10-15: guide implementation and reanalysis of existing public records. No new GPU experiment, quantization or benchmark run; reader evaluation was not performed.

<!-- case014-entry -->
## Separate0.6B compensation study

[Case014](../../cases/014-quantization-compensation/README.md) · [Pages](https://munsik-kim.github.io/inference-lab/en/case014.html#overview). This separate Qwen3-0.6B quality study does not change or contribute numbers to the Qwen3-4B recommendations/tables above.
