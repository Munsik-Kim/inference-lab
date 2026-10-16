# Start here: DIOVA

English | [한국어](../ko/START_HERE.md) · [Home](../../README.md)

## 1. What problem does this address?

A model may take less space yet give worse answers or fail to finish the requested work. DIOVA checks these issues together. I built tools that inspect and restart saved models, compare answers and execution costs on the same questions, and check whether an AI actually made and validated a file change. The records use public Qwen models on an RTX 5080 16GB. You can read them without specialist AI knowledge or a GPU.

For hiring or technical collaboration, this is evidence of work in local AI execution, making trained models run (inference engineering), model optimization, checking for quality losses after changes (regression testing), and evaluating tool automation. Tasks to discuss include diagnosing local execution failures, comparing model settings, building answer/action checks, and reporting measurements with traceable sources. [The portfolio](PORTFOLIO.md) links each capability to code.

## 2. Read online first

Open the [public English home](https://munsik-kim.github.io/inference-lab/en/index.html) without installing anything. It shows recorded results, not a live model demo. Continue to the [16GB GPU model guides](https://munsik-kim.github.io/inference-lab/en/llm-guide.html) for Qwen3-4B execution, quality and failures. The [guide entry](LLM_GUIDE.md) also explains local opening instructions.

Choose a route by your purpose:

- **Run a local model:** start with [model guides](LLM_GUIDE.md) to see what ran, what you need and what failed.
- **Assess fit for a role or technical task:** read [featured A/B/C and code evidence](PORTFOLIO.md). A connects [Case008 Q storage/reload](https://munsik-kim.github.io/inference-lab/en/case008.html#track-q) → [Case009 actual requests](https://munsik-kim.github.io/inference-lab/en/case009.html#serving).
- **Compare your own saved answers:** run the [diova-compare example with invented answers](../../packages/diova-compare/examples/compare/README.md), then put your records into its documented format. This independent CPU tool does not require a model.
- **Recheck an experiment:** choose a historical CPU audit in [developer reproduction](GETTING_STARTED.md). This is separate from comparing your own records.

Featured C tests [available tool names, call-error explanations and actual action completion](../../research/agent_tool_contract_repair_20261014/README.md). It reuses eight existing tasks used while adjusting settings (DEV). D16 stores weights as four-bit integers (INT4) with BF16 calculation inputs; D4 uses the same weights and also quantizes inputs to INT4. Quantized values are restored to BF16 for quality checks (reference computation). Results on new tasks and speed from directly operating on packed INT4 remain unverified.

## 3. Choose an instruction path

| Your question | Entry point |
|---|---|
| What ran for this model, and where did it fail? | [Model guides](LLM_GUIDE.md): full Qwen3-4B detail in Korean, short English summaries |
| How do memory, speed and quality numbers differ? | [How to read the measurements](https://munsik-kim.github.io/inference-lab/en/measurements.html) |
| How can I check the records and tools locally? | [Developer reproduction](GETTING_STARTED.md); the existing `guide.html` has this role too |
| What does a term mean? | [Glossary](GLOSSARY.md) |

Fitting in memory, serving faster and doing your task correctly are different decisions. Results from different experiments must not be combined into a single configuration's performance.

## 4. How do the cases connect?

Each case has its own model, inputs and conditions; you do not need to read all of them first.

- [001 Execution compatibility](CASEBOOK.md#case-001)
- [002 Official BF16/FP8 document extraction](CASEBOOK.md#case-002)
- [003 References for numerical approximation](CASEBOOK.md#case-003)
- [004 Complete call cost](CASEBOOK.md#case-004)
- [005 Speed and error settings](CASEBOOK.md#case-005)
- [006 Scores and changed answers](CASEBOOK.md#case-006)
- [007 Physically smaller matrices](CASEBOOK.md#case-007)
- [008 Quantized storage/reload and separate structural reconstruction](CASEBOOK.md#case-008)
- [009 Serving cost and official quality](CASEBOOK.md#case-009)
- [010 State storage, restart and memory horizon](CASEBOOK.md#case-010)
- [011 Changing the readout of the same state](CASEBOOK.md#case-011)
- [012 Image-generation budget allocation](CASEBOOK.md#case-012)
- [013 Weight error, answers and tool use](CASEBOOK.md#case-013)

Case008 R removes part of one block in the smaller Qwen3-0.6B model (pruning), then fits its remaining output weights. It is separate from the Qwen3-4B compression path. Use the [Casebook](CASEBOOK.md) for exact results and the [portfolio](PORTFOLIO.md) for implementation and external attribution.

## 5. Download or check records when needed

Read online, then choose the [offline explorers](GETTING_STARTED.md#offline-explorers) or [CPU record checks](GETTING_STARTED.md#cpu-checks). Public ZIPs do not include model weights. Public-record checks and GPU reruns have different prerequisites.

## 6. Related implementations that change one layer

<!-- claims: c007-transfer c007-quality c006-native c006-readout -->
**Case 007:** one MLP, a large block that transforms information inside the model, is divided into 16 channel groups. The selectors score groups separately or include their interactions when choosing what to remove. After cutting the matrices, the study checks 192 inputs unused for selection. Removing 25% kept the block slightly closer to its original output with interaction-aware selection; at 50%, both methods chose the same structure. The block itself ran faster, while whole-model measurements did not establish a speedup. Closeness to the original output and correct-answer probability scores also gave different conclusions. [Exact results](CASEBOOK.md#case-007).

**Case 006:** one attention operation, which combines information from the question, is changed while scores and choices among four answers are compared on identical questions. Two settings changed 5 and 8 choices out of 192 standard inputs, respectively. Neither lost a baseline-correct answer in that set, but losses occurred in a separate stress set. A follow-up examined equal top scores and the final word-score calculation on the same inputs. It is labelled separately from the original study. [Individual outcomes and limits](CASEBOOK.md#case-006).

<!-- case014-entry -->
## Deeper research: quality compensation at identical INT4 storage

[Case014](CASEBOOK.md#case-014) · [Online](https://munsik-kim.github.io/inference-lab/en/case014.html#overview) · [CPU checks](GETTING_STARTED.md#case014-cpu). This separate Qwen3-0.6B study tests whether correcting compression error improves whole-model text prediction on fresh documents while keeping the converted matrices’ storage unchanged. PG19 gains and ArXiv limits are reported separately from the Qwen3-4B practical guide.
