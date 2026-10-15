# Start here: DIOVA

English | [한국어](../ko/START_HERE.md) · [Home](../../README.md)

## 1. What problem does this address?

Running a lighter model on your GPU requires checking storage, request cost and answers separately. DIOVA builds artifact inspection and reload tools, matched answer/cost runners and checks of actual tool-task completion, using recorded open-weight model runs on an RTX 5080 16GB. Reading the results requires neither an AI degree nor a GPU.

## 2. Read online first

Open the [public English home](https://munsik-kim.github.io/inference-lab/en/index.html) without installing anything. It shows recorded results, not a live model demo. Continue to the [16GB GPU model guides](https://munsik-kim.github.io/inference-lab/en/llm-guide.html) for Qwen3-4B execution, quality and failures. The [guide entry](LLM_GUIDE.md) also explains local opening instructions.

Start with one of these three problems:

- **Make model files smaller and reload them:** [Case008 Q](https://munsik-kim.github.io/inference-lab/en/case008.html#track-q) connects existing GPTQ conversion with artifact inspection, copying, fresh-process reload and runtime-path checks. Weight file size and whole-device memory are separate measurements.
- **Check whether INT4 speeds up your requests:** [Case009](https://munsik-kim.github.io/inference-lab/en/case009.html#serving) compares matched lengths and execution modes. Gains depend on request conditions; separate quality regressions and abnormal process exits remain.
- **Check whether a claimed success actually used the tools:** the [Case013 follow-up report](../../research/agent_tool_contract_repair_20261014/records/REPORT_KO.md) records registered-name constraints, error feedback and checks of actual patch/validation events. Previously observed development tasks were reused; generalization to new inputs is unverified. This is dequantized BF16 reference computation, not packed INT4 acceleration. The report and full guide are in Korean.

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

Case008 R is separate Qwen3-0.6B pruning/reconstruction, not Qwen3-4B quantization repair. Use the [Casebook](CASEBOOK.md) for exact results and the [portfolio](PORTFOLIO.md) for implementation and external attribution.

## 5. Download or check records when needed

Read online, then choose the [offline explorers](GETTING_STARTED.md#offline-explorers) or [CPU record checks](GETTING_STARTED.md#cpu-checks). Public ZIPs do not include model weights. Public-record checks and GPU reruns have different prerequisites.

## 6. Related implementations that change one layer

<!-- claims: c007-transfer c007-quality c006-native c006-readout -->
**Case 007:** one MLP, a large block that transforms information inside the model, is divided into 16 channel groups. The selectors score groups separately or include their interactions when choosing what to remove. After cutting the matrices, the study checks 192 inputs unused for selection. Removing 25% kept the block slightly closer to its original output with interaction-aware selection; at 50%, both methods chose the same structure. The block itself ran faster, while whole-model measurements did not establish a speedup. Closeness to the original output and correct-answer probability scores also gave different conclusions. [Exact results](CASEBOOK.md#case-007).

**Case 006:** one attention operation, which combines information from the question, is changed while scores and choices among four answers are compared on identical questions. Two settings changed 5 and 8 choices out of 192 standard inputs, respectively. Neither lost a baseline-correct answer in that set, but losses occurred in a separate stress set. A follow-up examined equal top scores and the final word-score calculation on the same inputs. It is labelled separately from the original study. [Individual outcomes and limits](CASEBOOK.md#case-006).

<!-- case014-entry -->
## Recently completed: quality compensation at identical INT4 storage

[Case014](CASEBOOK.md#case-014) · [Online](https://munsik-kim.github.io/inference-lab/en/case014.html#overview) · [CPU checks](GETTING_STARTED.md#case014-cpu). A fresh-document0.6B whole-model quality study, distinct from the Qwen3-4B practical guide.
