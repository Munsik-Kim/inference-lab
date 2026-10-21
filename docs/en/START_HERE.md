# Start here: DIOVA

English | [한국어](../ko/START_HERE.md) · [Home](../../README.md)

## 1. What problem does this address?

Choose a route for running a model, reviewing the work, comparing your results, or auditing preserved evidence. Reading saved records and using the CPU comparison tool requires no GPU.

## 2. Read online first

Open the [public English home](https://munsik-kim.github.io/inference-lab/en/index.html) without installing anything. It shows recorded results, not a live model demo. Continue to the [16GB GPU model guides](https://munsik-kim.github.io/inference-lab/en/llm-guide.html) for Qwen3-4B execution, quality and failures. The [guide entry](LLM_GUIDE.md) also explains local opening instructions.

Choose a route by your purpose:

- **Run a local model:** start with [model guides](LLM_GUIDE.md) to see what ran, what you need and what failed.
- **Assess fit for a role or technical task:** read [featured A/B/C and code evidence](PORTFOLIO.md). A connects [Case008 Q storage/reload](https://munsik-kim.github.io/inference-lab/en/case008.html#track-q) → [Case009 actual requests](https://munsik-kim.github.io/inference-lab/en/case009.html#serving).
- **Compare your own saved answers:** run the [diova-compare example with invented answers](../../packages/diova-compare/examples/compare/README.md), then put your records into its documented format. This independent CPU tool does not require a model.
- **Recheck an experiment:** choose a historical CPU audit in [developer reproduction](GETTING_STARTED.md). This is separate from comparing your own records.

Start featured work C with [Case017 results and recorded episodes](../../cases/017-agent-contract-cost-bench/REPORT.md). Continue to [CPU replay without a model](../../cases/017-agent-contract-cost-bench/REPRODUCE.md) or the [earlier Case013 follow-up](../../research/agent_tool_contract_repair_20261014/README.md).

## 3. Reference guides

[Read memory, speed and answer metrics](https://munsik-kim.github.io/inference-lab/en/measurements.html) · [Glossary](GLOSSARY.md). File size, runtime memory and answer quality are different measurements.

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

<!-- case015-entry -->
<a id="case015-cpu"></a>
## API rename regression testing

Check whether an AI finishes the same task after a tool rename. The example is **read item count and price per item → calculate the purchase total → save it → validate the record**. Functions and input formats stay fixed; only names change, such as `read_order` to `readOrder`. Across 16 problems, BF16 was **15/16→16/16**, RTN **16/16→16/16**, and D16 **12/16→12/16**. The [task and terminology guide](../../cases/015-tool-interface-sensitivity/README.md) explains orders as purchase records, unit price as price per item, and how to read the results. The 96 runs of the earlier P1 study and the new A1 application are counted separately.

<a id="3-choose-an-instruction-path"></a>

## Case017 — Agent Contract–Cost Bench

We built a benchmark that checks how tool call policies change completion of saved state, observed validation and final answers. It uses owned synthetic records for product totals, inventory transfers and document updates.

With BF16 weights and native NVFP4 KV, grammar changes completion 7/24→15/24 and correct saved states 7/24→22/24 on 24 new synthetic tasks: 8 recoveries and 0 losses. Added execution costs and remaining failures are reported alongside the gain. More tasks completed, while total execution costs also increased. Dividing all 24 runs, including failures, by completions gives approximately 3.62% more input tokens, 12.36% more output tokens and 11.27% more episode-wall time per completion.

Protocol deviation: BF16 KV with free output completed 5/8 DEV tasks, below the original 6/8 expansion criterion. The local plan and execution gate omitted this success threshold and proceeded to P2 after checking completion, freezing and sealing. No separate waiver approval was found. We disclose this deviation and present P2 as an exploratory measurement on new synthetic tasks.

[Report](../../cases/017-agent-contract-cost-bench/REPORT.md) · [CPU command and scope](../../cases/017-agent-contract-cost-bench/REPRODUCE.md) · [Strict parser](../../cases/017-agent-contract-cost-bench/src/agent_contract_cost/contracts.py) · [Independent auditor](../../cases/017-agent-contract-cost-bench/src/agent_contract_cost/audit.py)
