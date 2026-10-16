# Terms for reading DIOVA

English | [한국어](../ko/GLOSSARY.md) · [Home](../../README.md)

You do not need these terms to explain your problem. Start with the [introduction](START_HERE.md), then use this page when reading code or measurements. For the main A/B/C stories, useful entries are [saved model files](#inference), [existing setting-development tasks](#held-out), [reference computation](#quantization) and [D16/D4](#number-formats). Linked cases retain exact settings, mathematical definitions and sources.

<a id="llm-transformer"></a>
## LLM and Transformer

An LLM is a language model trained on large amounts of text to score or generate text. It works with text units called tokens, which may be words or parts of words. Qwen uses the Transformer architecture: repeated information-combining (attention) and feed-forward (MLP) blocks transform the numerical representation of those tokens.

[Related case 006](CASEBOOK.md#case-006).

<a id="inference"></a>
## Inference

Running an already trained model to obtain scores or outputs. The studies here change execution or computation, rather than train a new model.

[Related case 001](CASEBOOK.md#case-001).

**Saved model files: artifact, checkpoint and reload**

In the model-conversion stories, an artifact is the converted and saved model: weight arrays, settings and information needed to read them. A checkpoint is a saved model state. Reload means reading it back; a fresh-process check starts another program so leftover memory from the builder cannot make a broken save appear usable. Elsewhere, an artifact may mean a saved result file; the context states which.

[Case008 storage and restart](CASEBOOK.md#case-008).

<a id="gpu"></a>
## GPU

A processor that runs many numerical operations in parallel. These cases use an RTX 5080; viewing the saved reports and explorers requires no GPU.

[Related case 001](CASEBOOK.md#case-001).

<a id="quantization"></a>
## Quantization

Representing numbers with a restricted set of values, often using fewer bits. This can reduce storage or computation cost, while changing represented values; speed also depends on the available execution path.

[Related case 002](CASEBOOK.md#case-002).

**Reference calculation and packed execution**

A reference is a stated comparison calculation. In Case013, its follow-up and Case014, compressed numbers are restored to BF16 and used in matrix calculations to check quality. Packed storage means several low-bit values are stored together, for example two INT4 values in one byte. Actually executing GPU operations on that representation is a separate path. Successful storage and reference-quality checks do not establish faster packed serving.

[Case013 checks and limits](CASEBOOK.md#case-013).

<a id="number-formats"></a>
## BF16, FP8 and INT8

These names describe how numbers are represented. BF16 and FP8 are 16-bit and 8-bit floating-point formats; INT8 and INT4 use 8-bit and 4-bit integers. FP32/FP64 use 32/64-bit floating-point formats. Their ranges and spacing differ. A model’s learned values (weights) can use one format while the inputs to an operation (activations) use another. W4A16 means four-bit weights with 16-bit calculation inputs; the exact format and implementation still matter.

[Related case 006](CASEBOOK.md#case-006).

**RTN, D16 and D4 in Case013**

RTN rounds each weight to the nearest permitted quantized value and is a comparison method. D16 uses row/column scaling from the public-equation reference variant, stores weights as INT4 and keeps calculation inputs in BF16. D4 uses those same weights and also quantizes the inputs to INT4. The quality calculations restore quantized values to BF16. D16 and D4 are experiment names; they do not mean moderate and strong weight compression.

[Case013](CASEBOOK.md#case-013) · [Tool-action follow-up](../../research/agent_tool_contract_repair_20261014/README.md).

<a id="attention"></a>
## Attention

An operation that combines information from input positions using calculated weights. Case 006 changes this operation at only one layer when processing the prompt.

[Related case 006](CASEBOOK.md#case-006).

<a id="mlp"></a>
## MLP and SwiGLU

An MLP is a feed-forward calculation block inside the model. In Qwen’s SwiGLU form, the up matrix expands an input, the gate controls which resulting channels contribute, and the down matrix brings it back to the output size. These learned matrix transformations are called projections. Case007 removes matching channels, or intermediate components, from all three matrices.

[Related case 007](CASEBOOK.md#case-007).

<a id="prefill"></a>
## Prefill

The initial processing of the observed prompt. Model-prefill timing measures this model boundary; it is different from timing one operation inside it.

[Related case 006](CASEBOOK.md#case-006).

<a id="decode"></a>
## Decode

Producing subsequent tokens using the tokens already available and cached state. In Case 006 the changed attention path is limited to prefill; decode stays BF16.

[Related case 006](CASEBOOK.md#case-006).

**Changing memory: state, cache and readout**

State is information a model carries from earlier inputs and updates as it continues. A cache stores intermediate information for reuse. A readout converts the state to answer scores. Cases010/011 distinguish storing this state from changing how an answer is read: a better probability score need not delay the first wrong answer. A patch saves changed parameters; it does not by itself imply a complete new model.

[Case010](CASEBOOK.md#case-010) · [Case011](CASEBOOK.md#case-011).

<a id="kernel"></a>
## Kernel and complete-call cost

A kernel implements a numerical operation on the device. A complete call can also prepare, quantize or convert data. Timing the kernel alone omits that surrounding work.

[Related case 004](CASEBOOK.md#case-004).

<a id="latency"></a>
## Latency

Elapsed time for a stated piece of work. Always read the boundary: one MLP, an attention call, a full prompt forward, or a generated request. Speed ratios from different boundaries are not interchangeable.

[Related case 007](CASEBOOK.md#case-007).

**Request serving, graph/eager, throughput and TPOT**

Serving means accepting requests and producing model answers. Concurrency is how many requests can be in progress together. Throughput is work completed per unit time. TPOT is the average time per output token after the first token under the recorded measurement definition. Graph mode reuses prepared GPU execution sequences; eager mode dispatches operations as they occur. Compare models within the same mode and read each timing boundary.

[Case009 request measurements](CASEBOOK.md#case-009).

<a id="accuracy"></a>
## Accuracy and gold answers

Accuracy counts correct answers under a task’s scoring rule. Gold is the independently established answer key. A baseline-correct answer becoming wrong is a regression; a previously wrong answer becoming correct is a gain.

[Related case 006](CASEBOOK.md#case-006).

**Tool-call rules: schema, contract, enum and validation**

A schema states the allowed fields and value types in a request or result. A contract is the set of rules expected by the caller and tool. An enum is a fixed list of allowed values; the Case013 follow-up limits a call’s tool name to registered names. Validation checks a specific requirement. Here, actual file changes and check events are inspected separately from the AI’s final message claiming success.

[Tool-action follow-up](../../research/agent_tool_contract_repair_20261014/README.md).

<a id="probability"></a>
## Probability

The model assigns probability to possible next tokens. In four-choice scoring, probabilities are also normalized over those four labels; this can differ from probability across the entire vocabulary.

[Related case 006](CASEBOOK.md#case-006).

<a id="nll"></a>
## Negative log-likelihood (NLL)

A penalty based on how much probability the model gives the correct answer or next token: lower is better. A candidate-minus-baseline NLL change is worse when positive. In four-choice studies, choice NLL uses probabilities within the four allowed labels; full-vocabulary NLL also reflects probability assigned outside them. Cross-entropy (CE) is the related correct-answer probability loss used in Case011. These scores are different from counting correct answers.

[Related case 007](CASEBOOK.md#case-007).

**Text-prediction score: perplexity (PPL)**

PPL summarizes how much probability a model gives the actual next tokens in a text. Lower is better on the same text and scoring setup. It is not task accuracy, and different texts or tokenization rules are not a common ability ranking. Case014’s PG19 result concerns book excerpts; the same gain was not established on ArXiv articles.

[Case013 text scores](CASEBOOK.md#case-013) · [Case014 document comparison](CASEBOOK.md#case-014).

<a id="kl"></a>
## KL divergence

A directional comparison of probability distributions. Here KL(B ∥ candidate) compares the original model B with a changed model. A smaller value means closer distributions, not necessarily a better answer to the task.

[Related case 007](CASEBOOK.md#case-007).

<a id="pruning"></a>
## Structured pruning

Removing groups of model parameters and their matching computation. Case 007 physically slices the chosen MLP matrices; its 25% or 50% removal refers to that block’s channel groups, not the whole model.

[Related case 007](CASEBOOK.md#case-007).

**Fitting a smaller block: ridge regression**

Fit weights so a smaller block’s output approaches a target output, while penalizing excessively large weights. Case008 R uses this method on a fixed smaller output matrix; it does not restore removed channels or train a new foundation model. FP64 and Cholesky describe the numerical precision and solver used for that fit.

[Case008 R](CASEBOOK.md#case-008).

<a id="baseline"></a>
## Baseline

The original configuration used for comparison. B is the unpruned or unmodified BF16 control in Cases 007 and 006. Keeping its answer unchanged is different from being correct against gold.

[Related case 007](CASEBOOK.md#case-007).

<a id="held-out"></a>
## Held-out inputs

Inputs reserved from calibration and selection. Case 007 selects groups first, then evaluates 192 held-out prompts. Reopening these published records later does not make them fresh evaluation data.

[Related case 007](CASEBOOK.md#case-007).

**Setting and evaluation inputs: CAL, DEV and TEST**

Calibration (CAL) inputs supply statistics for conversion or fitting. Development (DEV) inputs help adjust settings or check implementation. Held-out/TEST inputs are reserved from those decisions under a stated protocol. Case013’s eight DEV tasks were already used while developing settings; repeating them does not test performance on new tasks. Case014’s P0/P1/P1b/P2 names identify research stages, not difficulty or product tiers.

[Case013 scope](CASEBOOK.md#case-013) · [Case014 stages](CASEBOOK.md#case-014).

<a id="post-hoc"></a>
## Post-hoc and readout diagnostic

Post-hoc analysis is motivated after results are known. Readout maps the final model state to vocabulary scores. Case 006’s FP32 readout supplement reuses the same inputs and states and remains separate from the native study.

[Related case 006](CASEBOOK.md#case-006).

**Checking a record: hash, manifest, audit and confidence interval**

A hash is a file fingerprint used to detect changes. A manifest lists files or data and their identities. An audit repeats stated checks or calculations; its scope may be saved values rather than a new model run. A confidence interval describes uncertainty under the stated sampling procedure. An interval including zero does not establish that two methods are equivalent. Status labels such as `NOT_ASSESSED` mean that the named decision, such as deployment suitability, was not evaluated.

[What local checks cover](GETTING_STARTED.md).

<a id="local-error"></a>
## Local output error

A numerical difference at a specified operation, relative to a stated reference. It describes that computation, not a percentage-point loss in task accuracy. Read which reference and aggregation were used.

[Related case 005](CASEBOOK.md#case-005).
