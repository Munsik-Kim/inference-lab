# Terms for reading DIOVA

English | [한국어](../ko/GLOSSARY.md) · [Home](../../README.md)

These 20 entries explain terms used in this repository. Start with the [five-minute introduction](START_HERE.md); the linked cases provide mathematical definitions, settings and sources.

<a id="llm-transformer"></a>
## LLM and Transformer

An LLM is a language model trained on large amounts of text. Qwen uses the Transformer architecture, which repeats attention and MLP blocks to process token representations.

[Related case 006](CASEBOOK.md#case-006).

<a id="inference"></a>
## Inference

Running an already trained model to obtain scores or outputs. The studies here change execution or computation, rather than train a new model.

[Related case 001](CASEBOOK.md#case-001).

<a id="gpu"></a>
## GPU

A processor that runs many numerical operations in parallel. These cases use an RTX 5080; viewing the saved reports and explorers requires no GPU.

[Related case 001](CASEBOOK.md#case-001).

<a id="quantization"></a>
## Quantization

Representing numbers with a restricted set of values, often using fewer bits. This can reduce storage or computation cost, while changing represented values; speed also depends on the available execution path.

[Related case 002](CASEBOOK.md#case-002).

<a id="number-formats"></a>
## BF16, FP8 and INT8

BF16 and FP8 are 16-bit and 8-bit floating-point formats; INT8 uses 8-bit integers. These formats have different ranges and spacing. A weight format and the format used by one operation can differ.

[Related case 006](CASEBOOK.md#case-006).

<a id="attention"></a>
## Attention

An operation that combines information from input positions using calculated weights. Case 006 changes this operation at only one layer when processing the prompt.

[Related case 006](CASEBOOK.md#case-006).

<a id="mlp"></a>
## MLP and SwiGLU

An MLP is the feed-forward block that transforms each token representation. Qwen’s SwiGLU form uses gate, up and down projections. Case 007 removes matching channels from these matrices.

[Related case 007](CASEBOOK.md#case-007).

<a id="prefill"></a>
## Prefill

The initial processing of the observed prompt. Model-prefill timing measures this model boundary; it is different from timing one operation inside it.

[Related case 006](CASEBOOK.md#case-006).

<a id="decode"></a>
## Decode

Producing subsequent tokens using the tokens already available and cached state. In Case 006 the changed attention path is limited to prefill; decode stays BF16.

[Related case 006](CASEBOOK.md#case-006).

<a id="kernel"></a>
## Kernel and complete-call cost

A kernel implements a numerical operation on the device. A complete call can also prepare, quantize or convert data. Timing the kernel alone omits that surrounding work.

[Related case 004](CASEBOOK.md#case-004).

<a id="latency"></a>
## Latency

Elapsed time for a stated piece of work. Always read the boundary: one MLP, an attention call, a full prompt forward, or a generated request. Speed ratios from different boundaries are not interchangeable.

[Related case 007](CASEBOOK.md#case-007).

<a id="accuracy"></a>
## Accuracy and gold answers

Accuracy counts correct answers under a task’s scoring rule. Gold is the independently established answer key. A baseline-correct answer becoming wrong is a regression; a previously wrong answer becoming correct is a gain.

[Related case 006](CASEBOOK.md#case-006).

<a id="probability"></a>
## Probability

The model assigns probability to possible next tokens. In four-choice scoring, probabilities are also normalized over those four labels; this can differ from probability across the entire vocabulary.

[Related case 006](CASEBOOK.md#case-006).

<a id="nll"></a>
## Negative log-likelihood (NLL)

A loss based on the probability assigned to the correct answer: lower is better. A candidate-minus-baseline NLL change is worse when positive. The choice NLL used here conditions on the four allowed labels.

[Related case 007](CASEBOOK.md#case-007).

<a id="kl"></a>
## KL divergence

A directional comparison of probability distributions. Here KL(B ∥ candidate) compares the original model B with a changed model. A smaller value means closer distributions, not necessarily a better answer to the task.

[Related case 007](CASEBOOK.md#case-007).

<a id="pruning"></a>
## Structured pruning

Removing groups of model parameters and their matching computation. Case 007 physically slices the chosen MLP matrices; its 25% or 50% removal refers to that block’s channel groups, not the whole model.

[Related case 007](CASEBOOK.md#case-007).

<a id="baseline"></a>
## Baseline

The original configuration used for comparison. B is the unpruned or unmodified BF16 control in Cases 007 and 006. Keeping its answer unchanged is different from being correct against gold.

[Related case 007](CASEBOOK.md#case-007).

<a id="held-out"></a>
## Held-out inputs

Inputs reserved from calibration and selection. Case 007 selects groups first, then evaluates 192 held-out prompts. Reopening these published records later does not make them fresh evaluation data.

[Related case 007](CASEBOOK.md#case-007).

<a id="post-hoc"></a>
## Post-hoc and readout diagnostic

Post-hoc analysis is motivated after results are known. Readout maps the final model state to vocabulary scores. Case 006’s FP32 readout supplement reuses the same inputs and states and remains separate from the native study.

[Related case 006](CASEBOOK.md#case-006).

<a id="local-error"></a>
## Local output error

A numerical difference at a specified operation, relative to a stated reference. It describes that computation, not a percentage-point loss in task accuracy. Read which reference and aggregation were used.

[Related case 005](CASEBOOK.md#case-005).
