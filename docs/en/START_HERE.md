# Start here: Inference Lab

English | [한국어](../ko/START_HERE.md) · [Home](../../README.md)

## 1. What is this project?

Inference Lab studies how calculations and answers change when we try to make an AI model lighter or faster. It combines comparison runners, code that cuts model matrices, and viewers for individual recorded results, measured locally on an RTX 5080. Reading and viewing the results needs neither an AI degree nor a GPU.

## 2. Why is making a model faster difficult?

A model runs many calculations in sequence. Shrinking one may add data-preparation work, or other calculations may take most of the time. The project therefore measures the small operation and the whole model separately. It also compares representing numbers with fewer bits and removing parts of a calculation as distinct approaches.

## 3. Why check speed and accuracy separately?

A faster calculation can return different numbers. Different numbers may still produce the same answer. Even a similar total number of correct answers can hide newly correct questions and previously correct questions that became wrong. Answer comparisons use an independently computed correct answer (gold), the original model's answer, and the probability assigned to the correct choice.

## 4. How do the seven cases connect?

The questions grew from execution to number formats, cost, answers and structural compression. This is a reading order through separate studies with different conditions.

- [001 Execution](CASEBOOK.md#case-001): check whether a model runs before and after an upstream path fix.
- [002 Number formats](CASEBOOK.md#case-002): compare two official models' memory, time and document extraction.
- [003 Numerical approximation](CASEBOOK.md#case-003): check error against different numerical references.
- [004 Call cost](CASEBOOK.md#case-004): measure speed with preparation included.
- [005 Settings](CASEBOOK.md#case-005): check speed and error requirements together.
- [006 Answers](CASEBOOK.md#case-006): track scores and changed choices on the same questions.
- [007 Structural compression](CASEBOOK.md#case-007): physically shrink matrices and compare outcomes.

## 5. Start with two examples

<!-- claims: c007-transfer c007-quality c006-native c006-readout -->
**Case 007:** one MLP, a large block that transforms information inside the model, is divided into 16 channel groups. The selectors score groups separately or include their interactions when choosing what to remove. After cutting the matrices, the study checks 192 inputs unused for selection. Removing 25% kept the block slightly closer to its original output with interaction-aware selection; at 50%, both methods chose the same structure. The block itself ran faster, while whole-model measurements did not establish a speedup. Closeness to the original output and correct-answer probability scores also gave different conclusions. [Exact results](CASEBOOK.md#case-007).

**Case 006:** one attention operation, which combines information from the question, is changed while scores and choices among four answers are compared on identical questions. Two settings changed 5 and 8 choices out of 192 standard inputs, respectively. Neither lost a baseline-correct answer in that set, but losses occurred in a separate stress set. A follow-up examined equal top scores and the final word-score calculation on the same inputs. It is labelled separately from the original study. [Individual outcomes and limits](CASEBOOK.md#case-006).

## 6. How can I view results without running code?

Follow the [download and opening instructions](GETTING_STARTED.md#offline-explorers): download a ZIP, extract it, and open its HTML in a browser. The existing English explorers show saved results. GitHub's HTML source page displays code rather than the working screen. Opening an explorer does not run a model.

## 7. Where can I inspect the implementation?

- **How is only one operation changed?** The [Case 006 intervention](../../cases/006-attention-decision-stability/src/intervention.py) changes the selected layer and restores the original path on exit or exceptions.
- **How is a block made smaller?** The [Case 007 surgery code](../../cases/007-interaction-aware-mlp-pruning/src/surgery.py) applies the same channel indices to the corresponding matrix rows and columns.
- **How are records checked again?** The [CPU guide](GETTING_STARTED.md#cpu-checks) separates checks of public records from model execution.

Use the [glossary](GLOSSARY.md) for terms, the [Casebook](CASEBOOK.md) for numbers and conditions, and the [portfolio](PORTFOLIO.md) for implementation contributions and upstream attribution.
