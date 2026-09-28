# Case 011 — Frozen-State Readout Adaptation and Length Extrapolation

[한국어](README.ko.md) · [DIOVA home](../../README.md)

Built tools that replace only the final answer-reading layer, save its two tensors as a patch, and apply the patch in a fresh process. The comparison measures gold-label scores and first errors while each readout uses the same evolving state.

[What was built](#implementation) · [Key findings](#results) · [Run and inspect](#reproduce) · [Design and experiment](#design) · [Sources](#sources)

<a id="implementation"></a>
## What was built

**1,158 final-layer values · 0 extra recurrent-state bytes · Fresh-process patch reload**

The state continues to evolve with every input. Within each storage mode, ORIGINAL, SHORT and MIXED read the same state and feature at each step, and their answers never change the next state. Native and INT8 states are separate trajectories.

- [Feature-boundary adapter](source/adapter.py): splits the existing GELU output from the final Linear. [State/feature tests](tests/test_adapter.py).
- [Two-tensor patch loader](source/head_patch.py): verifies the base checkpoint hash, shapes and dtypes before applying weight and bias. [Artifact/error tests](tests/test_fitting_patch.py).
- [Paired runner](source/run_eval.py) and [separate auditor](analysis/audit.py): compare predictions, scores and first errors on the same inputs. [Aggregation tests](tests/test_aggregate.py).

Python · NumPy · PyTorch. The zero-byte figure concerns **additional recurrent state**; patch files and fitting features have [separate ledger entries](results/byte_ledger.json).

<a id="results"></a>
## Key findings

**Gold-label scores improved, but the average uninterrupted correct prefix became shorter.** With INT8 MIXED, gold cross-entropy (CE, lower is better) decreased beyond fitting support, at positions 257–2,048, for all three checkpoints. Whole-sequence token accuracy rose in two of three; mean uninterrupted correct length (RMST0, higher is better) fell in all three.

| Checkpoint | ORIGINAL | MIXED | Change |
| --- | --- | --- | --- |
| 0 | 285.84 | 275.96 | -9.89 |
| 1 | 264.59 | 254.19 | -10.40 |
| 2 | 239.80 | 226.31 | -13.49 |

Units are **symbolic tokens continuously correct before the first error**, averaged over the same 1,024 sequences per checkpoint through 2,048 steps. This is symbolic tracking, not a natural-language context-length measurement. [Primary intervals and Native results](REPORT.md#results).

SHORT is a control whose final weights stayed identical under the fixed stopping rule. The selected MIXED fits for seeds 0/1 reached the 200-iteration cap. [Selection and solver status](REPORT.md#experiments).

<a id="reproduce"></a>
## Run and inspect

From the repository or extracted public ZIP root, use Python with NumPy to audit public predictions and scalars. Choose a new external output directory.

```bash
python -B cases/011-frozen-state-readout-adaptation/publication/run_checks.py \
  --output /tmp/diova-case011-publication-check-new
```

[Model-free and synthetic checks](REPRODUCTION.md) · [Result walkthrough](https://munsik-kim.github.io/inference-lab/en/case011.html#overview) · [Original offline table](demo/en.html)

The public package contains code and measurement records. Applying an actual patch needs the original checkpoint and private head tensors; those weights are excluded from the ZIP.

<a id="design"></a>
## Design and experiment

Final-layer supervised adaptation uses three existing CKDA symbolic-tracking checkpoints and two fixed storage modes. SHORT and MIXED fit 16,384 Native feature rows from positions 1–32 and 1–256, respectively, then select by DEV gold CE. Three readouts share features from each of six TEST state rollouts.

[Full report](REPORT.md) · [Methods](METHODS.md) · [Protocol](configs/protocol.json) · [Post-hoc analysis of the same records](REPORT.md#analysis)

<a id="sources"></a>
## Sources

[NOTICE](NOTICE.md) distinguishes the ComplexKDA model/task, Case010 state codecs, PyTorch/NumPy and this project’s intervention, storage and audit work, including Codex assistance. [Publication records](publication/README.md) trace the original review documents and verification scopes.

[Download code and recorded measurements (ZIP; no model weights)](https://github.com/Munsik-Kim/inference-lab/raw/refs/heads/main/downloads/case011_readout_adaptation_reviewed_publication_v1.zip)
