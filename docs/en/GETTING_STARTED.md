# Read, explore, recalculate, reproduce

English | [한국어](../ko/GETTING_STARTED.md) · [Home](../../README.md)

The repository contains individual studies, not one installable service or a shared model-serving CLI. Choose the level of access you need. Reading and viewing recorded results do not require a GPU.

## 1. Read the evidence

Start with the [Casebook](CASEBOOK.md), then follow a case's result and methods links. The [portfolio guide](PORTFOLIO.md) connects technical skills to source files. The earlier [extraction-results discussion](../../notes/case002-results-and-discussion.md) remains available as a short case-specific reading. Original case documentation remains the technical record; these guides do not replace its protocols or measurements.

<a id="offline-explorers"></a>
## 2. Open the recorded explorers

<!-- claims: c006-package c006-browser -->
Use the [current combined Case006 ZIP](../../downloads/case006_decision_stability_with_readout_20260917_docfix1.zip) and its [size/hash metadata](../../downloads/case006_decision_stability_with_readout_20260917_docfix1.json). On GitHub's file page, choose the download action rather than the HTML source preview. The older combined ZIP without `docfix1` is retained as a historical version; the current version fixes reproduction paths without changing the scientific evidence. Cases001, 002 and 004 also have recorded ZIPs linked from their case pages. There is no dedicated Case003 or Case005 ZIP linked here.

Unzip to a local directory and open either file in a browser:

```text
006-attention-decision-stability/demo/index.html
006-attention-decision-stability/supplemental/readout-ties-v1/demo/index.html
```

Readout maps the final hidden state to vocabulary scores. The first explorer shows the original (native) output computation; the second compares it with a separate diagnostic (shadow) computation on the same inputs, explicitly post-hoc. Both embed recorded data; they do not run the model. GitHub's HTML source view is not a running demo, and no Pages service is required.

B is the BF16 baseline; A_PUBLIC and V4 are two low-precision attention settings within the same BF16-weight model. Gold is the independently computed correct answer.

**A short walkthrough:** start with STANDARD, select a task and item ID, compare gold with B/A_PUBLIC/V4 scores, then inspect the outcome type. In the supplement, compare native and shadow readouts without treating them as independent samples. Inspect the separately selected stress set only after noting its different sampling rule. An empty category means none observed, not zero future risk.

The supplement has recorded Windows Edge `file://` checks at desktop and emulated tablet viewports. That is not a real iPad test: **real iPad NOT_TESTED**. File previewers that disable JavaScript may not run the controls. The Markdown, figures and JSON remain readable separately. See the [actual browser record](../../cases/006-attention-decision-stability/supplemental/readout-ties-v1/provenance/browser_validation.json), not an implied cross-device guarantee.

<a id="case007-explorer"></a>
### Case007: inspect compression and its trade-offs

<!-- claims: c007-package -->
Download the [reviewed Case007 ZIP](../../downloads/case007_mlp_pruning_reviewed_publication_v1.zip) and check its [size/hash metadata](../../downloads/case007_mlp_pruning_reviewed_publication_v1.json). Extract it and open `007-interaction-aware-mlp-pruning/demo/index.html` locally. The English UI uses retained measurements only. Choose a method and deletion budget, inspect the prompt ID and removed groups, and separate local reconstruction error from gold-answer transitions. Its headline status is the frozen **COMPLETED_NO_CLEAR_TRANSFER**, not deployment approval. [Result interpretation](CASEBOOK.md#case-007).

The original and current publication manifests have different scopes. The existing Case007 package verifier uses the Python standard library. From the checkout root (or the extracted case directory, omitting `cd`), use a new external output file:

```bash
cd cases/007-interaction-aware-mlp-pruning
python -B scripts/verify_publication.py --case . --output /tmp/inference-lab-case007-docs-check-new.json
```

This checks preservation and retained scalar/table consistency, not private vectors or GPU execution. For full CPU reanalysis and the distinct GPU procedure, use [Case007 reproduction instructions](../../cases/007-interaction-aware-mlp-pruning/REPRODUCTION.md). The case-only ZIP is not a repository backup; links to earlier cases need the repository. Its historical browser checks are not rerun by this guide, and real iPad remains **NOT_TESTED**.

## 3. Check Case006, then reanalyze on CPU

<!-- claims: c006-verification -->
From a repository checkout's root, use an available CPU Python environment. This minimal package check uses the Python standard library; no GPU, model or new installation is needed. The output must be a **new file outside the case**; choose another name if it exists.

```bash
cd cases/006-attention-decision-stability
python -B scripts/verify_publication.py --output /tmp/inference-lab-case006-docs-check-new.json
```

For the extracted ZIP, first enter its `006-attention-decision-stability` directory instead. The same verifier command then applies. It checks identities and inventory, not the scientific arithmetic.

To recalculate scores and intervals, follow [CPU reanalysis commands](../../cases/006-attention-decision-stability/REPRODUCTION.md#cpu-analysis-of-retained-measurements). The recorded analysis environment uses Python 3.12.14 and NumPy 2.3.5; figure generation also used Matplotlib 3.10.8 and Pillow 12.3.0. This guide adds no installation command or dependency update. Large reanalyses and GPU examples were not run for this documentation revision; only the minimal package verifier was exercised with the existing CPU environment.

`SHA256SUMS` describes the original reviewed snapshot. `PUBLICATION_SHA256SUMS` covers the current combined publication tree, excluding itself. The supplement retains its own frozen manifest. The historical 59-test exact-inventory suite belongs to the original-only archive, not the combined tree; do not weaken it to accommodate new files. [Verification scopes](../../cases/006-attention-decision-stability/REPRODUCTION.md#publication-package-verification).

## 4. Treat GPU reproduction as a separate experiment

Recreating kernel outputs requires the pinned model, sources, binaries and compatible hardware environment. Public scalar records cannot reconstruct excluded hidden vectors or full Q/K/V. The [pinned-local GPU instructions](../../cases/006-attention-decision-stability/REPRODUCTION.md#pinned-local-gpu-replay) and [original-to-readout directory contract](../../cases/006-attention-decision-stability/REPRODUCTION.md#connecting-original-replay-to-the-readout-supplement) describe those requirements. They are future reproduction instructions, not steps executed for this guide. CPU recalculation establishes consistency of saved evidence, not independent GPU replication or deployment readiness.
