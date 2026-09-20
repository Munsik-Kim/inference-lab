# Read, explore, recalculate, reproduce

English | [한국어](../ko/GETTING_STARTED.md) · [Home](../../README.md)

[New here?](START_HERE.md) · [Plain-language glossary](GLOSSARY.md)

Read the studies, explore recorded inputs, or run a small CPU check. GPU reproduction has its own environment and commands.


<a id="case008-paths"></a>
## Case008: choose what to run

| Your purpose | Entry point |
|---|---|
| Inspect results and recalculate recorded scores | [Q/R result screen](https://munsik-kim.github.io/inference-lab/en/case008.html), then the CPU evidence command below. |
| Try the implementation without a pretrained model | [Tiny CPU save–reload example](#tiny-model-demo). |
| Build a full Q or R model artifact | [Pinned environments and complete build procedures](../../cases/008-build-reconstruct-reload/REPRODUCTION.md). Model weights are separate local prerequisites. |
| Reproduce the frozen study | [Protocol](../../cases/008-build-reconstruct-reload/configs/protocol_draft.json), fixed inputs and [historical snapshot restoration](../../cases/008-build-reconstruct-reload/publication/README.md). |

From the repository root, use Python 3.12.14 with NumPy 2.3.5 and a **new external output directory**:

```bash
OUT=$(mktemp -d)
python3 -B tools/case008_checks/run.py --mode evidence --output "$OUT/evidence"
```

This runs the publication tests, original analysis, scalar and timing audits, and the same-record NLL/recovery diagnostic. Regenerated summary, additional/local tables and post-hoc JSON are compared byte-for-byte with preserved files. It does not rerun model inference or reconstruct excluded full vectors. The `Case 008 CPU contracts` workflow separates this job from modelpack tests; Portfolio CPU checks remains the documentation/site suite.

<a id="tiny-model-demo"></a>
## Run a tiny model's save–reload contract

Use a separate CPU environment with the [fixture dependency versions](../../tools/modelpack_demo/requirements-cpu.txt): Python 3.12.14, PyTorch 2.13.0, Transformers 5.17.0, safetensors 0.8.0 and NumPy 2.3.5. The CI job installs the CPU-only PyTorch wheel; an existing compatible environment also works. No pretrained weights, tokenizer or GPU is required. Environment setup details are in the [demo README](../../tools/modelpack_demo/README.md).

```bash
OUT=$(mktemp -d)
python3 -B tools/modelpack_demo/roundtrip.py --output "$OUT/tiny-roundtrip"
python3 -B -m unittest discover -s tests/modelpack_demo -v
```

The command creates a random two-layer Qwen fixture on CPU, reduces layer 1's intermediate width from 64 to 48, saves it and lets the builder process exit. A second process loads a copied artifact from another directory, with access to the original builder directory blocked. It compares complete finite outputs, tensor shapes and tied embedding/head identity. It reuses the frozen Case008 loader; the wrapper does not alter research code.

The checked summary includes `TINY_RANDOM_CPU_FIXTURE`, widths `[64,48]`, gate/up `[48,32]`, down `[32,48]`, matching logits and shared weights. Generated tiny weights and process logs stay in the external output directory. This is a serialization test, with no trained-model quality or speed claim. To run the historical modelpack tests and this demo together:

```bash
OUT=$(mktemp -d)
python3 -B tools/case008_checks/run.py --mode modelpack --output "$OUT/modelpack"
```

The wrapper restores the hash-checked historical tree before its exact-inventory tests. It also checks missing/invalid artifact behavior; it never weakens a frozen inventory test for the current publication tree.


## Open the published result viewer

[English home](https://munsik-kim.github.io/inference-lab/en/index.html) · [Case007: structured pruning](https://munsik-kim.github.io/inference-lab/en/case007.html) · [Case006: decisions and ties](https://munsik-kim.github.io/inference-lab/en/case006.html). These Pages screens use saved measurements and require no installation or GPU. Use the local build below for offline viewing.

<a id="local-showcase"></a>
## Open the bilingual showcase

The new screen shows Case007 group selection and Case006 answer scores. It reads retained measurements, with separate native and post-hoc readout views. From this repository's root, use Python 3.12 and choose a new external output directory:

```bash
OUT=$(mktemp -d)
python3 -B tools/showcase/build.py --output "$OUT/site" --base-path /inference-lab/
python3 -B tools/showcase/check.py --site "$OUT/site" --output "$OUT/site-check.json"
```

Open `$OUT/site/index.html` in a browser, then choose English or 한국어. The build uses the Python standard library. All scripts and display data are local; `file://` works without a server. The same relative resource paths also support a site mounted at `/inference-lab/`. The published Pages viewer and a local build use the same static presentation code.

Read each detail page in order: **objective → dataset → assumptions and theory → experiment design → validation → results → interpretation → conclusion**. Use the contents links or **Jump to individual results**. The complete-study tables above are separate from the filtered subset statistics below. The [GitHub casebook](CASEBOOK.md) uses the same structure for all eight cases.

In Case007, choose 25% or 50% deletion, then a task or input ID. The table compares B with both selectors at the same budget. In Case006, choose standard or selected stress, then native or the labelled FP32 diagnostic; each candidate stays paired with B under that readout. Filters and the chosen ID persist in the URL and across language changes. JSON download includes the displayed comparison at its original numeric precision. Guided examples use the first fixed ID per task, not selected success stories.

The display omits deterministic irrelevant filler and token arrays, and links the exact public input and score records. Case007 timing is a separate six-input aggregate, not a measurement of each displayed item. Four option logits permit choice-score recalculation; retained full-vocabulary KL and full-output validity flags have a wider original measurement scope.

The local screen was exercised with existing Windows Edge through CDP at widths 390, 768 and 1280 pixels, including `file://`, filters, empty results, URL restoration, language switches and JSON download. These are emulated viewports; **real iPad NOT_TESTED**. The original explorers below retain their historical browser records. [Optional browser checker](../../tools/showcase/browser_check.cjs) · [Builder and display checks](../../tools/showcase/check.py).

<a id="cpu-selector"></a>
## Rerun the Case007 selector on CPU

Use an existing Python 3.12.14 environment with NumPy 2.3.5; Torch, Transformers and model files are unnecessary. The checked example was run with that environment. From the checkout root, let `CPU_PY` identify that interpreter and use a new output file:

```bash
CPU_PY=python3
OUT=$(mktemp -d)
"$CPU_PY" -B tools/showcase/replay_selection.py --output "$OUT/selection.json"
"$CPU_PY" -B -m unittest discover -s tests/showcase -v
```

The wrapper imports the preserved selector under a unique module name and evaluates the calibration Q matrix. It verifies source hashes, finite symmetric 16×16 Q, contiguous group mapping, frozen budgets and the original lexicographic tie rule. It does not access held-out statistics for selection.

| Removal budget | Recomputed groups; match with saved selection |
|---|---|
| 4/16 | INDEPENDENT `[8,12,13,14]`; PAIRWISE `[8,13,14,15]`; both match |
| 8/16 | Both `[7,8,9,11,12,13,14,15]`; both match |

Each method evaluates 1,820 or 12,870 small CPU objective values. **Model forwards: 0.** The JSON retains objective values and definitions: independent uses the selected diagonal sum; pairwise uses the entire selected submatrix sum. They are different objectives. This checks the recorded selection logic; GPU quality and timing remain the original experiment.


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

<a id="cpu-checks"></a>
## 3. Check Case006, then reanalyze on CPU

<!-- claims: c006-verification -->
From a repository checkout's root, use an available CPU Python environment. This minimal package check uses the Python standard library; no GPU, model or new installation is needed. The output must be a **new file outside the case**; choose another name if it exists.

```bash
cd cases/006-attention-decision-stability
python -B scripts/verify_publication.py --output /tmp/inference-lab-case006-docs-check-new.json
```

For the extracted ZIP, first enter its `006-attention-decision-stability` directory instead. The same verifier command then applies. It checks identities and inventory, not the scientific arithmetic.

To recalculate scores and intervals, follow [CPU reanalysis commands](../../cases/006-attention-decision-stability/REPRODUCTION.md#cpu-analysis-of-retained-measurements). The recorded analysis environment uses Python 3.12.14 and NumPy 2.3.5; figure generation also used Matplotlib 3.10.8 and Pillow 12.3.0. This guide adds no installation command or dependency update. The showcase checks run selection replay, package verification and selected scalar audits. Full study bootstrap/figure regeneration and GPU examples remain separate procedures.

`SHA256SUMS` describes the original reviewed snapshot. `PUBLICATION_SHA256SUMS` covers the current combined publication tree, excluding itself. The supplement retains its own frozen manifest. The historical 59-test exact-inventory suite belongs to the original-only archive, not the combined tree; do not weaken it to accommodate new files. [Verification scopes](../../cases/006-attention-decision-stability/REPRODUCTION.md#publication-package-verification).

## 4. Treat GPU reproduction as a separate experiment

Recreating kernel outputs requires the pinned model, sources, binaries and compatible hardware environment. Public scalar records cannot reconstruct excluded hidden vectors or full Q/K/V. The [pinned-local GPU instructions](../../cases/006-attention-decision-stability/REPRODUCTION.md#pinned-local-gpu-replay) and [original-to-readout directory contract](../../cases/006-attention-decision-stability/REPRODUCTION.md#connecting-original-replay-to-the-readout-supplement) describe those requirements. They are future reproduction instructions, not steps executed for this guide. CPU recalculation establishes consistency of saved evidence, not independent GPU replication or deployment readiness.

The earlier beginner-layer edit checked documentation, links and preservation only. The current Case008 CPU commands were separately executed and checked; GPU reproduction and full-model builds were not rerun.

## Case 008: report and standalone evidence bundle

Read the [structured report](../../cases/008-build-reconstruct-reload/REPORT.md), then download the [code/recipe/evidence ZIP](../../downloads/case008_build_reconstruct_reload_reviewed_publication_v2.zip) and [metadata](../../downloads/case008_build_reconstruct_reload_reviewed_publication_v2.json). The extracted root includes tools/modelpack and the case. Open `cases/008-build-reconstruct-reload/demo/index.html` locally for its original English viewer. The bilingual site summary shows Q storage/memory and R before/after repair; the original viewer retains detailed records. [CPU and historical-restore commands](../../cases/008-build-reconstruct-reload/REPRODUCTION.md) are distinct from future GPU commands and do not include model weights.
