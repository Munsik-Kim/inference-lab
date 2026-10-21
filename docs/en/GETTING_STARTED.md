# Read, explore, recalculate, reproduce

English | [한국어](../ko/GETTING_STARTED.md) · [Home](../../README.md)

[New here?](START_HERE.md) · [Plain-language glossary](GLOSSARY.md)

Choose what you want to do before choosing a command:

| Your purpose | Start here | What it gives you |
|---|---|---|
| Understand the work or assess a role/project fit | [Portfolio](PORTFOLIO.md) or the [published screen](https://munsik-kim.github.io/inference-lab/en/index.html) | The problem, tools built, related fields and source evidence; no installation |
| Find which saved answers changed | [Compare your results](#compare-your-results) | A reusable CPU tool that compares two result files |
| Check the project’s historical numbers | The relevant case’s CPU instructions below | Recalculation of saved results, without running a model |
| Repeat actual model execution | Each case’s linked reproduction document | Separate model files, fixed software versions and compatible hardware |

The examples below are technical instructions for the tools I built. A successful saved-record check confirms the stated file or arithmetic checks; it does not repeat the original GPU experiment. Here, a model artifact is a converted and saved model file, a scalar is one numerical value, and a hash/manifest is a file fingerprint/list used to check preservation. [Other terms](GLOSSARY.md) can be looked up as needed.


<a id="compare-your-results"></a>
## Compare your own saved answers first

The [diova-compare user example](../../packages/diova-compare/examples/compare/README.md) starts with two small files of invented answers: check the input format (`validate`) → compare matching questions (`compare`) → read the reports. It runs independently on CPU in a separate Python 3.10+ environment. Put your saved answers and correctness judgments into the same documented format (JSONL: one JSON record per line). Raw model scores and model downloads are unnecessary. The later commands recheck this project’s historical experiments.

## Open the published result viewer

[English home](https://munsik-kim.github.io/inference-lab/en/index.html) · [Case008 Q: smaller files and reload](https://munsik-kim.github.io/inference-lab/en/case008.html#track-q) · [Case009: serving comparison](https://munsik-kim.github.io/inference-lab/en/case009.html#serving) · [Case017 CPU replay without a model](../../cases/017-agent-contract-cost-bench/REPRODUCE.md). These Pages screens use saved measurements and require no installation or GPU. Use the local build below for offline viewing.

Use the [model guide entry](LLM_GUIDE.md) for execution paths, failures and mitigations, then the measurement page for interpreting numbers. This document covers downloads, static builds and CPU checks.

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

The command creates a small, randomly initialized two-layer Qwen test model (fixture) on CPU, reduces layer 1’s intermediate size from 64 to 48, saves it and closes the builder program. A second process reads a copied model from another directory while access to the original directory is blocked. It checks all output values, the dimensions of saved arrays (tensor shapes), and whether the input and output layers still share the same weights (tied embedding/head). It calls the preserved Case008 loader without changing that research code.

The checked summary includes `TINY_RANDOM_CPU_FIXTURE`, widths `[64,48]`, gate/up `[48,32]`, down `[32,48]`, matching raw output scores (logits) and shared weights. Generated tiny weights and process logs stay in the external output directory. This tests saving and restoring data (serialization); it does not test trained-model quality or speed. To run the historical modelpack tests and this demo together:

```bash
OUT=$(mktemp -d)
python3 -B tools/case008_checks/run.py --mode modelpack --output "$OUT/modelpack"
```

The helper restores a fingerprint-checked copy of the historical files before tests that require the exact original file list. It also checks that missing or invalid saved models are rejected. The original tests keep their original expectations.


<a id="local-showcase"></a>
## Open the bilingual showcase

The new screen shows Case007 group selection and Case006 answer scores. It reads retained measurements, with separate native and post-hoc readout views. From this repository's root, use Python 3.12 and choose a new external output directory:

```bash
OUT=$(mktemp -d)
python3 -B tools/showcase/build.py --output "$OUT/site" --base-path /inference-lab/
python3 -B tools/showcase/check.py --site "$OUT/site" --output "$OUT/site-check.json"
```

Open `$OUT/site/index.html` in a browser, then choose English or 한국어. Use a compatible CPU environment with Python 3.12 and NumPy; Case013 retained-record checks require NumPy. All scripts and display data are local; `file://` works without a server. The same relative resource paths also support a site mounted at `/inference-lab/`. The published Pages viewer and a local build use the same static presentation code.

Read each detail page in order: **objective → dataset → assumptions and theory → experiment design → validation → results → interpretation → conclusion**. Use the contents links or **Jump to individual results**. The complete-study tables above are separate from the filtered subset statistics below. The [GitHub casebook](CASEBOOK.md) uses the same structure for each case.

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

The helper loads the preserved group-selection code and repeats its decision using the setting data’s 16×16 table of group contributions (calibration Q matrix). It checks source-file fingerprints, valid finite symmetric values, group mapping, removal counts and the original rule for resolving tied choices by ordered group IDs (lexicographic tie rule). Inputs reserved for evaluation (held-out) are not used to select groups.

| Removal budget | Recomputed groups; match with saved selection |
|---|---|
| 4/16 | INDEPENDENT `[8,12,13,14]`; PAIRWISE `[8,13,14,15]`; both match |
| 8/16 | Both `[7,8,9,11,12,13,14,15]`; both match |

Each method evaluates 1,820 or 12,870 small candidate scores on CPU. **Model forwards: 0**—the model is never run. The JSON retains objective values and definitions: independent uses the selected diagonal sum; pairwise uses the entire selected submatrix sum. They are different objectives. This checks the recorded selection logic; GPU quality and timing remain the original experiment.


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
Download the [reviewed Case007 ZIP](../../downloads/case007_mlp_pruning_reviewed_publication_v1.zip) and check its [size/hash metadata](../../downloads/case007_mlp_pruning_reviewed_publication_v1.json). Extract it and open `007-interaction-aware-mlp-pruning/demo/index.html` locally. The English UI uses retained measurements only. Choose a method and deletion budget, inspect the prompt ID and removed groups, and separate local reconstruction error from gold-answer transitions. Its original status, **COMPLETED_NO_CLEAR_TRANSFER**, means the study finished without establishing a clear benefit under the preset comparison rule; deployment was not assessed. [Result interpretation](CASEBOOK.md#case-007).

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

## Installable CPU result comparison

[diova-compare](../../packages/diova-compare/README.md) validates paired inputs and computes correctness transitions and score changes. Install it in a dedicated CPU environment without a model runtime. The `evidence_kind` field says whether a file contains a new measurement or a recalculation of historical records.

## New graph-serving and official-task study

[Case009 reproduction](../../cases/009-q-serving-quality/REPRODUCTION.md) separates the preserved serving environment, a new harness environment and the CPU package. Its [full curves and quality report](../../cases/009-q-serving-quality/REPORT.md) also records abnormal finalization in some quality processes. Public scalar reanalysis does not require downloading the benchmark datasets or models.


<a id="case010-cpu"></a>
## Case010: state and result checks without a model

[Unified reproduction guide](../../cases/010-ckda-finite-precision-memory-horizon/REPRODUCTION.md) separates historical-layout restoration, scalar audits and synthetic failure-aware restart tests. These need no model or GPU; checkpoint-based research replay is a separate path.

```bash
python3 -B cases/010-ckda-finite-precision-memory-horizon/scripts/verify_unified.py
python3 -B cases/010-ckda-finite-precision-memory-horizon/scripts/restore_workspace.py --output /tmp/diova-case010-restored-new
```

<a id="case011-cpu"></a>
## Case011: saved-record audits and readout patch checks

Start with the [walkthrough](https://munsik-kim.github.io/inference-lab/en/case011.html#overview), then choose saved-record or synthetic checks in the [reproduction guide](../../cases/011-frozen-state-readout-adaptation/REPRODUCTION.md). From a repository or public ZIP root, this command needs Python/NumPy and a new output directory.

```bash
python -B cases/011-frozen-state-readout-adaptation/publication/run_checks.py \
  --output /tmp/diova-case011-publication-check-new
```

Add `--synthetic` for small tests of input/storage rules using invented data; they require Torch. The original 12 fresh-checkpoint reloads and 30 saved-boundary checks are records of past checks; this command does not run the model again. The public ZIP excludes original checkpoints and fitted head weights, so model replay requires the recorded environment and separate files.

<!-- case014-entry -->
<a id="case014-cpu"></a>
## Case014 — Recalculate P0–P2 without a model

[ZIP](../../downloads/case014_compensation_unified_publication_v1.zip) · [bytes/SHA256](../../downloads/case014_compensation_unified_publication_v1.json) · [Dependencies and scope](../../cases/014-quantization-compensation/REPRODUCE.md)

From repository root or archive top level, select an existing NumPy/CPU Torch interpreter as CPU_PY and use a fresh output path.

```bash
OUT=$(mktemp -d)
"$CPU_PY" -B cases/014-quantization-compensation/publication/scripts/run_checks.py --output "$OUT/case014" --tests
```

Restores the preserved file layouts for the P0/P1/P1b/P2 research stages, then checks 2880/3360/320 raw rows, the original setting choices, uncertainty intervals, display data and CPU tests. It uses saved records, without a model, excluded private arrays/source text or new GPU inference. Building the whole portfolio needs a full repository copy (clone), not this case-only ZIP.

<!-- case015-entry -->
<a id="case015-cpu"></a>
## API rename regression testing

Check whether an AI finishes the same task after a tool rename. The example is **read item count and price per item → calculate the purchase total → save it → validate the record**. Functions and input formats stay fixed; only names change, such as `read_order` to `readOrder`. Across 16 problems, BF16 was **15/16→16/16**, RTN **16/16→16/16**, and D16 **12/16→12/16**. The [task and terminology guide](../../cases/015-tool-interface-sensitivity/README.md) explains orders as purchase records, unit price as price per item, and how to read the results. The 96 runs of the earlier P1 study and the new A1 application are counted separately.

```bash
OUT=$(mktemp -d)
python3 -B cases/015-tool-interface-sensitivity/publication/scripts/run_checks.py --output "$OUT/case015" --tests
```

<!-- case016-entry -->
<a id="case016-cpu"></a>
## Case016 — Native KV capacity and request costs at the same memory budget

Built native NVFP4 KV storage/read checks and a fixed-pool comparison tool on RTX5080. Token capacity was 3.56× at approximately4GiB; F4/B16 request-throughput ratios at16K concurrency2/4 were 1.21/1.35×. Other conditions were not uniformly faster; original agent-baseline failures remain preserved.

[Report](../../cases/016-kv-agent-bench/REPORT.md) · [CPU commands](../../cases/016-kv-agent-bench/REPRODUCE.md) · [Implementation](../../cases/016-kv-agent-bench/followups/nvfp4-technical-r1/src/nvfp4_technical/runner.py)

```bash
python3 -B cases/016-kv-agent-bench/publication/run_checks.py --output /tmp/case016-check-new --tests
```

Bilingual Pages integration is prepared; manual deployment was not dispatched without separate approval.

## Case017 — Agent Contract–Cost Bench

We built a benchmark that checks how tool call policies change completion of saved state, observed validation and final answers. It uses owned synthetic records for product totals, inventory transfers and document updates.

With BF16 weights and native NVFP4 KV, grammar changes completion 7/24→15/24 and correct saved states 7/24→22/24 on 24 new synthetic tasks: 8 recoveries and 0 losses. Added execution costs and remaining failures are reported alongside the gain. More tasks completed, while total execution costs also increased. Dividing all 24 runs, including failures, by completions gives approximately 3.62% more input tokens, 12.36% more output tokens and 11.27% more episode-wall time per completion.

Protocol deviation: BF16 KV with free output completed 5/8 DEV tasks, below the original 6/8 expansion criterion. The local plan and execution gate omitted this success threshold and proceeded to P2 after checking completion, freezing and sealing. No separate waiver approval was found. We disclose this deviation and present P2 as an exploratory measurement on new synthetic tasks.

[Report](../../cases/017-agent-contract-cost-bench/REPORT.md) · [CPU command and scope](../../cases/017-agent-contract-cost-bench/REPRODUCE.md) · [Strict parser](../../cases/017-agent-contract-cost-bench/src/agent_contract_cost/contracts.py) · [Independent auditor](../../cases/017-agent-contract-cost-bench/src/agent_contract_cost/audit.py)

```bash
python3 -B cases/017-agent-contract-cost-bench/publication/run_checks.py --output /tmp/case017-public-check-new --tests
```

CPU replay of256 saved endpoints, without new inference. Use a fresh output directory.
