# DIOVA result showcase

DIOVA stands for Deep-learning Inference Optimization, Validation & Analysis.
The display name is DIOVA; the repository and existing URLs remain `inference-lab`.
Historical case documents and packages retain their original names.

This presentation layer reads pinned public Case006/007/008 evidence without editing it.
It provides English and Korean static pages, item filters, same-readout comparisons,
source links and JSON downloads. Case008 has separate Q storage/runtime and R fixed-structure reconstruction summaries. The home page does not load item datasets.

Published viewer: [한국어](https://munsik-kim.github.io/inference-lab/ko/index.html) · [English](https://munsik-kim.github.io/inference-lab/en/index.html). The same code supports the offline build below.

From the repository root, with Python 3.12:

```bash
OUT=$(mktemp -d)
python3 -B tools/showcase/build.py --output "$OUT/site" --base-path /inference-lab/
python3 -B tools/showcase/check.py --site "$OUT/site" --output "$OUT/check.json"
```

Open `index.html` in that output folder. All resource links are relative, supporting
`file://`, a loopback root and `/inference-lab/`. Builds refuse an existing output
or a destination inside the repository. The builder copies only its explicit asset
list; it extracts score/prompt projections rather than copying experiment trees.
Marked irrelevant filler and token arrays are omitted. Exact input files remain linked.

`case008_sources.json` pins the additional summary sources; it publishes no full vectors or token arrays.
`source_manifest.json` records the fixed experiment commit and hashes of read files.
`build_manifest.json` separately records presentation-source hashes, display counts and
artifact hashes. The scientific source commit does not claim to contain the separately versioned
presentation files. Repeated arms/readouts are not independent scenarios.
A current source change makes the original build stale; generate a new external build.

The CPU selector uses Python 3.12.14 and NumPy 2.3.5. It imports the original selector
by a unique module name, validates calibration Q/grouping/hashes and reruns both
frozen budgets. The filter tests use Node 24.12.0 with `node:test` and no npm packages.
Portfolio checks use these tools. The separate `case008-checks.yml` runs retained-evidence checks and modelpack CPU tests, including a random tiny model using CPU PyTorch/Transformers/safetensors. No checkpoint downloads or GPU research runs are in CI. [Tiny example](../tools/modelpack_demo/README.md).

Browser checks used an already installed Windows Edge through CDP, with a temporary
profile, at 390/768/1280 pixels. `browser_check.cjs` is optional and accepts a local
site base URL plus a new external report directory. It exercises native/shadow/set
separation, task/ID/empty filters, language/deep-link restoration, keyboard focus,
text-only script-string rendering, downloads and runtime requests. Screenshots here
show fixed first CODE IDs from actual local execution. Real iPad: NOT_TESTED.
The browser record above covers the original local checks. The published site and
remote CPU runs have separate GitHub deployment and Actions records.

The three `linguist-generated` paths in `.gitattributes` are produced by the preserved
`build_study_demo.py`, supplement `build_materials.py` and Case007 `report.py` scripts.
The handwritten presentation HTML template, JavaScript and CSS remain source files.
No text/eol filters are introduced.

GitHub deployment guidance: [custom Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages),
[Actions security](https://docs.github.com/en/actions/reference/security/secure-use),
[Linguist generated files](https://github.com/github-linguist/linguist/blob/main/docs/overrides.md).
The manual workflow requires main, publish=true and a checked site artifact. Pages
write/identity permissions belong only to the deploy job; setup leaves enablement false.
Remote push/PR/merge and the manual Pages run require explicit PUBLISH_APPROVED=true. Pages settings and protection rules remain unchanged.

<a id="practical-guides"></a>
## Practical model guides

The existing builder also renders `ko/llm-guide.html`, `ko/qwen3-4b.html` and
`ko/measurements.html`. The matching `en/` pages are explicitly short summaries,
with links to the complete Korean detail. [Online guide](https://munsik-kim.github.io/inference-lab/ko/llm-guide.html) · [Qwen3-4B](https://munsik-kim.github.io/inference-lab/ko/qwen3-4b.html) · [Measurements](https://munsik-kim.github.io/inference-lab/ko/measurements.html). Existing Case pages stay put.

Use an already compatible CPU interpreter (recorded Python 3.12.14 and NumPy 2.3.5).
Case013's retained-record audit imports NumPy, so current builds also need it.
Choose a new output outside this checkout; for a retained review use a persistent
external directory rather than only a temporary directory:

```bash
python3 -B tools/showcase/build.py --help
python3 -B tools/showcase/check.py --help
python3 -B tools/showcase/build.py --output ../diova-guide-review/inference-lab --base-path /inference-lab/
python3 -B tools/showcase/check.py --site ../diova-guide-review/inference-lab --output ../diova-guide-review/check.json
```

Open `../diova-guide-review/inference-lab/ko/index.html` or the language chooser
`../diova-guide-review/inference-lab/index.html` directly in a browser. This supports
`file://` without a server. For HTTP preview, serve the site's parent so the
`/inference-lab/` mount is preserved, or serve `site/` at the root. Use a server's
actual bound address; these instructions do not start one.

`presentation/guides/` holds narrative templates. `tools/showcase/practical.py`
extracts scalar values from existing Case008/009/013 projections plus the explicit
`practical_sources.json` allowlist. The original scientific manifests are unchanged.
The generated `data/practical-guide.json` records exact values, field paths, source
hashes and public evidence commit `09194c3f2e2c991feff5b6abab0cf8ab81a07052`.
New guide assets are an exact extension to the existing deploy allowlist: six HTML
pages, one stylesheet and one scalar JSON. No recursive source copying is added.

The optional `tools/showcase/guide_browser_check.cjs` uses an already installed
Edge via CDP, accepts a file or loopback base URL and a **new external report path**,
and captures 390/768/1280px screenshots. Desktop viewport emulation is not a real
iPad test or a reader study. See its usage comment for the Edge executable option.

After separate publication approval: review the local diff and evidence-preservation
report; commit only the documented presentation/docs/tests paths; push the feature
branch; create a PR; require the existing CPU/site workflows; merge the reviewed
head only after checks pass. With PUBLISH_APPROVED=true, the scoped push/PR/merge and existing manual Pages run
are authorized after required checks pass. Keep its main/publish gate, checked
artifact and scoped permissions; do not change settings or protection rules.


## Reader routes and publication gate

The home reads introduction → featured A/B/C implementations → model guides → implementation
collaboration tasks → contributions → capabilities/stack → separate Case014 research card → expandable case records. Existing section IDs remain.
The common header uses Home / Model guides / All cases / Reproduce. Individual
cases keep related-case links; the retained image-review tool keeps its own controls.
`guide.html` is the developer CPU build/check route. `llm-guide.html` is the model
entry; `measurements.html` explains how to interpret measurements. These roles do
not require a new framework or duplicated evidence.

`PUBLISH_APPROVED=false` limits a task to local checks and review files. A later
explicit publication instruction authorizes the scoped remote workflow below.
After explicit `PUBLISH_APPROVED=true`, publish the reviewed branch with a scoped PR,
pass required CI and repository policy, and merge the reviewed head. The existing
manual workflow is `.github/workflows/portfolio-pages.yml` (name: **Portfolio Pages
(manual)**). It accepts the boolean **publish** (default **false**) and builds/deploys
only on **main** with **publish=true**. With the newly reviewed changes merged and
PUBLISH_APPROVED=true, the exact dispatch is:

```bash
gh workflow run portfolio-pages.yml --repo Munsik-Kim/inference-lab --ref main -f publish=true
```

Do not change protection rules or Pages settings. Verify the new run's source SHA,
checked artifact and deployment status, then open the actual public home, menu and
Qwen guide. A successful Actions run and a verified public screen are separate
checks. A local preview does not establish that the public site was updated.

## Compare your saved answers

The home features Case008 Q + Case009 as A, **diova-compare** as B, and Case017
workflow completion and execution-cost auditing as C. Case013 tool-contract work
remains linked as earlier research. The short Case017 home copy is separate from
the unchanged detailed reports and keeps the expansion-gate deviation visible. Case014 is a separate research card after the technical
stack. Earlier implementation summaries remain in a native disclosure with their
existing anchors; the complete archive stays directly accessible.

The bilingual `compare.html` pages lead to the [four-pair synthetic CPU example](../packages/diova-compare/examples/compare/README.md).
They separate your own JSONL workflow from historical Case013 export. Two explicit
JSONL files are copied byte-for-byte via `COMPARE_EXAMPLES`; the checker extends its
exact deployment allowlist and checks those bytes. No experiment manifest changes.

The optional guide browser checker starts its own temporary Edge profile and uses
`--remote-debugging-port=0`, reading only that profile's DevToolsActivePort and
matching its browser identity. It does not attach to a fixed debugging port.
Browser screenshots, actual CLI reports and a reading demo belong to the external
review directory, with their execution/visual-review status recorded in HANDOFF.
