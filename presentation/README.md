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
Publication, merge, Pages setup and manual deployment require separate authorization.
