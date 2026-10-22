# Case018 — State Recovery and Final-Report Consistency — CPU replay

```bash
python3 -B cases/018-agent-state-recovery/publication/run_checks.py --output /tmp/case018-public-check-new --tests
```

Python3.12+ standard library only. A fresh external output is required; existing results/research paths are not written. Allow roughly1GiB RAM and1GiB restoration disk. No model/tokenizer/Torch/vLLM/xgrammar installation or download. This command has no GPU path.

1. Check manifest, XZ signature, every member path, uniqueness, regular type, sizes/hashes and alias coverage, then restore.
2. Run phase-specific frozen sources for26 P0/24 P1/96 P2 endpoints, observations, continuations, termination, budgets and SSE/IDs/usage.
3. Recompute token totals, cancellation missingness, worker costs and registered paired bootstrap.
4. Regenerate120 posthoc rows, tables, selected raw-text excerpts and SVG; compare public summaries exactly.
5. --tests runs17 new publication boundaries. Historical37/48/59 test receipts are reused, not claimed as new full-suite runs.

[Original mapping](publication/provenance/original_mapping.json) · [Public manifest](publication/PUBLIC_MANIFEST.json) · [Archive](publication/data/original-evidence.tar.xz) · [Redactions](publication/provenance/redactions.json)

2139 logical original files use1553 unique objects. All120 model episodes and all src Python bytes are original. Only82 ancillary documents/configs/receipts containing private absolute home prefixes are symbolic-root public copies, with old/new sizes/hashes, rules and counts. Original research/freezes were not edited. These redacted copies are not byte-identical originals. Publication wrappers rebind paths without altering endpoint algorithms. Entire environment/GPU reconstruction is outside this archive scope.

Original entry documents are in publication/originals/pre-publication. Base83e1cef did not contain the uncommitted Case018. P1 execution bundle e60dc421c2dc28c59e29ab26cb9faa6a2893208d5399577eed08d9745211ee7b, P2 44287b4091ac8e9e892a263b5770200a140f9d90f97ccef19dca203a1d5241a2 and seal d7afbef13bd8a46d46587348f17d25f26bf95b51ed55107a8b6fc64a46199ff5 differ from publication commit/build/deployment identities.

[External evidence omissions/duplicates](publication/provenance/omissions.json): in addition to2139 logical archived originals, one historical P0 model-status file is restored from an exact public copy. All1739 original Case018 files are represented; omitted external duplicates/preparation/temporary/full-profile files have reasons and CPU reproduction scope.
