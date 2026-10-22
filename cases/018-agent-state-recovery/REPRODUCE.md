# Case018 — Checking inventory recovery and completion reports — CPU record replay

```bash
python3 -B cases/018-agent-state-recovery/publication/run_checks.py --output /tmp/case018-public-check-new --tests
```

Use Python 3.12+ and its standard library. Choose a fresh external output directory that does not exist. Allow roughly 1 GiB each for restoration disk space and RAM. No model/tokenizer download or Torch/vLLM/xgrammar installation is needed.

1. Verify the manifest, full XZ paths, regular files, duplicates, bytes/hashes and alias coverage; restore the evidence.
2. Replay 120 model endpoints and the separate 26 P0 synthetic traces with phase-specific frozen sources.
3. Recalculate state, observations, continuations, termination, budgets, SSE/IDs/usage, costs and paired bootstrap.
4. Regenerate the existing 120-row posthoc analysis, selected excerpts and SVGs; compare with public summaries.
5. `--tests` runs 17 publication-boundary tests. Historical 37/48/59-test records remain in their original receipts.

CPU record replay has passed in a local public-only copy and on GitHub-hosted Ubuntu. The link below shows current main runs and receipts. This command replays saved execution records; retained receipts document the original tokenizer re-encoding and KV/grammar technical gates.

[CPU record replay · main](https://github.com/Munsik-Kim/inference-lab/actions/workflows/case018-checks.yml?query=branch%3Amain)

[Original mapping](publication/provenance/original_mapping.json) · [Public manifest](publication/PUBLIC_MANIFEST.json) · [Archive](publication/data/original-evidence.tar.xz) · [Redactions](publication/provenance/redactions.json)

2139 logical original files use1553 unique objects. All120 model episodes and all src Python bytes are original. Only82 ancillary documents/configs/receipts containing private absolute home prefixes are symbolic-root public copies, with old/new sizes/hashes, rules and counts. Original research/freezes were not edited. These redacted copies are not byte-identical originals. Publication wrappers rebind paths without altering endpoint algorithms. Entire environment/GPU reconstruction is outside this archive scope.

Original entry documents are in publication/originals/pre-publication. Base83e1cef did not contain the uncommitted Case018. P1 execution bundle e60dc421c2dc28c59e29ab26cb9faa6a2893208d5399577eed08d9745211ee7b, P2 44287b4091ac8e9e892a263b5770200a140f9d90f97ccef19dca203a1d5241a2 and seal d7afbef13bd8a46d46587348f17d25f26bf95b51ed55107a8b6fc64a46199ff5 differ from publication commit/build/deployment identities.

[External evidence omissions/duplicates](publication/provenance/omissions.json): in addition to2139 logical archived originals, one historical P0 model-status file is restored from an exact public copy. All1739 original Case018 files are represented; omitted external duplicates/preparation/temporary/full-profile files have reasons and CPU reproduction scope.
