# Case017 publication layer v1 (local candidate)

This layer selects public files, restores original bytes, and connects the existing DIOVA builder. It changes no frozen execution source, protocol, schema, budget, parser, scorer, original response or original endpoint score.

- `provenance/original_mapping.json`: all795 original file paths/sizes/hashes, selected storage/member or omission reason.529 files retained,266 omitted.
- `data/original-evidence.tar.xz`:418 regular members, lossless2,828,844 bytes. Includes all256 complete P1/P2 cells, raw SSE/token IDs, tool events, P0 records and historical receipts. No unsafe paths, links or repeated members are accepted by the restorer.
- Original root entry documents are retained under `originals/pre-publication/` before new public entry documents are written. Their old hashes remain unchanged.
- `wording.py`: shared bilingual prose derived from raw-backed group and comparison records; both the documents and existing site use it.
- `summary/`: raw-backed256-row ledger, policy/paired tables, display projection and zero-origin charts. Every projected row carries an original relative path and byte SHA256. The current-state Fresh flag is posthoc.
- `reviews/`: a clearly labelled path-sanitized copy of the earlier addendum, its transformation receipt, and a separately dated original-intent excerpt newly supplied on2026-10-21.
- `run_checks.py`: standard-library public restoration, fresh-fixture replay, original independent endpoint and bootstrap, display/review-ledger checks. CPU imports block model runtimes.
- `PUBLIC_MANIFEST.json`: separate public closed-file inventory, excluding itself to avoid self-hash cycles. Historical freezes/manifests are not loosened or rehashed.

Omissions are regenerable256 audit fixture states, old generated standalone HTML/JSON, and historical browser outputs/checker. All initial task/input/source records and full raw P1/P2 cells remain available for complete endpoint replay. Historical browser checks are not claimed as checks of this public page. No old review ZIP, model, tokenizer, environment, KV tensor, large profiler or private runtime is copied.

The prior research source is identified by `execution_source_sha256`, not the repository base commit. Public source hashes and the future deployment revision are separate; the research case was untracked at that base. New source links therefore use copies shipped with the actual page build rather than inventing old GitHub blob URLs.

Reuse: the restoration/mapping and manifest approach follows Case015/016 publication wrappers; Case017 original `audit`, `analysis`, `verify` and tests run unchanged in a restored view. The small CDP browser checker adapts Case016's isolated Node/Edge pattern. vLLM/FlashInfer/xgrammar and model rights remain upstream; see the original NOTICE and root Apache-2.0 license. New publication code used Codex assistance.

The initial preparation ended locally. `provenance/publication_identity.json` is that historical preparation receipt, including its then-unpublished status. A separate user instruction on 2026-10-21 authorizes publication, normal merge and the existing manual Pages deployment. Actual feature/main/deployment revisions are recorded by Git, the build manifest and external release receipts, without backdating this preparation record. No new model run or P3 is included.
