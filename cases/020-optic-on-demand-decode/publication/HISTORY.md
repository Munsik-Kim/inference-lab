# Case020 history and preservation boundaries

- Original research worktree: inference-lab-case020-optic-decode-20261026-1215; branch research/case020-optic-on-demand-decode-20261026-1215; Git base3e6f2c6b36fb2a3853cc3677bee84904d28567cf. Case was uncommitted; executions are identified by source bundles, not this base commit.
- P0/P1: pinned source/official CPU example; three-seed synthetic arithmetic, boundary/shape/order/serialization/zero/bias checks. [Original report](originals/pre-publication/REPORT.md).
- P2 first run20261026-1215:90 cell-blocks, later identified verification-order deviation. Diagnostic raw and freeze preserved.
- Corrected final P2 run20261026-1235:30 cells/90 cell-blocks. Historical host5/30/event4/30/common0/30 C wins remain unchanged.
- Post-hoc external CPU analysis: preserved under originals/external/case020-results-analysis-20261026-1418, with original docs/figures/receipts.
- P4 runp4-20261026-1451:30 timing workers/96 method-blocks,27 lifecycle,3 preflight+1 smoke+3 memory=64 workers. Post-P2 follow-up freeze states what had already been observed. C wins0/30 primary H blocks.
- Historical CPU-auditor draft variable shadowing and the initial P4A snapshot-versus-generated-summary scope error were corrected without changing frozen GPU source/raw. Original journal/receipts remain exact.
- Publication2026-10-26: new worktree/branch, exact1020-file Case copy plus19 external originals; plain bilingual docs, raw-derived direct calls/width storage/lifecycle analysis, CPU-only reproduction and dedicated CI. No new GPU/model execution.
- During publication calculator drafting, a fixture basename was initially written as normal instead of the actual random. The file-not-found check failed before producing summaries; corrected by reading the existing fixture name. No historical evidence changed.

[Preservation map](preservation_map.json) maps original paths to exact public paths/bytes/SHA256. No original file was omitted or transformed. [Original PUBLIC_MANIFEST](originals/pre-publication/PUBLIC_MANIFEST.json) and [P4 manifest](originals/pre-publication/followup/p4_speed_cost/PUBLIC_MANIFEST_P4.json) retain their historical identities. The new Case PUBLIC_MANIFEST covers current public packaging and excludes itself.

Original receipts' remote-CI/publication NOT_RUN states are historical. Current local verification is a new receipt and remote CI is assessed for the actual PR/main head. No Pages/DIOVA implementation or deployment is part of this publication.

## Publication integration blocker observed locally

Case020 CPU full checks and47 document regressions passed. The existing Portfolio site builder discovers every cases/*/README.md but explicitly accepts only IDs001–019. Adding GitHub-only Case020 caused13 setup errors in the local display regression (68 tests reached); the original failure receipt is retained separately. This is not a changed scientific result. No build.py/display registry or Pages implementation was modified because site implementation is explicitly outside publication authorization. A normal main merge must wait for a separately authorized resolution of this integration gate.

An initial local invocation used Python -I on the existing Portfolio runner, which requires its sibling common module on script sys.path. It was rerun using the workflow’s normal Python -B invocation without modifying that runner. The real registry failure above was retained rather than skipped.

The first local publication commit retained generated CSV CRLF and Matplotlib SVG trailing indentation. Those new publication outputs were normalized to LF/no trailing whitespace by their writers, with numeric tables and PNG content unchanged. Historical originals retain their exact original formatting. This was packaging-only and did not rerun GPU work.

## Follow-up: full Case020 site integration authorized

After PR40 exposed the Case001–019 display limit, the user explicitly requested fixing the failure and integrating Case020 alongside the existing19. This supersedes the prior site-implementation exclusion only for this integration. Case020 is registered normally with bilingual home/archive/detail pages, source-checked tables/figures and rejection of unknown Case021. The earlier proposed exclusion patch is not applied. Existing scientific results and all archived source/raw/receipts remain unchanged. Pages dispatch/deployment remains outside scope.
