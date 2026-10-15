# Stage map / 단계와 원형

One active Case014 connects four separately designed stages. P1b/P2 were designed after prior observations.

| Stage | Input / unit | Candidate / selection | Original documents and records |
| --- | --- | --- | --- |
| P0 | Rational fixtures;4-code chain;2000 stored noise seeds per dither condition | Three distinct mathematical models, no LM quality | [Original P0 report](publication/original_pre_p1/REPORT.ko.md), [raw](results/local_r1/p0_full.json) |
| P1 p1_r2 | PG19 CAL32/DEV16/EVAL32;ArXiv16;256-token windows |6 independent targets×5alphas;CAL raw SSE selector | [Original P1 report](publication/originals/pre_publication/REPORT.ko.md), [cells](results/p1_r2/cells/), [freeze](results/p1_r2/fit_frozen.json) |
| P1b p1b_r2 | New SELECT32/PG19 EVAL64/ArXiv16;known prior inputs excluded |Same6 K/30artifacts;SSE/NLL selectors;7 policies can alias | [Report](followups/p1b_selection/REPORT.ko.md), [cells](results/p1b_r2/cells/), [selection](results/p1b_r2/selection_frozen.json) |
| P2 p2_r1 |Original CAL32;separate fresh PG1964/ArXiv16 |Whole196 projection models;new own-path K for fixed.25/.5;QUARTER primary | [Report](followups/p2_core/REPORT.ko.md), [metrics](results/p2_r1/metrics/), [freeze](results/p2_r1/artifact_frozen.json) |

Each model/method uses original next-token targets, not teacher argmax labels. Different stages/cohorts are not pooled.
The six P1/P1b hybrid interventions are not six independent models or six times as many documents.

[Research-tree mapping](publication/provenance/ORIGINAL_MAP.json) preserves source sizes/hashes and retained paths.
[Historical archive mapping](publication/provenance/STAGE_ARCHIVES.json) restores exactly the original file set for each stage.
Root documents changed by integration are retained before editing. Personal model-path metadata has an explicit sanitized-copy mapping.
Historical NOT_RUN_PUBLICATION/storage-policy decisions remain historical records, not current publication status.

Restore into a new external folder before using an original closed-set verifier. Links in relocated byte-preserved documents retain
their original relative meaning in the restored view. The publication wrapper performs these restorations automatically.

```bash
python3 -B cases/014-quantization-compensation/publication/scripts/restore_stage.py --stage P1 --output "$NEW_VIEW"
```

Requested named independent human/AI review attachments were not present in the scoped local inputs. The historical executors'
receipts and new Codex-assisted CPU audits are labeled separately in [review availability](publication/reviews/availability.json).
