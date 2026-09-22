# Feedback implementation map

[English home](../README.md) · [한국어 홈](../README.ko.md) · [New study](../cases/009-q-serving-quality/README.md)

| Feedback | Implementation and evidence | State |
|---|---|---|
| 1. Show project work and upstream components | Bilingual contribution tables, unchanged 008→007→006 project order, smaller R emphasis on its loader and fixed-weight fit. [Related work](related-work/README.md) maps He/FLAP/Dutta/MARLIN to the actual code. | Implemented locally |
| 2. Measure longer decode and client concurrency | [Case009 serving runner](../cases/009-q-serving-quality/scripts/serving.py), frozen 60-cell graph/eager design, three paired server rounds, request/token checks, scheduler and device observations. | 60/60 cells completed; 7,296 timed requests |
| 3. Add official quality tasks | Pinned official harness, full-split one-attempt budgets, frozen fewshot/template/request hashes, separate metric definitions and paired analysis. [Report](../cases/009-q-serving-quality/REPORT.md) identifies native finalization failures; retained calculations do not imply clean process completion. | Runtime validation partial; exact task states in report |
| 4. Audit new synthetic inputs | [data-v2](../cases/009-q-serving-quality/scripts/data_v2.py) separates RNGs, validates a bounded oracle and checks trivial-rule baselines. Initial out-of-domain foils were retained and repaired before model evaluation. | CPU fixture tests complete; model evaluation NOT_RUN |
| 5. Describe R's implementation accurately | Per-layer shape metadata, meta skeleton, strict assignment, tied-weight/buffer restoration and fixed-structure ridge correction. [Portfolio](en/PORTFOLIO.md), [Korean notes](interview-notes.ko.md), [future multi-layer design](phase2a-design.md). | Existing implementation documented; Phase2-A NOT_RUN |
| 6. Make paired analysis installable | [diova-compare](../packages/diova-compare/README.md): stdlib core, versioned contracts, arbitrary label counts, extracted-answer handling, JSON/CSV/Markdown, source/wheel clean install and historical/official adapters. | Implemented; local tests and demonstrations recorded |
| 7. Strengthen verification and explanation | CPU contract workflow with read-only token and always-upload reports; independent scalar path; unchanged historical-source checks; twelve interview questions linked to actual code. | Local checks; remote CI/deployment NOT_RUN |

The original model artifacts, Cases001–008 and tools/modelpack remain historical sources. No new R fit, multi-layer compression, requantization, model download, cloud inference or deployment took place. The three timing rounds are not a service SLO study. Public task familiarity or a standard benchmark name does not establish absence from pretraining.

The new comparison package checks submitted scalar contracts and calculations. It cannot verify full vectors or hidden state that were not supplied. The quality-process exit problem remains a runtime limitation even where completed calculations can be independently audited. The reports retain unsuccessful attempts and do not repeat test splits to obtain a clean or favorable result.
