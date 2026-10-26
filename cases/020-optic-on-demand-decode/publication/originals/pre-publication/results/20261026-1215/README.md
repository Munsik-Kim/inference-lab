# Historical diagnostic run — not the primary result

This run's 90 cell-blocks passed their per-worker FP64 checks before timing, but direct P2 A/B/C comparisons were performed after the sweep. That order did not satisfy the requested protocol. The results, source bundle, worker wall and raw outputs are preserved without modification.

`protocol_deviation.json` records the issue. `summary_draft/decision.json` describes raw coverage and an early descriptive interpretation; its COMPLETE value does not establish protocol compliance. `figures_draft/` is historical derived output.

The primary run is `../20261026-1235/`, with two dedicated all-condition A/B/C correctness workers before smoke/main and per-worker preflight identity checks. The global budget at `../gpu_worker_budget.json` includes both runs. No shape, seed, ID, tolerance, timing loop or execution-order stream was selected from favorable results.
