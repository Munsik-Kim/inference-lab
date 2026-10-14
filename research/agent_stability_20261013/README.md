# Case 013 follow-up: same-history tool selection

Added a stdlib-only Case013 exporter to diova-compare0.1.2 and ran a bounded RTX5080 diagnostic using existing Qwen3-4B weights/artifacts. D16's wrong CamelCase tool names reproduced. Restoring one preselected projection did not rescue the failures; the equal-cost control restoration introduced two additional losses.

[한국어](README.ko.md) · [Results](records/report_ko.md) · [Recompute / rerun](README_REPRODUCE.md) · [Decision](records/decision.json)

| Original8 FREE DEV tasks | Successes | Gains vs new D16 | Losses |
|---|---:|---:|---:|
| I0: new D16 baseline | 6/8 | — | — |
| I1: layer2 down_proj restored | 6/8 | 0 | 0 |
| I2: layer35 matched control restored | 4/8 | 0 | 2 |

Natural generation, controlled-prefix name scoring and identical-error recovery are separate probes. The registered target uses last-input-position output SSE/BF16 energy, including upstream differences. No local-quantizer causal proof or held-out superiority follows from these selected DEV records.

[Exporter](../../packages/diova-compare/src/diova_compare/case013.py) · [Read-only GPU runner](worker.py) · [Whole-path restoration](intervention.py) · [Independent CPU audit](audit.py). Existing Case013/r2 records, models and environment are preserved. No fitting, downloads, packed-kernel work or remote publishing occurred. Serving benefit remains NOT_ESTABLISHED.

The public preparation contains redacted compact records with a [separate manifest](PUBLIC_MANIFEST.json), not model weights or full logits. Existing Case009/011 checks run on their exact restored historical package snapshot; current package0.1.2 is checked separately.
