# One Case015, separate stages

| Stage | Inputs / scope | Endpoint / denominator | Original location |
|---|---|---|---|
| P0 | Eight historical synthetic tasks, four interfaces, CPU equivalence | Contract tests; no model quality denominator | data/, specs/, tests/, results/p0/ if present; see original manifests |
| P1 | 8 previously observed4K/8K DEV tasks ×3 methods ×4 interfaces | FUNCTIONAL_SUCCESS;96 episodes | results/p1_r1/, src/interface_sensitivity/ |
| A1 | New one-workflow application,4 templates×4 fixtures×2 API names×3 methods | VERIFIED_WORKFLOW_SUCCESS;96 episodes | followups/order-api-regression-v1/ |
| A1 smoke | One distinct fixture, BF16 two APIs |2 smoke episodes, excluded from primary | followups/order-api-regression-v1/results/a1_r1/smoke/ |
| Publication | CPU raw replay and new display only | Both denominators retained; no additional GPU run | publication/ |

[P1 restoration map](publication/provenance/p1_original_map.json) preserves228 selected original files. Seven overwritten integration documents have exact copies in publication/originals/p1_review/. Original scientific source, specs, records, decisions, failed attempts and closed manifest retain their bytes. Restore an original-only view before running the old verifier.

A1 observation receipt is a stricter new contract. P1 execution-state equality does not establish equal information at generation time; the display labels this distinction without retroactive rescoring. A1 diagnostic annotations correct eight early-stop terminal labels and one omitted value-error label, without altering the frozen raw or primary verdicts.

[Original P1 report](publication/originals/p1_review/REPORT.md) · [A1 report](followups/order-api-regression-v1/REPORT.md) · [CPU commands](REPRODUCE.md)
