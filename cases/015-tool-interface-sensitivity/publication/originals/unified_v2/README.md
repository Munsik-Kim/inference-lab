# Case015 — Same Tools, Different Interfaces

Built tools that compare lookup, calculation, record updates and validation across an API rename. They record results received before generation separately from actual JSON state, then replay saved responses into fresh fixtures to audit success.

Across 96 new measured A1 episodes, BF16 changed15/16→16/16, RTN 16/16→16/16 and D16 12/16→12/16. There was one BF16 recovery and no rename-related losses. D16 failed on the same four fixtures on both APIs.

| Model | API_V1 verified /16 | API_V2 verified /16 | Both success | Loss | Gain | Both fail |
|---|---|---|---|---|---|---|
| BF16 | 15 | 16 | 15 | 0 | 1 | 0 |
| RTN | 16 | 16 | 16 | 0 | 0 | 0 |
| D16 | 12 | 12 | 12 | 0 | 0 | 4 |


One synthetic order workflow · four English instruction templates · 16 fixtures · Qwen3-4B-Instruct-2507. The 96 original P1 episodes keep a separate denominator. Quality uses dense BF16 reload of saved INT4 artifacts.

[Pages](https://munsik-kim.github.io/inference-lab/en/case015.html#overview) · [REPORT](REPORT.md) · [CPU / GPU](REPRODUCE.md) · [Stage map](STAGE_MAP.md) · [A1 runner](followups/order-api-regression-v1/src/order_regression/runner.py) · [Tests](followups/order-api-regression-v1/tests/test_order_contracts.py)

## Three implemented functions

1. Exact name↔semantic mapping and executed API equivalence.
2. A receipt contract from lookup to multiplication, write/validation and final answer.
3. Raw-response/state replay with per-model loss/gain regression tables.

## The original P1 is preserved

For eight previously observed DEV tasks × 3 methods×4 interfaces, the signed interaction was+8.33pp with 95% CI [-8.33,+25.00]pp, inconclusive. See the [original report](publication/originals/p1_review/REPORT.md) and [restoration map](publication/provenance/p1_original_map.json).
