# Case015 — Same Tools, Different Interfaces

## Abstract

Built tools that compare lookup, calculation, record updates and validation across an API rename. They record results received before generation separately from actual JSON state, then replay saved responses into fresh fixtures to audit success.

Across 96 new measured A1 episodes, BF16 changed15/16→16/16, RTN 16/16→16/16 and D16 12/16→12/16. There was one BF16 recovery and no rename-related losses. D16 failed on the same four fixtures on both APIs.

[1. Background](#section-1) · [2. Questions and hypotheses](#section-2) · [3. Principles and execution contract](#section-3) · [4. Method](#section-4) · [5. Experiment](#section-5) · [6. Results](#section-6) · [7. Failure and recovery analysis](#section-7) · [8. Conclusion](#section-8) · [9. Reproduction and sources](#section-9)

<a id="section-1"></a>
## 1. Background

Can an agent finish the same workflow when tool names change? P1 varied names/order on long-context DEV tasks and observed losses and gains. A1 is a new application designed after P1, without connecting to a business or payment system.

<a id="section-2"></a>
## 2. Questions and hypotheses

The primary display is each model’s paired API_V1→API_V2 both-success/loss/gain/both-fail table, without a directional superiority hypothesis. D16 change minus BF16 change is secondary and exploratory over one workflow and four templates.

<a id="section-3"></a>
## 3. Principles and execution contract

| Semantic ID | API_V1 | API_V2 |
|---|---|---|
| order.read | read_order | readOrder |
| amount.multiply | multiply_amount | multiplyAmount |
| order.write | write_order_total | writeOrderTotal |
| order.validate | validate_order | validateOrder |

Only names change to lowerCamelCase; descriptions, arguments and order stay fixed. The bijective registry rejects old names and typos. Every call within a response shares immutable available_result_event_ids. Multiplication must be generated after receipt of the read result, writing after receipt of the correct product. Write and validate may execute in that order within one response. Final true requires receipt of validation matching the current state version.

<a id="section-4"></a>
## 4. Method

VERIFIED_WORKFLOW_SUCCESS requires the executed read→multiply→write→validate chain and receipt order, actual integer total, other-order/immutable-field preservation throughout, received current validation and an exact three-field final JSON. Bool and int are distinct. Wrong schema-valid integers execute/store as given; the oracle does not repair them. STATE_ONLY_SUCCESS and FINAL_ANSWER_CORRECT remain separate.

<a id="section-5"></a>
## 5. Experiment

Seed 6101621 generated four templates × four fixtures = 16 scenarios and one distinct smoke fixture; IDs and amounts are independent. Quantities 2..40 and unit prices 100..20,000 are integers; initial totals are null or -1. Requests hide quantity/price. Initial lengths 506..518 tokens differ from P1’s 4K/8K contexts. The matrix is 16 × 2 APIs × 3 methods = 96 primary episodes, plus two separate BF16 smoke episodes. Seed 6101622 fixes within-method order; load order is BF16→D16→RTN.

Revision cdbee75f17c01a7cc42f958dc650907174af0554, greedy 192-token generation, enable_thinking=False, at most 8 requests/6 tool attempts/900 seconds, context 24576, BF16/SDPA, logits_to_keep=1, within-request KV ON/turn cache OFF. ENUM, enhanced feedback and aliases are OFF. Original 252-projection RTN/D16 artifacts are read-only; D16 beta_inverse/W_pre transport is preserved.

<a id="section-6"></a>
## 6. Results

### A1

| Model | API_V1 verified /16 | API_V2 verified /16 | Both success | Loss | Gain | Both fail |
|---|---|---|---|---|---|---|
| BF16 | 15 | 16 | 15 | 0 | 1 | 0 |
| RTN | 16 | 16 | 16 | 0 | 0 | 0 |
| D16 | 12 | 12 | 12 | 0 | 0 | 4 |

Secondary D16−BF16 API change: -6.25 pp; 95% exploratory CI [-18.75, +0.00] pp

Intervals use 10,000 template-group resamples, seed 6101623. They are exploratory and conditional on four templates and one fixed rename, not API-population deployment or equivalence guarantees.

### P1

| Model | I0 original /8 | I1 order /8 | I2 lowerCamel /8 | I3 opaque /8 |
|---|---|---|---|---|
| BF16 | 8 | 8 | 8 | 7 |
| RTN | 8 | 8 | 8 | 8 |
| D16 | 6 | 6 | 5 | 8 |

Original P1 primary interaction +8.33pp,95% CI [-8.33,+25.00]pp; losses and gains are retained separately. The two stages are not pooled into a 192-episode common success rate.

### Cost

687.452s; 463 model requests; 378 attempted / 368 executed tools; 16927 generated tokens. Smoke: 10 requests / 8 tools / 360 tokens. Peak allocated 7.802 GiB; reserved 8.021 GiB.

Worker wall includes model hashing/loading, artifact decoding, smoke, failures and I/O. Each response retains device free/total samples separately from Torch allocator peaks and Windows background use. This is not serving latency or native INT4 acceleration.

<a id="section-7"></a>
## 7. Failure and recovery analysis

BF16 / A1-06: V1 saved 191444 correctly but generated validation order ID...3659 instead of...3657, received an error and ended with validated:false. V2 validated the target ID and succeeded. D16 failed to complete write-call JSON for A1-00/06/08/14 on both APIs and stopped before saving. Unknown names and receipt-order violations were zero. The rename did not repair these D16 parse failures.

Post-hoc annotations separate diagnostic record defects. Eight malformed-only early stops retained the initial BUDGET_EXHAUSTED terminal label despite neither request nor tool budgets being exhausted; separate annotations explain PARSE_ERROR. The wrong-order-ID case was omitted from raw summary ARGUMENT_VALUE_WRONG but retained as UNKNOWN_ORDER/BAD_ARGUMENTS in its tool event, and is counted in the annotation. Original responses, frozen source and success verdicts remain unchanged; independent endpoint recomputation changes zero outcomes. Recovery after parse-only early stops was not observed.

P1 did not separately register equal pre-execution fixtures and equal pre-generation observations. The new display distinguishes them without rescoring old records under the A1 contract.

<a id="section-8"></a>
## 8. Conclusion

The tool now traces and replays rename regressions in one order workflow. V2 caused no losses and one BF16 recovery, while four D16 fixture failures remained. These are not automatically the same mechanism as P1 interface sensitivity. A next study would separately authorize/register P2 recovery diagnostics with controls matched on received information.

<a id="section-9"></a>
## 9. Reproduction and sources

[CPU / GPU commands](REPRODUCE.md) · [A1 raw](followups/order-api-regression-v1/results/a1_r1/episodes.jsonl) · [Independent audit](followups/order-api-regression-v1/results/a1_r1/independent_audit.json) · [Diagnostic annotation](followups/order-api-regression-v1/results/a1_r1/diagnostic_annotations.json) · [Source freeze](followups/order-api-regression-v1/provenance/source_freeze.json) · [Original P1](publication/originals/p1_review/REPORT.md) · [NOTICE](NOTICE.md)

The 27 frozen P1 regression methods and 18 new A1 unique tests are counted separately. CPU audit reconstructs 96 raw episodes, 368 executed calls and 463 requests with state/receipt verdicts; it does not reproduce GPU inference. Codex/ChatGPT assistance and separate CPU calculation review are not human peer review or author endorsement.

## Public display v2

After PR28 merged, a JavaScript syntax check of the standalone CPU viewer exposed an unescaped newline. The frozen experiment renderer and public v1 are preserved; postprocess/render_review.py repairs only presentation. Node syntax and real browser checks cover this boundary. There are zero additional model forwards or score changes.
