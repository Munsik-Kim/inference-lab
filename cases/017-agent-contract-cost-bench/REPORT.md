# Case017 — Agent Contract–Cost Bench

We tested an AI agent on synthetic records: calculating and saving order totals, transferring inventory, and updating document metadata. The benchmark checks the saved state, the validation results the agent observed, and its final answer. It compares output constraints and one retry after a contract error.

[Summary](#overview) · [Background](#background) · [Questions](#questions) · [Principles and execution contract](#contract) · [Methods](#method) · [Experiments](#experiments) · [Results](#results) · [Analysis](#analysis) · [Conclusion](#conclusion) · [Reproduction and sources](#reproduce)

## <a id="overview"></a>Summary

Qwen3-4B weights stay BF16 in both configurations. The KV cache stores intermediate attention values for preceding text. BF16 KV (K0) is the 16-bit baseline; native NVFP4 KV (K1) stores packed four-bit values with scales.

Free output (C0) gives format instructions in the prompt and stops at the first contract error. Grammar (G) applies a public JSON schema to every response. Neutral retry (N) and error retry (E) each offer one retry after the first error, using a neutral message or the public error code, path and expected type. The schema supplies neither correct values nor the next action.

| KV storage | C0 | G | N | E |
| --- | --- | --- | --- | --- |
| BF16 KV (K0) | 16/24 | 16/24 | 16/24 | 16/24 |
| NVFP4 KV (K1) | 7/24 | 15/24 | 9/24 | 13/24 |

With BF16 weights and native NVFP4 KV, applying a public JSON schema to every response increased verified completion from 7/24 to 15/24 on newly generated synthetic tasks: 8 recoveries and 0 losses relative to free output. Correct saved states increased from 7/24 to 22/24. A correct state without observed validation of the updated state did not count as verified completion. The completion difference is 33.33 percentage points (pp), with an exploratory 95% interval of [16.67, 50.00]pp.

**Protocol deviation: BF16 KV with free output completed 5/8 DEV tasks, below the original 6/8 expansion criterion. The local plan and execution gate omitted this success threshold and proceeded to P2 after checking completion, freezing and sealing. No separate waiver approval was found. We disclose this deviation and present P2 as an exploratory measurement on new synthetic tasks.** [PROTOCOL_DEVIATION.md](PROTOCOL_DEVIATION.md)

More tasks completed, while total execution costs also increased. Dividing all 24 runs, including failures, by completions gives approximately 3.62% more input tokens, 12.36% more output tokens and 11.27% more episode-wall time per completion.

## <a id="background"></a>Background

Case015 examined action-format and interface failures; Case016 verified native KV storage, capacity and serving costs. This case measures completion and costs of strict execution policies with two KV configurations on new synthetic tasks. Earlier selected-failure recovery rates and serving performance are not included in these denominators.

## <a id="questions"></a>Questions

Q1: How do output policies change completion of correct state and final answers? Q2: With BF16 weights fixed, how do BF16 KV and native NVFP4 KV differ in outcomes, tokens, calls and measured time? Improvement was not assumed.

## <a id="contract"></a>Principles and execution contract

Qwen3-4B weights stay BF16 in both configurations. The KV cache stores intermediate attention values for preceding text. BF16 KV (K0) is the 16-bit baseline; native NVFP4 KV (K1) stores packed four-bit values with scales.

Free output (C0) gives format instructions in the prompt and stops at the first contract error. Grammar (G) applies a public JSON schema to every response. Neutral retry (N) and error retry (E) each offer one retry after the first error, using a neutral message or the public error code, path and expected type. The schema supplies neither correct values nor the next action.

Every policy uses the same chat endpoint, template, single JSON envelope and strict parser. Extra prose, multiple objects, duplicate keys, wrong types and unregistered names are rejected. Schema-valid wrong values are passed to the fixture without hidden-gold repair. Validation checks only public invariants.

G leaves `uniqueItems` to the host, as registered before GPU execution because the backend did not support it. N/E retry only the first contract rejection, not business errors. A sixth final action is allowed; a sixth non-final action executes and then ends with TOOL_CALL_LIMIT under the original policy.

| KV | Bytes/token | Actual pool bytes | Token slots |
| --- | --- | --- | --- |
| K0 | 147456 | 4293918720 | 29120 |
| K1 | 41472 | 4294508544 | 103552 |

P0 retains native packed-storage/write/read evidence across 36 layers. The approximately 3.556× slot ratio describes pool capacity, not total VRAM reduction, a single 100K context or serving speedup.

## <a id="method"></a>Methods

A calculates and records integer totals from product quantities, per-item prices and discounts. B reads stock/revisions and performs an atomic warehouse transfer. C reads a document revision and updates status, tags and owner. DEV8 (A3/B3/C2), new HOLDOUT24 (eight/family) and P3-only COST6 are separate. HOLDOUT uses different templates, state, IDs and values, but is not a population of real companies.

V is the original `VERIFIED_WORKFLOW_SUCCESS`, S is `STATE_CORRECT`, and F is `FINAL_ANSWER_CORRECT`. Final must follow actual observation of successful validation of the updated current state. Protected fields, other records, dependency observations and strict final output are checked.

Fresh* is a **posthoc derived metric** from the review addendum. It checks target, SUCCESS, bool true, final-state digest/version and pre-final available_result_event_ids; if there is no final, it uses the last actual generation. Fresh alone does not guarantee task correctness.

## <a id="experiments"></a>Experiments

Qwen/Qwen3-4B-Instruct-2507, revision `cdbee75f17c01a7cc42f958dc650907174af0554`. BF16 weights, RTX 5080 16GB, vLLM0.32.0/FlashInfer/xgrammar, TP1, common GRAPH mode, episode C1, prefix cache OFF. Actual initial lengths are DEV1726–1784/HOLDOUT1734–1798 tokens.

Limits: 512 output tokens/request, 8 model attempts, 6 parsed tool/final attempts, 4096 output tokens/episode, one retry, request180s/episode900s, client context24576/server28672. TF32=False and BF16 reduced-precision reduction=True are common. Initial IDs match across configurations; free histories may diverge later.

P1/P2 COMPLETE means execution completed. Local records place HOLDOUT generation, sealing and P2 freeze before the first recorded response; 35 common frozen files are identical. These self-records are distinct from compliance with the omitted expansion criterion.

## <a id="results"></a>Results

### P1 DEV

| KV | C0 | G | N | E |
| --- | --- | --- | --- | --- |
| K0 | 5/8 | 5/8 | 5/8 | 5/8 |
| K1 | 2/8 | 5/8 | 2/8 | 4/8 |

### P2 HOLDOUT

| KV | Policy | V | S | F | Fresh* |
| --- | --- | --- | --- | --- | --- |
| K0 | C0 | 16 | 21 | 21 | 16 |
| K0 | G | 16 | 23 | 23 | 16 |
| K0 | N | 16 | 21 | 21 | 16 |
| K0 | E | 16 | 21 | 21 | 16 |
| K1 | C0 | 7 | 7 | 7 | 7 |
| K1 | G | 15 | 22 | 22 | 16 |
| K1 | N | 9 | 12 | 12 | 9 |
| K1 | E | 13 | 16 | 16 | 13 |

V/S/F/Fresh*: 24 tasks/cell, distinct outcome definitions. Fresh is posthoc.

| KV | Policy | A /8 | B /8 | C /8 |
| --- | --- | --- | --- | --- |
| K0 | C0 | 8 | 0 | 8 |
| K0 | G | 8 | 0 | 8 |
| K0 | N | 8 | 0 | 8 |
| K0 | E | 8 | 0 | 8 |
| K1 | C0 | 4 | 0 | 3 |
| K1 | G | 8 | 0 | 7 |
| K1 | N | 5 | 1 | 3 |
| K1 | E | 7 | 0 | 6 |

| Comparison | Recovery | Loss | Difference pp | 95% pp |
| --- | --- | --- | --- | --- |
| K0:G-C0 | 0 | 0 | 0.00 | [0.00, 0.00] |
| K0:N-C0 | 0 | 0 | 0.00 | [0.00, 0.00] |
| K0:E-C0 | 0 | 0 | 0.00 | [0.00, 0.00] |
| K0:E-N | 0 | 0 | 0.00 | [0.00, 0.00] |
| K1:G-C0 | 8 | 0 | 33.33 | [16.67, 50.00] |
| K1:N-C0 | 2 | 0 | 8.33 | [0.00, 20.83] |
| K1:E-C0 | 6 | 0 | 25.00 | [8.33, 41.67] |
| K1:E-N | 5 | 1 | 16.67 | [0.00, 33.33] |
| C0:K1-K0 | 0 | 9 | -37.50 | [-54.17, -20.83] |
| G:K1-K0 | 0 | 1 | -4.17 | [-12.50, 0.00] |
| N:K1-K0 | 1 | 8 | -29.17 | [-45.83, -12.50] |
| E:K1-K0 | 0 | 3 | -12.50 | [-25.00, 0.00] |

| KV | Policy | Requests | Input tokens | Output tokens | Episode wall (s) | Output tokens/verified job |
| --- | --- | --- | --- | --- | --- | --- |
| K0 | C0 | 102 | 201511 | 4996 | 77.864714401 | 312.250 |
| K0 | G | 110 | 217801 | 5399 | 84.562621448 | 337.438 |
| K0 | N | 104 | 205193 | 5139 | 80.317695542 | 321.188 |
| K0 | E | 108 | 213191 | 5546 | 85.310678150 | 346.625 |
| K1 | C0 | 51 | 98437 | 2470 | 39.188918025 | 352.857 |
| K1 | G | 110 | 218579 | 5947 | 93.437362778 | 396.467 |
| K1 | N | 88 | 171277 | 4355 | 68.265962043 | 483.889 |
| K1 | E | 102 | 200653 | 5001 | 79.399216725 | 384.692 |

More tasks completed, while total execution costs also increased. Dividing all 24 runs, including failures, by completions gives approximately 3.62% more input tokens, 12.36% more output tokens and 11.27% more episode-wall time per completion.

Input and output tokens are not interchangeable compute costs. Ratios divide all task costs, including failures, by completions; they are not mean latency of successful episodes, P3 throughput, billing or energy costs.

P0/P1/P2 recorded 22/233/775 requests (1030 total) and 678/11054/38853 output tokens (50585 total). P0 contains 12 probes plus ten requests in two smoke episodes. Every attempt received a response; infrastructure retries were zero. Worker walls are164.583118317/216.100527464/654.464649490s, totaling1035.148295271s. They include server load, compilation, shutdown and I/O, unlike summed episode walls.

## <a id="analysis"></a>Analysis

The 16 G warehouse failures are 14 stale-validation failures (seven tasks ×two KV configurations), plus two B05 runs with a missing SKU prefix and no transfer. These are different errors. The system prompt explicitly said `Inspect validation after the intended update.` and `An older validation is stale if a subsequent action changed the state.` ([source](src/agent_contract_cost/prompts.py)).

A00_K1_C0 was rejected at the first read because of an extra `}`; G completed calculation, writing, validation and final. B01_K0_G moved the correct quantities but observed only pre-transfer validation. C00_K1_G stored published when review was requested: Fresh1/S0/F0/V0. B03_K1_N was the only NVFP4 N warehouse completion, following neutral retry with transfer→validation→final. G/E validated before transfer on that task. N/E shared the first input and failed raw but later observations differed; this single case cannot isolate the causal effect of error-message information.

The preserved INCORRECT_FINAL label includes unmet observation obligations even when final numbers are correct. The secondary terminal_budget_or_infrastructure label is not evidence of actual budget exhaustion. A separate terminal-independent audit changed zero of 256 scores.

Intervals use the original family-stratified paired task bootstrap:10000 draws, seed6102013, retaining all eight conditions/task. These exploratory pointwise95% intervals do not certify external-business generalization, noninferiority or equivalence. The NVFP4 E−N interval includes zero.

## <a id="conclusion"></a>Conclusion

Under these conditions grammar increased NVFP4 workflow completions, while leaving semantic values and post-update validation planning unresolved. BF16 completions were unchanged, but correct-state counts rose21→23. Added token/episode-time costs and the omitted expansion criterion are disclosed together.

Scope: one model/device and short C1 synthetic tasks, comparing complete serving configurations. Other models, real business tasks, full quality equivalence, total VRAM savings, speed superiority and P3 concurrency remain unmeasured. P3 and new causal diagnostics are NOT_RUN.

## <a id="reproduce"></a>Reproduction and sources

[REPORT.md](REPORT.md) · [REPRODUCE.md](REPRODUCE.md) · [Runner](src/agent_contract_cost/runner.py) · [Full ledger](publication/summary/episode_ledger.csv)

The public layer losslessly compresses all256 complete raw responses, SSE, tokens and events. It restores original relative paths/bytes into a fresh directory and runs the frozen verifier. Original scorer/parser/protocol hashes are not rewritten for new documents. Publication source/build identities are separate from execution-source SHA.

Implementation and writing used Codex assistance. Separate CPU code is an AI-assisted audit, not human peer review, external GPU reproduction or author endorsement. ChatGPT review of the addendum CSV covers256-row scalar/paired arithmetic; full fixture replay is evidenced by local CPU receipts.

[NOTICE](NOTICE.md) · [REUSE_MAP](REUSE_MAP.md) · [Frozen protocol](specs/protocol.json) · [Execution freeze](provenance/execution_freeze_p2.json) · [Preservation map](publication/provenance/original_mapping.json)
