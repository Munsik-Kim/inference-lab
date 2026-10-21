# Case017 — Agent Contract–Cost Bench

We tested whether an agent both changes a synthetic record correctly and returns a verified final answer. Tasks cover integer order totals, atomic warehouse transfers, and document metadata updates. BF16 model weights stay fixed while four output-contract policies and two native KV-cache precisions vary.

[Overview](#overview) · [Question](#question) · [Implementation](#implementation) · [Method](#method) · [Results](#results) · [Analysis](#analysis) · [Conclusion](#conclusion) · [Reproduction](#reproduce)

## <a id="overview"></a>Overview

| KV | C0 FREE | G GRAMMAR | N NEUTRAL | E ERROR |
| --- | --- | --- | --- | --- |
| K0 | 16/24 | 16/24 | 16/24 | 16/24 |
| K1 | 7/24 | 15/24 | 9/24 | 13/24 |

The table reports verified completions on 24 new HOLDOUT tasks. Correct files alone or valid JSON alone do not count.

On NVFP4, G recovered 8 C0 failures with 0 losses: 33.33 percentage points, exploratory interval [16.67, 50.00]pp. All BF16 policies completed the same number. NVFP4 E-N had 5 recoveries and 1 loss; its interval includes zero.

P0: PASS; total actual model requests: **1030**; output tokens: **50585**; cumulative worker wall: **1035.148s**.

## <a id="question"></a>Background and question

Case015 selected-failure recovery and Case016 cache verification motivate this study; their results are not included here. Q1 concerns contract-policy completion and cost. Q2 concerns native NVFP4 versus BF16 KV with identical BF16 weights.

## <a id="implementation"></a>Technical principle and implementation contract

Our implementation supplies the strict parser, policy/budget runner, three safe fixtures, observation tracking, independent endpoint audit, and comparison tools. Engine, kernels, and constrained decoding come from vLLM, FlashInfer, and xgrammar. Full history is resent each turn; prefix caching is disabled.

| Policy | Contract |
| --- | --- |
| C0 | Free generation; first contract rejection terminates |
| G | Public branch-specific JSON schema on every response; no retry |
| N | Free generation; one neutral retry after first contract rejection |
| E | Free generation; one retry with public error code/path/type |

All policies share endpoint, template, one JSON envelope, and parser. G receives no gold IDs, amounts, or action order. `uniqueItems` remains a common host-only check because the installed backend does not support it. A sixth final is allowed; a sixth nonfinal executes then terminates. Final consumes an action, so there is no seventh generation.

| KV | Actual storage | Bytes/token | Actual pool bytes | Token slots |
| --- | --- | --- | --- | --- |
| K0 | BF16 | 147456 | 4293918720 | 29120 |
| K1 | packed uint8 + FP8 block scales | 41472 | 4294508544 | 103552 |

Capacity concerns token slots in the same approximately 4GiB KV pool; it is not total VRAM reduction or a 100K single-context claim.

## <a id="method"></a>Methods and experiments

Qwen/Qwen3-4B-Instruct-2507; revision `cdbee75f17c01a7cc42f958dc650907174af0554`; BF16 weights; vLLM 0.32.0; FlashInfer; xgrammar; TP1; GRAPH; episode concurrency1.

Actual numerical settings: TF32=False; BF16 reduced-precision reduction=True, identical across KV configurations.

DEV8 (A3/B3/C2) yields 64 episodes. After P1, the frozen generator seals HOLDOUT24 (eight per family), with different templates and structural states, yielding 192 episodes. COST6 has CPU gold paths only; P3 is not run. Initial prompts target approximately 2K tokens; actual lengths are recorded. Initial IDs must match across KV/policies; later free histories may diverge.

Limits: 512 tokens/request; 8 model attempts; 6 parsed tool/final attempts; 4096 output tokens; one retry; request180s/episode900s; client context24576; server context28672.

Output-token cost counts returned raw token IDs and is checked against usage; EOS/control IDs are retained. Episode wall includes local tokenization, tools, transport, and failures. Worker wall additionally includes server load, compile, and shutdown; warmup forwards are not additional API generation requests.

Pre-generation observation IDs are immutable. Dependent calculations/updates require observed reads; final requires the latest observed validation. Validation checks public invariants, never hidden gold. A separate CPU oracle evaluates the endpoint.

## <a id="results"></a>Experimental results

### P1 — COMPLETE

| KV | C0 FREE | G GRAMMAR | N NEUTRAL | E ERROR |
| --- | --- | --- | --- | --- |
| K0 | 5/8 | 5/8 | 5/8 | 5/8 |
| K1 | 2/8 | 5/8 | 2/8 | 4/8 |

| KV | Policy | Requests | Output tokens | Retries | Episode wall sum (s) | Tokens / verified job |
| --- | --- | --- | --- | --- | --- | --- |
| K0 | C0 | 26 | 1184 | 0/0 | 18.752 | 236.8 |
| K0 | G | 36 | 1648 | 0/0 | 26.561 | 329.6 |
| K0 | N | 29 | 1387 | 3/3 | 21.630 | 277.4 |
| K0 | E | 35 | 1850 | 3/3 | 28.408 | 370.0 |
| K1 | C0 | 15 | 710 | 0/0 | 11.230 | 355.0 |
| K1 | G | 36 | 1732 | 0/0 | 27.951 | 346.4 |
| K1 | N | 23 | 1042 | 6/6 | 16.538 | 521.0 |
| K1 | E | 33 | 1501 | 6/6 | 24.464 | 375.2 |

![P1 verified success](figures/p1_success.svg)

| KV | Policy | A: totals | B: transfer | C: metadata |
| --- | --- | --- | --- | --- |
| K0 | C0 | 3/3 | 0/3 | 2/2 |
| K0 | G | 3/3 | 0/3 | 2/2 |
| K0 | N | 3/3 | 0/3 | 2/2 |
| K0 | E | 3/3 | 0/3 | 2/2 |
| K1 | C0 | 1/3 | 0/3 | 1/2 |
| K1 | G | 3/3 | 0/3 | 2/2 |
| K1 | N | 1/3 | 0/3 | 1/2 |
| K1 | E | 3/3 | 0/3 | 1/2 |

| Comparison | Pairs | Both success | Loss | Recovery | Both fail | Difference (pp) | 95% interval (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K0:G-C0 | 8 | 5 | 0 | 0 | 3 | 0.00 | [0.00, 0.00] |
| K0:N-C0 | 8 | 5 | 0 | 0 | 3 | 0.00 | [0.00, 0.00] |
| K0:E-C0 | 8 | 5 | 0 | 0 | 3 | 0.00 | [0.00, 0.00] |
| K0:E-N | 8 | 5 | 0 | 0 | 3 | 0.00 | [0.00, 0.00] |
| K1:G-C0 | 8 | 2 | 0 | 3 | 3 | 37.50 | [12.50, 62.50] |
| K1:N-C0 | 8 | 2 | 0 | 0 | 6 | 0.00 | [0.00, 0.00] |
| K1:E-C0 | 8 | 2 | 0 | 2 | 4 | 25.00 | [0.00, 37.50] |
| K1:E-N | 8 | 2 | 0 | 2 | 4 | 25.00 | [0.00, 37.50] |
| C0:K1-K0 | 8 | 2 | 3 | 0 | 3 | -37.50 | [-62.50, -12.50] |
| G:K1-K0 | 8 | 5 | 0 | 0 | 3 | 0.00 | [0.00, 0.00] |
| N:K1-K0 | 8 | 2 | 3 | 0 | 3 | -37.50 | [-62.50, -12.50] |
| E:K1-K0 | 8 | 4 | 1 | 0 | 3 | -12.50 | [-25.00, 0.00] |

### P2 — COMPLETE

| KV | C0 FREE | G GRAMMAR | N NEUTRAL | E ERROR |
| --- | --- | --- | --- | --- |
| K0 | 16/24 | 16/24 | 16/24 | 16/24 |
| K1 | 7/24 | 15/24 | 9/24 | 13/24 |

| KV | Policy | Requests | Output tokens | Retries | Episode wall sum (s) | Tokens / verified job |
| --- | --- | --- | --- | --- | --- | --- |
| K0 | C0 | 102 | 4996 | 0/0 | 77.865 | 312.2 |
| K0 | G | 110 | 5399 | 0/0 | 84.563 | 337.4 |
| K0 | N | 104 | 5139 | 2/2 | 80.318 | 321.2 |
| K0 | E | 108 | 5546 | 2/2 | 85.311 | 346.6 |
| K1 | C0 | 51 | 2470 | 0/0 | 39.189 | 352.9 |
| K1 | G | 110 | 5947 | 0/0 | 93.437 | 396.5 |
| K1 | N | 88 | 4355 | 16/16 | 68.266 | 483.9 |
| K1 | E | 102 | 5001 | 16/16 | 79.399 | 384.7 |

![P2 verified success](figures/p2_success.svg)

| KV | Policy | A: totals | B: transfer | C: metadata |
| --- | --- | --- | --- | --- |
| K0 | C0 | 8/8 | 0/8 | 8/8 |
| K0 | G | 8/8 | 0/8 | 8/8 |
| K0 | N | 8/8 | 0/8 | 8/8 |
| K0 | E | 8/8 | 0/8 | 8/8 |
| K1 | C0 | 4/8 | 0/8 | 3/8 |
| K1 | G | 8/8 | 0/8 | 7/8 |
| K1 | N | 5/8 | 1/8 | 3/8 |
| K1 | E | 7/8 | 0/8 | 6/8 |

| Comparison | Pairs | Both success | Loss | Recovery | Both fail | Difference (pp) | 95% interval (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K0:G-C0 | 24 | 16 | 0 | 0 | 8 | 0.00 | [0.00, 0.00] |
| K0:N-C0 | 24 | 16 | 0 | 0 | 8 | 0.00 | [0.00, 0.00] |
| K0:E-C0 | 24 | 16 | 0 | 0 | 8 | 0.00 | [0.00, 0.00] |
| K0:E-N | 24 | 16 | 0 | 0 | 8 | 0.00 | [0.00, 0.00] |
| K1:G-C0 | 24 | 7 | 0 | 8 | 9 | 33.33 | [16.67, 50.00] |
| K1:N-C0 | 24 | 7 | 0 | 2 | 15 | 8.33 | [0.00, 20.83] |
| K1:E-C0 | 24 | 7 | 0 | 6 | 11 | 25.00 | [8.33, 41.67] |
| K1:E-N | 24 | 8 | 1 | 5 | 10 | 16.67 | [0.00, 33.33] |
| C0:K1-K0 | 24 | 7 | 9 | 0 | 8 | -37.50 | [-54.17, -20.83] |
| G:K1-K0 | 24 | 15 | 1 | 0 | 8 | -4.17 | [-12.50, 0.00] |
| N:K1-K0 | 24 | 8 | 8 | 1 | 7 | -29.17 | [-45.83, -12.50] |
| E:K1-K0 | 24 | 13 | 3 | 0 | 8 | -12.50 | [-25.00, 0.00] |

| Stage | Terminal reason | Episodes |
| --- | --- | --- |
| P1 | CONTRACT_FAILURE | 22 |
| P1 | INCORRECT_FINAL | 4 |
| P1 | INCORRECT_STATE | 5 |
| P1 | VERIFIED_SUCCESS | 33 |
| P2 | CONTRACT_FAILURE | 40 |
| P2 | INCORRECT_FINAL | 35 |
| P2 | INCORRECT_STATE | 9 |
| P2 | VERIFIED_SUCCESS | 108 |

## <a id="analysis"></a>Analysis and uncertainty

Functional failures, contract rejections, and unobserved infrastructure outcomes are separate. Intervals use 10,000 family-stratified paired task draws (seed6102013), preserving all eight cells. These synthetic tasks are not independent companies or an external business population. Zero-width intervals do not establish equivalence. N/E are independent free policies, not an isolated feedback-information experiment. KV changes may include kernel-path differences.

- [DEV-A00_K1_C0](results/p1_r1/cells/DEV-A00_K1_C0.json): `CONTRACT_FAILURE`; first unmet `read`.
- [DEV-A00_K1_N](results/p1_r1/cells/DEV-A00_K1_N.json): `CONTRACT_FAILURE`; first unmet `read`.
- [DEV-A01_K1_C0](results/p1_r1/cells/DEV-A01_K1_C0.json): `CONTRACT_FAILURE`; first unmet `read`.
- [DEV-A01_K1_N](results/p1_r1/cells/DEV-A01_K1_N.json): `CONTRACT_FAILURE`; first unmet `read`.
- [DEV-B00_K0_C0](results/p1_r1/cells/DEV-B00_K0_C0.json): `CONTRACT_FAILURE`; first unmet `source_read`.
- [DEV-B00_K0_E](results/p1_r1/cells/DEV-B00_K0_E.json): `INCORRECT_STATE`; first unmet `source_read`.
- [DEV-B00_K0_G](results/p1_r1/cells/DEV-B00_K0_G.json): `INCORRECT_STATE`; first unmet `source_read`.
- [DEV-B00_K0_N](results/p1_r1/cells/DEV-B00_K0_N.json): `CONTRACT_FAILURE`; first unmet `source_read`.

Example: [HOLDOUT-A00 / NVFP4 C0](results/p2_r1/cells/HOLDOUT-A00_K1_C0.json) generated an extra closing brace after its first read JSON, so nothing executed. [G on that task](results/p2_r1/cells/HOLDOUT-A00_K1_G.json) actually completed read→calculate→write→validate→final. This is a conditional serialization-bottleneck recovery.

[HOLDOUT-B01 / BF16 G](results/p2_r1/cells/HOLDOUT-B01_K0_G.json) validated before transfer and completed without observing validation of the changed state. Its file and final numbers were correct but its observation contract failed. [HOLDOUT-C00 / NVFP4 G](results/p2_r1/cells/HOLDOUT-C00_K1_G.json) stored schema-valid `published` instead of requested `review`. G does not supply the correct status; this semantic error remains a failure.

## <a id="conclusion"></a>Conclusion and scope

The paired table gives exact preservation, loss, and recovery counts. Syntactic validity does not guarantee correct values, state, or observed validation. Costs sum episode walls and generated tokens including failures; this is not a serving-throughput or speedup benchmark.

| Scope | Status |
| --- | --- |
| P0 native cache / grammar | Measured; see P0 gates and kernel/finite receipts |
| P1 DEV / P2 HOLDOUT | Separate measured denominators and stored-trace audits |
| P3 COST / concurrent serving | NOT_RUN |
| Weight INT4 / training / other models | NOT_RUN |
| Git publication / Pages | NOT_RUN; separate approval required |

Q1: Grammar increased actual completions on NVFP4 but not BF16, and did not guarantee semantic values or current-state validation. E has a higher point estimate than N, with an interval including zero and independent free-policy histories. Q2: The serving configurations differ despite identical BF16 weights. G narrows the observed gap without establishing equivalence, noninferiority, or general business superiority. Future approved work could diagnose N/E at shared failure observations or post-update validation planning; neither is run here.

## <a id="reproduce"></a>Reproduction and references

[CPU/GPU reproduction](REPRODUCE.md) · [Protocol](specs/protocol.json) · [Reuse map](REUSE_MAP.md) · [Notice](NOTICE.md) · [Roadmap](ROADMAP.md)

[P0 source freeze](provenance/execution_freeze_p0_r2.json) · [P1 source freeze](provenance/execution_freeze_p1.json) · [Model identity](provenance/model_identity.json)

Implementation and documentation used Codex assistance; a separate CPU calculation checks the endpoint. This is not human peer review or external GPU replication. CPU auditing replays saved text, SSE, token IDs, and events; it does not regenerate model outputs without weights.
