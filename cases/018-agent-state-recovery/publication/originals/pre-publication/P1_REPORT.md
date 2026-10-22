# Case018 P1 — Server gates and DEV24

The fresh-validation completion guard did not improve the primary recovery endpoint in this DEV run. With BF16 weights fixed, BF16 KV achieved Q=6/6 under both R0 and R1; native NVFP4 KV achieved Q=5/6 under both policies. Verified recovery workflow W was 2/6 for both BF16 policies and 0/6 for both NVFP4 policies.

| KV | Policy | D | B | P | Fresh | C1 | C2 | Q | W | Requests | Input tokens | Output tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | R0 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 2 | 44 | 86,059 | 2,866 |
| BF16 | R1 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 2 | 44 | 86,059 | 2,866 |
| NVFP4 | R0 | 6 | 5 | 6 | 5 | 5 | 5 | 5 | 0 | 44 | 84,733 | 2,298 |
| NVFP4 | R1 | 6 | 5 | 6 | 5 | 5 | 5 | 5 | 0 | 48 | 96,781 | 2,662 |

Each row contains the same six synthetic tasks (two wrong-target, two duplicate, two partial effects), rather than six new independent samples per policy. Faults are injected deterministically on CPU. This does not measure natural model error frequency.

Q requires restored business values, protected records/history, and correct outcomes in two fixed CPU continuation programs started from independent copies of the terminal state. W additionally requires an accepted, accurate final and observation of successful validation of the current state. These continuations are evaluator programs, not autonomous model actions or proof of equivalence for all future workloads. A rejected recovered:true final still counts as a declaration D, not verified success.

There were 18 Q1/W0 episodes: their saved state and registered continuations were correct, but their final quantities differed from the actual terminal quantities. In `DEV-A00_K1_R0`, the repaired x/y quantities were 42/26 while the final reported 36/32, despite fresh successful validation. In `DEV-C01_K1_R0`, compensating both sides of a partial debit caused an incorrect destination quantity. Repeated validation returned false; the model still declared recovery. R1 rejected its three final attempts and stopped at GUARD_LIMIT without correcting the state. No observed model case had B1 with failed continuations only, or Fresh1/B0; corresponding P0 scripted counterexamples are not added to model results.

Within each KV, R1−R0 had no Q or W recoveries or losses. For NVFP4−BF16, each policy had one Q loss and two W losses, with no recoveries. These are descriptive DEV6 paired counts, not equivalence or general quality claims. NVFP4 R1 added four requests and 364 output tokens without improving Q/W. The comparison includes kernel-path differences, not just quantization rounding.

All technical gates passed in the current runtime: actual BF16 weights and tokenizer identities; the real V2 xgrammar mask; all 36 cache layers, packed uint8 NVFP4 data and FP8 block scales; actual NVFP4 write/read kernels; finite outputs from all36 attention implementations in explicit technical prefill/decode probes; raw SSE/token/usage accounting, deadlines and observed engine drain. Main serving used the same GRAPH/FlashInfer/TP1 configuration, 28672 context, four sequences, 2048 batched tokens, no prefix cache and C1 episodes. The uncompiled finite-output probes were technical-only and restored their bindings.

| KV | Actual pool bytes | Bytes/token | Blocks (16) | Token slots |
|---|---:|---:|---:|---:|
| BF16 | 4,293,918,720 | 147,456 | 1,820 | 29,120 |
| NVFP4 | 4,294,508,544 | 41,472 | 6,472 | 103,552 |

The approximately3.556× slot ratio concerns a fixed approximately4GiB KV pool, not total GPU VRAM savings, a100K single context, quality equivalence or serving throughput. We reused Qwen3-4B-Instruct-2507 revision cdbee75f17c01a7cc42f958dc650907174af0554 and the existing vLLM0.32.0 environment. No installation, download or weight quantization was performed. Small model/runtime file hashes and weight stats were checked; full weight files were not rehashed.

DEV generated180 complete responses with353,632 input and10,692 output tokens, including controls. Episode wall sum was165.105584630s. There were six TECH generation attempts across preserved attempts: four complete responses (3,868 input/52 output tokens) and two intended timeouts without SSE/usage. Their engine token costs remain unknown. Total generation attempts were186, complete responses184, and observed output tokens at least10,744. Twelve explicit technical forwards and uncounted internal engine warmups are separate from generation API attempts.

All-attempt worker wall was439.692244164s, including185.476104969s of pre-DEV technical failures and254.216139195s of the final attempt. This leaves3160.307755836s of the common3600s budget. These intervals include load, compile, failed attempts, I/O and shutdown and overlap episode/continuation costs. They are not model busy time or a throughput claim. Cost per W is undefined for NVFP4 because W=0.

The final model-free audit replayed all24 episodes, reassembled every DEV SSE stream, and found zero endpoint or raw-trace differences. Separate CPU rendering/encoding, special-token decode and original ordered-schema grammar-prefix checks passed for all180 requests. An earlier post-hoc checker incorrectly recompiled the sorted-key freeze schema, changing xgrammar property order; its failure and replacement checker are retained without changing experiment source, raw or scores. CPU tests comprise37 existing plus11 new unique tests, executed as a44-test batch and four individual additions, not one48-test run. The26 frozen P0 traces were replayed once with their original source.

Execution status is COMPLETE independently of the model score. The uncommitted execution source identity is `e60dc421c2dc28c59e29ab26cb9faa6a2893208d5399577eed08d9745211ee7b`; the Git base remains `83e1cef6dd60384931a38e016ea5e8d069d40674`. All first DEV outputs were unobserved at freeze, while six TECH attempts were explicitly disclosed. Pre-DEV observer fixes and their source snapshots remain in r1–r4 and amendments01–04. No DEV cell was regenerated.

Original P0 source/spec/data/results are recoverable from the229-file exact-byte snapshot. Only the permitted runner/audit/CLI/REUSE_MAP extensions changed existing Case018 files; selected81 Case017 files, eight imported helpers and selected runtime/model identities remain unchanged. All owned servers were stopped; no owned PID or port8196 listener remained. Git staging/publication, HOLDOUT generation/sealing, P2 and P3 are NOT_RUN. See the [Korean report](P1_REPORT.ko.md), [CPU/GPU command scopes](REPRODUCE.P1.ko.md), [machine summary](results/p1_r5/analysis/summary.json) and [execution freeze](results/p1_r5/execution_freeze.json).

Owned process groups received SIGTERM and API parents exited0. Upstream teardown logs nevertheless show forced termination of their own remaining EngineCore child and semaphore-cleanup warnings. These warnings are retained separately from the confirmed request drain and absence of residual owned processes; warning-free graceful teardown is not claimed.
