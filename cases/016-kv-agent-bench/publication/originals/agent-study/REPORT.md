# Case016 — Can a Smaller KV Cache Complete More Jobs Correctly?

Implemented real compressed-KV vLLM execution and a long-context workflow harness/auditor. On the frozen 96 DEV conditions, B16 completed 0/48 verified workflows and NVFP4 6/48. These reuse 12 fixtures under 4 menu/cap combinations. The baseline expansion gate is NOT_MET. P2/P3 are outside this run and are not executed.

[Overview](#overview) · [The job](#business) · [KV and baselines](#cache) · [Experiment](#experiment) · [Success and cost](#results) · [Failure events](#events) · [Reproduce](#reproduce) · [Sources](#sources)

<a id="overview"></a>

## Overview

The question is how smaller KV affects completion of long-context jobs. Completion of the experiment does not require a positive quality result. All 96 conditions were observed, with failures retained. Future 48 EVAL fixtures were generated and CPU-oracle-checked, never model-evaluated.

<a id="business"></a>

## The job

The four families cover quotes, latest versions, delayed arrivals and corrections/cancellations. A valid path selects the source using the policy, reads it, calculates, writes the target, validates current state and then answers. The gold path needs 4 tools and 5 generations, within budget.

Money uses integer minor units to avoid floating-point rounding. `quantity` is the count and `unit_price_minor` is price per unit. The rule is `floor((quantity*price*(100-discount)+50)/100)`. UTC cutoff, version and cancellation precedence are visible in the initial evidence.

Selection policy, quantity/discount and source candidates are placed at offsets3600/5500/7350 in8192-token inputs. Menus8/32 share lengths and evidence offsets. Irrelevant synthetic padding controls position; representativeness of natural-document workloads is not measured. Initial facts are available from generation0; there are no hidden reread obligations beyond the stated source verification.

<a id="cache"></a>

## KV and baselines

| KV | storage | actual pool B | capacity tokens | layers | max context |
|---|---|---|---|---|---|
| B16 | BF16 | 4,293,918,720 | 29,120 | 36 | 28672 |
| F8 | FP8 E4M3 | 4,293,918,720 | 58,240 | 36 | 28672 |
| F4 | NVFP4 packed + FP8 block scales | 4,294,508,544 | 103,552 | 36 | 28672 |

The requested pool is4,294,967,296B for every arm; the table shows allocations after page alignment. Shared storage views are not double-counted. Per-token bytes are147,456 for B16,73,728 for FP8 and36,864 packed+4,608 block scales=41,472 for NVFP4. Capacity is not throughput. F8 uses unchanged global scales1.0 (F8_DEFAULT_SCALE), in P0 only.

NVFP4 uses real uint8 packed data and FP8 E4M3 block scales. The GPU trace records its writer, FA2 prefill reading FP4 cache and the decode kernel. BF16 query/output does not imply BF16-cache fallback. Exact names are in `results/p0_diagnostics_r2/F4_kernel_profile.json`.

<a id="experiment"></a>

## Experiment

P0 technically executed8 SMOKE episodes per KV. The two decoders of the same compressed cache agreed exactly; relative norm error against original BF16 was about 0.095. Separately, the exact upstream test body passed 4 layouts; the full upstream pytest suite is NOT_RUN. There are 34 pre-run Case016 CPU boundary tests and 3 separately reported post-run CPU review properties.

| KV | SMOKE observed | verified workflow |
|---|---|---|
| B16 | 8 | 1 |
| F8 | 8 | 1 |
| F4 | 8 | 0 |

P1 is12 DEV×2 KV×2 menus×2 caps=96. Common settings are BF16 weights,TP1,FlashInfer,eager,async scheduling OFF,prefix caching OFF,chunked prefill2048,max-num-seqs4,context28672 and clientC1. Budgets are8192 generated tokens,8 model calls,6 tool attempts,180s/request and900s/episode,temperature0 with EOS honored. Tasks/rules were not tuned after model outcomes. Setup/instrumentation fixes are separately recorded in amendments01–08.

![All eight P1 cells](figures/p1_success.svg)

Each turn resends the full history and repeats prefill with prefix caching OFF. Autoregressive KV is used within each HTTP request and released afterward; an episode does not permanently occupy its KV throughout the workflow.

<a id="results"></a>

## Success and cost

| KV | menu | cap | verified/12 | state/12 | final/12 | false done |
|---|---|---|---|---|---|---|
| B16 | 8 | 256 | 0/12 | 2/12 | 0/12 | 4 |
| B16 | 8 | 1024 | 0/12 | 2/12 | 0/12 | 4 |
| B16 | 32 | 256 | 0/12 | 4/12 | 0/12 | 2 |
| B16 | 32 | 1024 | 0/12 | 4/12 | 0/12 | 2 |
| F4 | 8 | 256 | 0/12 | 2/12 | 0/12 | 3 |
| F4 | 8 | 1024 | 0/12 | 2/12 | 0/12 | 3 |
| F4 | 32 | 256 | 3/12 | 5/12 | 5/12 | 1 |
| F4 | 32 | 1024 | 3/12 | 5/12 | 5/12 | 1 |

| comparison | both success | loss | recovery | both fail |
|---|---|---|---|---|
| B16/menu8: cap256 to1024 | 0 | 0 | 0 | 12 |
| B16/menu32: cap256 to1024 | 0 | 0 | 0 | 12 |
| F4/menu8: cap256 to1024 | 0 | 0 | 0 | 12 |
| F4/menu32: cap256 to1024 | 3 | 0 | 0 | 9 |
| menu8/cap256: B16 toF4 | 0 | 0 | 0 | 12 |
| menu8/cap1024: B16 toF4 | 0 | 0 | 0 | 12 |
| menu32/cap256: B16 toF4 | 0 | 0 | 3 | 9 |
| menu32/cap1024: B16 toF4 | 0 | 0 | 3 | 9 |

Baseline usable: **NOT_MET**; B16/menu8/cap1024 = **0/12**, families {'A': 0, 'B': 0, 'C': 0, 'D': 0}. The registered9/12 and at least2/3 per family criterion is an expansion-management proposal. The weak baseline limits interpretation of these DEV paired changes as practical superiority, equivalence or noninferiority.

| KV | model requests | tool attempts | events | output tokens | worker attempt s | peak allocated GiB | peak reserved GiB |
|---|---|---|---|---|---|---|---|
| B16 | 266 | 284 | 284 | 13164 | 513.292 | 12.601 | 12.688 |
| F4 | 212 | 260 | 251 | 11916 | 568.285 | 12.601 | 12.688 |

P1 worker clock: **1084.509s**. This includes load, failures, I/O and metrics; it is neither GPU busy nor a serving speedup. C1 mixed-DEV makespan/correct-job ledgers are in summary.attempts. Seconds/tokens per correct job are null when success is0. Received tokens from cancel probes are preserved, but total engine generation before abort is not exactly observed.

<a id="events"></a>

## Failure events

| terminal reason | episodes |
|---|---|
| TOOL_CALL_LIMIT | 62 |
| FINAL | 30 |
| PARSE_STOP | 4 |

| first unmet obligation | episodes |
|---|---|
| validation | 20 |
| write | 6 |
| source | 48 |
| calculation | 16 |
| none | 6 |

Tool attempts count slots admitted within the host budget. Generated over-budget slots remain in raw responses. Event counts include registry/argument rejections; successful callable events are separately counted in diagnostics.

Public valid:true checks schema/invariants, not the gold amount or correct version selection. B16 DEV-B00/menu8/cap1024 selected version2 rather than3, wrote348142 and received valid:true. The independent gold is253559, so failure is retained. The full record is linked; this is not a selected sole representative of all conditions.

All calls in one response share the same pre-generation observations. Write→validate can execute sequentially, but read/calculation-dependent actions must be generated after those results were received. First-unmet follows the registered DAG order, not a unique diagnosis of internal model causality. Diverged free rollouts are not claimed to have identical later observations.

[All 96 raw records](results/p1_r1/episodes/) · [Family/parse/length diagnostics](results/p1_r1/diagnostics.json) · [Paired CSV](results/p1_r1/paired.csv)

<a id="reproduce"></a>

## Reproduce

[Commands and CPU audit scope](REPRODUCE.md). Reassemble raw SSE and replay state/observations/endpoints on fresh fixtures. A scalar audit does not reproduce model inference. No model download or quantizer is needed for the CPU path.

<a id="sources"></a>

## Sources

Model revision: `cdbee75f17c01a7cc42f958dc650907174af0554`. P1 execution source SHA256: `ab8463e9bfde41564e44a863bc1159dfecf26cada54a7fdceea5ebec127e53ee`. Git base: `0c86afa4728cbf901f6c3df64802810db6581a00`.

| scope | status |
|---|---|
| vLLM/NVFP4 storage + 36 layers | observed |
| P1 DEV96 | completed |
| EVAL48 /16K /C2,C4 /P2,P3 | NOT_RUN |
| weight W4 /D16 /training | NOT_RUN |
| Git stage/commit/push/PR/Pages | NOT_RUN |
| full model logits independently checked finite | NOT_RUN; small cache decode finite checked |

P2/P3 require review of baseline failures, separate approval and a newly frozen version/plan. No prompt/tool/budget sweep was added to improve this run.

[vLLM0.32 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0), [NVFP4 SM8x/SM12x support](https://github.com/vllm-project/vllm/pull/46963), [SM12x XQA decode](https://github.com/vllm-project/vllm/pull/60452), [Qwen function calling](https://qwen.readthedocs.io/en/latest/framework/function_call.html). [Source audit](provenance/source_audit.json), [reuse map](REUSE_MAP.md), [NOTICE](NOTICE.md).

Upstream engine/kernels/model are distinguished from the implemented fixtures/client/runner/scoring/audit. Codex assistance and separate scalar calculations are disclosed; they are not human peer review or external GPU reproduction.
