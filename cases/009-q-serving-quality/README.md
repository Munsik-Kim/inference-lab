# Case 009 — Qwen serving cost and official quality

[한국어](README.ko.md) · [Full report](REPORT.md) · [CPU reanalysis](REPRODUCTION.md)

Measured the existing Qwen3-4B-Instruct-2507 BF16 and GPTQ W4A16 artifacts with CUDA Graph execution, 256-token decode and four client-concurrency limits. The same-input runner retains latency, throughput, token counts, scheduler observations and process identities. Official quality tasks run separately through pinned lm-evaluation-harness definitions.

On RTX 5080, the primary input128/output256 graph conditions gave paired BF16/W4 TPOT ratios of **2.113, 2.094, 1.934, 1.656** at concurrency 1/4/16/32. These are complete-request decode timing observations with three server rounds; the full curves, eager control, quality scores and native exit failures belong to the same report.

## Serving results

![All primary concurrency values, with three-round ranges](figures/serving-L128.png)

| Client concurrency | TPOT BF16 / W4 (ms/token) | Paired time ratio BF16/W4 |
|---|---:|---:|
| 1 | 12.155 / 5.754 | 2.113 |
| 4 | 12.622 / 6.030 | 2.094 |
| 16 | 13.664 / 7.068 | 1.934 |
| 32 | 14.792 / 8.931 | 1.656 |

TPOT cells average round medians; ratio cells average paired round ratios, so they need not equal the quotient of displayed rounded times. Whiskers show raw round min–max, not confidence intervals. [Both input lengths, paired intervals, TTFT, throughput, memory and preemptions](REPORT.md#results). Client concurrency is not GPU batch size. The 60 cells retain 7,296 timed requests and 120 fixed warmups. There are 512 unique synthetic workload prompts reused across settings.

## Official quality results and execution status

| Official task / metric | BF16 | W4 | Paired W4−BF16 [pointwise 95% interval] |
|---|---:|---:|---:|
| arc_challenge **process status** | COMPLETE | FAILED | retained computation below; not clean execution |
| arc_challenge/none acc | 507/1172 (43.26%) | 515/1172 (43.94%) | +0.68 [-1.02, +2.39] pp |
| arc_challenge/none acc_norm | 502/1172 (42.83%) | 493/1172 (42.06%) | -0.77 [-2.47, +0.85] pp |
| wikitext **process status** | FAILED | FAILED | retained computation below; not clean execution |
| WikiText-2 word perplexity | 13.1081 | 14.5194 | word NLL +0.10226 [+0.09620, +0.10893] nats |
| WikiText-2 byte perplexity | 1.6180 | 1.6493 | separate original UTF-8 denominator |
| gsm8k **process status** | COMPLETE | COMPLETE | original clean process exits |
| gsm8k/flexible-extract exact_match | 1211/1319 (91.81%) | 1191/1319 (90.30%) | -1.52 [-2.96, -0.15] pp |
| gsm8k/strict-match exact_match | 1145/1319 (86.81%) | 1028/1319 (77.94%) | -8.87 [-11.14, -6.60] pp |
| mmlu **process status** | COMPLETE | FAILED | retained computation below; not clean execution |
| MMLU acc (57 subjects) | 8720/14042 (62.10%) | 7616/14042 (54.24%) | -7.86 [-9.05, -6.87] pp |


**Execution qualification:** arc_challenge/W4, wikitext/BF16, wikitext/W4, mmlu/W4 exited abnormally after all calculations and result files were written. Their status remains `COMPUTED_WITH_PROCESS_FAILURE`. Pairing, finite scores and official aggregates were checked, but these are not clean process completions. No full split was repeated. [Attempt ledger](provenance/quality_attempts.json) · [Item transitions, MMLU subjects and generation caps](results/derived/quality_summary.json).

GSM strict-match and flexible-extract are two official views of the same 1,319 outputs. ARC acc and acc_norm are separate official metrics. WikiText uses original-word/byte denominators over 62 documents. Benchmark scores are not pooled. Full-vocabulary KL is NOT_RUN; choice-score diagnostics use normalized continuation likelihoods.

## Tools and evidence

- [Request runner](scripts/serving.py): matched token IDs, graph/eager process order and measured SSE timing.
- [Official-task adapter](scripts/quality.py): frozen request/few-shot checks before model calls and retained generation limits.
- [Installable paired CLI](../../packages/diova-compare/README.md): CPU contracts, correctness transitions, NLL/Brier and explicitly scoped KL; source-process failures remain visible.
- [Methods](METHODS.md), [protocol](configs/serving_protocol.json), [quality protocol](configs/quality_protocol.json), [reproduction](REPRODUCTION.md), [attribution](NOTICE.md).

The existing BF16/W4 files, Cases001–008 and modelpack code remain unchanged. This case did not rebuild weights or run R reconstruction. The new data-v2 fixtures received CPU checks only. Multi-layer R is a [future design](../../docs/phase2a-design.md). Deployment: **NOT_ASSESSED**. Publication: **LOCAL_REVIEW**. The [run state](RUN_STATE.json) separates completed measurement from unresolved runtime validation.

## Reviewed software and diagnostics

[CLI 0.1.1 wheel](../../downloads/diova_compare-0.1.1-py3-none-any.whl) · [Public ZIP](../../downloads/case009_serving_quality_reviewed_publication_v1.zip) · [Metadata](../../downloads/case009_serving_quality_reviewed_publication_v1.json) · [Review follow-up](publication/README.md)

MMLU declined in all 57 subjects; WikiText likelihood declined on all 62 documents. The detailed report separates these observations, the original failed process exits and the new short lifecycle probes.
