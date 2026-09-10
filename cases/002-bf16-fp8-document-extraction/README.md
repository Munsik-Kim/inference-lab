# Comparing official BF16 and FP8 Qwen3-4B on Korean document extraction

**Accuracy was low for both configurations: BF16 31/100 documents, FP8 32/100.** FP8 used less GPU memory and completed requests sooner in this RTX 5080 run, but neither configuration is supported for unattended extraction under these conditions.

The task extracts an active request's owner, exact task name, confirmed date and KRW amount from synthetic Korean records containing other requests and versioned changes. Each model is scored against independently constructed gold, not against BF16 output. There are 20 development documents and 100 evaluation documents; the three rounds do not create 300 independent quality samples.

## Primary accuracy

| Primary round 1 | BF16 | FP8 |
|---|---|---|
| All (n=100) | 31/100 (31%) | 32/100 (32%) |
| Short (n=50) | 18/50 (36%) | 17/50 (34%) |
| Long (n=50) | 13/50 (26%) | 15/50 (30%) |
| JSON valid | 100/100 | 100/100 |
| Schema valid | 100/100 | 100/100 |

FP8−BF16 document accuracy: **+1 percentage point**, paired stratified bootstrap 95% interval **[-2, 5] percentage points** (10,000 resamples, preserve 50/50 short/long). This does not establish superiority, equivalence, non-inferiority or preserved quality.

|  | FP8 correct | FP8 wrong |
|---|---|---|
| BF16 correct | 30 | 1 |
| BF16 wrong | 2 | 67 |

| Field | BF16 all | FP8 all | BF16 null / non-null | FP8 null / non-null |
|---|---|---|---|---|
| owner | 64/100 (64.0%) | 65/100 (65.0%) | 1/20 (5.0%) / 63/80 (78.8%) | 0/20 (0.0%) / 65/80 (81.2%) |
| task | 70/100 (70.0%) | 70/100 (70.0%) | 0/10 (0.0%) / 70/90 (77.8%) | 0/10 (0.0%) / 70/90 (77.8%) |
| due_date | 68/100 (68.0%) | 72/100 (72.0%) | 7/10 (70.0%) / 61/90 (67.8%) | 7/10 (70.0%) / 65/90 (72.2%) |
| amount_krw | 78/100 (78.0%) | 77/100 (77.0%) | 16/20 (80.0%) / 62/80 (77.5%) | 15/20 (75.0%) / 62/80 (77.5%) |

Pooled null-field accuracy was 24/60 (40.0%) for BF16 and 22/60 (36.7%) for FP8, versus 256/340 (75.3%) and 262/340 (77.1%) for non-null fields. Missing fields did not inflate the overall field score: for example, neither model correctly returned null for any of the ten missing task names. Both often copied the literal `미기재` (“not recorded”), or mixed fields from another request. All 200 round-1 outputs were valid four-field JSON objects; syntactic validity did not mean correct extraction. Length-specific field/null breakdowns are in [aggregate.json](results/aggregate.json).

[All three discordant cases and two simultaneous failures](results/error_examples.md) show full documents, gold, evidence lines and both raw outputs. [The full list](results/failures.jsonl) contains 70 documents where at least one model was wrong. No responses were repaired or excluded.

## Latency and memory

The following timings are **median / p95**, measured by a sequential localhost streaming client after a short and long warmup. Each model/bucket contains 150 requests across three rounds, repeatedly measuring the same 50 documents. Output-token columns show medians; full distributions are in the aggregate file.

| Bucket | BF16 TTFT ms | FP8 TTFT ms | BF16 total ms | FP8 total ms | BF16 / FP8 output tokens |
|---|---|---|---|---|---|
| short | 51.5 / 52.9 | 26.2 / 34.1 | 658.4 / 704.3 | 399.0 / 459.4 | 54 / 49 |
| long | 355.5 / 359.2 | 173.4 / 175.7 | 974.4 / 1032.0 | 562.3 / 604.7 | 52 / 48 |

For matched document/round pairs, the median FP8/BF16 total-latency ratio was **0.630 short** and **0.572 long**. Output lengths were equal in only 87/150 short and 57/150 long pairs. Therefore the total-time difference is not a pure decode-speed comparison. The approximate post-first-token time per output token was BF16/FP8 11.66/7.62 ms short and 12.28/8.27 ms long; it includes final transport overhead and is not token-level profiling. SSE chunks are not counted as tokens.

FP8 was not faster for every document. In round 1, `eval-short-045` took 764.5 ms with FP8 versus 654.9 ms with BF16: FP8 incorrectly included the rest of a record line in `task`, generating 98 tokens versus 54. This same document was slower under FP8 in all three rounds.

Round-level median request completion times:

| Round | Order | Short total ms BF16 / FP8 | Long total ms BF16 / FP8 |
|---|---|---|---|
| 1 | BF16 → FP8 | 655.1 / 395.2 | 967.4 / 556.8 |
| 2 | FP8 → BF16 | 658.8 / 405.1 | 982.1 / 572.3 |
| 3 | BF16 → FP8 | 662.4 / 392.6 | 982.1 / 561.0 |

Each model's raw outputs were identical to its own round-1 outputs for all 100 documents in both later rounds. That is repeatability on this machine, not independent reproduction or a quality guarantee.

| Memory measure (GiB) | BF16 | FP8 |
|---|---|---|
| Engine model-loading allocation, reported | 7.64 | 4.32 |
| Explicit BF16 KV cache budget | 1.000 | 1.000 |
| Whole GPU idle median, range across rounds | 2.008–2.008 | 2.008–2.008 |
| Whole GPU ready median, range across rounds | 11.099–11.104 | 7.741–7.742 |
| Whole GPU measured sampled peak, range across rounds | 11.623–11.624 | 8.211–8.211 |

The difference between median per-run sampled peaks was **3.412 GiB (29.4% of BF16 whole-device peak)**. Both engines reported 7,280 cache tokens under the same 1 GiB budget. NVML sampled the entire device every nominal 100 ms; Windows/background allocations are included. These are sampled peaks, not guaranteed instantaneous peaks or per-process/torch-allocator measurements. The engine model-loading allocation is not the same quantity as file size or total GPU use.

Startup-to-health ranged 17.0–19.1 seconds in evaluation, with compilation caches already populated during development. Load, JIT, graph capture, warmup and shutdown records are retained separately; this is not a fresh-machine cold-start test. All 600 measured requests generated content and finished with `stop`; no request exceptions, truncations or timeouts occurred. All six evaluation servers exited with code 0 after the runner requested shutdown. Existing shutdown warnings remain visible.

## Tested configuration and provenance

- RTX 5080 16GB / SM120, WSL2 Ubuntu 24.04.4, NVIDIA driver 610.47.
- Python 3.12.14; official vLLM 0.29.0 wheel (release commit `98dff2a81d747d1dba01a47f939f48c3526d4206`); PyTorch 2.13.0+cu130; Triton 3.7.1. An isolated baseline clone was used; no CASE 001 patch was applied.
- BF16: [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554), revision `cdbee75f17c01a7cc42f958dc650907174af0554`.
- FP8: [`Qwen/Qwen3-4B-Instruct-2507-FP8`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8/tree/8591804019c8b22094c3b5b4454e0edc05dffc98), revision `8591804019c8b22094c3b5b4454e0edc05dffc98`.
- Same non-thinking architecture, tokenizer and chat template; all actual input token IDs matched. FP8 config: dynamic activation quantization, e4m3, 128×128 weight blocks and module exclusions. The FP8 files also contain F16 tensors. This compares official deployment artifacts, without independently establishing exact upstream tensor conversion history.
- TP=1, concurrency=1, `max_num_seqs=1`, length limit 5120, generation limit 256, temperature 0, fixed seed and stop tokens. Engine dtype and unquantized KV cache dtype: bfloat16.
- Normal API server with separate V1 EngineCore, FlashAttention 2, compile/CUDA graphs `FULL_AND_PIECEWISE` with capture sizes [1,2]; **not eager**. CPU offload, prefix caching, chunked prefill, async scheduling, speculation and constrained JSON decoding are disabled. FlashInfer sampler is disabled. Development failures and the one shared prompt adjustment are documented in [development evidence](results/development/README.md).

## Reproduce and inspect

Start with [methods and commands](METHODS.md) and [environment installation/provenance](provenance/INSTALL.md). The supplied [experiment spec](configs/experiment_spec.json) was frozen before evaluation: SHA256 `7a7602ad33bc57eb2a346c8b06d1811ef9442a9301a677a0f93d7eb38d576d26`. Dataset seed: 20260910; facts version 1, prompt/token-count metadata version 2. Fixed token lengths are 498–559 short and 4009–4069 long for evaluation.

The scripts prepare pinned models, reconstruct/check synthetic data, stream requests, score, export path-redacted evidence and calculate paired results. Model weights, environments, caches and private user reports are not included. [models.json](provenance/models.json) and [local_model_manifest.json](provenance/local_model_manifest.json) record revisions and checked files; [export_manifest.json](results/export_manifest.json) connects public copies to original evidence hashes.

## Interpretation and limits

There is a resource-efficiency reason to investigate this FP8 deployment further: lower observed device memory, TTFT and request completion latency. **There is not enough accuracy to recommend either frozen setup for unattended extraction.** FP8's +1-point difference is small and uncertain; lower latency partly coincides with different output lengths. Improving task reliability would require a separately defined follow-up, not relabeling these outputs or changing this gold after evaluation.

The 100 documents are synthetic and share generation rules with development data. Missing values, distractor requests and version ordering dominate this particular task. Results do not establish general Korean-language quality, performance of all FP8 methods, other GPUs, long-term stability, multi-GPU behavior, or a controlled effect of only weight precision. No fresh-machine or independent reproduction was performed. The initial default V2 runner and FlashInfer sampler did not work in this WSL setup; the recorded adjustments bound applicability.

OpenAI Codex assisted with synthetic data, code, execution, analysis and documentation. No human annotation review, upstream adoption, new kernel or new quantization method is claimed. See [NOTICE](NOTICE.md) and [LICENSE](LICENSE).
