# Matched serving and official quality evaluation

## Question and fixed artifacts

Compare the existing Qwen3-4B-Instruct-2507 BF16 checkpoint with the existing GPTQ W4A16 checkpoint under longer decode and controlled client concurrency. This is a new workload study, not a correction to Case 008's eager, eight-token requests. No model was requantized, calibrated, trained or downloaded for this study.

The original model revision is `cdbee75f17c01a7cc42f958dc650907174af0554`. [Artifact provenance](provenance/artifacts.json) checks 22 retained files against Case 008's recorded identities. The tokenizer and chat template are shared; `lm_head` remains excluded from GPTQ. The model has 36 layers, hidden size 2560 and intermediate size 9728. Its 252 unfused projection modules map to 144 vLLM fused projection modules; these counts are not GPU call counts.

## Serving design

[The frozen protocol](configs/serving_protocol.json) records the complete process and condition order. [Requests](configs/requests.json) retain exact token IDs and hashes, made before timing. The default graph/compile path uses identical settings for both models: vLLM 0.29.0, TP=1, BF16 activations and KV, 4 GiB KV cache, maximum context 2048, maximum sequences 32 and batched-token limit 2048. Prefix caching, speculative decoding, chunked prefill and async scheduling are disabled. The sampler and V1 runner choices are recorded in [the runner](scripts/serving.py). No profiler is enabled during primary timing.

Inputs have exactly 128 or 1024 tokens before submission; outputs have exactly 256 generated tokens with greedy sampling and `ignore_eos`. This fixed-workload measurement does not use a chat template or pretend that forced-length text is a quality task. Client maximum concurrency is 1, 4, 16 or 32, with saturation submission and `max(64, 8*C)` requests per cell. Request ID/order/token IDs are identical between arms. Two fixed warmup requests precede each cell and are separately retained. Returned token IDs and API usage must agree; streaming chunks are never used as token counts.

Three paired server-process rounds alternate arm order, with a fixed condition permutation shared by both arms. The graph sweep has 48 cells. A matched eager ablation adds 12 cells at input128/output256, C=1/16. Planned timed work is 7,296 requests and 1,867,776 output tokens, plus 120 warmup requests. The manifest contains 512 distinct synthetic workload prompts, reused across arms, conditions and rounds; repeated requests are not additional independent quality samples. The same model is loaded alone in each process. Client concurrency, scheduler running sequences and a Linear kernel's token-row dimension are distinct.

The common KV capacity was chosen in DEV: 3 GiB first, then one symmetric increase to 4 GiB after observing headroom. DEV includes 19 BF16 and 18 W4 requests across three server starts. With long inputs, C=32 can exceed the approximately 29,120-token active KV capacity. Preemptions, queueing and any incomplete cells remain in results; capacity failure is not an infinite speedup. No maximum-capacity expansion arm was added.

## Timing definitions and uncertainty

The client uses `perf_counter`. TTFT is submission to the first choice-bearing SSE event; request latency ends at the last choice-bearing event. TPOT is `(last − first)/(generated_tokens − 1)`. HTTP completion time is retained separately. These follow the installed vLLM serving benchmark's timestamp convention; network/chunk buffering can affect observed inter-token timing. ITL was not collected and is unavailable. Throughput is total actual output tokens divided by measured block elapsed time.

Each server-round cell produces p50/p95 TTFT, TPOT and request latency. Display tables average these three round summaries, not all requests as independent trials. Time ratios are BF16/W4; throughput ratios are W4/BF16. A 5,000-replicate paired server-round bootstrap (seed909221) gives pointwise descriptive intervals. Only three independent process rounds are available. Plot whiskers show raw round min–max, not confidence intervals.

Whole-device memory is sampled at 1 Hz, including server startup, warmup, desktop and background allocations. It is neither a Torch allocated/reserved peak nor exclusive model memory; subsecond peaks may be missed. WSL desktop activity and CPU document/test preparation occurred during this session, especially the first graph round. The host was not an isolated benchmark appliance. These conditions and all rounds are retained.

## Runtime evidence

[DEV trace summaries](provenance/dev_profile_summary.json) preserve observed CUDA graph launches and W4 Marlin CUDA kernel names at C=1/16. Empty/non-worker traces remain visible. Capture logs alone are not replay proof; the trace covers the sampled DEV steps, not every operation of every timed request. Kernel counts are observations, not isolated kernel speedups. Profiling startup warnings and full trace files remain private. No attribution of end-to-end differences to one kernel is inferred from these counts.

## Official quality tasks

Quality uses the same artifacts and vLLM version in a separate environment and separate runs. The separate environment has NumPy 2.5.3, while serving uses 2.3.5; an optional cuteDSL low-latency BF16 GEMM import is unavailable during quality DEV. Both quality arms share that environment, and quality execution time is not used as a serving measurement. [Environment notes](provenance/quality_environment_notes.json) retain the warning. The pinned [lm-evaluation-harness commit](https://github.com/EleutherAI/lm-evaluation-harness/tree/d6de81643928d653435c431bae19945d41d32520) owns task construction, multi-token likelihood, generation extraction and official aggregation. [Input freeze](configs/quality_input_freeze.json) records document/fewshot/request hashes before quality outcomes. Each request is compared to the frozen task/doc/index/arguments identity before model execution.

- ARC-Challenge: official `arc_challenge`, 0-shot, all 1,172 test items; `acc` and character-length-normalized `acc_norm` remain distinct.
- WikiText-2: official `wikitext`, `EleutherAI/wikitext_document_level`, `wikitext-2-raw-v1`, all 62 test documents. No chat template. `loglikelihood_rolling` uses disjoint prediction windows, maximum length minus two, context length one. Word counts are whitespace splits of the original document before detokenization; byte counts are original UTF-8 bytes. Word/byte perplexity is `exp(-sum(loglikelihood)/sum(denominator))`.
- GSM8K: official generative task, 5-shot worked-answer examples, all 1,319 test items, greedy, maximum 1,024 generated tokens. Official strict-match and flexible-extract filters are reported separately; truncation and finish reasons remain visible.
- MMLU: official group, 5-shot, 57 subjects and 14,042 test items. Subject metrics and official group aggregation remain available.

The pinned instruction template is applied once to ARC/GSM/MMLU; fewshot examples form one official context block (`fewshot_as_multiturn=False`). Seeds are909222. Maximum context8192, chunked prefill enabled, maximum sequences16 and the same 4 GiB KV budget bound quality execution. These quality settings are not performance cells. API/parser DEV uses two validation documents from ARC and WikiText per arm; full test attempts are bounded to one per task and model after DEV. No task subset is presented as a full split.

[The analysis plan](configs/quality_analysis_plan.json) was fixed before quality results. Paired item resampling stays within task/filter; MMLU's aggregate interval resamples paired subjects and weights by their sampled item counts. WikiText resamples paired documents and recomputes denominators. Intervals are pointwise, descriptive and based on 5,000 replicates (seed909223). Benchmarks and GSM filters are never pooled into an extra sample count or a single headline score.

Correctness flips count correct→wrong plus wrong→correct. All-answer disagreement also includes changed wrong answers. Choice-score NLL/Brier/KL normalize full-continuation option likelihoods; they are diagnostics, not full-vocabulary next-token metrics. Full-vocabulary KL is **NOT_RUN** because complete vectors were not collected. [Definitions and related work](../../docs/related-work/README.md).

## Data-v2 and independent CPU checks

[The new generator](scripts/data_v2.py) separates label permutation, program facts and ordering RNGs. A bounded integer-loop oracle is cross-checked against a closed form. Distractors occur below and above the answer; duplicate choices are rejected. A count→label baseline fitted on calibration is evaluated on held-out inputs, alongside constant/index/min/max rules. Because the answer is interior by design, this set is not cue-free. These are CPU input audits, **not model evaluations**, and they do not replace standard tasks or alter earlier generators.

Public CPU analyses reconstruct serving summaries and official-task scalar scores; excluded profiler traces, full generated rationales and model tensors require private evidence. The installable [paired CLI](../../packages/diova-compare/README.md) provides a separate comparison path. No claims of third-party GPU replication or deployment equivalence follow from CPU agreement.
