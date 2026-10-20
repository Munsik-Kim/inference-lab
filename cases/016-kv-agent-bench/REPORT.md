# Case016 — NVFP4 KV Cache on RTX5080: Storage, Correctness, and Serving Performance

Built tools integrating vLLM0.32.0 and FlashInfer native compressed KV to verify storage/read correctness, fixed-pool capacity and concurrent-request costs. Qwen3-4B weights remain BF16.

At the same approximately4GiB pool, B16 has29,120 slots and NVFP4 has103,552, a token-capacity ratio of 3.5560×. Serving gains were observed in every registered16K/C2 and C4 paired round.

1. [Background](#background)
2. [Questions](#questions)
3. [Technical principles](#principles)
4. [Method](#method)
5. [Experiment](#experiment)
6. [Results](#results)
7. [Analysis](#analysis)
8. [Conclusion](#conclusion)
9. [Reproduction and sources](#sources)

<a id="background"></a>

## 1. Background

The original long-context agent exploration had a weak baseline. Preserving those failures, this follow-up focuses on native compressed-KV storage/read, capacity and serving costs. It adds no agent task, training or weight quantization.

<a id="questions"></a>

## 2. Questions

TECH_PATH_VALID checks native packed reads/writes on SM120 across36 layers. TECH_NUMERIC_VALID checks two dequant interpretations and finite actual attention outputs. CAPACITY_VALID checks actual bytes and slots at the same pool. SERVING_MEASURED requires the fixed request set across all conditions. Faster performance is not required for completion.

<a id="principles"></a>

## 3. Technical principles

B16 uses2×36×8×128×2=147456B/token. F4 uses36864 packed-data bytes plus4608 FP8 block-scale bytes, totaling41472B/token. The arithmetic ratio3.5556 differs slightly from page-aligned slot capacity. Shared-storage views are counted once. F4 uses packed uint8 values, FP8 E4M3 block scales and fixed global scale1. Short profiles show the NVFP4 writer and FlashInfer FA2 reading compressed data, rather than BF16 storage fallback.

<a id="method"></a>

## 4. Method

Model revision is`cdbee75f17c01a7cc42f958dc650907174af0554`, with36 layers,Q32/KV8 and head_dim128. Both use a4GiB target pool, context28672,max_num_seqs4,chunked prefill2048,prefix caching OFF,async scheduling OFF and BF16 weights. Both CUDA-graph support gates passed; actual FULL_AND_PIECEWISE compilation is common. Both B16 and F4 use XQA decode; F4 prefill uses NVFP4 FA2. XQA receives packed NVFP4 data and block scales. Neither`nvfp4_fa2=True`nor`impl.use_xqa=None`implies XQA decode is unused.

<a id="experiment"></a>

## 5. Experiment

Exactly8192/16384 input IDs include the full Qwen template, roles and assistant prefix, sent to the completions endpoint. Only synthetic filler was adjusted before registration; facts/questions were never truncated. The same8 prompts/length are used at C1/2/4 for three rounds. KV order is B16→F4 / F4→B16 / B16→F4. Throughput uses ignore_eos=True and128 output tokens, separate from natural-end sanity. The180-second request timeout includes client-slot waiting.

Makespan is launch→last client-response terminal for the closed8-request set. The timestamp includes raw SSE parsing after HTTP receipt; exact HTTPclose is not separately timestamped. req/s=successful requests/makespan; output tokens/s=observed output tokens/makespan. Client TTFT is dispatch→first nonempty token-ID event. Mean client decode is(last−first token-bearing event)/(n−1). SSE chunk gaps are not individual-token ITL. Engine queue/decode comes from dedicated-server histogram sum/count interval differences. Device memory includes background use and is separate from Torch allocated/reserved.

<a id="results"></a>

## 6. Results

| KV | bytes/token | actual pool bytes | blocks ×16 | token slots |
|---|---|---|---|---|
| B16 | 147456 | 4293918720 | 1820 | 29120 |
| F4 | 41472 | 4294508544 | 6472 | 103552 |

| Input | C | B16 req/s | F4 req/s | F4/B16 median [3-round range] | success B16 / F4 |
|---|---|---|---|---|---|
| 8K | 1 | 0.396 | 0.397 | 1.001 [1.000, 1.003] | 24/24 / 24/24 |
| 8K | 2 | 0.563 | 0.561 | 0.996 [0.992, 1.000] | 24/24 / 24/24 |
| 8K | 4 | 0.630 | 0.712 | 1.125 [1.125, 1.131] | 24/24 / 24/24 |
| 16K | 1 | 0.251 | 0.247 | 0.986 [0.985, 0.991] | 24/24 / 24/24 |
| 16K | 2 | 0.251 | 0.303 | 1.209 [1.205, 1.211] | 24/24 / 24/24 |
| 16K | 4 | 0.251 | 0.340 | 1.351 [1.350, 1.357] | 24/24 / 24/24 |

| Input | C | B16/F4 median TTFT s | B16/F4 decode ms/token | B16/F4 queue s | B16/F4 preemptions | B16/F4 device peak MiB |
|---|---|---|---|---|---|---|
| 8K | 1 | 0.844/0.923 | 13.182/12.534 | 0.000/0.000 | [0.0, 0.0, 0.0] / [0.0, 0.0, 0.0] | 15102/15103 |
| 8K | 2 | 0.873/0.973 | 21.015/20.326 | 0.083/0.092 | [0.0, 0.0, 0.0] / [0.0, 0.0, 0.0] | 15096/15100 |
| 8K | 4 | 1.895/1.353 | 28.072/33.124 | 1.402/0.629 | [0.0, 0.0, 0.0] / [0.0, 0.0, 0.0] | 15101/15102 |
| 16K | 1 | 2.121/2.378 | 14.646/13.093 | 0.000/0.000 | [0.0, 0.0, 0.0] / [0.0, 0.0, 0.0] | 15102/15101 |
| 16K | 2 | 6.100/2.467 | 14.735/32.372 | 3.337/0.273 | [0.0, 0.0, 0.0] / [0.0, 0.0, 0.0] | 15097/15104 |
| 16K | 4 | 13.987/3.020 | 14.662/67.868 | 8.804/1.718 | [0.0, 0.0, 0.0] / [0.0, 0.0, 0.0] | 15102/15103 |

| Input | B16 copy exact /4 | F4 copy exact /4 | equal token sequences /8 |
|---|---|---|---|
| 8K | 4 | 4 | 4 |
| 16K | 4 | 4 | 4 |

| Phase | model request attempts | responses received | observed output tokens |
|---|---|---|---|
| p0_numeric | 2 | 2 | 64 |
| p0_graph | 8 | 8 | 128 |
| warmup | 12 | 12 | 1536 |
| sanity | 32 | 32 | 921 |
| timed | 288 | 288 | 36864 |

| Owned server | Mode | Load/compile/warmup ready seconds | Total attempt wall seconds |
|---|---|---|---|
| p0_B16 | EAGER | 23.310 | 29.545 |
| p0_F4 | EAGER | 41.275 | 48.242 |
| graph_B16 | GRAPH | 36.330 | 37.096 |
| graph_F4 | GRAPH | 35.291 | 36.040 |
| R1_B16 | GRAPH | 18.284 | 198.163 |
| R1_F4 | GRAPH | 17.284 | 184.655 |
| R2_F4 | GRAPH | 17.284 | 152.574 |
| R2_B16 | GRAPH | 16.290 | 166.072 |
| R3_B16 | GRAPH | 17.285 | 167.496 |
| R3_F4 | GRAPH | 17.292 | 152.470 |


The small NHD/HND checks found zero maximum difference between two dequant interpretations of the same compressed cache. Relative norm errors versus original BF16 were 0.095461, 0.095341. These are quantization errors, not decoder disagreements. Actual attention outputs were finite across36 layers for both conditions in the short512-token P0 eager probe with32 generated tokens. Independent full-model-logit finiteness and full KL/PPL were not measured.

<a id="analysis"></a>

## 7. Analysis

F4 changes free slot capacity, normal attention-kernel selection, queueing and time within the same pool. These are observed configuration differences, not an isolated arithmetic cause. All three round ratios/ranges are shown; no request-latency bootstrap fabricates a narrow makespan interval. Repeated8 prompts are not24 independent tasks.

Instrumentation note: the reused CacheInspector`cudagraph_policy` is an old constant eager-description string. Raw fields remain unchanged; graph status uses actual`compilation_mode`, startup capture logs and frozen launch settings. Parameter signatures sample the first/last8 values of each named parameter, not full GPU tensors. File-level model-shard hashes were rechecked against the original snapshot.

<a id="conclusion"></a>

## 8. Conclusion

Serving gains were observed in every registered16K/C2 and C4 paired round.

The original agent study retains B16 0/48,F4 6/48 across the same12 DEV tasks repeated by menu/cap. NOT_MET baseline gate, failures,amendments01–08,source hashes and zero CPU-audit score changes are preserved. Three successful F4 tasks do not establish general workflow superiority.

| Scope | Status |
|---|---|
| Native NVFP4 KV storage/read ·36 layers | VERIFIED |
| Fixed4GiB capacity ·36 timed blocks/288requests | MEASURED |
| Copy canary/open-text sanity | LIMITED ·16 paired prompts |
| Full KL/PPL · agent-quality improvement ·100K context | NOT_EVALUATED |
| Agent P2/P3 · weight W4 · training · prefix reuse | NOT_RUN |
| GitHub Pages | BUILDER_READY · MANUAL_DEPLOY_NOT_RUN |


<a id="sources"></a>

## 9. Reproduction and sources

[CPU / GPU commands](REPRODUCE.md) · [Stage map](STAGE_MAP.md) · [Raw rows](followups/nvfp4-technical-r1/results/) · [Protocol](followups/nvfp4-technical-r1/specs/protocol.json) · [Execution source freeze](followups/nvfp4-technical-r1/provenance/execution_freeze.json)

Execution source SHA256: `2c36f2cb0f0394f4f9661733344226d05c099f262e248dd4f94ec135e4fc4aec`

Git base,execution source hash,model revision and page-build revision identify different objects. Of449 original files,447 are publicly retained byte-identically or as lossless gzip. Two packaging helpers with personal scratch paths are omitted, retaining original hashes. Existing files in the user’s original worktree are unchanged.

[vLLM v0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0) · [Pinned FlashInfer backend](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/attention/backends/flashinfer.py) · [Completion protocol](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/entrypoints/openai/completion/protocol.py) · [NOTICE](NOTICE.md)

Upstream vLLM/FlashInfer provide the NVFP4 kernels. Contributions are integration,path/capacity inspection,fixed-workload measurement,CPU auditing and documentation. Codex assisted implementation and separate calculation review; this is not human peer review or external GPU reproduction.

Cumulative GPU-worker wall: 1183.314s /7200s. Includes load,small checks,graph gates,warmup,sanity,metrics and I/O; it is not GPU-busy time or serving speedup. No installs,model downloads or weight-artifact builds were performed.
