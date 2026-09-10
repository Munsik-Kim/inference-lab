# Methods and reproduction

This is a local comparison of two official deployment artifacts, using public synthetic Korean business records. OpenAI Codex generated and checked the data, wrote the runner and scorer, executed the local workflow, and assisted with analysis/documentation. No human annotation review or independent reproduction is claimed.

## Data and gold

The dataset contains 20 development documents (10 short, 10 long) and 100 evaluation documents (50 short, 50 long). Scenario IDs, document IDs and document text do not overlap between splits. Both splits share the same synthetic generation grammar; they are not a representative sample of Korean business work.

`scripts/data.py` creates structured initial facts and versioned events before rendering text. Confirmed events update only their stated fields; proposals and withdrawn changes do not update the state. The target is always an existing, active request. Other requests, including cancelled ones, are distractors. Missing fields are explicit or absent values represented by `null`; zero is a real amount. No model output or LLM judge determines gold.

The public files are separate:

- `data/dev_inputs.jsonl` and `data/eval_inputs.jsonl`: document, target ID, case metadata and actual token counts.
- `data/dev_gold.jsonl` and `data/eval_gold.jsonl`: gold values and verbatim supporting lines.
- `data/facts.jsonl`: initial structured facts, event statuses and versions.
- `data/manifest.json`: seed, split counts, lengths and hashes.

The request builder uses only document text, target ID, extraction instructions and the four-field output format. It does not serialize split/bucket/scenario metadata, gold or evidence into model messages. `verify.py` independently parses the rendered target lines, applies confirmed versions, compares all 480 gold fields and their quotes, checks split uniqueness, and tests this input allowlist with canaries. All 120 requests produce identical token IDs under the two pinned fast tokenizers.

The short target is about 512 input tokens and the long target about 4096, including the chat template and generation prompt. All final inputs are within ±10%. Unique distractor records add length; target evidence is not truncated. The documents use recurring synthetic layouts and a finite vocabulary, which limits generalization.

One shared prompt revision was made using development data: the target ID moved after the document, and the instructions more explicitly prohibited filling fields from another request. Documents, facts, events and gold were preserved. Only input-token metadata changed. Both models scored 7/20 documents under each development prompt. This low score was recorded, not used to reject either model or search for favorable evaluation results.

## Strict scoring

The raw response must parse as a JSON object with exactly `owner`, `task`, `due_date`, and `amount_krw`. No code-fence stripping, JSON repair, fuzzy matching, semantic judge, automatic retry or response-constrained decoding is used.

- Duplicate, missing and extra keys have separate error classifications.
- Owner and task must be strings or null. Dates must be null or a valid `YYYY-MM-DD` date. Amounts must be integers or null; booleans and floating-point numbers are rejected.
- Only Unicode NFC and leading/trailing whitespace removal are allowed for strings. JSON whitespace and key order do not matter.
- Empty strings, the string `"null"`, and zero do not count as missing values.
- A schema error makes all four primary fields wrong. An execution/transport error also makes the document and its primary fields wrong. Failures remain in the denominator.

JSON validity, schema validity, individual field accuracy and four-field document accuracy are reported separately. Null and non-null gold subsets have their own numerators and denominators. The scorer has 19 fixed checks including wrong values, null/zero confusion, malformed dates, missing/extra/duplicate keys, truncated JSON and Unicode normalization.

## Execution and timing

The fixed settings are in `configs/common.json` and the immutable `configs/experiment_spec.json`. The actual environment, wheel origin, model revisions and startup adjustments are in `provenance/`.

Both models run one at a time on one GPU, TP=1, with a normal localhost API server and a separate V1 EngineCore process. Concurrency and `max_num_seqs` are 1. Maximum model length is 5120; maximum generation is 256 tokens. The engine dtype and unquantized KV cache dtype are bfloat16. Temperature is 0, the seed is fixed, and both models share sampling and stop-token settings. CPU offload, prefix caching, chunked prefill, asynchronous scheduling, speculative decoding and JSON constrained decoding are disabled. FlashAttention 2 and the same default `FULL_AND_PIECEWISE` CUDA graph configuration are retained; this is not an eager run. FlashInfer sampling is disabled after its development JIT link failure. The initial V2 runner failed on WSL UVA initialization; it is not part of the measured comparison.

Each server receives one short and one long development warmup request before measurement. The three evaluation rounds use BF16→FP8, FP8→BF16, and BF16→FP8. Within each round, the same seeded document permutation is used for both models. There are 600 measured requests but only 100 unique evaluation documents. Round 1 is the primary quality result; rounds 2–3 describe timing variation and repeated-output consistency. No best-round selection is performed.

`run.py` makes direct sequential HTTP/SSE calls to `127.0.0.1`. It saves the raw output, each nonempty content chunk and its arrival time, usage token counts, finish reason and any error. TTFT starts before sending the request and ends at the first nonempty generated content chunk; role-only and empty chunks are excluded. Total latency ends at SSE `[DONE]`. These are client timings, including local HTTP overhead. Output token counts come from engine usage and include its completion accounting, not a count of SSE chunks.

The optional post-first-token quantity is `(total_latency - TTFT) / (output_tokens - 1)`. It includes final transport/completion overhead and can group multiple tokens in a chunk; it is not a kernel profiler or an exact token-by-token decode measurement. A shorter response can reduce total latency without improving token processing, so output lengths and matched-document timing differences are reported together.

## Memory and uncertainty

Both engines receive an explicit 1 GiB KV cache budget. In installed vLLM 0.29.0, `kv_cache_memory_bytes` overrides automatic cache sizing from `gpu_memory_utilization`. Both engines report capacity for 7,280 tokens, enough for the configured 5,120-token sequence. The model-loading memory reported by vLLM is kept separate from cache allocation, model file bytes, and whole-device occupancy.

NVML samples whole-GPU used/free memory and utilization at a nominal 100 ms interval. Phases include idle before server start, startup, server-ready (`loaded`), development warmup, measured requests, and shutdown. The ready phase includes completed engine initialization, compilation and cache allocation; it does not isolate weights. The reported maximum is a sampled peak, not a guaranteed instantaneous peak. WSL did not provide usable process memory attribution, so Windows/display/background use remains included. No torch allocator allocated/reserved values are invented. Observed idle values and per-run variation are retained.

Server startup-to-health and engine loading/compilation messages are separate from warmed request latency. Compilation caches were reused after development. These are not fresh-machine cold-start benchmarks. Shutdown warnings, including worker/semaphore messages, remain in the logs.

Accuracy uncertainty uses 10,000 paired bootstrap samples of document IDs, separately within short and long strata, preserving their 50/50 weights. The reported difference is FP8 minus BF16 in percentage points. An interval containing zero is not evidence of equivalence, non-inferiority or preserved quality. A degenerate interval only reflects the observed synthetic sample. Latency medians/p95 are descriptive; pooled requests are repeated observations of the same documents, not independent quality samples.

## Commands

Run from this case directory. Set `CASE_ENV`, `MODEL_CACHE`, `RUNS`, and `JIT_CACHE` to your isolated Python environment and local storage outside the tracked case. Do not overwrite existing run directories; each run ID must be new.

```bash
PYTHON="$CASE_ENV/bin/python"
"$PYTHON" scripts/prepare_models.py --cache-dir "$MODEL_CACHE" --output "$RUNS/model_paths.json" --verify-weights
"$PYTHON" scripts/verify.py --model-paths "$RUNS/model_paths.json"
"$PYTHON" scripts/score.py
"$PYTHON" scripts/run.py --model-paths "$RUNS/model_paths.json" --work "$RUNS" --cache "$JIT_CACHE" --config configs/common.json --stage dev --run-id my-dev
"$PYTHON" scripts/run.py --model-paths "$RUNS/model_paths.json" --work "$RUNS" --cache "$JIT_CACHE" --config configs/common.json --stage eval --run-id my-evaluation
```

The supplied experiment spec is already frozen; do not run `freeze.py` over it. Evaluation validates all frozen source/data/provenance hashes before starting. To reproduce dataset construction separately without touching the frozen files:

```bash
"$PYTHON" scripts/data.py --tokenizer "$BF16_SNAPSHOT" --root "$RUNS/regenerated-data"
```

The original local run used the same `run.py` command with run ID `case002-v1-20260910`. Original private paths were replaced by stable labels only in public evidence copies. Raw output and numeric measurements remain unchanged. `results/export_manifest.json` records both original and public-copy hashes.

To collect a new run's evidence and analyze it, use a separate copy of the case so the supplied evidence is preserved:

```bash
"$PYTHON" scripts/export_evidence.py --root "$NEW_CASE_COPY" --work "$RUNS" --cache "$JIT_CACHE" --model-paths "$RUNS/model_paths.json"
"$PYTHON" scripts/analyze.py --root "$NEW_CASE_COPY"
```

Ensure the analysis copy contains exactly one complete six-server evaluation schedule. The analysis deliberately refuses an incomplete or duplicated schedule. Public results contain no model weights or environment/cache copies.
