# Two independent build-and-reload tracks

This is a local experiment on an RTX 5080. Q and R have different models and runtimes; their effects are summarized separately. The [build design freeze](configs/build_freeze.json) records the initial settings; track evaluation freezes record implementation and input hashes after development. These are local freezes, not external preregistration.

## Inputs and scoring

Fresh English synthetic scenarios cover six-record retrieval, comparison of two amounts, and bounded integer programs. Gold answers come from the generator. Code uses a bounded loop and an independently checked closed-form sum; model-generated code is never executed. Shared task templates limit generalization. Scenario IDs, latent facts, text, full chat-template token IDs, gold labels, and label token IDs are retained in `inputs/`. Complete prompts are used without filler or truncation. The caps are Q 1,024 and R 512 tokens; actual lengths are much shorter and recorded per input.

The answer interface is exactly one next token. Tokenization of the full prompt plus each label must preserve the prompt prefix and append one distinct token. No gold continuation enters the forward. Exact maximum ties use the first option in A/B/C/D order. Native full-vocabulary outputs are checked before CPU FP64 statistics. Widening native values does not restore lost precision.

The primary model score is the task-balanced mean paired change in **full-vocabulary gold-token NLL**, candidate minus baseline, in nats. Positive is worse. Conditional four-choice NLL, unnormalized four-choice Brier, gold margin, allowed-label mass, full-vocabulary argmax membership, correctness transitions, and exact ties are separate diagnostics. Full KL is KL(baseline || candidate), computed from full vectors retained privately. Public option logits and normalizers support scalar score checks but cannot independently reconstruct that full KL.

There are 192 held-out scenarios per track, 64 per task. No held-out correctness filtering or selector tuning is allowed. Paired bootstrap resamples base scenarios within task, keeping arms paired: seed 808191, 5,000 replicates, pointwise 95% percentile intervals. A zero-event bootstrap interval does not establish zero population risk or equivalence. Deployment utility tolerances were not supplied; deployment remains NOT_ASSESSED.

## Q: GPTQ, standard checkpoint, vLLM reload

Qwen/Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554` uses BF16 activations with one weight-only GPTQ recipe: 4 bits, symmetric groups of 128, block size 128, dampening 0.01, no actorder. Linear layers are targeted except the tied output head; embeddings and normalization remain native. Calibration contains 256 prompts, development 64, smoke 6. The sequential compressor offloads Hessians and blocks to CPU. No training-gradient fine-tuning occurs.

The compressor runs in a separate local environment. The existing vLLM environment is unchanged. Packed tensor/scale shapes and finite positive scales are checked before BUILD_COMPLETE. The standard checkpoint is copied to another absolute directory and loaded by two fresh vLLM processes with local/offline flags. The original build process is not reused. Packed runtime parameter inventories before and after requests, selected quantization classes, and an untimed actual CUDA profiler request distinguish compressed storage, runtime support, and measured cost.

The existing WSL runtime requires its previously used V1 model runner and non-FlashInfer sampler settings. Initialization/instrumentation failures are retained in provenance. Q uses vLLM raw final-token logits, BF16 KV cache fixed at 1 GiB, no prefix caching, no chunked prefill, batch 1, eager execution, and the explicit FLASH_ATTN backend. No whole-model dense reconstruction is accepted as a successful W4 runtime.

## R: fixed pruning followed by ridge reconstruction

Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca` has 28 layers, hidden size 1,024, intermediate width 3,072, SiLU SwiGLU and no MLP biases. Only zero-based layer 13 changes. The 16 contiguous groups each contain 192 channels. Frozen removed sets from Case 007 are I25 `[8,12,13,14]`, P25 `[8,13,14,15]`, and S50 `[7,8,9,11,12,13,14,15]`. Each has an uncorrected and repaired arm; R-B is the original model.

The exact boundary is the input to layer 13's MLP, after post-attention layer normalization. Native BF16 MLP output is the teacher Y. H comes from the **actual smaller** gate/up matrices and SiLU/product calculation. The student H is not assumed identical to a slice of full-width activations. Smoke compares sliced and masked computation (relative norm tolerance 0.01 fixed before model data), exact same-shape local/integrated output, input immutability, and full model outputs.

Calibration has 128 prompts and 32 fixed positions `floor(i*(L-1)/31)`, including the last prompt token: 4,096 vectors, clustered in 128 prompts. Represented H/Y values are promoted to CPU FP64. With row-sample H, G = HᵀH/N and B = YᵀH/N. Solve

`W (G + lambda I) = B + lambda W0`, where `lambda = eta * trace(G)/width`.

A Cholesky factorization and two solves replace an explicit inverse. Residuals must be below 1e-9. Zero trace/nonfinite data block fitting. Eta is selected from `[1e-4,1e-3,1e-2,1e-1,1]` using 48 DEV prompts: mean per-prompt relative squared error, then equal average over three structures. One common eta is selected; exact ties choose larger eta. Every fit is evaluated with its represented **BF16** weights. Only the smaller down projection changes during repair.

Calibration readout reuses privately stored full-shape H, avoiding another transformer pass. The held-out model evaluation loads each standalone checkpoint. Full model, norm, MLP and final vocabulary outputs must be finite. Identical token/input hashes pair arms. Per-prompt local sufficient statistics cover the same 32 positions; tokens are not independent examples.

Primary reconstruction is pooled recovery: `1 - sum(repaired SSE)/sum(uncorrected SSE)`. Positive means lower deletion error energy; it is not task accuracy. Bootstrap recomputes both energy sums each draw. Median/p95 relative local error, worse prompts and model scores remain visible.

## Standalone dense serialization

All needed BF16 weights are saved in safetensors, together with original config/tokenizer bytes, retained indices, per-layer width override, recipe, loader version, tensor hashes and exact file checksums. A meta skeleton gets the small layer before strict assignment. Tied embeddings are restored explicitly; rotary nonpersistent buffers are derived from config. No original checkpoint is loaded during reload. Missing keys, wrong shapes, corrupt hashes and invalid loader versions fail. Repaired and uncorrected artifacts have identical shapes and parameter counts and differ only in the selected down-projection weight.

## Cost boundaries

Twelve held-out IDs (four hash-selected per task), three process rounds, five warmups and five blocks of three calls are fixed. Arms are loaded sequentially. R measures pretokenized prefill plus argmax and fixed eight-token greedy requests. Q measures blocking local vLLM one-token and eight-token requests, including scheduling/sampling/readback; a pure transformer prefill is **not isolated** in that API boundary. Every request starts with an empty cache; EOS is ignored for the fixed-length timing only.

Synchronized wall time is primary. Profiler, full-output hooks, hashing and file writes stay outside timing. Timing resamples process rounds, scenarios within task and blocks, preserving paired arms. Block means are not a user-request p95. Allocator allocated/reserved peaks and resident parameter bytes have different scopes. Whole-device time-series peak and CUDA-event timing are not measured. Process startup includes uncontrolled OS caches and is not a fresh-machine cold start.
