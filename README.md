# Inference Lab

Reproducible case studies of local LLM execution on an **RTX 5080 16GB**. The project investigates execution compatibility, memory use, latency, and task quality through recorded experiments and inspectable evidence.

**Three completed cases are documented here.** They cover execution compatibility, an official BF16/FP8 comparison on synthetic Korean document extraction, and an independent EFQ-Softmax numerical audit on Qwen attention traces.

| Case | Question | Recorded result | Materials |
|---|---|---|---|
| 001 — Validating vLLM W8A8 INT8 fallback on RTX 5080 | Can an existing kernel-selection guard unblock a real W8A8 model on SM120? | Before: initialization error, exit 1. After: text generation, exit 0; 96 INT8 modules selected the Triton kernel class. | [Case and reproduction instructions](cases/001-sm120-int8-fallback/README.md) · [Download ZIP](downloads/step04_sm120_int8_repro.zip) |
| 002 — Official BF16 vs FP8 Qwen3-4B document extraction | How do memory, latency, and extraction accuracy compare between the official BF16 and FP8 releases? | BF16 31/100 vs FP8 32/100 correct; difference +1 pp, 95% paired interval −2 to +5 pp. FP8 used less memory and had lower median latency; both accuracies were too low for unattended extraction. | [Case, results and reproduction](cases/002-bf16-fp8-document-extraction/README.md) · [Download ZIP](downloads/case002_bf16_fp8_document_extraction.zip) |
| 003 — Independent EFQ-Softmax numerical audit | How robust are the published EFQ settings on real Qwen attention inputs? | EFQ-Mean had 11.49–14.49% median attention-output error. Development calibration reduced it to 6.51–8.09% and passed the predefined same-scale screen; headroom-scale nearest rounding was still more accurate. | [Case, methods and evidence](cases/003-efq-softmax-numerical-audit/README.md) |

The unmodified vLLM 0.29.0 wheel selected a CUTLASS implementation that does not support INT8 on SM120. Applying the Python runtime guard from [hclsys's PR #54316](https://github.com/vllm-project/vllm/pull/54316) let the selector use the existing Triton implementation. The pinned `RedHatAI/Qwen2.5-0.5B-Instruct-quantized.w8a8` model then generated “The capital of France is Paris.” and exited normally, with 8 generated tokens including EOS.

The recorded setup was WSL2 Ubuntu 24.04.4, TP=1, a single-process V1 runner, eager execution, and one short request. Only the PR's runtime hunk was applied to the official wheel; the complete PR checkout was not built or tested. The 96-module observation records `TritonInt8ScaledMMLinearKernel` and INT8 weights on CUDA, not GPU kernel call counts. Performance and task quality were not measured.

The case includes the tested patch, dependency locks, model preparation and run commands, and separate [original](cases/001-sm120-int8-fallback/evidence/original.json) and [replay](cases/001-sm120-int8-fallback/evidence/replay.json) evidence with logs. The replay reused the same PC's existing environments and checkpoint; it is not an independent reproduction or a fresh installation. Model weights are obtained separately. Exact revisions, hashes, and settings are in the case files.

The [posted RTX 5080 validation comment](https://github.com/vllm-project/vllm/pull/54316#issuecomment-5614091211) connects this case to the upstream discussion and [issue #54311](https://github.com/vllm-project/vllm/issues/54311). The repository and download contain the reviewed documentation revision; executable files and execution evidence are unchanged from the earlier ZIP attached to that comment.

Case 002 used 100 fixed evaluation documents and three paired timing rounds on the same RTX 5080. With a shared 1 GiB BF16 KV cache, whole-device sampled peak was about 11.62 GiB for BF16 and 8.21 GiB for FP8. The [case](cases/002-bf16-fp8-document-extraction/README.md) reports output lengths, null-field errors and uncertainty alongside latency. This is a completed comparison, not a recommendation to deploy either setup unchanged. A [short result and discussion](notes/case002-results-and-discussion.md) explains the findings and possible follow-up work.

Case 003 used post-QK-normalization/post-RoPE traces from Qwen3-0.6B on 8 development and 16 held-out synthetic documents. It isolates probability approximation with fixed Q/K/V and FP32 arithmetic; it does not measure model quality or packed-FP4 acceleration. The [case](cases/003-efq-softmax-numerical-audit/README.md) reports published parameters, calibration transfer, error tails and explicit implementation choices.

Future work may address extraction reliability or a limited kernel-feasibility experiment under a separately fixed protocol; those follow-ups have not been run. The [case template](cases/TEMPLATE.md) provides a starting point.

The original guard and its branch tests are hclsys's work. This project's contribution is RTX 5080 execution validation and reproducible evidence. OpenAI Codex assisted with the reproducer and documentation. See the case [NOTICE](cases/001-sm120-int8-fallback/NOTICE.md) for attribution and the [Apache-2.0 license](LICENSE).

Case 002 uses synthetic data and independent fact-derived gold. Codex assisted with its code, local execution and analysis; see its [NOTICE](cases/002-bf16-fp8-document-extraction/NOTICE.md).

EFQ-Softmax and its published parameters are Han et al.'s work. Case 003 contributes independent implementation and numerical validation. Codex assisted with inputs, implementation, analysis and documentation; see its [NOTICE](cases/003-efq-softmax-numerical-audit/NOTICE.md).
