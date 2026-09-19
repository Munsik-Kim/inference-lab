# Case 008 — Build, Reconstruct, Reload

English | [한국어](README.ko.md)

Built a GPTQ W4A16 checkpoint for Qwen3-4B-Instruct-2507 and standalone layer-13 pruned/reconstructed checkpoints for Qwen3-0.6B. Both tracks reload in fresh processes from a different directory. The W4 safetensors occupy 2,651,839,568 bytes versus 8,044,982,000 bytes for BF16. R evaluates whether fitting only the smaller down projection recovers native BF16 output error on 192 held-out scenarios. The tracks use different models and runtimes; their quality and cost are reported separately.

[Recorded item explorer](demo/index.html) · [CPU reanalysis](REPRODUCTION.md) · [Methods](METHODS.md) · [Detailed results](ANALYSIS.md)

## Q: packed weights through an actual runtime

RTX 5080; 252 target Linear modules; GPTQ W4A16, symmetric group 128; native BF16 embeddings/norm/output head. A separate pinned build environment produces a standard compressed-tensors checkpoint. The unchanged vLLM runtime executes Marlin kernels. Both reload processes have identical smoke scores and greedy tokens. Weight bytes, allocator memory and request time have distinct scopes.

| Arm / 192 scenarios | Correct | Δ full gold NLL (nats; 95% CI) |
| --- | ---: | --- |
| Q-BF16 | 137/192 | +0.000000 [+0.000000, +0.000000] |
| Q-W4 | 137/192 | -0.591295 [-0.723442, -0.470060] |

Positive NLL change means less probability on the independently computed correct token. Conditional four-choice NLL instead changed by +0.379465 [+0.248205, +0.497079] nats (positive worse). Mean allowed-label mass was 0.666963 for BF16 and 0.673368 for W4. Thus the full-vocabulary and conditional choice scores moved in opposite directions; equal accuracy does not summarize this difference. These are short synthetic retrieval, comparison and integer-program prompts, not a general language-quality benchmark. Exact transitions and baseline task strata are in [analysis](ANALYSIS.md).

![Q paired gold-score change](figures/q_score_boundaries.png)

## R: same smaller matrices, fitted down projection

The fixed I25/P25/S50 deletion sets come from Case 007. Actual gate/up rows and down columns are removed. The common DEV-selected ridge strength, η=0.01, fits only the remaining down weights; inference stays BF16. Repaired and uncorrected pairs have identical shapes and parameter counts, with no extra inference matrix multiplication.

| Fixed structure | Pooled recovery (95% CI) | Improved prompts |
| --- | ---: | ---: |
| I25 | 95.40% [95.25, 95.55] | 192/192 |
| P25 | 94.96% [94.79, 95.13] | 192/192 |
| S50 | 94.09% [93.90, 94.28] | 192/192 |

Recovery is the fraction of deletion **squared output-error energy** removed, pooled over the same 32 fixed positions per prompt. It is not an accuracy increase. Each interval resamples the 192 prompts, not their 6,144 dependent positions. R-B is the native model; the six other arms are three structures before/after repair.

![R pooled error energy recovery](figures/r_recovery.png)

| Arm / 192 scenarios | Correct | Δ full gold NLL (nats; 95% CI) |
| --- | ---: | --- |
| I25 | 128/192 | -0.583823 [-0.660297, -0.505045] |
| I25-R | 124/192 | -0.058966 [-0.082720, -0.034713] |
| P25 | 130/192 | +0.774289 [+0.727695, +0.820419] |
| P25-R | 123/192 | -0.134708 [-0.155610, -0.113348] |
| R-B | 124/192 | +0.000000 [+0.000000, +0.000000] |
| S50 | 107/192 | +2.267479 [+2.169380, +2.371095] |
| S50-R | 122/192 | -0.139956 [-0.173521, -0.106027] |

Compared with its uncorrected version, repair changed mean full gold-token NLL by +0.524857 nats for I25 (worse), −0.908996 for P25 and −2.407435 for S50 (better). Correct-answer counts changed 128→124, 130→123 and 107→122, respectively. Both model baselines scored 16/64 on code, limiting task-utility conclusions there. Teacher-output fidelity and gold-based quality remain separate. Deployment is **NOT_ASSESSED**; there is no equivalence or universal quality gate.

## Measured request cost

| Pretokenized Q request | BF16 / W4 mean ms | BF16 / W4 speed ratio [95% CI] |
| --- | ---: | --- |
| fixed_8_token_request | 164.711 / 150.969 | 1.0910 [0.8790, 1.2899] |
| one_token_request | 21.638 / 23.329 | 0.9275 [0.8474, 1.0095] |

Q measures blocking local vLLM requests, including sampling/scheduling/readback. Both speed-ratio intervals include 1. The 1 GiB KV-cache allocation is fixed in both arms. Smaller weight files do not establish lower request latency.

| Same-size repair pair | Prefill + argmax ratio [95% CI] | Fixed 8-token ratio [95% CI] |
| --- | ---: | --- |
| I25 | 0.9678 [0.8668, 1.0718] | 1.0295 [0.9727, 1.0905] |
| P25 | 1.0450 [0.8942, 1.2265] | 1.0088 [0.9609, 1.0598] |
| S50 | 0.9313 [0.8329, 1.0374] | 0.9981 [0.9631, 1.0328] |

R ratios are uncorrected / repaired time at the same smaller width; values above 1 favor the repaired arm. Each track uses 12 fixed scenarios, three process rounds and paired blocks. They are separate runtime comparisons. Full per-arm timing and allocator memory scopes are in [analysis](ANALYSIS.md).

## Read and run

The [offline explorer](demo/index.html) uses retained scalar data, with task/ID/arm filters and JSON export. Open it from disk; GitHub's source preview does not execute HTML. [Reproduction](REPRODUCTION.md) separates CPU reanalysis, private artifact inspection, GPU reload and fresh-output timing commands. Model weights/full hidden or vocabulary vectors are excluded from this review bundle.

The shared CLI lives in [tools/modelpack](../../tools/modelpack/). [Source hashes and reuse](provenance/reuse.json), [environment](provenance/environment.json), [validation](validation.json) and [limits](LIMITATIONS.md) describe exactly what was checked. This is a **local review candidate**, not a published model service.

## Attribution

Qwen, PyTorch/Transformers, LLM Compressor, compressed-tensors and vLLM/Marlin supply the upstream models and runtime algorithms. This case implements the artifact/structure loader, fixed-set ridge reconstruction, comparisons and CPU checks. Codex assisted the work. See [NOTICE](NOTICE.md) and [LICENSE](LICENSE).
