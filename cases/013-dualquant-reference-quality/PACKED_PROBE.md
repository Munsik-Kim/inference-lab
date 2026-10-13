# Packed W4A16 probe / 커널 실행과 수치 대응

SM120에서 실제 Marlin CUDA kernel 실행을 확인했습니다. 그러나 전체 모델의 수치 대응 기준은 미통과입니다. 같은4개 greedy 출력이 나왔다고 그 기준을 통과했다고 바꾸지 않습니다. 이 prototype의 시간·메모리는 serving 이득의 근거로 승격하지 않습니다.

## Actual implementation

Existing vLLM0.29.0 `_custom_ops.gptq_marlin_repack` and `_custom_ops.marlin_gemm`; scalar type `uint4b8`. The [inspected final probe source](provenance/source_snapshot/probe_public_marlin.py) preserves the executed source bytes/hash, while [source freeze](records/source_freeze_final.json) identifies the final code. It is a historical inspection entrypoint: its original module-level ACTIVE_RUN lookup is not advertised as a portable verified rerun CLI.

For codes[O,I], signed c becomes unsigned c+8, transpose to[I,O], pack eight nibbles per int32 to[I/8,O], then `gptq_marlin_repack(packed, empty_g_idx, I, O, 4)`. Combine a·γ in FP32, transpose to[G,O], cast BF16 and apply `marlin_permute_scales(..., group_size=64)`. Beta remains FP32. Input z=BF16(FP32(X)·beta), flattened to[M,I].

The GEMM arguments are `z, None, repacked, None, scales, None, None, empty, empty, empty, workspace, uint4b8, M, O, I, True, False, True, False`; exact positional meanings depend on the installedvLLM0.29.0 API and are preserved in source. A BF16 bias is added when present. This is packedINT4 weights with fused unpack and **BF16 tensor-core MMA**, not nativeINT4 MMA, packedW4A4 or the authors' Nunchaku path.

The projection fixture uses the0.6B layer0 q_proj [2048,1024], BF16 input[16,1024]. A first block and32-token full logits use the same0.6B checkpoint and frozen first32WikiText tokens. Model revision and input-array hashes are in [experiment manifest](records/experiment_manifest.json). The final dense comparison preserves input rescale and disables BF16 reduced-precision reduction, matching the main quality runner; TF32 is disabled. Model/kernel tensors are not public payloads.

The final profiler includes `marlin::Marlin<...>`; the full CUDA symbol, memory, repeat times and errors are in [probe03 receipt](records/diagnostics/packed_precision_aligned_03.json). Numeric error is `sqrt(sum((candidate.float−reference.float)^2)/sum(reference.float^2))`. GEMM outputs, block outputs and logits have different dimensions/denominators; their errors are not a causal layer-by-layer curve.

## Sequence and limits

| Probe | Reference/control | Full-output relative error | Process exit | Semantic status |
|---|---|---:|---:|---|
|01 | Initial D16 dense; original probe reduction setting |0.0255330186 |0 |BLOCKED |
|02 | BF16 combined-scale matched dense diagnostic |0.0225979742 |0 |BLOCKED |
|03 | Primary dense, reduction flag alignedFalse |0.0260543320 |1 |BLOCKED |

Probe01 throws before writing the final comparison field; its0.025533 value is retained in the exception/log and the subsequent primary-comparison receipt. It is not an independently complete short-decode record. Probe02's matched-dense versus primary-dense difference was0.025813; its smaller comparison error did not validate the primary route. The first two handlers wrote semanticBLOCKED with exit0; the final CLI is fail-fast exit1. [01](records/packed_probe.json) · [02](records/diagnostics/packed_format_matched_02.json) · [03 command/exit](records/attempts/packed03_command.json)

Final GEMM relativeRMSE0.0030783622 and one-block0.0056733498 pass their2% gates. Full32-token model error0.0260543320 exceeds the predeclared0.02 limit. Greedy4-token suffix matches; forward argmax differs at1/32 positions (0.03125). These metrics measure different things. The threshold was not expanded after observing outputs.

Three32input/4output repeats, Torch allocated/reserved and kernel names remain in the raw receipts. There is no accepted quality-equivalent serving latency/capacity result. The authors' official packed path remains unverified; stockGPTQ serving is NOT_RUN in this study.

## Next cause separation — NOT_RUN

A: original D16 dense. B: dense reconstruction of the actual packed codes/scales/beta. C: Marlin on that exact packed artifact. A↔B primarily diagnoses format conversion; B↔C execution. Match dtype, reduction settings and input rescale in both comparisons. A complete controlled A/B/C block trace has not been performed; current probes do not isolate a unique cause. [Roadmap](ROADMAP.md).
