# Reuse and implementation boundaries

| Component | Reused source | This follow-up adds |
|---|---|---|
| NVFP4 writer and attention kernels | vLLM0.32.0 / FlashInfer0.7.0.post1 | Read-only tensor/layout inspection and short kernel-name profile; no kernel changes |
| Qwen architecture/tokenizer/template | Qwen3-4B pinned revision | Exact template-inclusive input IDs; unchanged BF16 weights |
| Server ownership | Original Case016 server_manager pattern; Case009 serving code read as reference | Owned loopback PID groups, common CUDA-graph support gate, separate new scratch |
| Cache inspection/profile | Original Case016 CacheInspector/ProfileInspector, imported read-only | Actual compilation-mode metadata, parameter sampling signatures and temporary36-layer finite-output hooks |
| Small numeric fixture | Original fixed Case016 kernel_check.py | One unchanged rerun; two dequantizers of the same cache, not two whole decoders |
| Completion recording | Official v0.32 completion protocol | Token-ID stream reconstruction, lossless prompt deduplication, attempt/response counters |
| Measurement | Fixed36-block protocol | Closed8-request queue, matched rounds, raw metrics/telemetry, independent stdlib auditor and CSV/SVG |
| Original agent audit | Unchanged baseline-audit ZIP | Public restoration wrapper and rerun of all96 stored episodes |

Old Case009 legacy runner flags,attention backend,context length and event-based TPOT are not copied. The technical benchmark has no tool parser/grammar/agent fixture. GPU model weights and KV tensors are not redistributed. The inherited `cudagraph_policy` text is a stale constant; actual compilation mode and startup logs take precedence, with the raw field preserved.
