# Source → implementation → evidence

| Primary source / fixed identity | Relevant contract | P4 application / evidence |
|---|---|---|
| [OPTIC paper v1](https://arxiv.org/html/2610.31309v1) | Shared token codes, layer affine decoders; E4B latent width644 | Synthetic execution choices only; Z644 is not full E4B reproduction |
| [Pinned OPTIC](https://github.com/VILA-Lab/optic/tree/344c712aeb6620e39eb78930fb922447f0e123f3) | Apache-2.0 source, no learned artifact used | Historical `../../vendor/optic/` and original source manifest preserved |
| `src/ple_eval/compact_adapter.py`: `load_compact_ple`, `CompactOptic.forward`, `_matmul_fp32` | Discrete8-bit/K256; ID flatten→codebook lookup→stacked GEMM→FP32 bias→BF16; no request unique | `src/base.py` imports unchanged loader; worker logs actual official import path |
| Same file: `compact_resident_bytes` | Excludes non-persistent buffers | Actual parameter/buffer storage counts include group_offsets; old accounting remains historical |
| Same file: `install_compact_ple` | sqrt(H) deployment scale | Not used; A/B/C/S use raw unscaled output |
| [Pinned deployment documentation](https://github.com/VILA-Lab/optic/blob/344c712aeb6620e39eb78930fb922447f0e123f3/docs/deployment.md) | Full-model deployment context | No vLLM install/model load; deployment resource claims are not operator requirements |
| [Pinned CPU example](https://github.com/VILA-Lab/optic/blob/344c712aeb6620e39eb78930fb922447f0e123f3/examples/decode_optic.py) | Small synthetic decode path | Already executed in original P1; original stdout/CPU checks remain unchanged |
| [PyTorch CUDA notes](https://docs.pytorch.org/docs/stable/notes/cuda.html) | Async execution, synchronization, allocated/reserved, numerical flags | Explicit H/D/E fences, separate peaks and FP32 accumulation policy |
| [PyTorch benchmark utilities](https://docs.pytorch.org/docs/stable/benchmark_utils.html) | Warm-up/timing scope | Context for custom declared measurement intervals, not a claim of using torch.utils.benchmark.Timer |
| Installed Torch2.13 API/docstrings | memory history, snapshots, stream/events, rusage boundary | `_record_memory_history(state,all,max_entries512)` and public `memory_snapshot`; actual C++ allocation stacks preserved |

Stable docs redirected to2.14 on2026-10-26. Installed runtime stayed2.13.0+cu130;2.13 versioned web pages were unavailable, so actual installed signatures/docstrings were checked. No stable-doc-driven runtime upgrade occurred.

Independent FP64 formula: for token t, group g, group coordinate q, `latent[t,4*g+q] = codebook[g,codes[t,g],q]`; for layer l, hidden h, `out[t,l,h] = sum_z latent[t,z]*W[l,h,z] + bias[l,h]`. Stored FP16 values are promoted to FP64. Token/layer/hidden order is preserved. Chunked reference uses128 tokens and never creates `[M,L,H,Z]`.

Initial P4 freeze identifies the executed GPU harness/config/inputs; final P4 manifest separately identifies later CPU aggregation, figures and prose. Git base is not the uncommitted source bundle identity.
