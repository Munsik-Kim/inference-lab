# Reuse and implementation boundary

| Component | Origin | P4 handling |
|---|---|---|
| OPTIC format, codebook, affine decode, FP16 GEMM/FP32 output, bias/BF16 cast | VILA-Lab/optic pinned344c712aeb6620e39eb78930fb922447f0e123f3 | Unchanged historical vendor; no kernel or bias/group-lookup optimization |
| CUDA GEMM, unique/sort/index_select, streams, allocator, memory snapshot | Installed PyTorch/CUDA/cuBLAS | Library APIs reused; not authored here |
| Original LARGE/Z32 artifact, int64 ID files, numerical policy,256-row full-decode setup | Original Case020 | Exact file and payload hashes reused, no original file updated |
| Z128/Z644 synthetic fixtures | P4 `base.py`/`prepare.py` | Original random distribution/FP16 scale/bias rule retained, separate deterministic width seeds |
| A dense reconstruction+gather wrapper | Original design, P4 harness | Official decoder full table in fixed chunks; setup/cleanup timed separately |
| B request decode | Official upstream forward | Direct call, no patch or dtype fallback |
| C unique/decode/inverse wrapper | Original design, P4 wrapper | unique computed every call, M-row output, no caching |
| S sort/decode/inverse diagnostic | P4 wrapper | Decode M rows, one duplicate-free core input only; not a dedup improvement or pure unique-cost control |
| H/D/E interleaving and lifecycle clocks; serial worker budget/journal | P4 `worker.py`/`run.py` | New measurement harness, frozen before main GPU comparison |
| FP64 independent affine reference | NumPy formula, own reference code | No Torch/upstream decoder inside reference; CPU auditor independently evaluates sampled outputs |
| Hash/storage/statistics/interval/budget audit, mutation tests, tables, figures and reports | P4 `audit.py`, `figures.py`, tests and documents | CPU-only; later source identified by PUBLIC_MANIFEST_P4 separately from GPU freeze |

No new codec, GEMM kernel, allocator, custom kernel, trained model or compression-quality result is claimed. Upstream source/license/metadata already preserved in the original Case; no duplicate checkout, model, dataset, wheel cache or dense table is included in P4 public files.
