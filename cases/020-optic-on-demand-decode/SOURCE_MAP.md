# Sources and evidence map

| Source | Reused fact / function | Evidence and application |
|---|---|---|
| [OPTIC v1](https://arxiv.org/html/2610.31309v1) | Token codes shared across PLE layers; layer-specific affine decoder | Synthetic operator execution study, not trained quality/full-model reproduction |
| [Pinned OPTIC commit](https://github.com/VILA-Lab/optic/tree/344c712aeb6620e39eb78930fb922447f0e123f3) | load_compact_ple, CompactOptic.forward, _matmul_fp32, compact_resident_bytes | [Exact vendor source](publication/originals/pre-publication/vendor/optic/src/ple_eval/compact_adapter.py); raw unscaled BF16, no install_compact_ple scale |
| Same pinned source | FP16 mm with FP32 out_dtype on CUDA; FP32 CPU path; no forward request unique | B unchanged; own A/C wrappers; S is an order diagnostic |
| [Deployment docs](https://github.com/VILA-Lab/optic/blob/344c712aeb6620e39eb78930fb922447f0e123f3/docs/deployment.md) | Full-model deployment context | Not an operator dependency requirement; no vLLM/model installation |
| [PyTorch CUDA notes](https://docs.pytorch.org/docs/stable/notes/cuda.html) | Async execution, synchronization, numerical flags, allocated/reserved | Historical runtime2.13.0+cu130; stable docs redirected2.14; no upgrade |
| [PyTorch timing notes](https://docs.pytorch.org/docs/stable/benchmark_utils.html) | Timing and warm-up boundaries | H/D/E contracts remain explicit; no claim of using Timer |
| Original Case020 P2/P4 records | Numerical gates, raw timing, tensor inventories and worker walls | [Exact source/input/raw/history](publication/originals/pre-publication/README.md), [mapping](publication/preservation_map.json) |
| Publication calculator | Alternative NumPy statistics, stored-array FP64 slice, unique-storage calculation | [analyze.py](publication/analyze.py), [run_checks.py](publication/run_checks.py) |

Original Git base3e6f2c6b36fb2a3853cc3677bee84904d28567cf is not the uncommitted execution bundle. P2/P4 freezes remain exact under the archive; publication has a new commit and separate PUBLIC_MANIFEST. Historical documents and receipts are retained as history, including NOT_RUN states.
