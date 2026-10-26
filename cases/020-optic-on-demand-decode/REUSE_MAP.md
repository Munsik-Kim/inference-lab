# What was built and what was reused

| Part | Origin | Public code/evidence |
|---|---|---|
| Compression representation, codebook lookup, layer decoder and bias/BF16 path | VILA-Lab/optic pinned344c712aeb6620e39eb78930fb922447f0e123f3 | [Preserved vendor](publication/originals/pre-publication/vendor/optic/src/ple_eval/compact_adapter.py) |
| GEMM, unique/sort/index_select, CUDA events, allocator | Installed Torch/CUDA/cuBLAS | Versions/policies/import paths remain in raw; no custom kernel or allocator authored |
| Synthetic fixtures, A/C wrappers, independent FP64 reference | Original Case020 harness | [P2 source](publication/originals/pre-publication/src/), [P4 source](publication/originals/pre-publication/followup/p4_speed_cost/src/) |
| S diagnostic, balanced same-process H/D/E, actual lifecycle, stage/stack accounting | Case020 P4 harness | Exact frozen source and raw kept; no caching/fused-kernel modification |
| Historical CPU auditors | Original Case020/P4 | Reused unchanged in full mode on restored/mapped scope; not presented as fully independent new implementations |
| Publication raw calculator and alternative statistics/storage checks | New publication layer | [analyze.py](publication/analyze.py), [tests](tests/test_publication.py) |
| Exact public mapping/manifests, fresh-output CPU CLI, plain bilingual docs and four figures | New publication layer | [verify.py](publication/verify.py), [run_checks.py](publication/run_checks.py), [figures.py](publication/figures.py) |

OpenAI Codex assisted implementation, execution analysis, validation and documentation. The work does not claim authorship of OPTIC's codec, Torch GEMM/allocator, or a new compression algorithm/kernel. No learned-model quality improvement is inferred from synthetic reconstruction gates.
