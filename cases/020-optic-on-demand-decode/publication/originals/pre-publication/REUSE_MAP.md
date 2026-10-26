# Reuse and contribution boundary

| Component | Owner / reuse | Case020 contribution |
|---|---|---|
| OPTIC representation/official forward | VILA-Lab/optic, Apache-2.0, pinned commit | Read, freeze, execute without modification |
| GEMM, unique, gather, allocator, profiler | PyTorch/CUDA and their upstream implementations | Operator invocation/policy selection/accounting; no new kernel, codec or allocator |
| Fixture/metadata and ID streams | Case020 | Small synthetic NPZs, nonzero layer biases, exact duplication and hash manifests |
| A/C wrappers | Case020 using official B decoder | Full reconstruction/gather and unique/decode/inverse with matched outputs |
| NumPy FP64 references | Case020 | Scalar loop, vector check, separate auditor formula over stored FP16 arrays |
| Measurement/audit/figures/docs | Case020, with Codex assistance | Isolated process budget ledger, storage inventories, raw timing, independent CPU regeneration and bilingual explanation |

The root inference-lab Apache-2.0 license applies to new case code. Upstream notices remain separate. Source/protocol/input identity is distinct from Git base; no commit, publication, remote CI or clean-environment validation is implied by this candidate.
