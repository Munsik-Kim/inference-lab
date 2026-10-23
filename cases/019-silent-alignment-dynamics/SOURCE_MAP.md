# Source and statement map

Simon Martin, Francis Bach, Giulio Biroli, [Silent Alignment in Two-Layer Linear Networks: A High-Dimensional Perspective, arXiv:2610.29864v1](https://arxiv.org/abs/2610.29864v1). Original reading records remain [P0](PAPER_SOURCE_MAP.md), [P1](P1_SOURCE_MAP.md), [P2](P2_SOURCE_MAP.md). No author code or full PDF/HTML is redistributed. Our figures show our saved finite-dimensional calculations.

| Statement | Evidence/definition | Scope |
|---|---|---|
| Loss can stay near its initial value while subspaces align | P0/P1 original ODE, structured initialization; publication/facts.json timing | Partial finite-dimensional numerical reproduction; larger rho reverses event ordering |
| Boundary slow direction and R2 tail | P1 mean/worst; P2 graph + selected RK4 | Own boundary-pair and posthoc R2 checks, not a new theorem |
| Direction intervention alters paths at fixed prediction size | P2 actual transformed checkpoints/128 continuations | Own oracle/ambient intervention; starting loss/gradient can change |
| Immediate and subsequent effects differ | Per-parent J/G/A in original fixed_time_paired CSV | Common original learnable-loss denominator; not a speedup percentage |
| Public replay | Closed manifest, explicit lossless evidence and phase source loaders | Stored-data CPU reconstruction; unsaved factor paths not restored |

The single current fact source is publication/facts.json, with original fields and denominators. Original phase documents retain their old NOT_RUN and failure histories. The current README/REPORT are new public prose; former entry documents are exactly recoverable at case/README*,case/P*_REPORT* in the evidence mapping. They were not edited in the research worktree. Base Git, uncommitted scientific bundles, current public bundle and presentation identity are distinct.
