# P1 source correspondence and reuse boundary

P0 [PAPER_SOURCE_MAP.md](PAPER_SOURCE_MAP.md) remains unchanged.
[Martin, Bach and Biroli, arXiv:2610.29864v1](https://arxiv.org/pdf/2610.29864v1)
§2/§3.1/§5.1/§5.2 supply the model and limiting context already checked in P0.
The v1 full PDF text and Theorem 10 were checked again for P1.

Theorem 10(i), printed pp.18–19 (zero-based PDF17–18), is a finite-dimensional
top-k subspace bound, with the small-rho limit taken before the late-time bound
and a strict boundary gap. Its exponential big-O is not an exact finite-rho
curve or a universal crossing time. Part(ii) adds a high-dimensional limit.
Our factor span at full rank m corresponds to the product range for t>0; the
zero product's t=0 SVD is never used as a direction. Our projector normalization
is 1/(2m); the new worst-angle observable tests the slowest direction separately.

The boundary-pair experiment, rho/gap pairing, 8 new R streams, worst-half
crossing, 160-time horizon, compact raw and amplitude-time fits are local
controlled designs, not copied paper experiments. A finite sample on one reused
P0 target basis is kept separate from asymptotic claims or new-data replication.

## Reuse

Read-only P0 dependencies: `math.scaled_rhs` and `physical_rhs`, `stiefel`,
`solvers.dop853`, `solvers.rk4`, `analysis.crossing`, and `io` serialization,
hash/output/environment helpers. P1's bundle lists all 12 original P0 scientific
files and their unchanged bytes, plus its own files. P0's bundle hash refers to
its fixed original list rather than a new recursive glob that would include P1.

New P1 code supplies boundary-pair inputs, compact diagnostics, a vectorized
independent scalar z equation, a cumulative worker budget, analysis, plotting
and replay. Physical RK4 and scaled DOP853 retain different RHS paths. The
32-mode scalar ODE reconstructs aligned Z/loss separately from factor dynamics.
All routes share NumPy and target/rho inputs. Checkpoint replay shares P1 metric
code; all-grid loss/norm/angle/eigenvalue recombination is a separate algebraic
check. Unsaved full factors are not reconstructed. These are same-machine CPU
checks, with no external human review or GPU/model execution.

No author code or other research code was copied. Existing NumPy, SciPy and
Matplotlib and the repository Apache-2.0 license are unchanged. Code was written
with Codex assistance. P2 direction interventions and P3 remain NOT_RUN.
