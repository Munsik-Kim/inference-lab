# Implementation and reuse

The independent scientific code implements the gradient flow, structured initialization, rank-aware subspace metrics, scalar and matrix checks, paired controls, checkpoint interventions and J/G/A. NumPy performs linear algebra, SciPy DOP853 performs adaptive integration, and Matplotlib renders plots. The paper supplies mathematical context, not copied implementation.

Historical phase files remain unchanged. P3 adds publication/verify.py (bounded lossless restoration), phase_check.py/recorded.py (phase-pinned recorded consumer), portable.py (predeclared derived numeric comparisons), run_checks.py (isolated orchestration), make_projection.py (actual saved-grid display), docs, manifests and DIOVA renderer/assets. These wrappers do not change the solver, seeds, thresholds or original endpoints. Checkpoint metric code and serialization are shared with the frozen phases; raw loss/norm and reference/graph calculations supply additional paths. This is not external human review.

P0/P1/P2 tests are loaded from separate copied frozen file lists to avoid cross-version imports; they run small original fixtures only. Original P2 graph is limited to U0=0 paths. GAUGE/CONTINUE full Z was compared during P2, but only error series and selected factors remain; public replay checks that retained scope. Former P0 REUSE_MAP bytes are preserved by the original mapping.
