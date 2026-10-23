# Paper source map — actual v1 access

Source: Simon Martin, Francis Bach, Giulio Biroli,
*Silent Alignment in Two-Layer Linear Networks: A High-Dimensional Perspective*,
[arXiv:2610.29864v1](https://arxiv.org/abs/2610.29864v1), submitted 2026-10-21.
[v1 PDF](https://arxiv.org/pdf/2610.29864v1), 86 PDF pages.

On 2026-10-23 KST the abstract and full PDF text were accessed through the web
reader. The following sections were read directly, including the formula text.
PDF image screenshots returned an internal tool error; figure numbering and
captions were checked in the full PDF text. No author code was downloaded or
copied. The PDF/HTML is not included in this payload.

| Verified location (printed page; zero-based PDF index) | Implementation correspondence | Local scope |
|---|---|---|
| §2.2, Eq.(2), p.5 (4); §2.3 Assumption 1.1/1.4, p.6 (5) | `math.physical_rhs`, `stiefel`, scaled factors | Sigma=I; prescribed finite C |
| §3.1 Proposition 1, Eq.(5)–(6), pp.9–10 (8–9) | `closed_gram`: symmetric eigenfunctions and solve | Nine small-matrix runs, four early checkpoints |
| §5.1.1 Proposition 7, Eq.(21)–(23), pp.16–17 (15–16) | `linearized_factors` | Fixed-time approximation diagnostic |
| §5.2.1 Theorem 11, pp.20–21 (19–20) | Scalar leading-log derivation and target-mode observations | Finite-rho pilot, not theorem verification |
| Appendix E, pp.84–85 (83–84) | Gaussian QR, paired C/R, saved-time SVD checks | DOP853/RK4 replace the paper's Euler simulations |
| Figure 1, p.8 (7) | Result/time-scale overview | No reconstructed paper figure |
| Figure 2, p.9 (8) | Dynamics curves | Paper d=500, ten repeats; own d=8/64, two/one random starts |

The factor projector normalization here is 1/(2m), and loss plots subtract the
rank-m floor. These displayed quantities differ from the paper's normalizations.
The local target spectra, orthogonal bases, aligned oracle controls, grids,
thresholds and numerical checks follow the user-specified finite CPU design.
No high-dimensional random-matrix solver is implemented.

## Claim map

| Kind | Claim / location | Evidence boundary |
|---|---|---|
| Paper | Two time scales and mode activation | Sections above, with their limiting assumptions |
| Own derivation | Whitened-data equivalence, scaled flow, scalar integral/log coefficient | `theory.md`, finite differences and separate scalar path |
| Own implementation | Rank-aware metrics, three solvers, Gram check, frozen replay | `src/`, `tests/`, execution freeze and receipts |
| Pilot observation | Timing, direction, amplitude and drift | Actual `results/` raw and regenerated summaries |
| Future hypothesis | Direction learning may prepare several abrupt changes | Unproven; aligned controls retain an alternative |

Existing repository Apache-2.0 applies to this independent code. Paper authors
receive mathematical attribution; their paper and code are not relicensed here.
