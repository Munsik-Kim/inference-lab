# P2 source correspondence and derivation

Source: Martin, Bach, Biroli, [arXiv:2610.29864v1](https://arxiv.org/pdf/2610.29864v1). P0/P1 access records remain unchanged. On2026-10-23 the full v1 PDF text for Proposition1, Appendix B.1/B.2 (printed35–37), and Theorem10 (printed18–19) was read again. No author code or PDF payload copied. These references motivate algebraic checking; the direction intervention is our controlled finite system experiment.

## Finite-rho original subspace derivation

Let W=[U;V], Imb=2rho I. Direct substitution of the original ODE gives Wdot=A_rho W-.5W(W^TW), where A_rho=[-rho I,C;C^T,rho I]. Set W=exp(t A_rho) W0 T(t). Tdot=-.5 T(W^TW), T(0)=I; finite-time T is invertible (its determinant satisfies a scalar linear log-derivative equation). Thus the nonlinear right factor does not change either block range. A_rho² is block diagonal with CC^T+rho²I and C^TC+rho²I. Acting on W0=[0;sqrt(2rho)R], the lower block is sqrt(2rho)Q diag(f_i)Q^TR with f_i=cosh(omega_i t)+(rho/omega_i)sinh(omega_i t), omega_i=hypot(sigma_i,rho). This retains finite rho.

In target coordinates B=Q^TR, split top m/bottom. If B_T is invertible, multiply the filtered basis on the right by B_T^{-1}diag(1/f_top); the graph becomes[I;H], H=diag(f_bottom) M diag(1/f_top), M=B_B B_T^{-1}. Compute M by solving B_T^T M^T=B_B^T. Principal squared sines are h_j²/(1+h_j²), from singular values of H. log(f)=x+log((1+a)/2+(1-a)exp(-2x)/2), x=omega*t,a=rho/omega. No large cosh intermediate or clipping. Rank/condition/residual failures are errors. Tiny fixtures compare direct filtered projectors and numerical ODE. This formula is never applied to nonzero-U checkpoint branches.

Theorem10's small-rho then late-time upper bound is not an exact finite-rho event-time curve. The exact graph for this original initialization checks finite-rho right spans; it does not prove a high-dimensional limit.

## Gauge and oracle construction

For orthogonal O, F_U(UO,VO)=F_U(U,V)O and similarly V, so prediction Z is unchanged along the flow; Grams conjugate. The ALIGN QR/small-SVD construction satisfies KuKv^T=diag(s), Ku^TKu=Ru^TRu and Kv^TKv=Rv^TRv. Target top singular pairs maximize the target inner product at fixed prediction singular values (trace inequality). The resulting instantaneous loss bound is checked separately from subsequent J/G/A. Ambient rotations preserve Grams and product singular values but change the target inner product.

## Actual reuse and independence

Read-only P0: FP64 physical/scaled RHS, hash/output/meter/environment utilities. Read-only P1: fixed25-file bundle and input loader; rank-aware direction/compact physical metrics; crossing/bracket and comparison-threshold helpers. P2 adds graph filtering, stable graph solve, persistence, exact transforms, nonzero-initial DOP853 and physical RK4 wrappers, source/input/order/budget contracts, analysis/figures/replay. Mathematical RHS functions and NumPy/SciPy are shared; the graph uses no stored ODE factors, and original physical RK4 is separate from the scaled primary solver. Replay reconstructs loss/norm algebraically and checkpoints from physical U/V, sharing metric definitions. It cannot reconstruct unsaved full factor paths. Same-machine CPU calculations, no new model inference or external human review. Source written with Codex assistance under the existing repository Apache-2.0 license. Oracle directions are diagnostic information, not a practical optimizer. P3 and other research remain outside scope.
