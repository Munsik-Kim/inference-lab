# Case019: direction and magnitude in a finite CPU model

This is an independent implementation of a whitened two-layer linear flow. The
notation below also gives the numerical contracts used before the pilot.
The reference is [Martin, Bach and Biroli, v1](https://arxiv.org/pdf/2610.29864v1).

## Physical model and time

For C in R^(p×d), U in R^(p×m), V in R^(d×m), set Z=UVᵀ and

    L = 1/2 ||Z-C||²_F
    Udot = (C-Z)V,        Vdot = (C-Z)ᵀU.

The time coordinate is the gradient-flow time of this loss. There is no factor
1/d or 1/m in the ODE. For n=d, X=√d I and Y=CX, the empirical loss
||UVᵀX-Y||²_F/(2n) and both gradients are exactly the same quantities.
The unit test checks this equality using rectangular factors.

Let U(0)=0 and V(0)=√(2ρ)R with RᵀR=I. Differentiating the two factor Grams gives

    d(VᵀV-UᵀU)/dt = 0,       VᵀV-UᵀU = 2ρ I.

Here ρ is half the imbalance, while √(2ρ) is the initial factor amplitude.
The primary solver uses U=√(2ρ)A, V=√(2ρ)B, hence

    Adot = CB-2ρ A(BᵀB),      Bdot = CᵀA-2ρ B(AᵀA).

All observables use the physical U,V,Z. The independent fixed-step solver uses
the original physical residual (C-UVᵀ), so it can expose errors in this scaling.
No clipping, Gram correction, jitter or renormalization is applied to factors.
Only the analytically bounded argument of arccos is bounded against roundoff in
the principal-angle diagnostic; the projector distance uses explicit projectors.

## What is alignment?

Use the top-m left and right singular subspaces of C as references. The right
distance is ||P_span(V)-P_C,right,m||²_F/(2m); the left distance is analogous.
Angles are principal angles in radians. Basis rotations and sign choices leave
the projectors unchanged. At t=0, U=0 has no left direction: its distance and
angles are NA. V's t=0 distance describes its factor subspace, not a direction
chosen from the SVD of the zero product Z.

Rank is measured on scaled factors with the predeclared absolute/relative
thresholds. Incomplete rank yields NA rather than completed QR columns. A
degenerate or numerically unresolved boundary singular gap makes the selected
top-m subspace unidentifiable and its distance NA. Degeneracies within a selected
subspace allow the subspace comparison but not identification of individual
singular directions. Condition/rank diagnostics are retained.

Sorted singular values of Z describe magnitude. The signed coefficients
l_iᵀ Z r_i separately describe target modes; sorted position alone does not
identify a target mode in a mixed state.

## Loss floor and timing

The rank-m optimum is Lmin=1/2 Σ_(i>m) σ_i². Plot the learnable residual
(L-Lmin)/(L0-Lmin), while retaining raw L and Lmin. A positive terminal loss can
therefore be the rank limit rather than a delayed-learning plateau.

T_visible10 is the first crossing of residual ≤0.9. T_align_half is the first
right-distance crossing of half its initial value. Aligned controls have
T_align_half=0 with status ALREADY_ALIGNED. Unreached thresholds have
NOT_REACHED, null time, and observation end 20. A crossing is linearly
interpolated within an explicitly retained saved-grid bracket. These are
descriptive finite-pilot indices. A noncrossing or reversed ordering is not a
solver failure.

## Scalar derivation and already-aligned control

For a single positive target σ, z=uv and v²-u²=2ρ. Since u,v start nonnegative,

    zdot = (u²+v²)(σ-z)
          = 2 sqrt(z²+ρ²) (σ-z),     z(0)=0.

If z is small relative to σ, replace σ-z by σ. Integrating
asinh(z/ρ)=2σt gives z≈ρ sinh(2σt). The approximation is plotted only where
the independent exact scalar trajectory has z≤0.01σ. It is not used as a
finite-rho solver oracle outside that region.

For fixed 0<a<σ, the exact arrival time has the integral expression

    T(a,ρ) = integral_0^a dz / [2 sqrt(z²+ρ²) (σ-z)].

Splitting 1/(σ-z)=1/σ + z/[σ(σ-z)] makes the first term
asinh(a/ρ)/(2σ). The remaining integral is bounded as ρ→0 for fixed a<σ.
Thus the leading coefficient of log(1/ρ) is 1/(2σ). The P0 fit uses three rho
values and a=0.1σ; it is a finite diagnostic, not an asymptotic-law proof.

In the matrix control R is the top-m right singular basis of C. Its product
starts at zero and its norm/imbalance match the random control. Its directions
already coincide with the target subspaces, yet its mode amplitudes still have
to grow. This oracle uses the target SVD; its measured CPU cost is reported.

## Independent Gram reference and small-rho diagnostic

Stack X=[U;V] and define M=[[-ρI,C],[Cᵀ,ρI]]. Direct substitution using the
imbalance gives Xdot=MX-(1/2)XXᵀX. For G=XXᵀ the Riccati equation is
Gdot=MG+GM-G². The reference computes

    G(t) = E X0 solve(K, X0ᵀ E),
    E = exp(tM),
    K = I + X0ᵀ [integral_0^t exp(2sM) ds] X0.

This is Proposition 1, Eq.(6), implemented through symmetric eigendecomposition
and a linear solve. No explicit inverse is formed. The eigenfunction for the
integral uses expm1(2tλ)/(2λ), with limit t at λ=0. Comparisons are restricted
in advance to block B at t=0.25,0.5,1,2; condition and symmetry errors are saved.
Large-time exponentials are not used as an unchecked exact reference.

For fixed t at small ρ, drop the nonlinear scaled-factor terms. With C=LΣWᵀ,
A_lin=L sinh(tΣ)WᵀR and B_lin=W cosh(tΣ)WᵀR, adding the right-null component
when C is rectangular. These produce the Gram/product limits in Proposition 7.
Their difference from finite-ρ dynamics is a scientific diagnostic, not a
numerical acceptance failure.

Theorem 11 concerns ρ→0 at t=s log(1/ρ), away from activation boundaries
s=(2σ_i)^(-1), while the number of active target modes fits the rank budget.
P0 retains finite dimensions and finite rho. The high-dimensional d→∞ limit,
the singular small-initialization limit and a smooth finite-rho transition are
different objects. In particular, O(1) in rho leaves constants depending on gap
and alignment tolerance.

## Next designs, not executed

P1 candidate: d=p=64,m=32, rho={1e-2,1e-4,1e-6}, boundary
gap={0.4,0.1,0.025}, eight paired R samples: 72 random trajectories plus nine
unique aligned controls. Hold the upper spectrum fixed and set the lower
spectrum's upper endpoint to 0.8-gap. Retain actual gaps, target norm and loss
floor. Pair the same R across all rho/gap conditions; do not replicate aligned
controls eight times. Resource review and a separate design freeze come first.

P2 candidate: direction interventions with matched singular values and norms,
including random directions, and explicit accounting of changed loss/imbalance.
P3 candidate: choose one later nonlinear, quantization or continual-learning
question separately. None is run here.

The user's long-term hypothesis is that directional structure may prepare
multiple abrupt performance changes. The aligned-amplitude control is retained
as an alternative mechanism. This finite pilot supplies observations and a
checked implementation rather than a universal explanation of phase changes.
