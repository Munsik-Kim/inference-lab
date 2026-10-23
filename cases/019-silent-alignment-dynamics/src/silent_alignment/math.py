"""Physical gradient flow, rank-aware observables, and independent formulas."""

import numpy as np


def validate(C, U, V):
    if C.ndim != 2 or U.ndim != 2 or V.ndim != 2:
        raise ValueError("MATRIX_REQUIRED")
    if C.shape != (U.shape[0], V.shape[0]) or U.shape[1] != V.shape[1]:
        raise ValueError("SHAPE_MISMATCH")
    if any(a.dtype != np.float64 or not np.isfinite(a).all() for a in (C, U, V)):
        raise ValueError("FINITE_FP64_REQUIRED")


def loss(C, U, V):
    return 0.5 * float(np.sum((U @ V.T - C) ** 2))


def physical_rhs(C, U, V):
    error = C - U @ V.T
    return error @ V, error.T @ U


def scaled_rhs(C, A, B, rho):
    return C @ B - 2 * rho * A @ (B.T @ B), C.T @ A - 2 * rho * B @ (A.T @ A)


def stiefel(d, m, seed):
    if not 0 < m <= d:
        raise ValueError("STIEFEL_SHAPE")
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    Q, R = np.linalg.qr(rng.standard_normal((d, m)), mode="reduced")
    # A positive diagonal R makes the Gaussian QR orthogonal factor Haar.
    signs = np.where(np.diag(R) < 0.0, -1.0, 1.0)
    return Q * signs


def target_diagnostics(sigma, m, tol):
    gap = float(sigma[m-1] - sigma[m]) if m < len(sigma) else None
    unique_subspace = bool(sigma[m-1] > tol["rank_scaled_absolute"] and
                           (gap is None or gap > tol["boundary_gap_warning"]))
    return {"boundary_gap": gap, "target_subspace_identifiable": unique_subspace,
            "boundary_gap_warning": bool(gap is not None and gap <= tol["boundary_gap_warning"])}


def subspace(F, target_basis, tol):
    """No QR completion: an incomplete rank has NA full-m distance/angles."""
    q, s, _ = np.linalg.svd(F, full_matrices=False)
    cutoff = max(tol["rank_scaled_absolute"], (s[0] if len(s) else 0) * tol["rank_relative"])
    rank = int(np.count_nonzero(s > cutoff))
    m = target_basis.shape[1]
    condition = float(s[0] / s[-1]) if rank == m and s[-1] > 0 else np.nan
    if rank != m:
        return np.nan, np.full(m, np.nan), rank, condition
    q = q[:, :m]
    # Explicit projectors avoid negative squared distances from trace subtraction.
    distance = float(np.sum((q @ q.T - target_basis @ target_basis.T) ** 2) / (2 * m))
    overlap = np.linalg.svd(q.T @ target_basis, compute_uv=False)
    # arccos's argument is bounded analytically. This only bounds roundoff in
    # an angle diagnostic; the factors, Gram matrices and flow are not altered.
    angles = np.arccos(np.minimum(1.0, np.maximum(0.0, overlap)))
    return distance, angles, rank, condition


def metrics(C, left, sigma, right, A, B, rho, tol):
    m = B.shape[1]
    U, V = np.sqrt(2 * rho) * A, np.sqrt(2 * rho) * B
    Z = U @ V.T
    gu, gv = U.T @ U, V.T @ V
    diff = gv - gu - 2 * rho * np.eye(m)
    drift = float(np.linalg.norm(diff))
    gram_scale = float(np.linalg.norm(gv) + np.linalg.norm(gu))
    lmin = 0.5 * float(sigma[m:] @ sigma[m:])
    l0 = 0.5 * float(sigma @ sigma)
    l = loss(C, U, V)
    dl, al, rl, cl = subspace(A, left[:, :m], tol)
    dr, ar, rr, cr = subspace(B, right[:, :m], tol)
    if not target_diagnostics(sigma, m, tol)["target_subspace_identifiable"]:
        dl, dr = np.nan, np.nan
        al, ar = np.full(m, np.nan), np.full(m, np.nan)
    return {"loss": l, "loss_min": lmin, "learnable_residual": (l - lmin) / (l0 - lmin) if l0 > lmin else np.nan,
            "z_frobenius": float(np.linalg.norm(Z)), "z_singular_sorted": np.linalg.svd(Z, compute_uv=False),
            "target_mode_signed": np.diag(left.T @ Z @ right),
            "gram_u": gu, "gram_v": gv, "imbalance_absolute": drift,
            "imbalance_relative_gram": drift / gram_scale if gram_scale > 0 else np.nan,
            "left_distance": dl, "right_distance": dr,
            "left_angles_radians": al, "right_angles_radians": ar,
            "left_rank": rl, "right_rank": rr,
            "left_condition": cl, "right_condition": cr}


def closed_gram(C, R, rho, t):
    """Prop.1 Eq.(6) via symmetric eigenfunctions and solve, never inverse."""
    p, d = C.shape
    M = np.block([[-rho * np.eye(p), C], [C.T, rho * np.eye(d)]])
    values, Q = np.linalg.eigh(M)
    X0 = np.vstack([np.zeros((p, R.shape[1])), np.sqrt(2 * rho) * R])
    W = Q.T @ X0
    integral = np.empty_like(values)
    small = np.abs(values) < 1e-14
    integral[small] = t
    integral[~small] = np.expm1(2 * t * values[~small]) / (2 * values[~small])
    K = np.eye(R.shape[1]) + W.T @ (integral[:, None] * W)
    X = Q @ (np.exp(t * values)[:, None] * W)
    G = X @ np.linalg.solve(K, X.T)
    return G, {"solve_condition": float(np.linalg.cond(K)),
               "gram_symmetry_error": float(np.linalg.norm(G - G.T)),
               "integral_min_eigenvalue": float(np.min(integral)),
               "finite": bool(np.isfinite(G).all())}


def linearized_factors(C, R, rho, t):
    """Fixed-t small-rho limit (Prop.7), not a finite-rho exact solution."""
    left, s, rt = np.linalg.svd(C, full_matrices=False)
    right = rt.T
    A = (left * np.sinh(t * s)) @ (right.T @ R)
    B = right @ (np.cosh(t * s)[:, None] * (right.T @ R))
    # Include the right-null component for rectangular matrices.
    B += R - right @ (right.T @ R)
    return np.sqrt(2 * rho) * A, np.sqrt(2 * rho) * B


def scalar_rhs(t, z, sigma, rho):
    return 2 * np.hypot(z, rho) * (sigma - z)
