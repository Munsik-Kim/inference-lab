"""Compact P1 metrics, including raw cosine range and worst-direction error."""

import numpy as np


def direction(F, target, tol):
    q, s, _ = np.linalg.svd(F, full_matrices=False)
    rank = int(np.count_nonzero(s > max(tol["rank_scaled_absolute"], s[0]*tol["rank_relative"])))
    m = target.shape[1]
    if rank != m:
        return {"distance": np.nan, "cosines": np.full(m, np.nan), "theta_max": np.nan,
                "worst": np.nan, "rank": rank, "condition": np.nan,
                "cosine_max_overshoot": np.nan, "cosine_out_of_range_count": 0}
    cosines = np.linalg.svd(q.T@target, compute_uv=False)
    minimum = float(np.min(cosines))
    overshoot = float(max(0., np.max(cosines)-1., -minimum))
    if overshoot > tol["cosine_roundoff_limit"]:
        raise ValueError("PRINCIPAL_COSINE_OUTSIDE_ROUNDOFF_RANGE")
    # The original singular values are stored unchanged. Only arccos's
    # diagnostic argument is bounded, with overshoot recorded explicitly.
    theta = float(np.arccos(min(1., max(0., minimum))))
    return {"distance": float(np.sum((q@q.T-target@target.T)**2)/(2*m)),
            "cosines": cosines, "theta_max": theta, "worst": 1-minimum**2,
            "rank": rank, "condition": float(s[0]/s[-1]),
            "cosine_max_overshoot": overshoot,
            "cosine_out_of_range_count": int(np.count_nonzero((cosines > 1) | (cosines < 0)))}


def measure(C, left, right, sigma, a, b, rho, tol):
    U, V = np.sqrt(2*rho)*a, np.sqrt(2*rho)*b
    Z = U@V.T
    gu, gv = U.T@U, V.T@V
    eu, ev = np.linalg.eigvalsh(gu), np.linalg.eigvalsh(gv)
    imbalance = gv-gu-2*rho*np.eye(b.shape[1])
    ei = np.linalg.eigvalsh(imbalance)
    drift = float(np.linalg.norm(imbalance))
    gram_scale = float(np.linalg.norm(gu)+np.linalg.norm(gv))
    L = .5*float(np.sum((Z-C)**2))
    lmin = .5*float(sigma[b.shape[1]:]@sigma[b.shape[1]:])
    l0 = .5*float(np.sum(C*C))
    out = {"loss": L, "learnable_residual": (L-lmin)/(l0-lmin),
           "z_frobenius": float(np.linalg.norm(Z)),
           "z_singular_sorted": np.linalg.svd(Z, compute_uv=False),
           "target_mode_signed": np.diag(left.T@Z@right),
           "gram_u_eigenvalues": eu, "gram_v_eigenvalues": ev,
           "gram_u_trace": float(np.trace(gu)), "gram_v_trace": float(np.trace(gv)),
           "gram_u_frobenius": float(np.linalg.norm(gu)), "gram_v_frobenius": float(np.linalg.norm(gv)),
           "gram_cross_trace": float(np.sum(gu*gv.T)),
           "imbalance_error_eigenvalues": ei, "imbalance_absolute": drift,
           "imbalance_relative_gram": drift/gram_scale}
    for side, f, reference in (("left", a, left[:, :b.shape[1]]), ("right", b, right[:, :b.shape[1]])):
        for k,v in direction(f, reference, tol).items():
            out[side+"_"+k] = v
    return out


def compact(C, left, right, sigma, A, B, rho, tol, budget):
    columns = {}
    for i,(a,b) in enumerate(zip(A,B)):
        if i % 16 == 0:
            budget.check()
        for k,v in measure(C,left,right,sigma,a,b,rho,tol).items():
            columns.setdefault(k,[]).append(v)
    return {k:np.asarray(v) for k,v in columns.items()}


def compare(C,left,right,sigma,rho,A,B,RA,RB,tol,budget):
    arrays = {k:[] for k in ["z_absolute", "z_reference_norm", "normalized_loss_difference",
                            "right_distance_reference", "left_distance_reference",
                            "right_worst_reference", "left_worst_reference",
                            "right_distance_difference", "left_distance_difference",
                            "right_worst_difference", "left_worst_difference"]}
    denominator = .5*float(np.sum(C*C))-.5*float(sigma[A.shape[2]:]@sigma[A.shape[2]:])
    for i,(a,b,ra,rb) in enumerate(zip(A,B,RA,RB)):
        if i%16 == 0:
            budget.check()
        z,zr=2*rho*a@b.T,2*rho*ra@rb.T
        arrays["z_absolute"].append(float(np.linalg.norm(z-zr)))
        arrays["z_reference_norm"].append(float(np.linalg.norm(zr)))
        arrays["normalized_loss_difference"].append(float(abs(.5*np.sum((z-C)**2)-.5*np.sum((zr-C)**2))/denominator))
        for side,f,rf,target in (("right",b,rb,right[:,:b.shape[1]]),("left",a,ra,left[:,:a.shape[1]])):
            obs,ref=direction(f,target,tol),direction(rf,target,tol)
            for metric in ["distance","worst"]:
                arrays[f"{side}_{metric}_reference"].append(ref[metric])
                if np.isfinite(obs[metric]) != np.isfinite(ref[metric]):
                    raise ValueError("PRIMARY_REFERENCE_RANK_DISAGREEMENT")
                arrays[f"{side}_{metric}_difference"].append(abs(obs[metric]-ref[metric]))
    arrays={k:np.asarray(v) for k,v in arrays.items()}
    return arrays, comparison_status(arrays,tol)


def comparison_status(arrays,tol):
    near=arrays["z_reference_norm"]<=tol["z_near_zero_norm"]
    rel=float(np.max(arrays["z_absolute"][~near]/arrays["z_reference_norm"][~near])) if (~near).any() else 0.
    absolute=float(np.max(arrays["z_absolute"][near])) if near.any() else 0.
    worst=float(max(np.nanmax(arrays["right_worst_difference"]),np.nanmax(arrays["left_worst_difference"])))
    distance=float(max(np.nanmax(arrays["right_distance_difference"]),np.nanmax(arrays["left_distance_difference"])))
    ld=float(np.max(arrays["normalized_loss_difference"]))
    return {"max_z_relative_frobenius":rel,"max_z_absolute_near_zero":absolute,
            "max_normalized_loss_difference":ld,"max_projector_distance_difference":distance,
            "max_worst_error_difference":worst,"pass":bool(rel<=tol["z_relative"] and absolute<=tol["z_absolute"] and
              ld<=tol["normalized_loss_absolute"] and distance<=tol["projector_absolute"] and worst<=tol["worst_absolute"])}
