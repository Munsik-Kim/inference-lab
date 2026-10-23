"""Independent graph algebra, exact interventions and general-initial-state ODEs.

The graph formula is restricted to the original U(0)=0 trajectory. The ODE
entry points accept physical nonzero U,V and do not impose isotropic imbalance.
"""
import time
import numpy as np
from scipy.integrate import DOP853
from ..math import physical_rhs, scaled_rhs
from ..p1.metrics import direction


def log_filter(sigma, rho, times):
    sigma, times = np.asarray(sigma), np.asarray(times)
    if rho <= 0 or np.any(sigma < 0) or np.any(times < 0):
        raise ValueError("GRAPH_DOMAIN")
    omega = np.hypot(sigma, rho)
    x = times[:, None] * omega
    a = rho / omega
    # Factor out exp(x) before evaluating; no cosh overflow or value clipping.
    return x + np.log(.5*(1+a) + .5*(1-a)*np.exp(-2*x))


def graph_setup(right, R, max_condition=1e12):
    m = R.shape[1]
    B = right.T @ R
    top, bottom = B[:m], B[m:]
    s = np.linalg.svd(top, compute_uv=False)
    rank = int(np.count_nonzero(s > s[0]*1e-12))
    condition = float(s[0]/s[-1]) if s[-1] > 0 else float("inf")
    if rank != m or not np.isfinite(condition) or condition > max_condition:
        raise ValueError("GRAPH_TOP_BLOCK_RANK_OR_CONDITION")
    M = np.linalg.solve(top.T, bottom.T).T
    residual = float(np.linalg.norm(M@top-bottom) / max(np.linalg.norm(bottom), 1e-300))
    if residual > 1e-10:
        raise ValueError("GRAPH_SOLVE_RESIDUAL")
    cosines = np.linalg.svd(top, compute_uv=False)
    return M, {"top_rank": rank, "top_min_singular": float(s[-1]),
               "top_condition": condition, "solve_relative_residual": residual,
               "initial_principal_cosines": cosines.tolist(),
               "graph_spectral_norm": float(np.linalg.norm(M, 2)),
               "graph_frobenius_norm": float(np.linalg.norm(M))}


def graph_metrics(sigma, rho, times, M, budget):
    lf = log_filter(sigma, rho, times)
    m = M.shape[1]
    mean, worst = [], []
    for i in range(len(times)):
        if i % 32 == 0:
            budget.check()
        H = np.exp(lf[i, m:, None] - lf[i, None, :m]) * M
        h = np.linalg.svd(H, compute_uv=False)
        # h is moderate for this registered target/initialization. Reject an
        # overflow rather than silently clipping it into a valid angle.
        hs = h*h
        sine = hs/(1+hs)
        if not np.isfinite(sine).all():
            raise ValueError("NONFINITE_GRAPH_ANGLE")
        mean.append(float(np.mean(sine))); worst.append(float(np.max(sine)))
    return np.asarray(mean), np.asarray(worst)


def rotations(seed, m=32, d=64):
    out = {}
    for stream, (name, size) in enumerate((("O",m),("O_L",d),("O_R",d)), 1):
        rng = np.random.default_rng(np.random.SeedSequence([seed,stream]))
        q,r = np.linalg.qr(rng.normal(size=(size,size)))
        sign = np.where(np.diag(r) < 0, -1., 1.)
        out[name] = q*sign
    return out


def transform(U, V, left, right, arm, rotation):
    if arm == "CONTINUE":
        return U.copy(), V.copy()
    if arm == "GAUGE":
        return U@rotation["O"], V@rotation["O"]
    if arm == "ROTATE":
        return rotation["O_L"]@U, rotation["O_R"]@V
    if arm != "ALIGN":
        raise ValueError("UNKNOWN_ARM")
    _,ru = np.linalg.qr(U, mode="reduced")
    _,rv = np.linalg.qr(V, mode="reduced")
    a,s,bt = np.linalg.svd(ru@rv.T, full_matrices=False)
    ku,kv = a.T@ru, bt@rv
    m = U.shape[1]
    return left[:,:m]@ku, right[:,:m]@kv


def relative_error(actual, reference, atol=1e-12, rtol=1e-10):
    error = float(np.linalg.norm(actual-reference))
    scale = float(np.linalg.norm(reference))
    return {"absolute_error": error, "reference_norm": scale,
            "limit": atol+rtol*scale, "pass": bool(error <= atol+rtol*scale)}


def matching(C, left, right, sigma, U, V, up, vp, arm, rotation, rho, tol):
    gu,gv = U.T@U,V.T@V
    expected_gu,expected_gv = gu,gv
    if arm == "GAUGE":
        o=rotation["O"];expected_gu=o.T@gu@o;expected_gv=o.T@gv@o
    before = np.linalg.svd(U@V.T,compute_uv=False)[:U.shape[1]]
    after = np.linalg.svd(up@vp.T,compute_uv=False)[:U.shape[1]]
    checks = {"gram_u":relative_error(up.T@up,expected_gu),
              "gram_v":relative_error(vp.T@vp,expected_gv),
              "imbalance":relative_error(vp.T@vp-up.T@up,expected_gv-expected_gu),
              "prediction_singular_values":relative_error(after,before),
              "prediction_norm":relative_error(np.array([np.linalg.norm(up@vp.T)]),np.array([np.linalg.norm(U@V.T)]))}
    if arm in ("CONTINUE","GAUGE"):
        checks["physical_Z"] = relative_error(up@vp.T,U@V.T)
    if arm == "ALIGN":
        checks["oracle_Z"] = relative_error(up@vp.T,(left[:,:len(after)]*before)@right[:,:len(after)].T)
        lower = .5*(float(np.sum(C*C))+float(before@before)-2*float(sigma[:len(before)]@before))
        checks["oracle_loss"] = relative_error(np.array([.5*np.sum((up@vp.T-C)**2)]),np.array([lower]))
    gu1,gv1=up.T@up,vp.T@vp
    z=up@vp.T;du,dv=physical_rhs(C,up,vp)
    info={"loss":float(.5*np.sum((z-C)**2)),"gradient_squared_norm":float(np.sum(du*du)+np.sum(dv*dv)),
          "loss_rate":-float(np.sum(du*du)+np.sum(dv*dv)),
          "z_singular_sorted":after.tolist(),"z_norm":float(np.linalg.norm(z)),
          "nominal_imbalance_absolute":float(np.linalg.norm(gv1-gu1-2*rho*np.eye(U.shape[1]))),
          "actual_imbalance":(gv1-gu1).tolist(),"target_mode_signed":np.diag(left.T@z@right).tolist(),
          "gram_u":gu1.tolist(),"gram_v":gv1.tolist(),
          "U_rank":int(np.linalg.matrix_rank(up)),"V_rank":int(np.linalg.matrix_rank(vp)),
          "U_condition":float(np.linalg.cond(up)),"V_condition":float(np.linalg.cond(vp)),
          "product_near_tie_min_gap":float(np.min(np.abs(np.diff(after))))}
    for name,f,target in (("left",up,left[:,:U.shape[1]]),("right",vp,right[:,:V.shape[1]])):
        q=direction(f/np.sqrt(2*rho),target,tol)
        info[name]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in q.items()}
    return {"checks":checks,"pass":all(x["pass"] for x in checks.values()),"metrics":info,
            "svd_convention":"descending singular values; sign ambiguity cancels in paired QR/SVD construction; no tie rotation chosen using outcomes"}


def integrate(C, U0, V0, rho, times, solver_config, budget):
    p,d=C.shape;m=U0.shape[1];size=p*m;scale=np.sqrt(2*rho)
    y0=np.r_[U0.ravel()/scale,V0.ravel()/scale]
    rhs_calls=0
    def f(t,y):
        nonlocal rhs_calls
        rhs_calls+=1;budget.check()
        a,b=y[:size].reshape(p,m),y[size:].reshape(d,m)
        da,db=scaled_rhs(C,a,b,rho)
        return np.r_[da.ravel(),db.ravel()]
    start=time.monotonic()
    s=DOP853(f,float(times[0]),y0,float(times[-1]),rtol=solver_config["rtol"],atol=solver_config["atol"])
    out=np.empty((len(times),len(y0)));out[0]=y0;cursor=1;steps=0
    while s.status=="running":
        message=s.step();steps+=1
        if s.status=="failed":raise RuntimeError("P2_DOP853_FAILED: "+str(message))
        dense=s.dense_output()
        while cursor<len(times) and times[cursor]<=s.t+1e-13:
            out[cursor]=dense(times[cursor]);cursor+=1
    if cursor!=len(times) or not np.isfinite(out).all():raise ValueError("P2_ODE_INCOMPLETE_NONFINITE")
    U=out[:,:size].reshape(len(times),p,m)*scale
    V=out[:,size:].reshape(len(times),d,m)*scale
    # Preserve exact supplied start, without any drift correction.
    U[0]=U0;V[0]=V0
    return U,V,{"solver":"P2_general_initial_scaled_DOP853","rhs_calls":rhs_calls,"nfev":s.nfev,
                "accepted_steps":steps,"rejected_steps":None,"rejected_steps_reason":"not exposed by SciPy public interface",
                "wall_seconds":time.monotonic()-start}


def reference(C, U0, V0, times, h, budget):
    """Independent physical U/V RK4 with arbitrary (including nonzero) start."""
    U,V=U0.copy(),V0.copy();us=[U.copy()];vs=[V.copy()];steps=0;now=float(times[0]);start=time.monotonic()
    for t in times[1:]:
        count=round((t-now)/h)
        if abs(now+count*h-t)>1e-10:raise ValueError("RK4_GRID_NOT_ON_STEP")
        for _ in range(count):
            if steps%64==0:budget.check()
            k1,l1=physical_rhs(C,U,V)
            k2,l2=physical_rhs(C,U+.5*h*k1,V+.5*h*l1)
            k3,l3=physical_rhs(C,U+.5*h*k2,V+.5*h*l2)
            k4,l4=physical_rhs(C,U+h*k3,V+h*l3)
            U=U+h*(k1+2*k2+2*k3+k4)/6
            V=V+h*(l1+2*l2+2*l3+l4)/6;steps+=1
        us.append(U.copy());vs.append(V.copy());now=float(t)
    us,vs=np.asarray(us),np.asarray(vs)
    if not np.isfinite(us).all() or not np.isfinite(vs).all():raise ValueError("RK4_NONFINITE")
    return us,vs,{"solver":"P2_general_initial_original_UV_RK4","h":h,"steps":steps,"nfev":4*steps,
                  "wall_seconds":time.monotonic()-start}
