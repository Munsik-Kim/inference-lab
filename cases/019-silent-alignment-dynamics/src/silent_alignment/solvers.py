"""DOP853 on scaled factors and independent RK4 on physical factors."""

import time
import numpy as np
from scipy.integrate import DOP853, solve_ivp

from .math import physical_rhs, scaled_rhs, scalar_rhs


def dop853(C, R, rho, times, solver_config, budget):
    p, d = C.shape
    m = R.shape[1]
    size = p * m
    y0 = np.r_[np.zeros(size), R.ravel()]
    rhs_calls = 0

    def f(t, y):
        nonlocal rhs_calls
        rhs_calls += 1
        budget.check()
        A, B = y[:size].reshape(p, m), y[size:].reshape(d, m)
        da, db = scaled_rhs(C, A, B, rho)
        return np.r_[da.ravel(), db.ravel()]

    solver = DOP853(f, float(times[0]), y0, float(times[-1]),
                    rtol=solver_config["rtol"], atol=solver_config["atol"])
    out = np.empty((len(times), (p + d) * m))
    out[0] = y0
    cursor = 1
    accepted = 0
    start = time.monotonic()
    while solver.status == "running":
        message = solver.step()
        if solver.status == "failed":
            raise RuntimeError("DOP853_FAILED: " + str(message))
        accepted += 1
        dense = solver.dense_output()
        while cursor < len(times) and times[cursor] <= solver.t + 1e-13:
            out[cursor] = dense(times[cursor])
            cursor += 1
    if cursor != len(times) or not np.isfinite(out).all():
        raise RuntimeError("INCOMPLETE_OR_NONFINITE_DOP853")
    A = out[:, :size].reshape(len(times), p, m)
    B = out[:, size:].reshape(len(times), d, m)
    return A, B, {"solver": "DOP853_scaled", "nfev": solver.nfev,
                  "rhs_calls": rhs_calls, "accepted_steps": accepted,
                  "rejected_steps": None,
                  "rejected_steps_reason": "SciPy DOP853 public interface does not expose rejected trial count",
                  "solver_wall_seconds": time.monotonic() - start}


def rk4(C, R, rho, times, h, budget):
    U = np.zeros((C.shape[0], R.shape[1]), dtype=np.float64)
    V = np.sqrt(2 * rho) * R.copy()
    us, vs = [U.copy()], [V.copy()]
    current = 0.0
    steps = 0
    start = time.monotonic()
    for target in times[1:]:
        # Every saved time is a multiple of both predeclared step lengths.
        count = round((target - current) / h)
        if abs(current + count * h - target) > 1e-10:
            raise ValueError("REFERENCE_GRID_NOT_ON_FIXED_STEP")
        for _ in range(count):
            if steps % 64 == 0:
                budget.check()
            k1, l1 = physical_rhs(C, U, V)
            k2, l2 = physical_rhs(C, U + 0.5 * h * k1, V + 0.5 * h * l1)
            k3, l3 = physical_rhs(C, U + 0.5 * h * k2, V + 0.5 * h * l2)
            k4, l4 = physical_rhs(C, U + h * k3, V + h * l3)
            U = U + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            V = V + (h / 6) * (l1 + 2 * l2 + 2 * l3 + l4)
            steps += 1
        current = float(target)
        us.append(U.copy())
        vs.append(V.copy())
    scale = np.sqrt(2 * rho)
    return np.asarray(us) / scale, np.asarray(vs) / scale, {
        "solver": "RK4_original_physical", "h": h, "steps": steps, "nfev": 4 * steps,
        "solver_wall_seconds": time.monotonic() - start}


def scalar_reference(sigma, rho, times, budget):
    def f(t, z):
        budget.check()
        return scalar_rhs(t, z, sigma, rho)
    sol = solve_ivp(f, (0, times[-1]), [0.0], t_eval=times,
                    method="DOP853", rtol=1e-12, atol=1e-14)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[0], {"solver": "DOP853_scalar_z", "nfev": sol.nfev}
