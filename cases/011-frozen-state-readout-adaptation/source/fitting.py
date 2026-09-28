"""Frozen-feature, centered-L2 final-head fitting; no recurrent model is imported.

All optimization uses full-batch CPU float64. Selection evaluates the returned
float32 head, with the four predeclared position bands equally weighted.
"""
from __future__ import annotations

import hashlib
import inspect
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F


FEATURES = 192
CLASSES = 6
BANDS = ((1, 32), (33, 64), (65, 128), (129, 256))
LAMBDAS = (1e-4, 1e-2, 1.0)


def _tensor(value, *, dtype, name):
    raw = torch.as_tensor(value).detach()
    if raw.is_complex() or raw.dtype == torch.bool:
        raise ValueError(f"{name} must be a real numeric array")
    tensor = raw.to(device="cpu", dtype=dtype).clone()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains a nonfinite value; no rows were removed")
    return tensor.contiguous()


def _data(phi, gold):
    x = _tensor(phi, dtype=torch.float64, name="phi")
    raw = torch.as_tensor(gold).detach().cpu()
    if raw.ndim != 1 or raw.dtype == torch.bool or raw.dtype.is_complex:
        raise ValueError("gold must be a one-dimensional integer label array")
    if not torch.isfinite(raw).all() or not torch.equal(raw, raw.to(torch.int64).to(raw.dtype)):
        raise ValueError("gold must contain finite integer labels")
    y = raw.to(torch.int64).clone()
    if x.ndim != 2 or x.shape[1] != FEATURES or x.shape[0] == 0 or len(y) != x.shape[0]:
        raise ValueError("phi must have nonzero shape (N,192), matched to gold (N,)")
    if not ((0 <= y) & (y < CLASSES)).all():
        raise ValueError("gold labels must be in [0,5]")
    return x, y


def _head(weight, bias, dtype=torch.float64):
    w = _tensor(weight, dtype=dtype, name="weight")
    b = _tensor(bias, dtype=dtype, name="bias")
    if tuple(w.shape) != (CLASSES, FEATURES) or tuple(b.shape) != (CLASSES,):
        raise ValueError("head must be weight (6,192), bias (6,)")
    return w, b


class _EvaluationBudget(Exception):
    pass


def fit_head(phi, gold, weight0, bias0, regularization, *, max_iter=200,
             max_evaluations=1000):
    """Fit one head from W0/b0; return FP32 tensors plus an auditable solver log.

    A hard objective-evaluation guard applies even during line search. If hit,
    the initial head is restored and the candidate is marked failed/ineligible.
    No partially completed line-search trial is accepted as a fitted head.
    Ordinary successful solves return the optimizer's final accepted iterate.
    """
    if isinstance(regularization, bool) or float(regularization) not in LAMBDAS:
        raise ValueError(f"regularization must be one of {LAMBDAS}")
    if (isinstance(max_iter, bool) or not isinstance(max_iter, int) or
            not 1 <= max_iter <= 200):
        raise ValueError("max_iter must be an integer in [1,200]")
    if (isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int) or
            not 1 <= max_evaluations <= 1000):
        raise ValueError("max_evaluations must be an integer in [1,1000]")
    x, y = _data(phi, gold)
    w0, b0 = _head(weight0, bias0)
    w, b = w0.clone().requires_grad_(), b0.clone().requires_grad_()
    lam = float(regularization)
    options = dict(lr=1.0, max_iter=max_iter, max_eval=max_evaluations,
                   tolerance_grad=1e-7, tolerance_change=1e-9,
                   history_size=100, line_search_fn="strong_wolfe")
    optimizer = torch.optim.LBFGS([w, b], **options)
    history = []
    evaluated = {}
    closure_calls = 0
    start = time.perf_counter()

    def key():
        return hashlib.sha256(w.detach().numpy().tobytes() + b.detach().numpy().tobytes()).hexdigest()

    def closure():
        nonlocal closure_calls
        closure_calls += 1
        if len(history) >= max_evaluations:
            raise _EvaluationBudget()
        optimizer.zero_grad(set_to_none=True)
        ce = F.cross_entropy(F.linear(x, w, b), y, reduction="mean")
        penalty = lam * ((w - w0).square().sum() + (b - b0).square().sum())
        loss = ce + penalty
        if not torch.isfinite(loss):
            raise ValueError("nonfinite fitting objective; fitting was not completed")
        loss.backward()
        if not torch.isfinite(w.grad).all() or not torch.isfinite(b.grad).all():
            raise ValueError("nonfinite fitting gradient; fitting was not completed")
        row = dict(evaluation=len(history) + 1, objective=float(loss.detach()),
                   ce=float(ce.detach()), centered_l2=float(penalty.detach()),
                   max_abs_gradient=float(torch.cat((w.grad.flatten(), b.grad)).abs().max()))
        history.append(row)
        evaluated[key()] = row
        return loss

    hit_guard = False
    try:
        optimizer.step(closure)
    except _EvaluationBudget:
        hit_guard = True
        with torch.no_grad():
            w.copy_(w0)
            b.copy_(b0)
    elapsed = time.perf_counter() - start
    state = optimizer.state[w]
    final = evaluated.get(key())
    if final is None:
        raise RuntimeError("optimizer returned an unevaluated iterate; refusing unlogged weights")
    grad_ok = final["max_abs_gradient"] <= options["tolerance_grad"]
    iterations = int(state.get("n_iter", 0))
    if hit_guard:
        status = "OBJECTIVE_EVALUATION_LIMIT_FAILED"
    elif grad_ok:
        status = "GRADIENT_TOLERANCE_MET"
    elif len(history) >= max_evaluations:
        status = "OBJECTIVE_EVALUATION_LIMIT_NOT_CONVERGED"
    elif iterations >= max_iter:
        status = "MAX_ITER_NOT_CONVERGED"
    else:
        status = "OPTIMIZER_SMALL_CHANGE_OR_DIRECTION_STOP"
    wf, bf = w.detach().to(torch.float32), b.detach().to(torch.float32)
    if not torch.isfinite(wf).all() or not torch.isfinite(bf).all():
        raise ValueError("head became nonfinite at its required FP32 execution boundary")
    optimizer_source = inspect.getsource(torch.optim.LBFGS).encode()
    return {
        "weight": wf, "bias": bf,
        "solver": {
            "algorithm": "torch.optim.LBFGS", "library_version": str(torch.__version__),
            "optimizer_source_sha256": hashlib.sha256(optimizer_source).hexdigest(),
            "fitting_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "device": "cpu", "fitting_dtype": "float64", "execution_dtype": "float32",
            "feature_scaling": "none", "trainable_parameter_count": 1158,
            "feature_rows": len(y), "feature_columns": FEATURES,
            "regularization": lam,
            "objective": "mean_cross_entropy + lambda*(sum((W-W0)^2)+sum((b-b0)^2))",
            "options": options, "iterations": iterations,
            "objective_evaluations": len(history), "closure_calls": closure_calls, "status": status,
            "eligible_for_dev": not hit_guard,
            "gradient_tolerance_met": grad_ok,
            "evaluation_limit_fallback": "restore_initial_head_and_reject_candidate" if hit_guard else None,
            "final_objective": final["objective"], "final_ce": final["ce"],
            "final_max_abs_gradient": final["max_abs_gradient"],
            "optimizer_signature": str(inspect.signature(torch.optim.LBFGS)),
            "elapsed_seconds": elapsed, "cpu_threads": torch.get_num_threads(),
            "objective_trace": history,
        },
    }


def equal_band_ce(phi, gold, positions, weight, bias):
    """Execute FP32 logits and equally weight mean CE in each declared DEV band."""
    x, y = _data(phi, gold)
    w, b = _head(weight, bias, dtype=torch.float32)
    p = torch.as_tensor(positions).detach().cpu()
    if p.shape != y.shape or p.dtype == torch.bool or p.dtype.is_complex:
        raise ValueError("positions must be an integer array matching gold")
    if not torch.isfinite(p).all() or not torch.equal(p, p.to(torch.int64).to(p.dtype)):
        raise ValueError("positions must contain finite integers")
    if not ((p >= 1) & (p <= 256)).all():
        raise ValueError("DEV positions must be in [1,256]; BOS is excluded")
    with torch.no_grad():
        logits = F.linear(x.to(torch.float32), w, b)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite FP32 DEV logits")
        # Logits execute in FP32; score accumulation uses FP64 logsumexp.
        losses = F.cross_entropy(logits.to(torch.float64), y, reduction="none")
    bands = []
    for lo, hi in BANDS:
        mask = (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            raise ValueError(f"DEV has no rows for position band [{lo},{hi}]")
        bands.append({"start": lo, "end": hi, "rows": n,
                      "ce": float(losses[mask].mean())})
    return {"equal_band_ce": math.fsum(row["ce"] for row in bands) / len(BANDS),
            "bands": bands, "logit_dtype": "float32", "score_dtype": "float64"}


def select_candidate(candidates, phi, gold, positions):
    """Select one candidate after FP32 casting; exact score ties prefer larger λ."""
    if not candidates:
        raise ValueError("at least one fitting candidate is required")
    rows = []
    seen = set()
    for index, candidate in enumerate(candidates):
        lam = float(candidate["solver"]["regularization"])
        if lam not in LAMBDAS or lam in seen:
            raise ValueError("candidate regularizations must be distinct declared lambdas")
        seen.add(lam)
        if candidate["solver"].get("eligible_for_dev") is False:
            rows.append({"candidate_index": index, "regularization": lam,
                         "eligible_for_dev": False, "status": candidate["solver"]["status"]})
            continue
        score = equal_band_ce(phi, gold, positions, candidate["weight"], candidate["bias"])
        rows.append({"candidate_index": index, "regularization": lam, "eligible_for_dev": True, **score})
    eligible = [row for row in rows if row["eligible_for_dev"]]
    if not eligible:
        raise ValueError("BLOCKED_FITTING: no solver candidate is eligible for DEV selection")
    selected = min(eligible, key=lambda r: (r["equal_band_ce"], -r["regularization"]))
    return {"candidate_index": selected["candidate_index"],
            "regularization": selected["regularization"], "dev_scores": rows,
            "rule": "minimum_FP32_head_equal_band_CE; exact_tie_larger_lambda"}
