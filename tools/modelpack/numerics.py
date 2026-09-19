"""Float64 sufficient statistics and prompt-paired descriptive analysis."""
from __future__ import annotations
import numpy as np
from .common import require


def finite(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    require(bool(np.isfinite(a).all()), "Nonfinite full array")
    return a


def fit_ridge(g: np.ndarray, b: np.ndarray, w0: np.ndarray, eta: float) -> tuple[np.ndarray, dict]:
    g, b, w0 = map(finite, (g, b, w0))
    require(g.ndim == 2 and g.shape[0] == g.shape[1] and b.shape == w0.shape
            and b.shape[1] == g.shape[0], "Ridge shape mismatch")
    require(np.allclose(g, g.T, rtol=0, atol=1e-10), "Gram not symmetric")
    require(np.isfinite(eta) and eta > 0, "Positive finite eta required")
    scale = float(np.trace(g) / len(g))
    require(scale > 0, "Zero/invalid Gram trace")
    lam = eta * scale
    system = g + lam * np.eye(len(g))
    cholesky = np.linalg.cholesky(system)
    rhs = b + lam * w0
    result = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, rhs.T)).T
    finite(result)
    residual = float(np.linalg.norm(result @ system - rhs) / np.linalg.norm(rhs))
    require(residual < 1e-9, "Normal-equation residual failed")
    return result, {"eta": eta, "lambda": lam, "trace_per_width": scale,
                    "normal_equation_relative_residual": residual, "solver": "CPU_FP64_Cholesky"}


def norm_record(actual: np.ndarray, reference: np.ndarray) -> dict:
    actual, reference = map(finite, (actual, reference))
    require(actual.shape == reference.shape and actual.size > 0, "Norm shape/size mismatch")
    diff = float(np.square(actual - reference).sum())
    ref = float(np.square(reference).sum())
    return {"squared_error": diff, "reference_squared_norm": ref, "elements": actual.size,
            "absolute_rms": float(np.sqrt(diff / actual.size)),
            "reference_rms": float(np.sqrt(ref / actual.size)),
            "relative_error": float(np.sqrt(diff / ref)) if ref > 0 else None}


def lse(values: np.ndarray) -> float:
    values = finite(values)
    m = float(values.max())
    return m + float(np.log(np.exp(values - m).sum()))


def score(logits: np.ndarray, labels: list[int], gold: int) -> dict:
    z = finite(logits); require(z.ndim == 1 and len(set(labels)) == 4 and 0 <= gold < 4, "Bad scoring geometry")
    s = z[labels]; lc = lse(s); lv = lse(z); q = np.exp(s - lc)
    pred = int(np.argmax(s)); target = np.eye(4)[gold]
    ties = np.flatnonzero(s == s.max()).tolist()
    return {"option_logits": s.tolist(), "choice_probabilities": q.tolist(),
            "prediction": pred, "gold": gold, "correct": pred == gold,
            "choice_nll": lc - float(s[gold]), "full_gold_nll": lv - float(s[gold]),
            "brier": float(np.square(q-target).sum()),
            "gold_margin": float(s[gold] - np.delete(s, gold).max()),
            "winner_gap": float(np.sort(s)[-1]-np.sort(s)[-2]), "top_ties": ties,
            "full_log_normalizer": lv, "choice_log_normalizer": lc,
            "allowed_mass": float(np.exp(lc-lv)), "full_argmax_allowed": int(np.argmax(z)) in labels}


def paired(baseline: dict, candidate: dict) -> dict:
    require(baseline['gold'] == candidate['gold'], "Gold mismatch")
    b, c = baseline['correct'], candidate['correct']
    flip = baseline['prediction'] != candidate['prediction']
    return {"flip": flip, "regression": b and not c, "gain": not b and c,
            "both_correct": b and c, "both_wrong": not b and not c,
            "wrong_to_different_wrong": not b and not c and flip,
            **{f"delta_{key}": candidate[key]-baseline[key] for key in
               ['full_gold_nll', 'choice_nll', 'brier', 'gold_margin']}}


def full_kl(baseline: np.ndarray, candidate: np.ndarray) -> float:
    baseline, candidate = map(finite, (baseline, candidate))
    require(baseline.shape == candidate.shape, "KL shape mismatch")
    lb, lc = baseline-lse(baseline), candidate-lse(candidate)
    return float(np.sum(np.exp(lb)*(lb-lc)))


def recovery(uncorrected: np.ndarray, repaired: np.ndarray) -> float | None:
    a, b = map(finite, (uncorrected, repaired))
    require(a.shape == b.shape and np.all(a >= 0) and np.all(b >= 0), "Invalid error energies")
    denom = float(a.sum())
    return 1-float(b.sum())/denom if denom > 0 else None


def bootstrap_indices(tasks: list[str], replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(np.array(tasks) == task) for task in sorted(set(tasks))]
    return np.concatenate([rng.choice(ids, (replicates, len(ids)), replace=True) for ids in groups], axis=1)
