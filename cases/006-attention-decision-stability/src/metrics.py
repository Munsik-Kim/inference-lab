"""FP64 score identities. Native model rounding has already happened."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def logsumexp(values: Any) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty logits")
    peak = float(x.max())
    return peak + float(np.log(np.exp(x - peak).sum()))


def score_options(logits: Any, gold: int, full_lse: float, full_argmax: int,
                  label_ids: list[int]) -> dict:
    x = np.asarray(logits, dtype=np.float64)
    if x.shape != (4,) or gold not in range(4) or len(set(label_ids)) != 4:
        raise ValueError("Four distinct option tokens and one gold index required")
    lse = logsumexp(x)
    if not math.isfinite(full_lse) or full_lse < lse - 1e-10:
        raise ValueError("Invalid full-vocabulary normalizer")
    logq = x - lse
    q = np.exp(logq)
    pred = int(x.argmax())  # Explicit earliest-label tie rule, never hidden.
    order = np.sort(x)
    target = np.zeros(4); target[gold] = 1
    nll_choice = float(-logq[gold])
    log_mass = float(lse - full_lse)
    return {"option_logits": x.tolist(), "gold_index": gold, "label_ids": label_ids,
            "log_q": logq.tolist(), "q": q.tolist(), "prediction": pred,
            "correct": pred == gold, "choice_nll": nll_choice,
            "choice_brier": float(np.square(q - target).sum()),
            "gold_margin": float(x[gold] - max(x[j] for j in range(4) if j != gold)),
            "winner_gap": float(order[-1] - order[-2]), "winner_tied": int((x == x.max()).sum()) > 1,
            "full_lse": float(full_lse), "full_gold_nll": float(full_lse - x[gold]),
            "log_label_mass": log_mass, "label_mass": math.exp(log_mass),
            "full_argmax": int(full_argmax), "full_argmax_allowed": int(full_argmax) in label_ids}


def score_full(logits: Any, label_ids: list[int], gold: int) -> dict:
    z = np.asarray(logits, dtype=np.float64)
    return score_options(z[label_ids], gold, logsumexp(z), int(z.argmax()), label_ids)


def kl_logits(a: Any, b: Any) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("KL requires the same complete vocabulary")
    la, lb = a - logsumexp(a), b - logsumexp(b)
    # Underflowed exp terms contribute zero; no probability clipping or epsilon.
    return float(np.sum(np.exp(la) * (la - lb)))


def paired(base: dict, candidate: dict) -> dict:
    if base["gold_index"] != candidate["gold_index"] or base["label_ids"] != candidate["label_ids"]:
        raise ValueError("Mismatched answer interface or gold")
    b_ok, c_ok = base["correct"], candidate["correct"]
    delta = np.subtract(candidate["option_logits"], base["option_logits"])
    oscillation = float(delta.max() - delta.min())
    gap = base["winner_gap"]
    flip = base["prediction"] != candidate["prediction"]
    sufficient = gap > oscillation
    if sufficient and flip:
        raise ValueError("Gap/oscillation invariant violated")
    return {"cell": "both_correct" if b_ok and c_ok else "regression" if b_ok else "gain" if c_ok else "both_wrong",
            "flip": flip, "wrong_to_wrong_flip": flip and not b_ok and not c_ok,
            "delta_choice_nll": candidate["choice_nll"] - base["choice_nll"],
            "delta_brier": candidate["choice_brier"] - base["choice_brier"],
            "delta_gold_margin": candidate["gold_margin"] - base["gold_margin"],
            "absolute_gold_margin_change": abs(candidate["gold_margin"] - base["gold_margin"]),
            "choice_kl": kl_logits(base["option_logits"], candidate["option_logits"]),
            "base_gap": gap, "option_delta": delta.tolist(), "oscillation": oscillation,
            "R": oscillation / gap if gap > 0 else None,
            "R_status": "DEFINED" if gap > 0 else "EXACT_TIE",
            "sufficient_unchanged": sufficient,
            "diagnostic_availability": "POST_HOC_REQUIRES_BOTH_OUTPUTS"}


def norm_evidence(actual: Any, reference: Any) -> dict:
    a, r = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if a.shape != r.shape or not a.size:
        raise ValueError("Norms require matching nonempty tensors")
    if not np.isfinite(a).all() or not np.isfinite(r).all():
        return {"valid": False, "relative_error": None, "reason": "NONFINITE"}
    e2, r2, a2 = float(np.square(a-r).sum()), float(np.square(r).sum()), float(np.square(a).sum())
    rms = math.sqrt(r2 / r.size)
    return {"valid": True, "count": r.size, "error_squared_sum": e2, "reference_squared_sum": r2,
            "actual_squared_sum": a2, "dot_sum": float(np.sum(a*r)),
            "reference_rms": rms, "absolute_rms": math.sqrt(e2/r.size),
            "relative_error": math.sqrt(e2/r2) if rms > 1e-6 else None,
            "cosine": float(np.sum(a*r))/math.sqrt(a2*r2) if a2 > 0 and r2 > 0 else None,
            "near_zero": rms <= 1e-6}


def pool_norms(units: list[dict]) -> dict:
    if not units or any(not u.get("valid") for u in units):
        raise ValueError("Cannot pool missing or invalid units")
    count = sum(u["count"] for u in units)
    e2 = math.fsum(u["error_squared_sum"] for u in units)
    r2 = math.fsum(u["reference_squared_sum"] for u in units)
    rms = math.sqrt(r2 / count)
    return {"count": count, "error_squared_sum": e2, "reference_squared_sum": r2,
            "absolute_rms": math.sqrt(e2/count), "reference_rms": rms,
            "relative_error": math.sqrt(e2/r2) if rms > 1e-6 else None,
            "definition": "RMS-weighted pooling: sqrt(sum squared error / sum squared reference)"}


def outcomes(pairs: list[dict]) -> dict:
    names = ("both_correct", "regression", "gain", "both_wrong")
    counts = {k: sum(p["cell"] == k for p in pairs) for k in names}
    n = len(pairs)
    correct, wrong = counts["both_correct"] + counts["regression"], counts["gain"] + counts["both_wrong"]
    def fraction(a: int, b: int) -> dict:
        return {"numerator": a, "denominator": b, "rate": a/b if b else None}
    return {"n": n, **counts, "flips": fraction(sum(p["flip"] for p in pairs), n),
            "wrong_to_wrong_flips": sum(p["wrong_to_wrong_flip"] for p in pairs),
            "regression_given_B_correct": fraction(counts["regression"], correct),
            "gain_given_B_wrong": fraction(counts["gain"], wrong),
            "regression_unconditional": fraction(counts["regression"], n),
            "gain_unconditional": fraction(counts["gain"], n),
            "accuracy_delta": (counts["gain"]-counts["regression"])/n if n else None,
            "zero_event_note": "Zero observed events is not zero population risk"}
