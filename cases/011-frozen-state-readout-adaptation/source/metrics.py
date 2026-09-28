"""Case 011 first-error metrics; the independent auditor does not import this module."""
from __future__ import annotations

import numpy as np

SUMMARY_POSITIONS = (32, 128, 256, 512, 1024, 2048)
PRIMARY_CONFIDENCE = 1.0 - 0.05 / 3.0


def _labels(value, name: str, ndim: int, *, invalid: bool = False):
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a {ndim}-dimensional integer array")
    if array.size == 0 or np.any(array < (-1 if invalid else 0)) or np.any(array > 5):
        raise ValueError(f"{name} contains missing/out-of-range S3 labels")
    return array


def summarize_head(predictions, gold, *, epsilon: float = 0.05,
                   positions=SUMMARY_POSITIONS):
    """Summarize every prefix. tau is 1-based; None means right-censored at Tmax.

    Readout errors are observations, not terminal-state events. A correct token
    after a first error contributes to token accuracy, never to survival.
    """
    pred = _labels(predictions, "predictions", 2, invalid=True)
    truth = _labels(gold, "gold", 2)
    if pred.shape != truth.shape:
        raise ValueError("prediction and gold shapes differ")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be between zero and one")
    n, length = pred.shape
    correct = pred == truth
    failed = ~correct
    has_failure = failed.any(axis=1)
    tau = np.where(has_failure, np.argmax(failed, axis=1) + 1, length + 1)
    continuous = tau - 1
    times = np.arange(1, length + 1)
    failures = (tau[:, None] <= times).sum(axis=0)
    allowed = np.flatnonzero(failures / n <= epsilon)
    empirical = int(allowed[-1] + 1) if len(allowed) else 0
    after_failure = times[None, :] > tau[:, None]
    recovered = correct & after_failure
    summary = {
        "n_sequences": n, "max_group_tokens": length,
        "tau": [int(t) if observed else None for t, observed in zip(tau, has_failure)],
        "right_censored_count": int((~has_failure).sum()),
        "rmst0": float(continuous.mean()),
        "rmst0_definition": "mean continuous correct group tokens before the first error; censored=Tmax",
        "empirical_t005" if epsilon == 0.05 else "empirical_t_epsilon": empirical,
        "epsilon": float(epsilon),
        "empirical_horizon_at_observation_limit": empirical == length,
        "confidence_supported_horizon": None,
        "horizon_scope": "empirical descriptive value only; no simultaneous confidence horizon",
        "token_accuracy": float(correct.mean()),
        "invalid_prediction_count": int((pred == -1).sum()),
        "failed_sequence_count": int(has_failure.sum()),
        "sequences_with_correct_token_after_first_error": int(recovered.any(axis=1).sum()),
        "correct_tokens_after_first_error": int(recovered.sum()),
        "tokens_after_first_error": int(after_failure.sum()),
        "survival_by_token": ((n - failures) / n).tolist(),
        "token_accuracy_by_token": correct.mean(axis=0).tolist(),
        "prefix_summary": [],
    }
    for t in positions:
        t = int(t)
        if not 1 <= t <= length:
            continue
        summary["prefix_summary"].append({
            "group_token": t, "failure_count": int(failures[t-1]),
            "failure_fraction": float(failures[t-1] / n),
            "survival": float((n - failures[t-1]) / n),
            "token_accuracy_at_position": float(correct[:, t-1].mean()),
            "token_accuracy_through_position": float(correct[:, :t].mean()),
            "rmst0_restricted": float(np.minimum(continuous, t).mean()),
            "denominator_sequences": n,
        })
    return summary


def summarize_predictions(predictions, gold, head_names=("ORIGINAL", "SHORT_REFIT", "MIXED_REFIT")):
    pred = _labels(predictions, "predictions", 3, invalid=True)
    truth = _labels(gold, "gold", 2)
    if pred.shape[1:] != truth.shape or len(head_names) != pred.shape[0]:
        raise ValueError("head names, predictions and gold have inconsistent dimensions")
    if len(set(head_names)) != len(head_names):
        raise ValueError("duplicate head names")
    return {str(name): summarize_head(p, truth) for name, p in zip(head_names, pred)}


def continuous_lengths(predictions, gold):
    pred = _labels(predictions, "predictions", 2, invalid=True)
    truth = _labels(gold, "gold", 2)
    if pred.shape != truth.shape:
        raise ValueError("prediction and gold shapes differ")
    mismatch = pred != truth
    return np.where(mismatch.any(axis=1), mismatch.argmax(axis=1), pred.shape[1]).astype(np.int64)


def paired_bootstrap_rmst(delta, *, seed: int, confidence: float = PRIMARY_CONFIDENCE,
                          n_bootstrap: int = 5000):
    """Resample paired base-sequence deltas, not tokens or separate head rows."""
    x = np.asarray(delta, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("paired RMST deltas must be a nonempty finite vector")
    if not 0 < confidence < 1 or n_bootstrap < 2 or seed < 0:
        raise ValueError("invalid bootstrap confidence, repetitions, or seed")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for first in range(0, n_bootstrap, 100):
        count = min(100, n_bootstrap - first)
        indices = rng.integers(0, len(x), size=(count, len(x)))
        means[first:first+count] = x[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    bounds = np.quantile(means, [tail, 1-tail], method="linear")
    return {"mean_delta_tokens": float(x.mean()), "interval_tokens": bounds.tolist(),
            "confidence": float(confidence), "n_paired_sequences": len(x),
            "bootstrap_repetitions": int(n_bootstrap), "bootstrap_seed": int(seed),
            "quantile_method": "linear", "resampling_unit": "paired base sequence",
            "interval_kind": "percentile bootstrap approximation"}


def primary_comparison(original, mixed, gold, checkpoint_seed: int):
    delta = continuous_lengths(mixed, gold) - continuous_lengths(original, gold)
    result = paired_bootstrap_rmst(delta, seed=63001 + int(checkpoint_seed))
    result.update({"comparison": "UNIFORM_8/MIXED_REFIT - UNIFORM_8/ORIGINAL",
                   "family_size": 3, "family_alpha": 0.05})
    return result


def score_logits(logits, gold):
    """Return CE and margins without reducing across sample/time dimensions.

    Nonfinite logits are execution observations, not valid score values: callers
    must retain invalid counts and represent unavailable scores as JSON null.
    """
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(gold)
    if x.ndim < 2 or x.shape[-1] != 6 or x.shape[:-1] != y.shape or y.dtype.kind not in "iu":
        raise ValueError("logit/gold shape or dtype mismatch")
    if np.any(y < 0) or np.any(y >= 6):
        raise ValueError("gold outside S3 label space")
    finite = np.isfinite(x).all(axis=-1)
    safe = np.where(finite[..., None], x, 0.0)
    shifted = safe - safe.max(axis=-1, keepdims=True)
    gold_logits = np.take_along_axis(safe, y[..., None], axis=-1)[..., 0]
    ce = np.log(np.exp(shifted).sum(axis=-1)) + safe.max(axis=-1) - gold_logits
    alternatives = safe.copy()
    np.put_along_axis(alternatives, y[..., None], -np.inf, axis=-1)
    gold_margin = gold_logits - alternatives.max(axis=-1)
    top = np.partition(safe, -2, axis=-1)[..., -2:]
    top_margin = top[..., 1] - top[..., 0]
    return {"finite": finite, "gold_ce": np.where(finite, ce, np.nan),
            "gold_margin": np.where(finite, gold_margin, np.nan),
            "top1_top2_margin": np.where(finite, top_margin, np.nan)}
