"""Sequence-level first-failure statistics with explicit right censoring.

tau is the first incorrect *predicted* time, starting at 1.  None means no
failure through max_horizon; it must never be replaced by max_horizon.  A
failure at max_horizon is an observed event.  Recovery can improve step accuracy
but cannot repair first-failure survival.  Inference assumes independent test
sequences for one fixed arm, condition, and model seed.
"""

import math
from numbers import Integral

import numpy as np


DEFAULT_HORIZONS = (32, 64, 128, 256, 512, 1024, 2048)
PRIMARY_N = 512


def _int(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _probability(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    value = float(value)
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return value


def _correct_array(correct, include_bos=False):
    correct = np.asarray(correct)
    if correct.ndim != 2 or correct.dtype.kind != "b":
        raise ValueError("correct must be a boolean [sequence, time] array")
    if include_bos:
        correct = correct[:, 1:]
    if correct.shape[0] < 1 or correct.shape[1] < 1:
        raise ValueError("at least one sequence and one predicted step are required")
    return correct


def correctness_from_labels(predicted, gold, n_classes=None):
    """Compare finite integer labels; invalid predictions count as failures.

    Nonfinite, fractional, negative, or out-of-range gold labels are an input
    error.  Such predictions are incorrect even when both arrays contain NaN.
    BOS, when present, is excluded by first_failure_times(include_bos=True).
    """
    predicted, gold = np.asarray(predicted), np.asarray(gold)
    if predicted.shape != gold.shape or gold.ndim != 2:
        raise ValueError("predicted and gold must have the same [sequence, time] shape")
    if gold.dtype.kind not in "iuf" or predicted.dtype.kind not in "iuf":
        raise ValueError("labels must be real numeric arrays")
    gold_valid = np.isfinite(gold) & (gold >= 0)
    pred_valid = np.isfinite(predicted) & (predicted >= 0)
    if gold.dtype.kind == "f":
        gold_valid &= gold == np.floor(gold)
    if predicted.dtype.kind == "f":
        pred_valid &= predicted == np.floor(predicted)
    if n_classes is not None:
        n_classes = _int(n_classes, "n_classes", 1)
        gold_valid &= gold < n_classes
        pred_valid &= predicted < n_classes
    if not np.all(gold_valid):
        raise ValueError("gold labels must be finite nonnegative integer class IDs")
    return pred_valid & (predicted == gold)


def first_failure_times(correct, include_bos=False):
    """Return observed 1-based failure times, or None for right censoring."""
    correct = _correct_array(correct, include_bos)
    failed = ~correct
    has_failure = failed.any(axis=1)
    first = failed.argmax(axis=1) + 1
    return [int(time) if observed else None for time, observed in zip(first, has_failure)]


def first_failure_from_labels(predicted, gold, n_classes=None, include_bos=False):
    return first_failure_times(correctness_from_labels(predicted, gold, n_classes), include_bos)


def _log_binomial_cdf(k, n, probability):
    """Stable log P[Binomial(n,p) <= k], used only inside CP inversion."""
    if k >= n or probability == 0:
        return 0.0
    if probability == 1:
        return -math.inf
    log_p, log_q = math.log(probability), math.log1p(-probability)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
             + i * log_p + (n - i) * log_q for i in range(k + 1)]
    largest = max(terms)
    return largest + math.log(math.fsum(math.exp(term - largest) for term in terms))


def clopper_pearson_upper(events, n, alpha=0.05):
    """Exact one-sided (1-alpha) binomial upper bound, no normal approximation."""
    n = _int(n, "n", 1)
    events = _int(events, "events")
    alpha = _probability(alpha, "alpha")
    if events > n:
        raise ValueError("events cannot exceed n")
    if events == n:
        return 1.0
    if events == 0:
        return -math.expm1(math.log(alpha) / n)
    lo, hi = 0.0, 1.0
    target = math.log(alpha)
    for _ in range(80):
        midpoint = (lo + hi) / 2
        if _log_binomial_cdf(events, n, midpoint) > target:
            lo = midpoint
        else:
            hi = midpoint
    return hi


def _validate_taus(taus, max_horizon):
    max_horizon = _int(max_horizon, "max_horizon", 1)
    taus = list(taus)
    if not taus:
        raise ValueError("at least one independent sequence is required")
    validated = []
    for tau in taus:
        if tau is None:
            validated.append(None)
        else:
            tau = _int(tau, "tau", 1)
            if tau > max_horizon:
                raise ValueError("tau exceeds the observation horizon; use None for censoring")
            validated.append(tau)
    return validated, max_horizon


def _validate_ids(sequence_ids, n):
    if sequence_ids is None:
        return None
    ids = list(sequence_ids)
    if len(ids) != n or any(not isinstance(key, str) or not key for key in ids):
        raise ValueError("sequence_ids must contain one nonempty string per sequence")
    if len(set(ids)) != n:
        raise ValueError("duplicate sequence IDs: stochastic or model-seed replicates cannot increase n")
    return ids


def _validate_horizons(horizons, max_horizon):
    if horizons is None:
        horizons = [t for t in DEFAULT_HORIZONS if t <= max_horizon]
        if max_horizon not in horizons:
            horizons.append(max_horizon)
    horizons = [_int(t, "horizon", 1) for t in horizons]
    if not horizons or horizons != sorted(set(horizons)) or horizons[-1] > max_horizon:
        raise ValueError("horizons must be unique, increasing, nonempty, and <= max_horizon")
    if horizons[-1] != max_horizon:
        raise ValueError("horizons must include max_horizon for censor-aware horizon reporting")
    return horizons


def _horizon_statement(horizons, values, epsilon, max_horizon, confidence_bound=False):
    eligible = [t for t, value in zip(horizons, values) if value <= epsilon]
    horizon = max(eligible) if eligible else None
    return {
        "epsilon": epsilon,
        "horizon": horizon,
        "relation": ">=" if horizon is not None and (confidence_bound or horizon == max_horizon) else "grid_estimate",
        "right_censored_at_max_horizon": horizon == max_horizon,
        "interpretation": (
            f"T_epsilon >= {max_horizon}; the finite observation window does not imply infinity"
            if horizon == max_horizon else
            "no evaluated grid horizon qualifies" if horizon is None else
            "simultaneous confidence-supported lower bound on T_epsilon"
            if confidence_bound else "largest qualifying evaluated grid horizon; no interpolation"
        ),
    }


def summarize_first_failures(taus, max_horizon, horizons=None, sequence_ids=None,
                             family_size=None, alpha=0.05, epsilons=(0.05, 0.01),
                             correct=None, model_seed=None, include_bos=False):
    """JSON-safe summary for a single model seed, arm, and condition.

    ``family_size`` is the *prespecified total* number of horizon x arm x
    condition comparisons (and model seeds when selecting across them).
    Default: one arm/condition, all supplied horizons.  Bonferroni adjusts the
    one-sided failure upper bounds before any horizon maximum is taken.  A
    caller comparing arms must supply the full family, not reuse the default.

    Restricted mean failure-free length = mean(min(tau - 1, max_horizon)),
    with None contributing max_horizon.  This discrete estimand counts correct
    predicted steps before first failure; it is not mean per-step accuracy.
    """
    taus, max_horizon = _validate_taus(taus, max_horizon)
    n = len(taus)
    ids = _validate_ids(sequence_ids, n)
    horizons = _validate_horizons(horizons, max_horizon)
    alpha = _probability(alpha, "alpha")
    family_size = len(horizons) if family_size is None else _int(family_size, "family_size", 1)
    if family_size < len(horizons):
        raise ValueError("family_size must cover every evaluated horizon before selecting a maximum")
    epsilons = [_probability(epsilon, "epsilon") for epsilon in epsilons]
    if len(set(epsilons)) != len(epsilons):
        raise ValueError("epsilons must be unique")
    if isinstance(model_seed, Integral):
        model_seed = _int(model_seed, "model_seed")
    elif model_seed is not None and not isinstance(model_seed, str):
        raise ValueError("model_seed must be one integer or string label, not pooled seed labels")
    event_times = np.asarray([max_horizon + 1 if tau is None else tau for tau in taus], dtype=np.int64)
    rows = []
    for horizon in horizons:
        events = int(np.sum(event_times <= horizon))
        upper = clopper_pearson_upper(events, n, alpha / family_size)
        rows.append({"horizon": horizon, "first_failures": events, "failure_probability": events / n,
                     "survival_probability": 1 - events / n, "failure_upper_bound": upper,
                     "survival_lower_bound": 1 - upper})
    result = {
        "n_sequences": n, "primary_n": PRIMARY_N, "is_primary_n": n == PRIMARY_N,
        "max_horizon": max_horizon, "model_seed": model_seed, "sequence_ids": ids,
        "tau": taus, "observed_failures": int(np.sum(event_times <= max_horizon)),
        "right_censored_sequences": int(np.sum(event_times > max_horizon)),
        "restricted_mean_failure_free_length": float(np.minimum(event_times - 1, max_horizon).mean()),
        "restricted_mean_definition": "mean(min(tau - 1, Tmax)); None contributes Tmax",
        "horizons": rows,
        "confidence": {"method": "exact one-sided Clopper-Pearson with Bonferroni",
                       "family_alpha": alpha, "family_size": family_size,
                       "per_comparison_alpha": alpha / family_size,
                       "scope": "prespecified horizons x arms x conditions (x selected model seeds)",
                       "model_seed_pooling": "forbidden; summarize each model seed separately"},
        "empirical_T_epsilon": [
            _horizon_statement(horizons, [row["failure_probability"] for row in rows], epsilon, max_horizon)
            for epsilon in epsilons],
        "confidence_supported_T_epsilon_lower_bound": [
            _horizon_statement(horizons, [row["failure_upper_bound"] for row in rows], epsilon, max_horizon, confidence_bound=True)
            for epsilon in epsilons],
        "step_accuracy": None,
    }
    if correct is not None:
        correct = _correct_array(correct, include_bos)
        if correct.shape != (n, max_horizon) or first_failure_times(correct) != taus:
            raise ValueError("correct must match tau and contain exactly max_horizon predicted steps")
        result["step_accuracy"] = {
            "overall": float(correct.mean()),
            "at_horizon": [{"horizon": t, "accuracy": float(correct[:, t - 1].mean())} for t in horizons],
            "prefix": [{"horizon": t, "accuracy": float(correct[:, :t].mean())} for t in horizons],
            "interpretation": "per-step correctness; recovery does not change first-failure survival",
        }
    return result


def summarize_by_model_seed(records_by_seed, **kwargs):
    """Keep model-seed strata distinct; never treat trained seeds as new n.

    Each mapping value supplies ``taus`` and ``sequence_ids`` and may supply
    ``correct``.  Shared sequence IDs across seeds are expected and preserved.
    """
    if not records_by_seed:
        raise ValueError("at least one model-seed stratum is required")
    summaries = {}
    for model_seed, records in records_by_seed.items():
        if str(model_seed) in summaries:
            raise ValueError("model-seed labels must remain distinct when serialized")
        if "sequence_ids" not in records:
            raise ValueError("model-seed strata require explicit sequence_ids")
        summaries[str(model_seed)] = summarize_first_failures(model_seed=model_seed, **records, **kwargs)
    return {"model_seed_summaries": summaries, "pooled_sequence_n": None,
            "interpretation": "model seeds are separate strata; no sequence-level pooling"}


def paired_bootstrap(taus_a, taus_b, *, sequence_ids_a, sequence_ids_b,
                     max_horizon, seed, horizons=None, n_bootstrap=2000, alpha=0.05):
    """Paired sequence percentile intervals for B - A, conditional on model seed.

    Bootstrap intervals are exploratory pointwise uncertainty summaries, not
    simultaneous confidence-supported T_epsilon or a new decision threshold.
    A required caller-supplied frozen seed makes resampling reproducible.
    """
    a, max_horizon = _validate_taus(taus_a, max_horizon)
    b, _ = _validate_taus(taus_b, max_horizon)
    ids_a = _validate_ids(sequence_ids_a, len(a))
    ids_b = _validate_ids(sequence_ids_b, len(b))
    if ids_a is None or ids_b is None or set(ids_a) != set(ids_b):
        raise ValueError("paired bootstrap requires the same explicit sequence IDs in both arms")
    lookup_b = dict(zip(ids_b, b))
    b = [lookup_b[key] for key in ids_a]
    horizons = _validate_horizons(horizons, max_horizon)
    seed = _int(seed, "seed")
    n_bootstrap = _int(n_bootstrap, "n_bootstrap", 2)
    alpha = _probability(alpha, "alpha")
    event_a = np.asarray([max_horizon + 1 if tau is None else tau for tau in a])
    event_b = np.asarray([max_horizon + 1 if tau is None else tau for tau in b])
    delta_survival = (event_b[:, None] > np.asarray(horizons)).astype(float) - (event_a[:, None] > np.asarray(horizons))
    delta_rmst = np.minimum(event_b - 1, max_horizon) - np.minimum(event_a - 1, max_horizon)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty((n_bootstrap, len(horizons) + 1))
    values = np.column_stack((delta_survival, delta_rmst))
    for draw in range(n_bootstrap):
        draws[draw] = values[rng.integers(len(a), size=len(a))].mean(axis=0)
    lower, upper = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    mean = values.mean(axis=0)
    return {
        "method": "paired sequence percentile bootstrap", "direction": "B - A",
        "n_sequences": len(a), "seed": seed, "n_bootstrap": n_bootstrap, "alpha": alpha,
        "scope": "pointwise exploratory intervals, conditional on a fixed model seed",
        "delta_survival": [{"horizon": t, "estimate": float(mean[i]), "lower": float(lower[i]),
                            "upper": float(upper[i])} for i, t in enumerate(horizons)],
        "delta_restricted_mean_failure_free_length": {
            "estimate": float(mean[-1]), "lower": float(lower[-1]), "upper": float(upper[-1])},
    }
