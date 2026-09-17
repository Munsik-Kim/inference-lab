"""Paired base-scenario resampling and fixed stress bins."""
from __future__ import annotations

import numpy as np

from .tasks import TASKS

GAPS = ((0., .1), (.1, .5), (.5, 1.), (1., float("inf")))


def gap_bin(gap: float) -> str:
    if not np.isfinite(gap) or gap < 0:
        raise ValueError("Invalid winner gap")
    if gap == 0:
        return "EXACT_TIE"
    for lo, hi in GAPS:
        if lo <= gap < hi:
            return f"[{lo},{hi})"
    raise ValueError("Gap outside bins")


def select_boundary(pool: list[dict]) -> dict:
    if any(r["arm"] != "B" or r["split"] != "boundary_pool" for r in pool):
        raise ValueError("Selection must use B-only pool scores")
    keys = [r["base_id"] for r in pool]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate boundary-pool base ID")
    groups: dict[str, list[str]] = {}
    for task in TASKS:
        for label in ["EXACT_TIE"] + [f"[{lo},{hi})" for lo, hi in GAPS]:
            groups[f"{task}/{label}"] = sorted(r["base_id"] for r in pool if r["task"] == task and gap_bin(r["score"]["winner_gap"]) == label)
    selected = [v for key, values in groups.items() if not key.endswith("EXACT_TIE") for v in values[:6]]
    return {"selected_ids": selected, "pool_counts": {k: len(v) for k, v in groups.items()},
            "groups": groups, "selection": "First six lexical IDs per task/non-tie bin; no refill",
            "evidence_kind": "BASELINE_CONDITIONED_STRESS", "candidate_scores_used": False}


def paired_join(records: list[dict], arms: tuple[str, ...] = ("B", "A_PUBLIC", "V4")) -> dict:
    index: dict[tuple, dict] = {}
    for r in records:
        if r.get("evidence_kind") != "gpu_measurement" or r.get("mock"):
            raise ValueError("Only actual GPU records may enter measured summaries")
        if r.get('arm') not in arms:
            raise ValueError('Unexpected arm')
        key = (r["item_id"], r["arm"])
        if key in index:
            raise ValueError("Duplicate measurement cell")
        index[key] = r
    groups = {}
    for item in sorted({r["item_id"] for r in records}):
        group = [index.get((item, arm)) for arm in arms]
        if any(x is None for x in group):
            raise ValueError("Missing arm; no implicit deletion")
        for field in ("token_hash", "design_hash", "manifest_hash", "base_id", "task", "split", "length"):
            if len({r[field] for r in group}) != 1:
                raise ValueError(f"Mismatched pair field: {field}")
        groups[item] = dict(zip(arms, group))
    return groups


def scenario_draws(rows: list[dict], seed: int = 606901, repetitions: int = 5000) -> dict:
    """The same task-stratified scenario draws serve every candidate and length."""
    ids = {task: sorted({r["base_id"] for r in rows if r["task"] == task}) for task in TASKS}
    if any(not x for x in ids.values()):
        raise ValueError("Task-balanced inference requires every declared task")
    rng = np.random.default_rng(seed)
    return {"ids": ids, "indices": {task: rng.integers(len(keys), size=(repetitions, len(keys))).tolist()
                                    for task, keys in ids.items()}, "seed": seed, "repetitions": repetitions}


def balanced_interval(rows: list[dict], metric: str, draws: dict) -> dict:
    # Average repeated observations within scenario before giving each task equal weight.
    means = {}
    for task, ids in draws["ids"].items():
        means[task] = np.array([np.mean([r[metric] for r in rows if r["base_id"] == sid and r["task"] == task]) for sid in ids])
        if not np.isfinite(means[task]).all():
            raise ValueError("Missing/invalid scenario; cannot silently change denominator")
    per_task = {task: float(v.mean()) for task, v in means.items()}
    boot = np.mean([values[np.asarray(draws["indices"][task])].mean(axis=1) for task, values in means.items()], axis=0)
    return {"mean": float(np.mean(list(per_task.values()))), "per_task": per_task,
            "ci95": np.quantile(boot, [.025, .975], method="linear").tolist(),
            "independent_scenarios": sum(len(v) for v in means.values()),
            "scope": "Pointwise paired task-stratified scenario-cluster percentile interval"}


def zero_event_upper(n: int, confidence: float = .95) -> dict:
    if n <= 0:
        return {"upper": None, "n": n, "assumption": "Undefined empty denominator"}
    return {"upper": float(1 - (1-confidence)**(1/n)), "n": n,
            "assumption": "One-sided exact zero-event bound assumes independent identically distributed Bernoulli trials; not valid for pooled reused lengths or selected stress strata"}
