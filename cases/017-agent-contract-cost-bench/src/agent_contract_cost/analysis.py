"""Exploratory paired policy/KV analysis from independently audited rows only.

No submitted summary is read. Missing observations are null, never successes or
failures. Bootstrap draws retain family membership and a task's whole condition
vector. Intervals describe this small synthetic menu, not business populations.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


POLICIES = ("C0", "G", "N", "E")
KVS = ("K0", "K1")


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _quantile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    low = int(position)
    high = min(low + 1, len(values) - 1)
    fraction = position - low
    return values[low] * (1 - fraction) + values[high] * fraction


def _observed(row):
    return row is not None and row.get("audit_pass") is True and row.get("observation_status") == "OBSERVED" and type(row.get("verified")) is bool


def analyze(rows, out, *, tasks=None, stage=None, bootstrap_count=10000, seed=6102013, include_synthetic=False):
    """Create tables/JSON in a new output directory; no GPU/model dependencies.

    Pass the registered task list (not only completed cells) so the planned
    denominator and missing bounds cover every task. Synthetic contract tests
    are excluded unless explicitly enabled, and then label the whole analysis.
    """
    if isinstance(rows, (str, Path)):
        with Path(rows).open(encoding="utf-8") as stream:
            rows = json.load(stream)
    if isinstance(tasks, (str, Path)):
        with Path(tasks).open(encoding="utf-8") as stream:
            tasks = json.load(stream)
    if isinstance(tasks, dict):
        tasks = tasks["tasks"]
    if type(bootstrap_count) is not int or bootstrap_count < 1:
        raise ValueError("bootstrap count must be positive")
    if stage is None:
        stages = {row["stage"] for row in rows}
        if len(stages) != 1:
            raise ValueError("one stage per analysis; do not pool P0/P1/P2")
        stage = stages.pop()
    excluded, selected = [], []
    for row in rows:
        if row["stage"] != stage:
            excluded.append({"task_id": row["task_id"], "reason": "different stage"})
        elif row.get("evidence_kind") != "measurement" and not include_synthetic:
            excluded.append({"task_id": row["task_id"], "reason": "not model measurement"})
        else:
            selected.append(row)
    coverage = "REGISTERED_TASK_LIST" if tasks is not None else "OBSERVED_TASK_IDS_ONLY_INCOMPLETE_COVERAGE"
    task_family = {task["task_id"]: task["family"] for task in tasks} if tasks is not None else {row["task_id"]: row["family"] for row in selected}
    lookup = {}
    for row in selected:
        key = (row["task_id"], row["kv"], row["policy"])
        if key in lookup:
            raise ValueError("duplicate condition cell")
        if row["task_id"] not in task_family or task_family[row["task_id"]] != row["family"]:
            raise ValueError("unregistered task/family")
        if row["kv"] not in KVS or row["policy"] not in POLICIES:
            raise ValueError("unregistered KV/policy")
        lookup[key] = row
    ordered_tasks = sorted(task_family)
    families = {family: [task_id for task_id in ordered_tasks if task_family[task_id] == family] for family in sorted(set(task_family.values()))}
    groups, family_groups, missing = [], [], []

    def group(kv, policy, task_ids, family=None):
        records = [lookup.get((task_id, kv, policy)) for task_id in task_ids]
        observed = [row for row in records if _observed(row)]
        present = [row for row in records if row is not None]
        success = sum(row["verified"] for row in observed)
        costs = {name: sum(row.get("counters", {}).get(name, 0) for row in present) for name in
                 ("model_request_attempts", "model_responses_received", "generated_output_tokens", "parsed_tool_attempts", "contract_rejections", "callable_executions", "side_effect_writes", "final_attempts", "retry_triggered", "retry_consumed")}
        walls = [row["wall_seconds"] for row in present if type(row.get("wall_seconds")) in (int, float)]
        wall = sum(walls)
        result = {"stage": stage, "kv": kv, "policy": policy, "family": family,
                  "planned": len(task_ids), "present": len(present), "observed": len(observed),
                  "missing_or_unobserved": len(task_ids) - len(observed), "verified_success": success,
                  "state_correct": sum(row["state_correct"] is True for row in observed),
                  "final_correct": sum(row["final_correct"] is True for row in observed),
                  "observed_success_rate": success / len(observed) if observed else None,
                  "planned_success_lower": success / len(task_ids) if task_ids else None,
                  "planned_success_upper": (success + len(task_ids) - len(observed)) / len(task_ids) if task_ids else None,
                  "accepted_actions": sum(row.get("accepted_actions", 0) for row in present),
                  "wall_sum_seconds": wall, "wall_mean_seconds": wall / len(walls) if walls else None,
                  "seconds_per_verified_success": wall / success if success else None,
                  "tokens_per_verified_success": costs["generated_output_tokens"] / success if success else None,
                  "zero_success_cost_status": "DEFINED" if success else "UNDEFINED_ZERO_VERIFIED_SUCCESS",
                  "terminal_counts": dict(Counter(row["terminal_reason"] for row in present)), **costs}
        return result

    for kv in KVS:
        for policy in POLICIES:
            groups.append(group(kv, policy, ordered_tasks))
            for family, ids in families.items():
                family_groups.append(group(kv, policy, ids, family))
            for task_id in ordered_tasks:
                row = lookup.get((task_id, kv, policy))
                if not _observed(row):
                    missing.append({"task_id": task_id, "kv": kv, "policy": policy,
                                    "status": "NOT_RUN" if row is None else "AUDIT_FAILED" if not row.get("audit_pass") else row.get("observation_status"),
                                    "reason": "no stored cell" if row is None else row.get("audit_errors", [])})
    specifications = [(f"{kv}:{candidate}-{baseline}", (kv, candidate), (kv, baseline)) for kv in KVS for candidate, baseline in (("G", "C0"), ("N", "C0"), ("E", "C0"), ("E", "N"))]
    specifications += [(f"{policy}:K1-K0", ("K1", policy), ("K0", policy)) for policy in POLICIES]
    comparisons, per_task = [], []
    for comparison_id, candidate, baseline in specifications:
        differences, categories, bounds = {}, Counter(), []
        for task_id in ordered_tasks:
            changed = lookup.get((task_id, *candidate))
            original = lookup.get((task_id, *baseline))
            a = int(changed["verified"]) if _observed(changed) else None
            b = int(original["verified"]) if _observed(original) else None
            lower, upper = (0 if a is None else a) - (1 if b is None else b), (1 if a is None else a) - (0 if b is None else b)
            bounds.append((lower, upper))
            if a is None or b is None:
                category = "UNOBSERVED_PAIR"
            else:
                differences[task_id] = a - b
                category = "both_success" if a and b else "loss" if b else "recovery" if a else "both_fail"
            categories[category] += 1
            per_task.append({"comparison_id": comparison_id, "task_id": task_id, "family": task_family[task_id],
                             "candidate_success": a, "baseline_success": b, "category": category,
                             "difference": None if a is None or b is None else a - b,
                             "missing_lower": lower, "missing_upper": upper})
        comparisons.append({"comparison_id": comparison_id, "candidate_kv": candidate[0], "candidate_policy": candidate[1],
                            "baseline_kv": baseline[0], "baseline_policy": baseline[1], "planned_pairs": len(ordered_tasks),
                            "observed_pairs": len(differences), "effect_fraction": sum(differences.values()) / len(differences) if differences else None,
                            "effect_percentage_points": 100 * sum(differences.values()) / len(differences) if differences else None,
                            "missing_effect_lower": sum(a for a, _ in bounds) / len(bounds) if bounds else None,
                            "missing_effect_upper": sum(b for _, b in bounds) / len(bounds) if bounds else None,
                            "counts": dict(categories), "_differences": differences, "_bootstrap": []})
    rng = random.Random(seed)
    for _ in range(bootstrap_count):
        draw = [ids[rng.randrange(len(ids))] for ids in families.values() for _ in ids]
        for comparison in comparisons:
            values = [comparison["_differences"][task_id] for task_id in draw if task_id in comparison["_differences"]]
            if values:
                comparison["_bootstrap"].append(sum(values) / len(values))
    for comparison in comparisons:
        draws = comparison.pop("_bootstrap")
        comparison.pop("_differences")
        comparison["ci_95_fraction"] = [_quantile(draws, .025), _quantile(draws, .975)] if draws else None
        comparison["bootstrap_draws_with_observed_pairs"] = len(draws)
        comparison["ci_type"] = "exploratory pointwise family-stratified paired task bootstrap"
        comparison["ci_zero_width_is_not_equivalence"] = bool(draws and min(draws) == max(draws))
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    summary = {"stage": stage, "evidence_scope": "SYNTHETIC_CONTRACT_TEST" if include_synthetic else "model_measurement_stored_trace_audit",
               "coverage": coverage, "registered_tasks": len(ordered_tasks), "registered_family_counts": {family: len(ids) for family, ids in families.items()},
               "planned_cells": len(ordered_tasks) * 8, "present_cells": len(selected), "missing_or_unobserved_cells": len(missing),
               "status": "COMPLETE" if not missing and len(selected) == len(ordered_tasks) * 8 else "PARTIAL",
               "bootstrap": {"seed": seed, "draws": bootstrap_count, "paired_shared_task_indices": True,
                             "stratified_family": True, "missing_policy": "no imputation; observed pairs plus exact binary missing bounds",
                             "interpretation": "exploratory synthetic task menu; not equivalence/noninferiority or external business-population CI"},
               "cost_scope": "sum of measured episode walls including failures; not workload makespan or serving speedup",
               "groups": groups, "family_groups": family_groups, "comparisons": comparisons, "excluded": excluded}
    _write_json(output / "summary.json", summary)
    _write_json(output / "missing.json", missing)
    _csv(output / "policy_results.csv", groups)
    _csv(output / "family_results.csv", family_groups)
    _csv(output / "paired_comparisons.csv", comparisons)
    _csv(output / "paired_task_table.csv", per_task)
    return summary
