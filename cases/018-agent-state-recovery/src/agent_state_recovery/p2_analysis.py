"""Registered family-stratified paired task analysis. CPU and stdlib only."""
import csv
from collections import Counter
from pathlib import Path
import random
from .common import CASE_ROOT, read_json, write_new, fresh_dir
from .p2_contract import complete_token_sum


def percentile(values, p):
    a = sorted(values)
    rank = (len(a) - 1) * p
    low = int(rank)
    return a[low] + (a[min(low + 1, len(a) - 1)] - a[low]) * (rank - low)


def paired(ledger, replicates=10000, seed=61022018):
    matrix = {}
    for row in ledger:
        matrix.setdefault(row["task_id"], {})[(row["kv"], row["policy"])] = row
    comparisons = [("K0_R1-R0", ("K0", "R0"), ("K0", "R1")), ("K1_R1-R0", ("K1", "R0"), ("K1", "R1")),
                   ("R0_K1-K0", ("K0", "R0"), ("K1", "R0")), ("R1_K1-K0", ("K0", "R1"), ("K1", "R1"))]
    rows = []
    for name, base, candidate in comparisons:
        for endpoint in ("Q", "W"):
            counts = dict.fromkeys(("both_success", "loss", "recovery", "both_failure"), 0)
            for values in matrix.values():
                if base not in values or candidate not in values: continue
                a, b = values[base][endpoint], values[candidate][endpoint]
                counts["both_success" if a and b else "loss" if a else "recovery" if b else "both_failure"] += 1
            n = sum(counts.values())
            rows.append({"comparison": name, "endpoint": endpoint, "base": list(base), "candidate": list(candidate),
                "pairs_observed": n, "pairs_planned": 24, **counts,
                "difference_pp": 100 * (counts["recovery"] - counts["loss"]) / n if n else None})
    full = len(matrix) == 24 and all(len(r) == 4 for r in matrix.values())
    if full:
        strata = [[task for task in sorted(matrix) if task.startswith("HOLDOUT-" + f)] for f in "ABC"]
        if [len(s) for s in strata] != [8, 8, 8]: raise ValueError("bootstrap family denominator")
        rng, distributions = random.Random(seed), [[] for _ in rows]
        for _ in range(replicates):
            selected = [rng.choice(stratum) for stratum in strata for _ in range(8)]
            # The same selected tasks retain all4 conditions in all comparisons.
            for r, d in zip(rows, distributions):
                base, candidate, endpoint = tuple(r["base"]), tuple(r["candidate"]), r["endpoint"]
                d.append(100 * sum(matrix[t][candidate][endpoint] - matrix[t][base][endpoint] for t in selected) / 24)
        for r, d in zip(rows, distributions):
            r["exploratory_pointwise_95_lower_pp"] = percentile(d, .025)
            r["exploratory_pointwise_95_upper_pp"] = percentile(d, .975)
            r["CI_status"] = "COMPUTED"
    else:
        for r in rows: r.update(exploratory_pointwise_95_lower_pp=None, exploratory_pointwise_95_upper_pp=None, CI_status="NOT_COMPUTED_INCOMPLETE_MATRIX")
    return rows


def analyze(run, audit_file, output):
    run, output = Path(run), fresh_dir(output)
    audit = read_json(audit_file)
    endpoints = {r["episode_id"]: r["independently_recomputed"] for r in audit["rows"]}
    ledger, diagnostics = [], []
    for p in sorted((run / "episodes").glob('*.json')):
        row = read_json(p)
        e, a = endpoints[row["episode_id"]], row["accounting"]
        final = row["final_action"]
        final_args = final["arguments"] if final else None
        actual = {k: r["quantity"] for k, r in row["terminal_snapshot"]["records"].items()}
        declared = {r["record_id"]: r["quantity"] for r in final_args["quantities"]} if final_args else {}
        mismatch = bool(final and actual != declared)
        reasons = []
        if e["Q"] and not e["W"]:
            if mismatch: reasons.append("FINAL_QUANTITY_MISMATCH")
            if not e["F"]: reasons.append("FRESH_VALIDATION_MISSING")
            if not final: reasons.append("NO_ACCEPTED_FINAL")
            if final and not e["FINAL_CONTENT_CORRECT"] and not mismatch: reasons.append("OTHER_FINAL_CONTENT_ERROR")
        item = {"episode_id": row["episode_id"], "task_id": row["task_id"], "family": row["family"], "kv": row["kv"], "policy": row["policy"],
            **{k: int(e[k]) for k in ("D", "B", "P", "F", "C1", "C2", "Q", "W", "FINAL_CONTENT_CORRECT")},
            "requests": a["actual"]["model_request_attempts"], "responses": a["actual"]["model_responses_received"],
            "input_tokens": a["actual_input_tokens"], "output_tokens": a["actual_output_tokens"],
            "observed_input_tokens": a["observed_input_tokens"], "observed_output_tokens": a["observed_output_tokens"],
            "unobserved_response_cost": a["unknown_response_cost"], "guard_rejections": a["actual"]["guard_rejections"],
            "episode_wall_seconds": a["episode_wall_seconds"], "CPU_fixture_init_seconds": a["CPU_fixture_init_seconds"],
            "CPU_continuation_seconds": a["CPU_continuation_seconds"], "terminal_reason": row["terminal_reason"],
            "accepted_final": int(final is not None), "Q1_W0_reasons": ";".join(reasons)}
        ledger.append(item)
        diagnostics.append({"episode_id": row["episode_id"], "path": str(p.relative_to(run)), "Q1_W0_reasons": reasons,
            "terminal_vs_final_quantity_difference": {k: {"terminal": v, "final": declared.get(k)} for k, v in actual.items() if final and declared.get(k) != v},
            "invalid_validation_events": [ev["event_id"] for ev in row["events"] if ev["tool"] == "validate_inventory" and ev["result"].get("valid") is not True]})
    cells, families = [], []
    for kv in ("K0", "K1"):
        for policy in ("R0", "R1"):
            group = [r for r in ledger if r["kv"] == kv and r["policy"] == policy]
            r = {"kv": kv, "policy": policy, "observed": len(group), "planned": 24}
            for k in ("D", "B", "P", "F", "C1", "C2", "Q", "W", "FINAL_CONTENT_CORRECT", "requests", "responses", "guard_rejections", "episode_wall_seconds", "CPU_fixture_init_seconds", "CPU_continuation_seconds", "accepted_final"):
                r[k] = sum(x[k] for x in group)
            for k in ("input_tokens", "output_tokens"): r[k] = complete_token_sum([x[k] for x in group])
            for endpoint in ("Q", "W"):
                r["cost_per_" + endpoint] = {k: r[k] / r[endpoint] if r[endpoint] and r[k] is not None else None for k in ("input_tokens", "output_tokens", "episode_wall_seconds")}
            r["terminal_reasons"] = dict(Counter(x["terminal_reason"] for x in group))
            r["Q1_W0"] = sum(x["Q"] and not x["W"] for x in group)
            cells.append(r)
            for family in "ABC":
                part = [x for x in group if x["family"] == family]
                families.append({"kv": kv, "policy": policy, "family": family, "planned": 8, "observed": len(part),
                    **{k: sum(x[k] for x in part) for k in ("D", "B", "P", "F", "C1", "C2", "Q", "W", "FINAL_CONTENT_CORRECT")}})
    pairs = paired(ledger)
    totals = {k: complete_token_sum([r[k] for r in ledger]) for k in
              ("requests", "responses", "input_tokens", "output_tokens", "observed_input_tokens", "observed_output_tokens", "guard_rejections", "episode_wall_seconds", "CPU_fixture_init_seconds", "CPU_continuation_seconds")}
    examples = {}
    for name, pred in (("Q1_W0", lambda r: r["Q"] and not r["W"]), ("B1_C0", lambda r: r["B"] and not(r["C1"] and r["C2"])),
                       ("Fresh1_B0", lambda r: r["F"] and not r["B"]), ("D1_Q0", lambda r: r["D"] and not r["Q"])):
        examples[name] = next((r["episode_id"] for r in ledger if pred(r)), None)
    for name, rows in (("episode_ledger.csv", ledger), ("cells.csv", [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in cells]),
                       ("family.csv", families), ("paired.csv", [{k: v for k, v in r.items() if k not in ("base", "candidate")} for r in pairs])):
        with (output / name).open('x', encoding='utf-8', newline='') as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result = {"phase": "P2_HOLDOUT", "execution_status": audit["execution_status"], "status": "PASS",
        "independent_tasks": 24, "planned_model_episodes": 96, "observed_model_episodes": len(ledger),
        "cells": cells, "families": families, "paired": pairs, "totals": totals, "examples": examples,
        "bootstrap": read_json(CASE_ROOT / "specs/protocol.json")["analysis"],
        "Q1_W0_reason_episode_counts": dict(Counter(reason for r in ledger for reason in r["Q1_W0_reasons"].split(';') if reason)),
        "reason_counts_may_overlap": True, "P3": "NOT_RUN", "remote_publication": "NOT_RUN"}
    write_new(output / "summary.json", result)
    write_new(output / "posthoc_diagnostics.json", diagnostics)
    return result
