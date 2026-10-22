"""DEV-only exact counts/costs/paired changes from independently replayed rows."""
import csv
from collections import Counter
from pathlib import Path
from .common import read_json, write_new, fresh_dir


def analyze(run, audit_file, output):
    run, output = Path(run), fresh_dir(output)
    audit = read_json(audit_file)
    by_id = {r["episode_id"]: r for r in audit["rows"]}
    records = [read_json(p) for p in sorted((run / "episodes").glob("*.json"))]
    ledger, cells, families = [], [], []
    for row in records:
        endpoint = by_id[row["episode_id"]]["independently_recomputed"]
        a = row["accounting"]
        ledger.append({"episode_id": row["episode_id"], "task_id": row["task_id"], "family": row["family"],
            "kv": row["kv"], "policy": row["policy"], **{k: int(endpoint[k]) for k in ("D", "B", "P", "F", "C1", "C2", "Q", "W")},
            "requests": a["actual"]["model_request_attempts"], "responses": a["actual"]["model_responses_received"],
            "input_tokens": a["actual_input_tokens"], "output_tokens": a["actual_output_tokens"],
            "guard_rejections": a["actual"]["guard_rejections"], "episode_wall_seconds": a["episode_wall_seconds"],
            "CPU_continuation_seconds": a["CPU_continuation_seconds"], "terminal_reason": row["terminal_reason"]})
    for kv in ("K0", "K1"):
        for policy in ("R0", "R1"):
            group = [r for r in ledger if r["kv"] == kv and r["policy"] == policy]
            r = {"kv": kv, "policy": policy, "observed": len(group), "planned": 6}
            for k in ("D", "B", "P", "F", "C1", "C2", "Q", "W", "requests", "responses", "guard_rejections", "episode_wall_seconds", "CPU_continuation_seconds"):
                r[k] = sum(row[k] for row in group)
            for k in ("input_tokens", "output_tokens"):
                r[k] = sum(row[k] for row in group) if all(row[k] is not None for row in group) and group else None
            for endpoint in ("Q", "W"):
                r["cost_per_" + endpoint] = {k: r[k] / r[endpoint] if r[endpoint] and r[k] is not None else None
                                            for k in ("input_tokens", "output_tokens", "episode_wall_seconds")}
            r["terminal_reasons"] = dict(Counter(row["terminal_reason"] for row in group))
            cells.append(r)
            for family in "ABC":
                f = [row for row in group if row["family"] == family]
                families.append({"kv": kv, "policy": policy, "family": family, "planned": 2, "observed": len(f),
                                 **{k: sum(row[k] for row in f) for k in ("D", "B", "P", "F", "C1", "C2", "Q", "W")}})
    pairs = []
    comparisons = [("K0_R1-R0", ("K0", "R0"), ("K0", "R1")), ("K1_R1-R0", ("K1", "R0"), ("K1", "R1")),
                   ("R0_K1-K0", ("K0", "R0"), ("K1", "R0")), ("R1_K1-K0", ("K0", "R1"), ("K1", "R1"))]
    for name, base, candidate in comparisons:
        a = {r["task_id"]: r for r in ledger if (r["kv"], r["policy"]) == base}
        b = {r["task_id"]: r for r in ledger if (r["kv"], r["policy"]) == candidate}
        for endpoint in ("Q", "W"):
            counts = dict.fromkeys(("both_success", "loss", "recovery", "both_failure"), 0)
            for task in a.keys() & b.keys():
                x, y = a[task][endpoint], b[task][endpoint]
                counts["both_success" if x and y else "loss" if x else "recovery" if y else "both_failure"] += 1
            pairs.append({"comparison": name, "endpoint": endpoint, "pairs_observed": len(a.keys() & b.keys()),
                          "pairs_planned": 6, **counts, "difference_count": counts["recovery"] - counts["loss"]})
    examples = {}
    for label, predicate in (("Q1_W0", lambda r: r["Q"] and not r["W"]),
                             ("B1_continuation_failure", lambda r: r["B"] and not (r["C1"] and r["C2"])),
                             ("F1_B0", lambda r: r["F"] and not r["B"]),
                             ("declared_but_not_quality", lambda r: r["D"] and not r["Q"])):
        examples[label] = next((r["episode_id"] for r in ledger if predicate(r)), None)
    def csv_file(name, rows):
        if not rows: return
        with (output / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    csv_file("episode_ledger.csv", ledger)
    csv_file("cells.csv", [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in cells])
    csv_file("family.csv", families)
    csv_file("paired.csv", pairs)
    result = {"phase": "P1_DEV", "execution_status": audit["execution_status"], "cells": cells, "families": families,
              "paired": pairs, "recorded_examples": examples, "independent_tasks": 6, "planned_model_episodes": 24,
              "totals": {k: (sum(r[k] for r in ledger) if ledger and all(r[k] is not None for r in ledger) else None) for k in
                         ("requests", "responses", "input_tokens", "output_tokens", "guard_rejections", "episode_wall_seconds", "CPU_continuation_seconds")},
              "scope": "descriptive DEV6 paired engineering results; no confirmatory CI, equivalence or autonomous continuation claim",
              "P2": "NOT_RUN", "P3": "NOT_RUN"}
    write_new(output / "summary.json", result)
    return result
