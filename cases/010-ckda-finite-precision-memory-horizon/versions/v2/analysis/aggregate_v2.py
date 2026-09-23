"""Aggregate the complete frozen 15-cell fresh experiment, without inference.

No partial aggregation is supported. The independent auditor must pass before
tables are written. Paired intervals use the original frozen metrics function
and exactly the pairs, repeat count, and seed rule declared in protocol_v2.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

CASE = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
ARMS = ("NATIVE_FP32", "UNIFORM_8", "UNIFORM_5", "LOWRANK_4_8_R2", "MIXED_5_6_BUDGET")
PAIRS = (("LOWRANK_4_8_R2", "MIXED_5_6_BUDGET"),
         ("LOWRANK_4_8_R2", "UNIFORM_5"), ("UNIFORM_8", "NATIVE_FP32"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    require(bool(rows), "empty table")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_cell_identities(cells, protocol, *, protocol_sha256, input_manifest_sha256,
                             sample_ids_sha256, tokens_sha256):
    fresh = protocol["fresh"]
    require(fresh["arms"] == list(ARMS) and fresh["model_seeds"] == list(SEEDS), "frozen fresh arm/seed set changed")
    require((fresh["N"], fresh["T"], fresh["simultaneous_family"]) == (1024, 2048, 195), "frozen cohort/family changed")
    require(fresh["pairs"] == [list(pair) for pair in PAIRS] and fresh["paired_bootstrap_repeats"] == 5000,
            "frozen paired comparisons changed")
    require(fresh["paired_bootstrap_seed_rule"] == "60101+100*model_seed+pair_index", "frozen bootstrap seed rule changed")
    seen = set()
    for cell in cells:
        key = cell["model_seed"], cell["arm"]
        require(type(key[0]) is int and key not in seen, "duplicate or invalid model seed")
        seen.add(key)
        require(key in {(s, a) for s in SEEDS for a in ARMS}, "unexpected fresh cell")
        require(cell["schema"] == "case010-failure-aware-v2-cell-v1" and cell["cohort"] == "fresh", "wrong result schema/cohort")
        require(cell["kind"] == "FRESH_TEST_FIXED_CHECKPOINTS", "wrong fresh execution kind")
        require(cell["execution"]["status"] in ("COMPLETE", "COMPLETE_WITH_TERMINAL_STREAMS"), "incomplete fresh cell")
        require((cell["N"], cell["T"], cell["family_size"]) == (1024, 2048, 195), "cell changed its denominator/family")
        require(cell["checkpoint_sha256"] == protocol["checkpoint_sha256"][str(key[0])], "checkpoint identity mismatch")
        for field, expected in (("protocol_sha256", protocol_sha256), ("input_manifest_sha256", input_manifest_sha256),
                                ("sample_ids_sha256", sample_ids_sha256), ("tokens_sha256", tokens_sha256)):
            require(cell[field] == expected, f"cell {field} mismatch")
        require([r["horizon"] for r in cell["grid_rows"]] == fresh["confidence_grid"], "fresh grids differ")
        require(cell["alpha"] == fresh["alpha"] and cell["epsilon_primary"] == fresh["epsilon_primary"], "fresh inference settings differ")
    require(seen == {(s, a) for s in SEEDS for a in ARMS}, "all 15 fresh cells are required; partial intersections forbidden")


def build_tables(cells, protocol, paired_function):
    indexed = {(cell["model_seed"], cell["arm"]): cell for cell in cells}
    require(len(indexed) == len(cells) == 15, "all 15 unique cells are required")
    fresh_rows, byte_rows, pair_rows, pair_details = [], [], [], []
    for seed in SEEDS:
        for arm in ARMS:
            cell = indexed[(seed, arm)]
            ledger = cell["ledger"]
            fresh_rows.append({"model_seed": seed, "arm": arm, "N": cell["N"], "T": cell["T"],
                               "family_size": cell["family_size"], "RMST0": cell["RMST0"],
                               **{name: cell[name] for name in ("empirical_T05_all_tokens", "empirical_T01_all_tokens",
                                   "supported_T05_grid", "supported_T01_grid", "token_accuracy", "final_quarter_accuracy",
                                   "bos_accuracy", "first_failure_censored", "terminal_count", "invalid_prediction_tokens")},
                               "execution_status": cell["execution"]["status"],
                               "per_stream_bytes": ledger["per_stream_persistent_bytes"], "shared_bytes": ledger["shared_bytes"]})
            for n in (1, 16, 128):
                total = ledger["shared_bytes"] + n * ledger["per_stream_persistent_bytes"]
                require(total == ledger["total_bytes"][str(n)], "serialized byte arithmetic mismatch")
                byte_rows.append({"model_seed": seed, "arm": arm, "N_streams": n,
                                  "per_stream_bytes": ledger["per_stream_persistent_bytes"],
                                  "header_bytes": ledger["header_bytes"], "shared_bytes": ledger["shared_bytes"],
                                  "shared_codec_bytes": ledger["shared_codec_bytes"],
                                  "shared_token_table_bytes": ledger["shared_token_table_bytes"],
                                  "shared_runtime_metadata_bytes": ledger["shared_runtime_metadata_bytes"],
                                  "total_bytes": total, "amortized_bytes": total / n,
                                  "scope": "serialized cache/config/basis/table/runtime; common model weights and evaluator history excluded"})
        for pair_index, (candidate, baseline) in enumerate(PAIRS):
            bootstrap_seed = 60101 + 100 * seed + pair_index
            result = paired_function(indexed[(seed, candidate)], indexed[(seed, baseline)],
                                     bootstrap_seed=bootstrap_seed, repeats=5000)
            require(result["model_seed"] == seed and result["bootstrap_seed"] == bootstrap_seed and
                    result["bootstrap_repeats"] == 5000 and result["checkpoint_pooling"] is False,
                    "paired function changed frozen design")
            pair_details.append(dict(pair_index=pair_index, **result))
            pair_rows.append({"model_seed": seed, "pair_index": pair_index, "candidate": candidate, "baseline": baseline,
                              "N": result["N"], "RMST0_delta": result["RMST0_delta"],
                              "pointwise_95_CI_low": result["pointwise_95_CI"][0], "pointwise_95_CI_high": result["pointwise_95_CI"][1],
                              "positive_difference_sequences": result["positive_difference_sequences"],
                              "negative_difference_sequences": result["negative_difference_sequences"],
                              "bootstrap_seed": bootstrap_seed, "bootstrap_repeats": 5000,
                              "checkpoint_pooling": False, "interval_scope": result["interval_scope"]})
    return fresh_rows, pair_rows, byte_rows, pair_details


def run(results, output, case_root=CASE):
    case_root, results, output = Path(case_root).resolve(), Path(results).resolve(), Path(output).resolve()
    require(not output.exists(), "aggregate output must be a new directory")
    require(results.is_relative_to(case_root) and not output.is_relative_to(results), "invalid source/output location")
    protocol_path, manifest_path = case_root / "protocol_v2.json", case_root / "inputs/manifest.json"
    freeze_path = case_root / "provenance/protocol_freeze.json"
    protocol, manifest, freeze = [json.loads(p.read_text()) for p in (protocol_path, manifest_path, freeze_path)]
    require(sha(protocol_path) == freeze["protocol_sha256"], "protocol differs from its frozen identity")
    source_hashes = {p.relative_to(case_root).as_posix(): sha(p) for p in (protocol_path, manifest_path, freeze_path)}
    for name, expected in freeze["frozen_files"].items():
        require(sha(case_root / name) == expected, f"frozen artifact changed: {name}")
        source_hashes[name] = expected
    cells = []
    for path in sorted(results.rglob("summary.json")):
        cell = json.loads(path.read_text())
        require(path.parent.relative_to(results).as_posix() == f"seed{cell['model_seed']}/{cell['arm']}", "cell folder/identity mismatch")
        cells.append(cell)
        for artifact in (path, path.parent / "predictions.npz"):
            source_hashes[artifact.relative_to(case_root).as_posix()] = sha(artifact)
    cohort = manifest["cohorts"]["fresh"]
    validate_cell_identities(cells, protocol, protocol_sha256=sha(protocol_path), input_manifest_sha256=sha(manifest_path),
                             sample_ids_sha256=cohort["sample_ids_sha256"], tokens_sha256=cohort["tokens_file_sha256"])
    spec = importlib.util.spec_from_file_location("case010_independent_audit", case_root / "analysis/audit_v2.py")
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    audit = auditor.audit_results(results, case_root=case_root)
    require(audit["status"] == "PASS" and audit["verified_cells"] == 15, "complete independent audit is required")
    for cell in audit["cells"]:
        for artifact in cell["actual_byte_verification"]["artifacts"].values():
            source_hashes[artifact["path"]] = artifact["sha256"]
    sys.path.insert(0, str(case_root))
    from source import metrics as frozen_metrics
    require(Path(frozen_metrics.__file__).resolve() == (case_root / "source/metrics.py").resolve(), "metrics resolved from another case")
    require(sha(frozen_metrics.__file__) == freeze["frozen_files"]["source/metrics.py"], "paired metrics function differs from freeze")
    fresh_rows, pair_rows, byte_rows, pair_details = build_tables(cells, protocol, frozen_metrics.paired_summary)
    for path in (case_root / "analysis/audit_v2.py", Path(__file__).resolve()):
        source_hashes[path.relative_to(case_root).as_posix()] = sha(path)
    require(all(sha(case_root / name) == digest for name, digest in source_hashes.items()), "source changed during aggregation")
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "fresh_table.csv", fresh_rows)
    write_csv(output / "paired_table.csv", pair_rows)
    write_csv(output / "bytes_table.csv", byte_rows)
    write_json(output / "source_hashes.json", source_hashes)
    result = {"schema": "case010-v2-complete-fresh-aggregate-v1", "status": "PASS", "fresh_cells": 15,
              "sequences_per_checkpoint_arm": 1024, "pooled_checkpoint_estimate": False,
              "simultaneous_grid_family": 195, "paired_comparisons": pair_details,
              "fresh_table": fresh_rows, "bytes_table": byte_rows,
              "scope": "complete fresh cohort; 3 fixed checkpoints reported separately; paired RMST intervals are pointwise, not multiplicity-corrected superiority tests",
              "historical_or_diagnostic_sequences_added_to_fresh_N": 0,
              "independent_audit": {"status": audit["status"], "verified_cells": audit["verified_cells"],
                                    "auditor_sha256": audit["auditor_sha256"]},
              "paired_metrics_source_sha256": sha(frozen_metrics.__file__),
              "source_hashes_file": "source_hashes.json", "source_hashes_sha256": sha(output / "source_hashes.json"),
              "outputs": {name: sha(output / name) for name in ("fresh_table.csv", "paired_table.csv", "bytes_table.csv")}}
    write_json(output / "combined.json", result)
    print(json.dumps({"status": "PASS", "fresh_rows": len(fresh_rows), "paired_rows": len(pair_rows),
                      "byte_rows": len(byte_rows), "combined_sha256": sha(output / "combined.json")}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, default=CASE)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.results or args.case_root / "results/fresh", args.output, args.case_root)


if __name__ == "__main__":
    main()
