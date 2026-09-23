"""Read-only joint view of the original menu and the frozen budget supplement.

The same TEST sample is reused. This derived view re-expresses every arm's
binomial bound at family size 630; it does not modify the original 546-family
summaries, rerun a model, select a DEV winner, or measure additional timing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codec.survival import summarize_first_failures

PHASE = "POST_HOC_BUDGET_AUDIT_SAME_TEST"
STREAM_COUNTS = (1, 16, 128)
ORIGINAL_ARMS = ("NATIVE_FP32", *(f"UNIFORM_{b}" for b in range(2, 17)), "STOCHASTIC_4",
                 "FULL_RESIDUAL_4_4", "FULL_RESIDUAL_4_8", "FULL_RESIDUAL_4_FP32",
                 "LOWRANK_4_8_R1", "LOWRANK_4_8_R2", "LOWRANK_4_8_R4", "UNTRANSPORTED_4_4",
                 "MIXED_4_8", "MIXED_6_8")
SUPPLEMENT_ARMS = ("MIXED_4_8_TOP2", "MIXED_4_8_TOP5", "MIXED_6_8_TOP1", "MIXED_6_8_TOP4")
DESCRIPTIVE = "best feasible on the reused TEST sample; not DEV-selected, independent confirmation, or a superiority test"
TIE_RULE = "maximize the named metric, then RMST; minimize total bytes, then arm name lexicographically"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Sources:
    """Receipts use logical paths; never leak machine-specific input paths."""

    def __init__(self):
        self.receipts = {}

    def read(self, directory, relative, label, expected=None):
        directory = Path(directory)
        path = directory / relative
        require(path.resolve().is_relative_to(directory.resolve()), "source path escapes its directory")
        raw = path.read_bytes()
        digest = sha(raw)
        require(expected is None or digest == expected, f"source hash mismatch: {label}/{relative}")
        logical = f"{label}/{relative}"
        receipt = dict(file=logical, sha256=digest, size_bytes=len(raw))
        require(logical not in self.receipts or self.receipts[logical] == receipt, "source changed while being read")
        self.receipts[logical] = receipt
        return raw

    def json(self, directory, relative, label, expected=None):
        return json.loads(self.read(directory, relative, label, expected))


def read_policy(sources):
    policy = sources.json(ROOT / "configs", "budget_supplement_v1.json", "config")
    digest = sources.receipts["config/budget_supplement_v1.json"]["sha256"]
    sidecar = sources.read(ROOT / "configs", "budget_supplement_v1.sha256", "config").decode().strip()
    require(digest == sidecar, "supplement policy file hash mismatch")
    require(policy["schema"] == "case010-budget-supplement-v1" and policy["phase"] == PHASE,
            "wrong supplement policy")
    require(policy["model_seeds"] == [0, 1, 2] and [a["name"] for a in policy["arms"]] == list(SUPPLEMENT_ARMS),
            "wrong seed or supplement-arm family")
    require([(a["base_bits"], a["promoted_bits"], a["top_per_head"]) for a in policy["arms"]] ==
            [(4, 8, 2), (4, 8, 5), (6, 8, 1), (6, 8, 4)], "supplement allocations changed")
    test, evaluation = policy["test"], policy["evaluation"]
    require([test[k] for k in ("group", "split", "seed", "sequences", "length")] == ["S3", "TEST", 3001, 512, 2048],
            "supplement must retain the frozen TEST sample")
    require([evaluation[k] for k in ("horizons", "alpha", "epsilons", "family_size")] ==
            [[32, 64, 128, 256, 512, 1024, 2048], .05, [.05, .01], 630], "joint inference family changed")
    require(evaluation["all_arms_all_seeds"] and evaluation["all512_no_conditioning"], "no sequence or seed filtering")
    sources.read(ROOT / "configs", policy["budget_file"], "config", policy["budget_file_sha256"])
    return policy, digest


def close(actual, expected, label):
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"inconsistent retained statistic: {label}")


def horizon_value(summary, confidence, epsilon):
    name = "confidence_supported_T_epsilon_lower_bound" if confidence else "empirical_T_epsilon"
    entries = [row for row in summary[name] if row["epsilon"] == epsilon]
    require(len(entries) == 1, "one entry required for each epsilon")
    return entries[0]["horizon"]


def horizon_status(value, grid, confidence=False):
    if value is None:
        return f"no qualifying grid horizon (<{grid[0]}); not a measured zero horizon"
    if confidence:
        return f"simultaneous confidence-supported lower bound >= {value}"
    if value == grid[-1]:
        return f"empirical zero/few failures through Tmax={value}; population claim requires confidence field"
    return "largest qualifying empirical sample grid horizon"


def reexpress(summary, arm, seed, source_family, policy):
    """Recompute only first-failure inference; retain original sample and byte cap."""
    test, evaluation = policy["test"], policy["evaluation"]
    require(summary["model_seed"] == seed, "model seeds must remain separate strata")
    require((summary["n_sequences"], summary["max_horizon"], len(summary["tau"])) ==
            (test["sequences"], test["length"], test["sequences"]), "wrong TEST denominator or length")
    execution = summary["execution"]
    require(execution["status"] in {"COMPLETE", "NONFINITE_ROWS_ABSORBED", "NONFINITE_LOGITS_WITH_FINITE_STATE"}
            and execution["completed_writes"] == test["length"] + 1, "incomplete TEST execution")
    require(summary["confidence"]["family_size"] == source_family and
            summary["confidence"]["family_alpha"] == evaluation["alpha"], "wrong source confidence family")
    derived = summarize_first_failures(summary["tau"], test["length"], horizons=evaluation["horizons"],
        sequence_ids=summary["sequence_ids"], family_size=evaluation["family_size"], alpha=evaluation["alpha"],
        epsilons=evaluation["epsilons"], model_seed=seed)
    require(len(summary["horizons"]) == len(derived["horizons"]), "wrong retained horizon grid")
    for original, joint in zip(summary["horizons"], derived["horizons"]):
        for key in ("horizon", "first_failures"):
            require(original[key] == joint[key], f"retained {key} disagrees with tau")
        for key in ("failure_probability", "survival_probability"):
            close(original[key], joint[key], key)
    close(summary["restricted_mean_failure_free_length"], derived["restricted_mean_failure_free_length"], "RMST")
    require(summary["observed_failures"] == derived["observed_failures"] and
            summary["right_censored_sequences"] == derived["right_censored_sequences"], "censoring disagrees with tau")
    for epsilon in evaluation["epsilons"]:
        require(horizon_value(summary, False, epsilon) == horizon_value(derived, False, epsilon),
                "empirical horizon must remain unchanged")
    # Token recovery/accuracy cannot be reconstructed from tau. Keep the source
    # value as a distinct retained statistic, not as a first-failure estimate.
    derived["step_accuracy"] = summary.get("step_accuracy")
    derived["confidence"]["scope"] = "joint descriptive reused-TEST view: 30 arms x 7 horizons x 3 separately reported model seeds"
    ledger = summary["ledger"]
    stream, shared = ledger["per_stream_persistent_bytes"], ledger["shared_bytes"]
    require(type(stream) is int and stream >= 0 and type(shared) is int and shared >= 0, "byte ledger must use nonnegative integers")
    totals = {str(n): shared + n * stream for n in STREAM_COUNTS}
    require(all(ledger["total_bytes"][str(n)] == totals[str(n)] for n in STREAM_COUNTS), "actual byte ledger does not reconcile")
    result = dict(phase=PHASE, model_seed=seed, arm=arm, source_family_size=source_family,
        joint_family_size=evaluation["family_size"], summary=derived, per_stream_bytes=stream, shared_bytes=shared,
        total_bytes=totals, amortized_bytes={str(n): totals[str(n)] / n for n in STREAM_COUNTS},
        rmst=derived["restricted_mean_failure_free_length"], final_survival=derived["horizons"][-1]["survival_probability"],
        execution_status=execution["status"], original_failure_upper_bounds=[p["failure_upper_bound"] for p in summary["horizons"]],
        original_supported_horizons=summary["confidence_supported_T_epsilon_lower_bound"],
        additional_complete_call_timing="NOT_MEASURED")
    for suffix, epsilon in (("05", .05), ("01", .01)):
        result[f"empirical_T{suffix}"] = horizon_value(derived, False, epsilon)
        result[f"supported_T{suffix}"] = horizon_value(derived, True, epsilon)
    return result


def read_run(directory, kind, policy, policy_digest, sources):
    directory = Path(directory)
    # Seed is checked before it enters receipt names or strata.
    preliminary = json.loads((directory / "manifest.json").read_bytes())
    seed = preliminary["model_seed"]
    require(type(seed) is int and seed in policy["model_seeds"], "unexpected model seed")
    label = f"{kind}_seed{seed}"
    manifest = sources.json(directory, "manifest.json", label)
    index = sources.json(directory, "index.json", label)
    index_digest = sources.receipts[f"{label}/index.json"]["sha256"]
    require(sources.read(directory, "index.sha256", label).decode().strip() == index_digest, "index checksum mismatch")
    require(index["manifest"] == manifest, "index and manifest disagree")
    supplement = kind == "supplement"
    names = SUPPLEMENT_ARMS if supplement else ORIGINAL_ARMS
    source_family = 630 if supplement else 546
    require(manifest["phase"] == (PHASE if supplement else "FROZEN_PRIMARY") and index["family_size"] == source_family,
            "input is not the required complete TEST phase")
    identity = next(row for row in policy["seed_identities"] if row["model_seed"] == seed)
    require(manifest["checkpoint_sha256"] == identity["checkpoint_sha256"], "checkpoint differs from policy")
    if supplement:
        require(manifest["status"] == index["status"] == "COMPLETE" and manifest["policy_sha256"] == policy_digest,
                "supplement is incomplete or uses a different policy")
        require(manifest["same_test"] is True and manifest["independent_confirmation"] is False and
                manifest["complete_call_timing"] == "NOT_MEASURED", "supplement interpretation changed")
        sources.json(directory, "budget_supplement_v1.json", label, policy_digest)
        require(sources.json(directory, "timing.json", label)["status"] == "NOT_MEASURED", "supplement timing must be unmeasured")
    else:
        require(manifest["protocol_sha256"] == policy["original_protocol_sha256"], "original protocol changed")
        require(manifest["runtime_sha256"] == policy["runtime_sha256"], "original runtime changed")
        for key in ("calibration_sha256", "token_table_config_sha256", "token_table_data_sha256", "source_sha256"):
            require(manifest[key] == identity[key], f"original identity mismatch: {key}")
    protocol = sources.json(directory, "protocol.json", label, policy["original_protocol_sha256"])
    require(protocol["arms"] == list(ORIGINAL_ARMS), "original 26-arm menu changed")
    sources.read(directory, "evaluation_runtime.json", label, policy["runtime_sha256"])
    inputs = sources.json(directory, "TEST_inputs.json", label, identity["test_inputs_file_sha256"])
    test = policy["test"]
    require([inputs[key] for key in ("group", "split", "seed", "sequences", "group_length")] ==
            [test[key] for key in ("group", "split", "seed", "sequences", "length")], "TEST law changed")
    require(all(inputs[key] == test[key] for key in ("token_sha256", "gold_sha256")), "TEST token/gold hashes changed")
    require(set(index["splits"]["TEST"]) == set(names) and
            {p.stem for p in (directory / "TEST").glob("*.json")} == set(names), "missing, extra, or incomplete arm family")
    records = []
    for arm in names:
        entry = index["splits"]["TEST"][arm]
        relative = f"TEST/{arm}.json"
        require(entry["file"] == relative, "unexpected arm result path")
        summary = sources.json(directory, relative, label, entry["sha256"])
        require(summary["sequence_ids"] == inputs["sequence_ids"], "arm reordered or filtered TEST sequences")
        if supplement:
            require(summary["phase"] == PHASE and summary["supplement_policy_sha256"] == policy_digest and
                    summary["independent_confirmation"] is False, "supplement summary identity mismatch")
        record = reexpress(summary, arm, seed, source_family, policy)
        record.update(source_kind=kind, source_json=f"{label}/{relative}", source_json_sha256=entry["sha256"])
        close(entry["rmst"], record["rmst"], "index RMST")
        close(entry["final_survival"], record["final_survival"], "index survival")
        require(entry["execution"] == record["execution_status"], "index execution mismatch")
        records.append(record)
    return records, dict(model_seed=seed, kind=kind, manifest=manifest, index_sha256=index_digest,
                         manifest_sha256=sources.receipts[f"{label}/manifest.json"]["sha256"])


def combine(learned, supplements, policy, policy_digest, sources):
    require(len(learned) == len(supplements) == len(policy["model_seeds"]), "supply three original and three supplement directories")
    records, metadata = [], {"original": {}, "supplement": {}}
    for kind, directories in (("original", learned), ("supplement", supplements)):
        for directory in directories:
            rows, receipt = read_run(directory, kind, policy, policy_digest, sources)
            seed = receipt["model_seed"]
            require(seed not in metadata[kind], "duplicate model-seed directory; replicates cannot increase n")
            metadata[kind][seed] = receipt
            records.extend(rows)
    for seed in policy["model_seeds"]:
        original, supplement = metadata["original"][seed], metadata["supplement"][seed]
        require(supplement["manifest"]["primary_index_sha256"] == original["index_sha256"] and
                supplement["manifest"]["primary_manifest_sha256"] == original["manifest_sha256"],
                "supplement references a different primary result set")
    records.sort(key=lambda row: (row["model_seed"], row["arm"]))
    return records, metadata


def is_baseline(arm):
    return arm.startswith(("NATIVE_", "UNIFORM_", "MIXED_", "STOCHASTIC_"))


def best_feasible(records, metric, n):
    eligible = [row for row in records if row[metric] is not None]
    return min(eligible, key=lambda row: (-row[metric], -row["rmst"], row["total_bytes"][str(n)], row["arm"])) if eligible else None


def budget_rows(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["model_seed"]].append(row)
    result = []
    for seed, peers in sorted(grouped.items()):
        baselines = [row for row in peers if is_baseline(row["arm"])]
        for candidate in peers:
            grid = [point["horizon"] for point in candidate["summary"]["horizons"]]
            for n in STREAM_COUNTS:
                cap = candidate["total_bytes"][str(n)]
                feasible = [row for row in baselines if row["total_bytes"][str(n)] <= cap]
                row = dict(phase=PHASE, model_seed=seed, candidate=candidate["arm"], N_streams=n,
                    candidate_total_byte_cap=cap, candidate_amortized_bytes=cap / n,
                    candidate_stream_bytes=candidate["per_stream_bytes"], candidate_shared_bytes=candidate["shared_bytes"],
                    family_size=candidate["joint_family_size"], feasible_baseline_count=len(feasible),
                    feasible_baselines=";".join(sorted(p["arm"] for p in feasible)),
                    native_fp32_feasible=any(p["arm"] == "NATIVE_FP32" for p in feasible),
                    baseline_selection_scope=DESCRIPTIVE, tie_rule=TIE_RULE,
                    comparison_scope="lower-bound or point-estimate ordering does not establish statistical superiority",
                    additional_complete_call_timing="NOT_MEASURED", candidate_source_sha256=candidate["source_json_sha256"])
                for metric in ("empirical_T05", "supported_T05", "empirical_T01", "supported_T01", "rmst", "final_survival"):
                    winner = best_feasible(feasible, metric, n)
                    row[f"candidate_{metric}"] = candidate[metric]
                    row[f"best_TEST_{metric}_baseline"] = None if winner is None else winner["arm"]
                    row[f"best_TEST_{metric}"] = None if winner is None else winner[metric]
                    row[f"best_TEST_{metric}_total_bytes"] = None if winner is None else winner["total_bytes"][str(n)]
                    row[f"best_TEST_{metric}_source_sha256"] = None if winner is None else winner["source_json_sha256"]
                    if "T0" in metric:
                        row[f"candidate_{metric}_status"] = horizon_status(candidate[metric], grid, metric.startswith("supported"))
                        row[f"best_TEST_{metric}_status"] = ("no feasible baseline" if not feasible else
                            horizon_status(None if winner is None else winner[metric], grid, metric.startswith("supported")))
                result.append(row)
    return result


def arm_rows(records):
    result = []
    for record in records:
        summary = record["summary"]
        grid = [point["horizon"] for point in summary["horizons"]]
        row = {key: record[key] for key in ("phase", "model_seed", "arm", "source_kind", "source_family_size",
               "joint_family_size", "execution_status", "per_stream_bytes", "shared_bytes", "rmst", "final_survival",
               "source_json", "source_json_sha256", "additional_complete_call_timing")}
        row.update(n_sequences=summary["n_sequences"], max_horizon=summary["max_horizon"],
                   observed_failures=summary["observed_failures"], right_censored_sequences=summary["right_censored_sequences"],
                   final_failure_probability=summary["horizons"][-1]["failure_probability"])
        for metric in ("empirical_T05", "supported_T05", "empirical_T01", "supported_T01"):
            row[metric] = record[metric]
            row[f"{metric}_status"] = horizon_status(record[metric], grid, metric.startswith("supported"))
        for n in STREAM_COUNTS:
            row[f"total_bytes_N{n}"] = record["total_bytes"][str(n)]
            row[f"amortized_bytes_N{n}"] = record["amortized_bytes"][str(n)]
        result.append(row)
    return result


def curve_rows(records):
    return [dict(phase=PHASE, model_seed=record["model_seed"], arm=record["arm"],
        n_sequences=record["summary"]["n_sequences"], source_family_size=record["source_family_size"],
        joint_family_size=record["joint_family_size"], source_failure_upper_bound=original_upper,
        source_json_sha256=record["source_json_sha256"], **point)
        for record in records for original_upper, point in zip(record["original_failure_upper_bounds"], record["summary"]["horizons"])]


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    require(bool(rows), "cannot write an empty report table")
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output, records, metadata, policy_digest, sources):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for relative in ("scripts/summarize_budget_supplement.py", "codec/survival.py"):
        sources.read(ROOT, relative, "summary_source")
    report = dict(schema="case010-budget-supplement-summary-v1", phase=PHASE,
        independent_confirmation=False, same_test=True, original_results_modified=False,
        interpretation=DESCRIPTIVE, no_global_budget_optimality_claim=True,
        model_seed_pooling="forbidden; each arm retains its original sequence denominator",
        primary_family_size_retained=546, joint_family_size=630,
        family_scope="30 arms x 7 horizons x 3 model seeds; re-expressed bounds for all 90 arms",
        budget_rule="candidate cap = shared codec/basis/token-table bytes + N * per-stream bytes; no artificial padding",
        common_weights="common model checkpoint/weight bytes excluded from codec caps and retained separately in source manifests",
        verification_scope="index/source hashes, matched checkpoint/TEST identities, tau-to-summary consistency, and ledger arithmetic; retained correctness arrays are checked by the separate auditor",
        tie_rule=TIE_RULE, additional_complete_call_timing="NOT_MEASURED",
        timing_scope="no timing is imported or substituted in this joint derived view; original timing files remain separate",
        policy_file_sha256=policy_digest, source_hash_file="source_hashes.json", seeds=metadata,
        records=records, outputs=dict(arms="all_arms.csv", horizons="horizon_curves.csv", budgets="budget_comparisons.csv"))
    write_csv(output / "all_arms.csv", arm_rows(records))
    write_csv(output / "horizon_curves.csv", curve_rows(records))
    write_csv(output / "budget_comparisons.csv", budget_rows(records))
    write_json(output / "summary.json", report)
    write_json(output / "source_hashes.json", dict(schema="case010-budget-supplement-summary-sources-v1", phase=PHASE,
        hash_kind="SHA256 of exact file bytes, including any terminal newline", sources=sorted(sources.receipts.values(), key=lambda row: row["file"]),
        outputs=[dict(file=path.name, sha256=sha(path.read_bytes()), size_bytes=path.stat().st_size)
                 for path in sorted(output.iterdir()) if path.is_file()]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned", type=Path, action="append", required=True)
    parser.add_argument("--supplement", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), "output must be a fresh directory")
    require(all(not args.output.resolve().is_relative_to(path.resolve()) for path in args.learned + args.supplement),
            "output must not be inside an input result directory")
    sources = Sources()
    policy, policy_digest = read_policy(sources)
    records, metadata = combine(args.learned, args.supplement, policy, policy_digest, sources)
    write_outputs(args.output, records, metadata, policy_digest, sources)
    print(json.dumps(dict(phase=PHASE, model_seeds=policy["model_seeds"], arms=len(records), family_size=630,
                          additional_complete_call_timing="NOT_MEASURED")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
