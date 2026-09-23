"""CPU-only independent audit of supplementary bytes and retained correctness.

Does not import the runner, model, codec, checkpoint, or survival implementation.
CAL rankings are reconstructed with Python sorting, and scalar/CP checks reuse
the independent retained-record auditor. Receipt paths are logical and relative.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_records import (Artifacts, HORIZONS, LEARNED_ARMS, canonical, check_online_ledger,
    check_summary, check_token_table, close, correctness, hash_id, require, sha)

PHASE = "POST_HOC_BUDGET_AUDIT_SAME_TEST"
ALLOCATIONS = {"MIXED_4_8_TOP2": (4, 2, [("LOWRANK_4_8_R1",16),("LOWRANK_4_8_R1",128)]),
               "MIXED_4_8_TOP5": (4, 5, [("LOWRANK_4_8_R2",16)]),
               "MIXED_6_8_TOP1": (6, 1, [("LOWRANK_4_8_R4",128)]),
               "MIXED_6_8_TOP4": (6, 4, [("LOWRANK_4_8_R4",16)])}


def audit_supplement(directory):
    preliminary = Artifacts(directory, "supplement")
    seed = preliminary.json("manifest.json")["model_seed"]
    require(type(seed) is int and seed in (0,1,2), "supplement: invalid seed")
    store = Artifacts(directory, f"budget_supplement_seed{seed}")
    manifest = store.json("manifest.json")
    require(manifest["phase"] == PHASE and manifest["status"] == "COMPLETE", "supplement: incomplete or wrong phase")
    require(manifest["same_test"] is True and manifest["independent_confirmation"] is False and
            manifest["no_calibration_refit"] is True and manifest["original_results_modified"] is False,
            "supplement: interpretation or no-refit boundary changed")
    require(manifest["family_size"] == 630 and manifest["original_family_size"] == 546,
            "supplement: confidence family changed")
    policy = store.json("budget_supplement_v1.json", manifest["policy_sha256"])
    require(store.read("budget_supplement_v1.sha256").decode().strip() == manifest["policy_sha256"],
            "supplement: policy checksum mismatch")
    require(policy["phase"] == PHASE and policy["schema"] == "case010-budget-supplement-v1" and
            policy["model_seeds"] == [0,1,2] and policy["shape"] == [12,16,16], "supplement: policy scope mismatch")
    require(policy["evaluation"]["family_size"] == 630 and policy["evaluation"]["horizons"] == HORIZONS,
            "supplement: policy family/horizons mismatch")
    require([a["name"] for a in policy["arms"]] == list(ALLOCATIONS), "supplement: hidden arm selection")
    for definition in policy["arms"]:
        low, top, targets = ALLOCATIONS[definition["name"]]
        require((definition["base_bits"],definition["promoted_bits"],definition["top_per_head"],
                [(t["candidate"],t["streams"]) for t in definition["targets"]]) == (low,8,top,targets),
                "supplement: allocation or target cap changed")
    identities = policy["seed_identities"]
    require([row["model_seed"] for row in identities] == [0,1,2] and all(
        row["lowrank_test_results_present_at_freeze"] is False and not any(
            name.startswith("LOWRANK_") for name in row["completed_test_arm_names_at_freeze"]) for row in identities),
        "supplement: pre-lowrank-result freeze identity invalid")
    identity = identities[seed]
    primary = store.json("primary-manifest.json",manifest["primary_manifest_sha256"])
    require(primary["model_seed"] == seed and primary["checkpoint_training_status"] == "TRAINING_COMPLETE" and
            primary["checkpoint_updates"] == 20000, "supplement: primary checkpoint not final")
    for key in ("checkpoint_sha256","calibration_sha256","token_table_config_sha256","token_table_data_sha256","source_sha256"):
        require(primary[key] == identity[key], f"supplement: frozen primary identity mismatch: {key}")
    require(manifest["checkpoint_sha256"] == identity["checkpoint_sha256"] and
            manifest["source_sha256"] == identity["source_sha256"], "supplement: evaluated identity mismatch")
    store.json("protocol.json",policy["original_protocol_sha256"])
    store.json("evaluation_runtime.json",policy["runtime_sha256"])
    require(primary["protocol_sha256"] == policy["original_protocol_sha256"] and
            primary["runtime_sha256"] == policy["runtime_sha256"], "supplement: primary protocol changed")
    store.source_files(primary["source_sha256"])
    required = {"scripts/run_budget_supplement.py","scripts/benchmark_isolated.py",
                "scripts/audit_records.py","scripts/audit_budget_supplement.py"}
    require(set(manifest["supplement_source_sha256"]) == required, "supplement: wrapper source set changed")
    for name, digest in manifest["supplement_source_sha256"].items():
        store.read(f"supplement-source/{name}", digest)
    require(manifest["supplement_source_sha256"]["scripts/benchmark_isolated.py"] == policy["benchmark_helper_sha256"],
            "supplement: evaluator loader changed")
    for key in ("primary_index_sha256","primary_manifest_sha256","isolated_timing_index_sha256","calibration_cost_index_sha256"):
        hash_id(manifest[key], f"supplement {key}")
    training = store.json("training-result.json")
    require(training["checkpoint_sha256"] == identity["checkpoint_sha256"] and training["seed"] == seed and
            training["completed_updates"] == 20000 and training["status"] == "TRAINING_COMPLETE", "supplement: training identity mismatch")
    table_size = check_token_table(store.read("token_coefficients.json",identity["token_table_config_sha256"]),
                                  store.read("token_coefficients.bin",identity["token_table_data_sha256"]))
    require(table_size == primary["shared_token_table_bytes"], "supplement: token table ledger mismatch")
    inputs = store.json("TEST_inputs.json",identity["test_inputs_file_sha256"])
    require([inputs[key] for key in ("group","split","seed","sequences","group_length")] == ["S3","TEST",3001,512,2048],
            "supplement: wrong TEST denominator or law")
    for key in ("token_sha256","gold_sha256"):
        require(inputs[key] == policy["test"][key], "supplement: changed TEST sample")
    raw_cal = store.read("calibration.npz",identity["calibration_sha256"])
    with np.load(io.BytesIO(raw_cal),allow_pickle=False) as archive:
        scores = archive["mixed_scores"]
    require(scores.shape == (12,16) and np.isfinite(scores).all(), "supplement: invalid CAL scores")
    require(sha(scores.astype("<f8").tobytes()) == identity["mixed_scores_le_f64_sha256"], "supplement: score byte identity mismatch")
    rankings = [sorted(range(16),key=lambda key:(-float(row[key]),key)) for row in scores]
    require(rankings == identity["rankings"], "supplement: ranking or tie order mismatch")
    csv_raw = store.read(policy["budget_file"],policy["budget_file_sha256"])
    csv_rows = list(csv.DictReader(io.StringIO(csv_raw.decode())))
    require(csv_rows == [{key:str(value) for key,value in row.items()} for row in policy["budget_rows"]],
            "supplement: CSV/JSON budgets disagree")
    expected_targets = {(s,name,c,n) for s in (0,1,2) for name,(_,_,targets) in ALLOCATIONS.items() for c,n in targets}
    require(len(policy["budget_rows"]) == 15 and {(r["model_seed"],r["arm"],r["candidate"],r["streams"])
        for r in policy["budget_rows"]} == expected_targets, "supplement: incomplete or duplicate budget targets")
    references = store.json("primary-dev-ledgers.json",manifest["primary_dev_ledgers_sha256"])
    require(set(references) == set(LEARNED_ARMS), "supplement: primary ledger family incomplete")
    for name, row in references.items():
        config = store.read(f"primary-codecs/{name}.json",row["config_sha256"])
        basis = b"" if json.loads(config)["basis_shape"] is None else store.read(f"primary-codecs/{name}.basis.bin",row["basis_sha256"])
        require(sha(basis) == row["basis_sha256"], "supplement: primary basis identity mismatch")
        check_online_ledger(row["ledger"],config,basis,count=128,table_bytes=table_size)
    index = store.json("index.json")
    require(store.read("index.sha256").decode().strip() == sha(store.read("index.json")), "supplement: index checksum mismatch")
    require(index["manifest"] == manifest and index["family_size"] == 630 and index["phase"] == PHASE and
            index["status"] == "COMPLETE", "supplement: index identity mismatch")
    require(set(index["splits"]) == {"TEST"} and set(index["splits"]["TEST"]) == set(ALLOCATIONS) and
            {p.stem for p in (store.root/"TEST").glob("*.json")} == set(ALLOCATIONS), "supplement: missing or extra TEST arms")
    results = {}
    for name,(low,top,_) in ALLOCATIONS.items():
        entry = index["splits"]["TEST"][name]
        require(entry["file"] == f"TEST/{name}.json", "supplement: unsafe result reference")
        summary = store.json(entry["file"],entry["sha256"])
        require(summary["phase"] == PHASE and summary["supplement_policy_sha256"] == manifest["policy_sha256"] and
                summary["independent_confirmation"] is False and summary["model_seed"] == seed and
                summary["sequence_ids"] == inputs["sequence_ids"], "supplement: summary identity mismatch")
        artifact = summary["correctness_artifact"]
        require(artifact["file"] == f"{name}.correctness.npz" and artifact["shape"] == [512,2048] and
                artifact["bitorder"] == "little" and artifact["bos_included"] is False, "supplement: correctness metadata mismatch")
        correct = correctness(store,f"TEST/{artifact['file']}",expected_sha=artifact["sha256"])
        results[name] = check_summary(summary,correct,family_size=630)
        config_raw = store.read(f"codecs/{name}.json",summary["config_sha256"])
        require(sha(config_raw) == identity["codec_sha256"][name], "supplement: allocation identity differs from freeze")
        config = json.loads(config_raw)
        expected_map = [[8 if key in ranking[:top] else low for key in range(16)] for ranking in rankings]
        require(config["state"]["shape"] == [12,16,16] and config["state"]["bits"] == low and
                config["state"]["mixed_bits"] == expected_map and config["state"]["stochastic"] is False and
                config["residual"] is None and config["basis_shape"] is None, "supplement: wrong fixed CAL allocation")
        require(summary["basis_sha256"] == sha(b""), "supplement: unexpected residual basis")
        check_online_ledger(summary["ledger"],config_raw,b"",table_bytes=table_size)
        rows = [row for row in policy["budget_rows"] if row["model_seed"] == seed and row["arm"] == name]
        require(summary["target_budgets"] == rows, "supplement: target budget selection changed")
        for row in rows:
            n = str(row["streams"])
            cap = references[row["candidate"]]["ledger"]["total_bytes"][n]
            bit,total = max((b,references[f"UNIFORM_{b}"]["ledger"]["total_bytes"][n]) for b in range(2,17)
                            if references[f"UNIFORM_{b}"]["ledger"]["total_bytes"][n] <= cap)
            ledger = summary["ledger"]
            require((cap,bit,total,cap-total,ledger["per_stream_persistent_bytes"],ledger["shared_bytes"],ledger["total_bytes"][n],
                     cap-ledger["total_bytes"][n]) == tuple(row[key] for key in ("candidate_cap_bytes","highest_feasible_uniform_bits",
                     "uniform_total_bytes","original_gap_bytes","supplement_per_stream_bytes","supplement_shared_bytes",
                     "supplement_total_bytes","remaining_gap_bytes")) and row["remaining_gap_bytes"] >= 0,
                     "supplement: byte gap or target feasibility mismatch")
        close(entry["rmst"],summary["restricted_mean_failure_free_length"],"supplement index RMST")
        close(entry["final_survival"],summary["horizons"][-1]["survival_probability"],"supplement index survival")
        require(entry["execution"] == summary["execution"]["status"], "supplement: execution status mismatch")
        hash_id(summary["final_payload_sha256"],"supplement final payload")
    timing = store.json("timing.json")
    require(timing["status"] == "NOT_MEASURED" and manifest["complete_call_timing"] == "NOT_MEASURED",
            "supplement: unmeasured complete-call timing must be explicit")
    return dict(model_seed=seed,phase=PHASE,verified_test_arms=4,sequences_per_arm=512,family_size=630,
        checkpoint_bytes_included=False,checkpoint_note="Identity verified against retained training result; no checkpoint needed for this scalar audit.",
        final_payload_note="Hash IDs and storage formulas verified; final-state byte arrays are not retained.",
        arms=results,artifacts=list(store.receipts.values()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipts = [audit_supplement(path) for path in args.supplement]
    require(len({row["model_seed"] for row in receipts}) == len(receipts), "supplement: duplicate seed directory")
    result = dict(schema="case010-budget-supplement-audit-v1",status="PASS",auditor_sha256=sha(Path(__file__).read_bytes()),
        scalar_helper_sha256=sha((ROOT/"scripts/audit_records.py").read_bytes()),phase=PHASE,seeds=receipts)
    with args.output.open("x") as output:
        output.write(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status="PASS",seeds=[r["model_seed"] for r in receipts],verified_test_arms=4*len(receipts))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
