"""Explicit implementation errata, retaining every original Phase A attempt.

Re-execute only two batch-aborted numerical-failure arms and five stochastic
arms whose neighboring seed streams were correlated in v1.  No input, CAL
parameter, horizon, arm, or threshold is selected or changed from TEST results.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE))

from codec.groups import frozen_sequences
from codec.survival import first_failure_times, summarize_first_failures
from codec.toy import ToyCondition, canonical_json, evaluate_arm, make_adapter


FAILED_V1 = {("s4_rotated", "uniform_2"), ("c31_moving_plane", "uniform_2")}
RNG_V1 = "splitmix64_seed_next_counter_le_u64"
RNG_V2 = "splitmix64_seedkey_v2_next_counter_le_u64"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_correctness(path):
    with np.load(path, allow_pickle=False) as data:
        shape = tuple(int(value) for value in data["shape"])
        correct = np.unpackbits(data["packed"], axis=1, count=shape[1], bitorder="little").astype(bool)
    if correct.shape != shape:
        raise ValueError("invalid packed correctness shape")
    return correct


def condition_from_frozen(specification):
    return ToyCondition(specification["name"], specification["group"],
                        np.asarray(specification["operators"], dtype=np.float64),
                        np.asarray(specification["prototypes"], dtype=np.float64), specification["schedule"])


def validate_arm_configuration(condition, arm, calibration, original):
    expected = copy.deepcopy(original)
    if arm == "stochastic_4":
        if expected["state"]["rng"] != RNG_V1:
            raise ValueError("repair expects the archived v1 stochastic seed generator")
        expected["state"]["rng"] = RNG_V2
    actual = json.loads(make_adapter(condition, arm, calibration).config_bytes)
    if actual != expected:
        raise ValueError(f"unapproved configuration change for {condition.name}/{arm}")
    return actual


def repair(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    protocol_file = (source / "protocol-frozen.json").read_bytes()
    old_protocol = json.loads(protocol_file)
    canonical_protocol_sha256 = digest(canonical_json(old_protocol))
    if canonical_protocol_sha256 != (source / "protocol-frozen.sha256").read_text().strip():
        raise ValueError("original canonical protocol identity does not match")
    rows = read_rows(source / "results.jsonl")
    observed_failures = {(row["condition"], row["arm"]) for row in rows if row["status"] == "error"}
    if observed_failures != FAILED_V1:
        raise ValueError("repair is restricted to the two archived failed arms")
    selected = [row for row in rows if (row["condition"], row["arm"]) in FAILED_V1 or row["arm"] == "stochastic_4"]
    if len(selected) != 7 or any(row["split"] != "TEST" for row in selected):
        raise ValueError("expected exactly two failed and five stochastic TEST arms")
    conditions = {spec["name"]: condition_from_frozen(spec) for spec in old_protocol["conditions"]}
    config = old_protocol["splits"]["TEST"]
    if config != {"n": 512, "horizon": 2048, "seed": 3001} or old_protocol["family_size"] != 875:
        raise ValueError("repair must retain the original primary split and comparison family")
    if digest((CASE / "codec" / "groups.py").read_bytes()) != old_protocol["code_sha256"]["groups.py"]:
        raise ValueError("input/group generator source must remain identical")
    frozen_configs = {}
    for condition in conditions.values():
        calibration = old_protocol["calibrations"][condition.name]
        basis = np.asarray(calibration["basis"], dtype="<f4")[None, :, :].tobytes()
        if digest(basis) != calibration["basis_bytes_sha256"]:
            raise ValueError("CAL basis bytes changed")
        for arm in old_protocol["arms"]:
            frozen_configs[f"{condition.name}/{arm}"] = validate_arm_configuration(
                condition, arm, calibration, old_protocol["arm_configurations"][condition.name][arm])
    # Generation is not model evaluation; exact input identities are frozen
    # before the first repaired TEST transition is executed.
    batches = {group: frozen_sequences(group, "TEST", config["n"], config["horizon"], seed=config["seed"])
               for group in sorted({condition.group_name for condition in conditions.values()})}
    input_identities = {group: {"tokens_sha256": digest(batch.tokens.astype("<i8").tobytes()),
                                "gold_sha256": digest(batch.gold.astype("<i8").tobytes()),
                                "sequence_ids_sha256": digest(canonical_json(batch.sequence_ids)), **config}
                        for group, batch in batches.items()}
    source_files = {f"codec/{name}": CASE / "codec" / name
                    for name in ("toy.py", "online.py", "packed.py", "groups.py", "survival.py")}
    source_files["scripts/repair_toys.py"] = Path(__file__)
    source_bytes = {name: path.read_bytes() for name, path in source_files.items()}
    previous_results_bytes = (source / "results.jsonl").read_bytes()
    prior_attempt_id = "phase-a-v1:" + canonical_protocol_sha256
    identity = {
        "schema": "case010-toy-implementation-errata-v1b-v2",
        "prior_attempt_id": prior_attempt_id,
        "canonical_protocol_sha256": canonical_protocol_sha256,
        "canonical_protocol_hash_kind": "SHA256 of canonical JSON without terminal newline",
        "protocol_file_sha256": digest(protocol_file),
        "protocol_file_hash_kind": "SHA256 of exact original protocol-frozen.json file bytes including newline",
        "prior_results_file_sha256": digest(previous_results_bytes),
        "old_protocol_unchanged": True,
        "reason": "executor violation of frozen finite-output policy and stochastic generator neighboring-seed correlation; implementation correction, no scientific selection or tuning",
        "changes": ["retain row-local first failure and continue healthy rows after nonrepresentable state",
                    "FP64 diagnostic reductions, with numerical terminals excluded from finite-value MSE",
                    "empirical sample horizon wording separated from confidence-supported population statement",
                    "stochastic v2 uses SplitMix64(SplitMix64(seed)+counter+index); seed/counter bytes unchanged"],
        "unchanged": ["input sequences and seeds", "CAL covariance/basis/ranking", "all arms and thresholds",
                      "all horizons and comparison family", "state bit layouts and scale/rounding algorithms except stochastic seed key"],
        "reevaluated_arms": [{"condition": row["condition"], "arm": row["arm"], "split": "TEST",
                              "version": "v2-stochastic-seedkey" if row["arm"] == "stochastic_4" else "v1b-row-local-numerical-failure"}
                             for row in selected],
        "derived_primary_composition": {"unchanged_v1_finite_nonstochastic": 118,
                                         "v1b_numerical_failure_repairs": 2, "v2_stochastic_reruns": 5},
        "family_size": old_protocol["family_size"], "inputs": input_identities,
        "arm_configurations": frozen_configs,
        "calibration_identity_file_sha256": digest((source / "calibration-identity.json").read_bytes()),
        "new_source_file_sha256": {name: digest(content) for name, content in source_bytes.items()},
        "identity_hash_kind": "SHA256 of exact erratum-identity.json file bytes including terminal newline",
    }
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("frozen-source", "correctness", "shared"):
        (output / directory).mkdir()
    for name, content in source_bytes.items():
        (output / "frozen-source" / name.replace("/", "--")).write_bytes(content)
    identity_bytes = canonical_json(identity) + b"\n"
    identity_hash = digest(identity_bytes)
    (output / "erratum-identity.json").write_bytes(identity_bytes)
    (output / "erratum-identity.sha256").write_text(identity_hash + "\n")
    (output / "original-protocol-frozen.json").write_bytes(protocol_file)
    with (output / "prior-attempts.jsonl").open("x") as old_log:
        for row in selected:
            old_log.write(json.dumps({"prior_attempt_id": prior_attempt_id, "result": row}, separators=(",", ":"), allow_nan=False) + "\n")
    started = time.monotonic()
    repaired = {}
    with (output / "rerun-results.jsonl").open("x") as results, (output / "rerun-tau.jsonl").open("x") as taus:
        for previous in selected:
            name, arm = previous["condition"], previous["arm"]
            condition = conditions[name]
            batch = batches[condition.group_name]
            result, correct, shared = evaluate_arm(condition, arm, old_protocol["calibrations"][name], batch,
                                                  old_protocol["family_size"], old_protocol["horizons"], old_protocol["alpha"])
            key = f"TEST--{name}--{arm}"
            version = "v2-stochastic-seedkey" if arm == "stochastic_4" else "v1b-row-local-numerical-failure"
            result.update(protocol_sha256=canonical_protocol_sha256, erratum_identity_sha256=identity_hash,
                          prior_attempt_id=prior_attempt_id, evaluation_version=version,
                          previous_result_sha256=digest(canonical_json(previous)), summary_version="v1b-empirical-wording")
            np.savez_compressed(output / "correctness" / f"{key}.npz", packed=np.packbits(correct, axis=1, bitorder="little"),
                                shape=np.asarray(correct.shape, dtype=np.int64))
            (output / "shared" / f"{key}.bin").write_bytes(shared)
            results.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
            results.flush()
            taus.write(json.dumps({"condition": name, "arm": arm, "split": "TEST", "sequence_ids": batch.sequence_ids,
                                   "tau": result["summary"]["tau"], "max_horizon": config["horizon"],
                                   "numerical_status": result["numerical_status"], "evaluation_version": version,
                                   "prior_attempt_id": prior_attempt_id, "erratum_identity_sha256": identity_hash}, separators=(",", ":")) + "\n")
            taus.flush()
            repaired[(name, arm)] = result
            print(json.dumps({"condition": name, "arm": arm, "evaluation_version": version,
                              "first_failures": result["summary"]["observed_failures"],
                              "numerical_terminals": result["numerical_status"]["n_terminal_sequences"]}), flush=True)
    with (output / "derived-results.jsonl").open("x") as derived, (output / "derived-tau.jsonl").open("x") as taus:
        for original_row_index, previous in enumerate(rows, start=1):
            pair = (previous["condition"], previous["arm"])
            result = copy.deepcopy(repaired.get(pair, previous))
            key = f"{result['split']}--{result['condition']}--{result['arm']}"
            owner = output if pair in repaired else source
            result["raw_artifact_owner"] = os.path.relpath(owner, output)
            result["original_result_row_1based"] = original_row_index
            if result["arm"] != "exact_symbolic_id":
                correct = load_correctness(owner / "correctness" / f"{key}.npz")
                observed_tau = first_failure_times(correct)
                if observed_tau != result["summary"]["tau"]:
                    raise ValueError(f"raw correctness/tau mismatch: {key}")
                result["summary"] = summarize_first_failures(observed_tau, config["horizon"], horizons=old_protocol["horizons"],
                    sequence_ids=result["summary"]["sequence_ids"], family_size=old_protocol["family_size"], alpha=old_protocol["alpha"],
                    epsilons=old_protocol["epsilons"], correct=correct)
                result.setdefault("evaluation_version", "v1-unchanged-finite-nonstochastic")
                result["summary_version"] = "v1b-empirical-wording"
                result["erratum_identity_sha256"] = identity_hash
                result["runtime_scope"] = "CPU diagnostic-loop timing; production complete-call not measured"
                taus.write(json.dumps({"condition": result["condition"], "arm": result["arm"], "split": result["split"],
                                       "tau": observed_tau, "sequence_ids": result["summary"]["sequence_ids"],
                                       "evaluation_version": result["evaluation_version"], "raw_artifact_owner": result["raw_artifact_owner"]},
                                      separators=(",", ":")) + "\n")
            else:
                result["evaluation_version"] = "v1-symbolic-capacity-control"
            derived.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    if (source / "results.jsonl").read_bytes() != previous_results_bytes or (source / "protocol-frozen.json").read_bytes() != protocol_file:
        raise ValueError("original attempt changed during repair")
    manifest = {"status": "complete", "erratum_identity_sha256": identity_hash,
                "identity_hash_kind": "exact file bytes including newline", "canonical_protocol_sha256": canonical_protocol_sha256,
                "protocol_file_sha256": digest(protocol_file), "prior_attempt_id": prior_attempt_id,
                "reevaluated_arms": len(repaired), "derived_primary_arms": 125, "capacity_controls": 5,
                "original_attempt_preserved": True, "runtime_seconds": time.monotonic() - started,
                "evaluation_versions": identity["derived_primary_composition"]}
    (output / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(repair(args.source, args.output), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
