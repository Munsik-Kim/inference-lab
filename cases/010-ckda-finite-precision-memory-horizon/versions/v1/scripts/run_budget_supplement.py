"""Evaluate four frozen CAL-ranked allocations on the already-used TEST sample.

One fresh CPU process handles one trained seed. No calibration fit or selection
occurs here. Run only after primary evaluation and both serial cost studies.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import benchmark_isolated as provenance

PHASE = "POST_HOC_BUDGET_AUDIT_SAME_TEST"
SPECS = (
    ("MIXED_4_8_TOP2", 4, 8, 2, (("LOWRANK_4_8_R1", 16), ("LOWRANK_4_8_R1", 128))),
    ("MIXED_4_8_TOP5", 4, 8, 5, (("LOWRANK_4_8_R2", 16),)),
    ("MIXED_6_8_TOP1", 6, 8, 1, (("LOWRANK_4_8_R4", 128),)),
    ("MIXED_6_8_TOP4", 6, 8, 4, (("LOWRANK_4_8_R4", 16),)),
)
NAMES = [row[0] for row in SPECS]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_policy(policy):
    actual = [(a["name"], a["base_bits"], a["promoted_bits"], a["top_per_head"],
               tuple((t["candidate"], t["streams"]) for t in a["targets"])) for a in policy["arms"]]
    if actual != list(SPECS) or policy["schema"] != "case010-budget-supplement-v1" or policy["phase"] != PHASE:
        raise ValueError("Supplement must contain exactly the four frozen allocations")
    if policy["shape"] != [12, 16, 16] or policy["model_seeds"] != [0, 1, 2]:
        raise ValueError("Supplement shape or seed family changed")
    test = policy["test"]
    if any(test[k] != v for k, v in dict(group="S3", split="TEST", seed=3001, sequences=512, length=2048).items()):
        raise ValueError("Supplement must reuse the original TEST law and sample")
    evaluation = policy["evaluation"]
    if (evaluation["horizons"], evaluation["alpha"], evaluation["epsilons"], evaluation["family_size"],
            evaluation["all_arms_all_seeds"], evaluation["all512_no_conditioning"], evaluation["retries"]) != (
            [32, 64, 128, 256, 512, 1024, 2048], .05, [.05, .01], 630, True, True, 0):
        raise ValueError("Supplement evaluation family changed")
    if [s["model_seed"] for s in policy["seed_identities"]] != [0, 1, 2] or any(
            s["lowrank_test_results_present_at_freeze"] for s in policy["seed_identities"]):
        raise ValueError("Invalid pre-lowrank-result freeze declaration")
    if policy["execution"]["cpu_threads"] != 2 or policy["execution"]["no_gpu"] is not True:
        raise ValueError("Supplement must use the frozen two-thread CPU policy")


def mixed_maps(scores):
    """Only fixed CAL scores enter allocation; no predictions or metrics argument."""
    scores = np.asarray(scores)
    if scores.shape != (12, 16) or not np.issubdtype(scores.dtype, np.floating) or not np.isfinite(scores).all():
        raise ValueError("Expected finite [12,16] CAL scores")
    ranks = np.argsort(-scores, axis=1, kind="stable")
    result = {}
    for name, low, high, top, _ in SPECS:
        bits = np.full((12, 16), low, dtype=np.uint8)
        np.put_along_axis(bits, ranks[:, :top], high, axis=1)
        result[name] = bits
    return ranks, result


def validate_primary(policy, inspection):
    validate_policy(policy)
    manifest = inspection["manifest"]
    identity = next(row for row in policy["seed_identities"] if row["model_seed"] == manifest["model_seed"])
    for key in ("checkpoint_sha256", "calibration_sha256", "token_table_config_sha256",
                "token_table_data_sha256", "source_sha256"):
        if manifest[key] != identity[key]:
            raise ValueError(f"Primary differs from the supplement freeze: {key}")
    if (manifest["protocol_sha256"], manifest["runtime_sha256"]) != (
            policy["original_protocol_sha256"], policy["runtime_sha256"]):
        raise ValueError("Primary protocol/runtime differs from supplement freeze")
    if provenance.sha256(provenance.__file__) != policy["benchmark_helper_sha256"]:
        raise ValueError("Frozen evaluator loader changed after supplement freeze")
    primary = Path(inspection["directory"])
    if provenance.sha256(primary/"TEST_inputs.json") != identity["test_inputs_file_sha256"]:
        raise ValueError("Retained TEST identity changed")
    with np.load(primary/"calibration.npz", allow_pickle=False) as stored:
        scores = stored["mixed_scores"]
    ranks, maps = mixed_maps(scores)
    if ranks.tolist() != identity["rankings"] or sha(scores.astype("<f8").tobytes()) != identity["mixed_scores_le_f64_sha256"]:
        raise ValueError("CAL scores/rankings differ from frozen allocation")
    return identity, maps


def completed_cost_gate(directory, schema, inspection):
    """Require all three cost processes to have finished before any forward call."""
    directory = Path(directory)
    index, digest = provenance.frozen_json(directory/"index.json")
    if (index.get("schema"), index.get("status"), index.get("serial"), set(index.get("seeds", {}))) != (
            schema, "COMPLETE", True, {"0", "1", "2"}):
        raise ValueError("All three serial cost runs must complete first")
    selected = inspection["manifest"]["model_seed"]
    for seed, row in index["seeds"].items():
        path = directory/row["file"]
        if not path.resolve().is_relative_to(directory.resolve()) or provenance.sha256(path) != row["sha256"]:
            raise ValueError("Cost completion receipt identity mismatch")
        record = json.loads(path.read_text())
        if record["model_seed"] != int(seed):
            raise ValueError("Cost completion seed mismatch")
        if int(seed) == selected and record["checkpoint_sha256"] != inspection["manifest"]["checkpoint_sha256"]:
            raise ValueError("Cost completion checkpoint mismatch")
    return digest


def build_arms(evaluation, maps, identity, policy, seed, table_bytes):
    arms = {name: evaluation.OnlineAdapter(evaluation.PackedCodec((12, 16, 16), bits=low,
            mixed_bits=maps[name])) for name, low, _, _, _ in SPECS}
    for name, adapter in arms.items():
        if sha(adapter.config_bytes) != identity["codec_sha256"][name]:
            raise ValueError("Supplement serialized allocation differs from freeze")
        for row in policy["budget_rows"]:
            if row["model_seed"] == seed and row["arm"] == name:
                stream, shared = adapter.bytes_per_stream, adapter.shared_bytes + table_bytes
                total = shared + row["streams"]*stream
                if (stream, shared, total, row["candidate_cap_bytes"]-total) != (
                        row["supplement_per_stream_bytes"], row["supplement_shared_bytes"],
                        row["supplement_total_bytes"], row["remaining_gap_bytes"]) or total > row["candidate_cap_bytes"]:
                    raise ValueError("Supplement byte budget differs from freeze")
    return arms


def preserve_inputs(output, primary, config, policy, inspection):
    for name in ("protocol.json", "protocol.sha256", "evaluation_runtime.json", "evaluation_runtime.sha256",
                 "training-result.json", "calibration.npz", "token_coefficients.json", "token_coefficients.bin", "TEST_inputs.json"):
        (output/name).write_bytes((primary/name).read_bytes())
    (output/"budget_supplement_v1.json").write_bytes(config.read_bytes())
    (output/"budget_supplement_v1.sha256").write_bytes(config.with_suffix(".sha256").read_bytes())
    budget_path = config.parent/policy["budget_file"]
    if sha(budget_path.read_bytes()) != policy["budget_file_sha256"]:
        raise ValueError("Frozen budget CSV changed")
    (output/policy["budget_file"]).write_bytes(budget_path.read_bytes())
    (output/"primary-manifest.json").write_bytes((primary/"manifest.json").read_bytes())
    ledgers = {}
    for name in inspection["protocol"]["arms"]:
        summary = json.loads((primary/"DEV"/f"{name}.json").read_text())
        ledgers[name] = {key: summary[key] for key in ("ledger", "config_sha256", "basis_sha256")}
        for source in (primary/"codecs").glob(f"{name}.*"):
            target = output/"primary-codecs"/source.name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(source.read_bytes())
    provenance.write_json(output/"primary-dev-ledgers.json", ledgers)
    for relative in inspection["manifest"]["source_sha256"]:
        target = output/"frozen-source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((primary/inspection["manifest"]["frozen_source_directory"]/relative).read_bytes())
    sources = {}
    for relative in ("scripts/run_budget_supplement.py", "scripts/benchmark_isolated.py",
                     "scripts/audit_records.py", "scripts/audit_budget_supplement.py"):
        raw = (ROOT/relative).read_bytes()
        target = output/"supplement-source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        sources[relative] = sha(raw)
    return sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("primary", "checkpoint", "upstream", "isolated-timing", "calibration-cost", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/budget_supplement_v1.json")
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES", ""):
        raise ValueError("Use an empty CUDA_VISIBLE_DEVICES for this CPU-only run")
    policy, policy_hash = provenance.frozen_json(args.config)
    inspection = provenance.inspect_evaluation(args.primary, args.checkpoint)
    identity, maps = validate_primary(policy, inspection)
    gate_hashes = dict(isolated_timing_index_sha256=completed_cost_gate(args.isolated_timing, "case010-isolated-timing-v1", inspection),
                       calibration_cost_index_sha256=completed_cost_gate(args.calibration_cost, "case010-calibration-cost-plan-v1", inspection))
    # Source isolation is established before creating the model; importing this
    # runner for allocation/audit tests never imports codec or torch modules.
    evaluation = provenance.load_frozen_evaluator(inspection)
    evaluation.torch.set_num_threads(2)
    primary = Path(inspection["directory"])
    seed = inspection["manifest"]["model_seed"]
    config_raw, table_raw = (primary/"token_coefficients.json").read_bytes(), (primary/"token_coefficients.bin").read_bytes()
    arms = build_arms(evaluation, maps, identity, policy, seed, len(config_raw)+len(table_raw))
    args.output.mkdir(parents=True, exist_ok=False)
    sources = preserve_inputs(args.output, primary, args.config, policy, inspection)
    manifest = dict(schema="case010-budget-supplement-result-v1", phase=PHASE, status="STARTED", model_seed=seed,
                    policy_sha256=policy_hash, family_size=630, original_family_size=546,
                    checkpoint_sha256=identity["checkpoint_sha256"], original_protocol_sha256=policy["original_protocol_sha256"],
                    primary_index_sha256=inspection["primary_index_sha256"], primary_manifest_sha256=inspection["manifest_sha256"],
                    source_sha256=inspection["manifest"]["source_sha256"], supplement_source_sha256=sources,
                    primary_dev_ledgers_sha256=provenance.sha256(args.output/"primary-dev-ledgers.json"),
                    started_utc=datetime.now(timezone.utc).isoformat(), fresh_process=True, no_calibration_refit=True,
                    original_results_modified=False, same_test=True, independent_confirmation=False,
                    gpu_used=False, complete_call_timing="NOT_MEASURED", **gate_hashes)
    provenance.write_json(args.output/"manifest.json", manifest)
    folder = args.output/"TEST"
    folder.mkdir()
    (args.output/"codecs").mkdir()
    for name, adapter in arms.items():
        (args.output/"codecs"/f"{name}.json").write_bytes(adapter.config_bytes)
    completed, current = {}, None
    try:
        model = evaluation.create_model(evaluation.load_upstream(args.upstream), checkpoint=args.checkpoint, device="cpu").eval()
        table = evaluation.token_table(model)
        if evaluation.table_bytes(table) != (config_raw, table_raw):
            raise ValueError("Reconstructed token table differs from retained primary bytes")
        batch = evaluation.frozen_sequences("S3", "TEST", 512, 2048, 3001)
        if evaluation.input_identity(batch) != json.loads((primary/"TEST_inputs.json").read_text()):
            raise ValueError("Regenerated TEST inputs differ from the frozen primary sample")
        for name, adapter in arms.items():
            current = name
            summary = evaluation.summarize_arm(model, table, adapter, batch, policy["evaluation"]["horizons"],
                630, seed, .05, (.05, .01), len(config_raw)+len(table_raw),
                correctness_output=folder/f"{name}.correctness.npz")
            summary.update(phase=PHASE, supplement_policy_sha256=policy_hash, independent_confirmation=False,
                           target_budgets=[row for row in policy["budget_rows"] if row["model_seed"] == seed and row["arm"] == name])
            path = folder/f"{name}.json"
            provenance.write_json(path, summary)
            completed[name] = dict(file=f"TEST/{name}.json", sha256=provenance.sha256(path),
                execution=summary["execution"]["status"], final_survival=summary["horizons"][-1]["survival_probability"],
                rmst=summary["restricted_mean_failure_free_length"])
            print(json.dumps(dict(model_seed=seed, arm=name, phase=PHASE, **completed[name])), flush=True)
    except Exception as exc:
        provenance.write_json(args.output/"failure.json", dict(status="FAILED_NO_RETRY", model_seed=seed,
            arm=current, error_type=type(exc).__name__, completed_arms=list(completed), all_records_retained=True))
        raise
    manifest.update(status="COMPLETE", finished_utc=datetime.now(timezone.utc).isoformat())
    provenance.write_json(args.output/"manifest.json", manifest)
    provenance.write_json(args.output/"timing.json", dict(status="NOT_MEASURED", phase=PHASE,
        reason="No complete-call timing collected for supplemental allocations; primary-arm times are not substituted."))
    provenance.write_json(args.output/"index.json", dict(schema=manifest["schema"], phase=PHASE, family_size=630,
        status="COMPLETE", manifest=manifest, splits={"TEST": completed}))
    (args.output/"index.sha256").write_text(provenance.sha256(args.output/"index.json")+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
