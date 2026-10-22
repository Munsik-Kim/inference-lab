"""P2 registration/gate/accounting only. P1 scientific files remain unchanged."""
import hashlib
import json
from pathlib import Path
from .common import CASE_ROOT, digest, read_json
from .generator import check_p2_gate


def file_row(path, root=None):
    path = Path(path)
    return {"path": str(path.relative_to(root)) if root else str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check_files(root, rows):
    for row in rows:
        p = Path(root) / row["path"]
        if not p.is_file() or file_row(p)["sha256"] != row["sha256"] or p.stat().st_size != row["bytes"]:
            raise ValueError("SOURCE_OR_INPUT_CHANGED:" + str(p))
    return True


def planned_keys():
    import random
    protocol = read_json(CASE_ROOT / "specs/protocol.json")
    rng = random.Random(protocol["seeds"]["P2_order"])
    planned = []
    for kv in protocol["future_execution"]["KV_blocks"]:
        block = [{"task_id": f"HOLDOUT-{family}{i:02d}", "kv": kv, "policy": policy,
                  "episode_id": f"HOLDOUT-{family}{i:02d}_{kv}_{policy}"}
                 for family in "ABC" for i in range(8) for policy in ("R0", "R1")]
        rng.shuffle(block)
        planned += block
    return planned


def validate_gate(receipt):
    check_p2_gate(receipt)
    for name in ("independent_audit_verified", "P1_preservation_verified", "registered_generator_verified"):
        if receipt.get(name) is not True:
            raise ValueError("P2_GATE_BLOCKED:" + name)
    return True


def p1_preconditions(parent):
    parent = Path(parent)
    run = CASE_ROOT / "results/p1_r5"
    freeze = read_json(run / "execution_freeze.json")
    check_files(CASE_ROOT, freeze["own"]["files"])
    for r in freeze["readonly_dependencies"]:
        check_files("/", [r])
    manifest = read_json(run / "run_manifest.json")
    expected = {f"DEV-{fam}{i:02d}_{kv}_{p}" for fam in "ABC" for i in range(2)
                for kv in ("K0", "K1") for p in ("R0", "R1")}
    if (manifest["execution_status"] != "COMPLETE" or set(manifest["ended"]) != expected
            or len(manifest["episodes"]) != 24 or manifest["missing"] or manifest["errors"]):
        raise ValueError("P2_GATE_BLOCKED:P1 matrix")
    audit = read_json(parent / "p1_final_audit.json")
    if (audit["status"] != "PASS" or audit["episodes"] != 24 or audit["endpoint_differences"] != 0
            or audit["raw_trace_differences"] != 0 or audit["source_sha256"] != freeze["execution_source_sha256"]):
        raise ValueError("P2_GATE_BLOCKED:P1 audit")
    delivery = read_json(parent / "final_delivery_verification.json")
    if delivery["status"] != "PASS" or not all(delivery["checks"].values()):
        raise ValueError("P2_GATE_BLOCKED:P1 preservation")
    raw = read_json(parent / "P1_RAW_MANIFEST.json")
    if len(raw["files"]) != 343 or sum(r["bytes"] for r in raw["files"]) != 111584798:
        raise ValueError("P2_GATE_BLOCKED:P1 raw identity")
    check_files(CASE_ROOT, raw["files"])
    protocol = read_json(CASE_ROOT / "specs/protocol.json")
    if protocol["gates"]["model_success_threshold"] is not None:
        raise ValueError("unexpected numeric gate")
    receipt = {"DEV24_complete": True, "completed_episode_count": 24,
        "execution_source_frozen": True, "execution_source_sha256": freeze["execution_source_sha256"],
        "actual_KV_verified": all(g["G3_actual_KV_finite"] == "PASS" for g in manifest["technical_gates"].values()),
        "grammar_verified": all(g["G2_actual_mask"] == "PASS" for g in manifest["technical_gates"].values()),
        "input_identity_verified": all(g["G1_model_input"] == "PASS" for g in manifest["technical_gates"].values()),
        "independent_audit_verified": True, "P1_preservation_verified": True,
        "registered_generator_verified": True, "model_success_threshold": None,
        "generator_sha256": file_row(CASE_ROOT / "src/agent_state_recovery/generator.py")["sha256"]}
    validate_gate(receipt)
    return receipt


def parent_debit(parent):
    parent = Path(parent)
    ledger = [json.loads(s) for s in (parent / "worker_ledger.jsonl").read_text().splitlines()]
    if sum(r["type"] == "WORKER_STARTED" for r in ledger) != sum(r["type"] == "WORKER_STOPPED" for r in ledger):
        raise ValueError("unclean parent ledger")
    debit = sum(r["elapsed_seconds"] for r in ledger if r["type"] == "WORKER_STOPPED")
    if abs(debit - read_json(parent / "cost_verification.json")["worker_wall_all_attempts_seconds"]) > 1e-8:
        raise ValueError("parent debit mismatch")
    return debit


def remaining_budget(parent_seconds, current_seconds=0):
    if parent_seconds < 0 or current_seconds < 0:
        raise ValueError("negative debit")
    return max(0.0, 3600 - parent_seconds - current_seconds)


def validate_runtime(runtime, parent_runtime):
    # Only process-owned paths and a vacant port change; scientific options do not.
    ignored = {"private_root", "port"}
    if {k: v for k, v in runtime.items() if k not in ignored} != {k: v for k, v in parent_runtime.items() if k not in ignored}:
        raise ValueError("P1 runtime semantic configuration changed")
    return True


def request_ceiling(rows, category):
    submitted = [r for r in rows if r["type"] == "REQUEST_SUBMITTED"]
    limits = {"TECH": 8, "HOLDOUT": 1536}
    if category not in limits or sum(r["category"] == category for r in submitted) >= limits[category] or len(submitted) >= 1544:
        raise ValueError("registered P2 request ceiling")
    return True


def complete_token_sum(values):
    return sum(values) if values and all(v is not None for v in values) else None


def scenario_group(task):
    order = task["public_request"]["record_ids"]
    incident = task["error_state"]["ledger"][task["public_request"]["incident_operation_id"]]
    # Family/template is shared by design; remove ID spelling to test distinct
    # numeric scenario instances, not merely different task names.
    return digest({"family": task["family"], "origin_quantities": [task["origin_state"]["records"][k]["quantity"] for k in order],
                   "incident_quantity": incident["quantity"], "phase": incident["phase"]})
