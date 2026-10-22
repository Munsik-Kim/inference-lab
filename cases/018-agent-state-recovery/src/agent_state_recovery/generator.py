"""Deterministic DEV inputs and a gated future HOLDOUT generator (never run in P0)."""
import copy
import random
from .common import CASE_ROOT, digest, read_json
from .world import World, business


def check_p2_gate(receipt):
    required = ("DEV24_complete", "execution_source_frozen", "actual_KV_verified",
                "grammar_verified", "input_identity_verified")
    missing = [key for key in required if receipt.get(key) is not True]
    if missing:
        raise ValueError("P2_GATE_BLOCKED:" + ",".join(missing))
    if receipt.get("completed_episode_count") != 24 or not receipt.get("execution_source_sha256"):
        raise ValueError("P2_GATE_BLOCKED:incomplete identity/count")
    # A recorded success count, including zero, is deliberately not a threshold.
    return True


def _task(split, family, index, rng):
    token = f"{rng.getrandbits(32):08x}"
    tid = f"{split}-{family}{index:02d}"
    ids = [f"r-{token}-{suffix}" for suffix in ("s", "d", "x", "y", "u")]
    src, dst, other_src, other_dst, unrelated = ids
    quantities = [rng.randint(40, 70), rng.randint(10, 25), rng.randint(30, 45), rng.randint(15, 30), 50]
    records = {k: {"quantity": quantity, "sku": "SKU-M" if i < 2 else "SKU-X" if i < 4 else "SKU-U",
                   "warehouse": f"W{i % 2}", "owner": "synthetic-owner", "protected": i == 4}
               for i, (k, quantity) in enumerate(zip(ids, quantities))}
    origin = {"scope_id": f"scope-{token}", "revision": 0, "records": records, "ledger": {},
              "dedup": {}, "active_operations": {},
              "stock_totals": {"SKU-M": sum(quantities[:2]), "SKU-X": sum(quantities[2:4]), "SKU-U": 50},
              "audit": [{"audit_id": "h000", "kind": "OPENING", "records": copy.deepcopy(records)}]}
    q = rng.randint(2, 6)
    normal = {"operation_id": f"delivery-{token}", "source_id": src, "destination_id": dst, "quantity": q}
    next_op = {"operation_id": f"next-{token}", "source_id": src, "destination_id": dst, "quantity": 1}
    shell = {"origin_state": origin}
    world = World(shell, origin)
    if family == "B":
        world.execute({"tool": "apply_transfer", "arguments": {**normal, "expected_revision": 0}}, 0, [])
    checkpoint = world.snapshot()
    fault_id = f"incident-{token}"
    error = copy.deepcopy(checkpoint)
    fsrc, fdst = (other_src, other_dst) if family == "A" else (src, dst)
    effect = {fsrc: [error["records"][fsrc]["quantity"], error["records"][fsrc]["quantity"] - q]}
    error["records"][fsrc]["quantity"] -= q
    if family != "C":
        effect[fdst] = [error["records"][fdst]["quantity"], error["records"][fdst]["quantity"] + q]
        error["records"][fdst]["quantity"] += q
    error["ledger"][fault_id] = {"source_id": fsrc, "destination_id": fdst, "quantity": q,
                                "phase": "DEBITED" if family == "C" else "COMPLETED", "compensations": []}
    error["dedup"][fault_id] = "ACTIVE"
    error["active_operations"]["stock-lane"] = fault_id
    error["revision"] += 1
    fault_event = {"audit_id": f"h{len(error['audit']):03d}", "kind": "FAULT_INJECTION",
                   "operation_id": fault_id, "changes": effect, "partial": family == "C"}
    error["audit"].append(fault_event)
    descriptions = {
        "A": "A transfer modified the wrong SKU records after the checkpoint. Undo that incident, including its active processing entry. Do not perform the intended SKU-M transfer yet.",
        "B": "One transfer was already completed at the checkpoint. A second operation applied the same effect once more. Compensate only the extra incident and clear its active processing entry; preserve the first completed operation.",
        "C": "After the checkpoint only the source debit of a transfer was executed. Undo the partial incident and clear its active processing entry. Do not finish the intended transfer yet."
    }
    task = {"task_id": tid, "split": split, "family": family, "generator_version": "inventory-recovery-v1",
            "origin_state": origin, "checkpoint": checkpoint, "error_state": error,
            "fault_trace": [{"evidence_kind": "CPU_FAULT_INJECTION", "state_before": checkpoint,
                             "action": fault_event, "state_after": error}],
            "public_request": {"task_id": tid, "scope_id": origin["scope_id"], "record_ids": ids,
                               "incident_operation_id": fault_id, "checkpoint_event_id": checkpoint["audit"][-1]["audit_id"],
                               "request": descriptions[family]},
            "allowed_compensation": {"stock_record_ids": [fsrc, fdst], "operation_id": fault_id,
                                     "stock_then_release": True, "explicit_values_required": True},
            "continuations": {"F1": [next_op] if family == "B" else [normal],
                              "F2": [normal, next_op]},
            "oracle": {"business_projection": business(checkpoint), "meaning": "restore checkpoint, not finish pending work"}}
    task["identity"] = {key: digest(task[key]) for key in
                        ("checkpoint", "error_state", "fault_trace", "public_request", "continuations", "oracle")}
    return task


def generate(split="DEV", gate_receipt=None):
    p = read_json(CASE_ROOT / "specs/protocol.json")
    if split not in ("DEV", "HOLDOUT"):
        raise ValueError("unknown split")
    if split == "HOLDOUT":
        check_p2_gate(gate_receipt or {})
    rng = random.Random(p["seeds"][split])
    return [_task(split, family, index, rng) for family in "ABC"
            for index in range(p["splits"][split]["per_family"])]


def dev_tasks():
    return [read_json(p) for p in sorted((CASE_ROOT / "data/dev").glob("DEV-*.json"))]
