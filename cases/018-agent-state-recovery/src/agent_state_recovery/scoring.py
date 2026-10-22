"""Runner endpoint; independent audit reimplements transitions and predicates."""
import copy
from .common import digest, strict_equal
from .world import World, business, protected_ok


def operational(state):
    keep = {op for op in state["ledger"] if state["dedup"].get(op) != "COMPENSATED"}
    return {"ledger": {op: state["ledger"][op] for op in sorted(keep)},
            "dedup": {op: state["dedup"][op] for op in sorted(keep)},
            "active_operations": state["active_operations"]}


def fresh_validation(events, available, state, scope):
    for event in reversed(events):
        r = event["result"]
        if (event["event_id"] in available and event["tool"] == "validate_inventory"
                and event["status"] == "SUCCESS" and r.get("scope_id") == scope
                and r.get("valid") is True and r.get("state_version") == state["revision"]
                and r.get("state_digest") == digest(state)):
            return event["event_id"]
    return None


def run_continuation(task, snapshot, name):
    """No model, conditional repair or outcome-dependent path choice."""
    world = World(task, snapshot)
    available = []
    for item in task["continuations"][name]:
        read = world.execute({"tool": "read_records", "arguments": {"record_ids": [item["source_id"], item["destination_id"]]}}, len(world.events), available)
        available.append(read["event_id"])
        # This public read's current revision is intentionally state-specific.
        action = {"tool": "apply_transfer", "arguments": {**copy.deepcopy(item), "expected_revision": read["result"]["revision"]}}
        event = world.execute(action, len(world.events), available)
        available.append(event["event_id"])
    read = world.execute({"tool": "read_records", "arguments": {"record_ids": task["public_request"]["record_ids"]}}, len(world.events), available)
    available.append(read["event_id"])
    world.execute({"tool": "validate_inventory", "arguments": {"scope_id": task["public_request"]["scope_id"]}}, len(world.events), available)
    return {"initial_snapshot_digest": digest(snapshot), "events": world.events, "final_state": world.snapshot(),
            "evidence_kind": "FIXED_CPU_CONTINUATION_NOT_MODEL_ACTIONS"}


def continuations(task, terminal):
    rows = {}
    for name in ("F1", "F2"):
        actual = run_continuation(task, terminal, name)
        reference = run_continuation(task, task["checkpoint"], name)
        states = [s for e in actual["events"] for s in (e["state_before"], e["state_after"])]
        no_error = all(e["status"] == "SUCCESS" for e in actual["events"])
        effects = lambda r: [(e["result"].get("applied"), e["result"].get("duplicate_ignored"))
                             for e in r["events"] if e["tool"] == "apply_transfer"]
        valid = actual["events"][-1]["result"].get("valid") is True
        ok = (no_error and valid and protected_ok(task, terminal, states)
              and strict_equal(business(actual["final_state"]), business(reference["final_state"]))
              and strict_equal(operational(actual["final_state"]), operational(reference["final_state"]))
              and effects(actual) == effects(reference))
        rows[name] = {"actual": actual, "reference": reference, "correct": bool(ok)}
    return rows


def assess(task, initial, terminal, events, final, final_available, last_available, reason, continuation_rows):
    final_args = final["arguments"] if final else None
    B = strict_equal(business(terminal), task["oracle"]["business_projection"])
    states = [initial, *[s for e in events for s in (e["state_before"], e["state_after"])], terminal]
    P = protected_ok(task, initial, states)
    available = final_available if final else last_available
    fresh_id = fresh_validation(events, available, terminal, task["public_request"]["scope_id"])
    F = fresh_id is not None
    quantities = final_args.get("quantities", []) if final_args else []
    expected = {k: r["quantity"] for k, r in terminal["records"].items()}
    actual = {r["record_id"]: r["quantity"] for r in quantities}
    final_ok = bool(final_args and final_args["task_id"] == task["task_id"]
                    and final_args["recovered"] is True and len(quantities) == len(actual)
                    and strict_equal(actual, expected) and final_args["validation_event_id"] == fresh_id)
    C1, C2 = (continuation_rows[name]["correct"] for name in ("F1", "F2"))
    Q = B and P and C1 and C2
    W = Q and reason == "ACCEPTED_FINAL" and final_ok and F
    return {"D": bool(final_args and final_args["recovered"] is True), "B": B, "P": P, "F": F,
            "C1": C1, "C2": C2, "Q": Q, "W": W, "FINAL_CONTENT_CORRECT": final_ok,
            "fresh_event_id": fresh_id, "F_context": "FINAL_PRE_GENERATION" if final else "LAST_GENERATION_DIAGNOSTIC",
            "failure_labels": [label for label, ok in (("BUSINESS_NOT_RESTORED", B), ("NEW_PROTECTED_DAMAGE", P),
                                ("F1_INCORRECT", C1), ("F2_INCORRECT", C2), ("FRESH_VALIDATION_MISSING", F),
                                ("FINAL_NOT_VERIFIED", W)) if not ok]}
