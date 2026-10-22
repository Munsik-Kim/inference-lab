"""Small JSON inventory world. Only registered IDs are interpreted, never paths."""
import copy
from pathlib import Path
from .common import canonical, digest, fresh_dir


def business(state):
    return {key: {"quantity": value["quantity"], "sku": value["sku"], "warehouse": value["warehouse"]}
            for key, value in sorted(state["records"].items())}


def invariants(state, opening):
    errors = []
    if state["scope_id"] != opening["scope_id"]:
        errors.append("SCOPE_ID")
    if set(state["records"]) != set(opening["records"]):
        return ["RECORD_MEMBERSHIP"]
    for key, row in state["records"].items():
        anchor = opening["records"][key]
        if set(row) != set(anchor) or any(row.get(k) != anchor[k] for k in anchor if k != "quantity"):
            errors.append("PROTECTED_FIELD")
        if type(row.get("quantity")) is not int or row["quantity"] < 0:
            errors.append("NONNEGATIVE_INTEGER")
        if row.get("protected") and row.get("quantity") != anchor["quantity"]:
            errors.append("PROTECTED_RECORD")
    sums = {sku: sum(row["quantity"] for row in state["records"].values() if row["sku"] == sku)
            for sku in opening["stock_totals"]}
    if sums != opening["stock_totals"] or state["stock_totals"] != opening["stock_totals"]:
        errors.append("CONSERVATION")
    for lane, op in state["active_operations"].items():
        if lane != "stock-lane" or op not in state["ledger"] or state["dedup"].get(op) != "ACTIVE":
            errors.append("ACTIVE_REFERENCE")
    for op, row in state["ledger"].items():
        if (row["source_id"] not in state["records"] or row["destination_id"] not in state["records"]
                or op not in state["dedup"]):
            errors.append("LEDGER_REFERENCE")
    return sorted(set(errors))


class World:
    def __init__(self, task, state, root=None):
        self.task = task
        self.state = copy.deepcopy(state)
        self.root = fresh_dir(root) if root is not None else None
        self.events = []
        self._persist()

    def snapshot(self):
        return copy.deepcopy(self.state)

    def _persist(self):
        if self.root is not None:
            if self.root.is_symlink() or (self.root / "state.json").is_symlink():
                raise ValueError("symlink fixture is forbidden")
            (self.root / "state.json").write_text(canonical(self.state) + "\n", encoding="utf-8")

    def _mutate(self, name, args, changes):
        self.state["revision"] += 1
        self.state["audit"].append({"audit_id": f"h{len(self.state['audit']):03d}",
                                    "kind": name, "arguments": copy.deepcopy(args), "changes": changes})

    def _call(self, name, a):
        s = self.state
        revision = s["revision"]
        if name == "read_history":
            return {"history": copy.deepcopy(s["audit"]), "revision": revision}
        if name == "read_records":
            if len(set(a["record_ids"])) != len(a["record_ids"]) or any(k not in s["records"] for k in a["record_ids"]):
                return {"error": "UNKNOWN_OR_DUPLICATE_RECORD", "revision": revision}
            return {"records": [{"record_id": k, **copy.deepcopy(s["records"][k])} for k in a["record_ids"]],
                    "revision": revision}
        if name == "read_operation":
            op = a["operation_id"]
            if op not in s["ledger"]:
                return {"error": "UNKNOWN_OPERATION", "revision": revision}
            return {"operation_id": op, "operation": copy.deepcopy(s["ledger"][op]),
                    "dedup_status": s["dedup"].get(op), "active_operations": copy.deepcopy(s["active_operations"]),
                    "revision": revision}
        if name == "validate_inventory":
            if a["scope_id"] != s["scope_id"]:
                return {"error": "UNKNOWN_SCOPE", "revision": revision}
            errors = invariants(s, self.task["origin_state"])
            return {"scope_id": a["scope_id"], "valid": not errors, "violations": errors,
                    "state_version": revision, "state_digest": digest(s)}
        if a["expected_revision"] != revision:
            return {"error": "REVISION_CONFLICT", "revision": revision}
        if name == "apply_transfer":
            op = a["operation_id"]
            previous = s["ledger"].get(op)
            if previous is not None:
                same = all(previous[k] == a[k] for k in ("source_id", "destination_id", "quantity"))
                if not same:
                    return {"error": "OPERATION_ID_CONFLICT", "revision": revision}
                if s["dedup"].get(op) in ("COMPLETED", "COMPENSATED"):
                    return {"applied": False, "duplicate_ignored": True, "revision": revision}
            if s["active_operations"]:
                return {"error": "ACTIVE_OPERATION", "revision": revision}
            src, dst = a["source_id"], a["destination_id"]
            if src not in s["records"] or dst not in s["records"] or src == dst:
                return {"error": "UNKNOWN_OR_SAME_RECORD", "revision": revision}
            left, right = s["records"][src], s["records"][dst]
            if left["protected"] or right["protected"] or left["sku"] != right["sku"]:
                return {"error": "PERMISSION_OR_SKU", "revision": revision}
            if left["quantity"] < a["quantity"]:
                return {"error": "INSUFFICIENT_STOCK", "revision": revision}
            old = [left["quantity"], right["quantity"]]
            left["quantity"] -= a["quantity"]
            right["quantity"] += a["quantity"]
            s["ledger"][op] = {k: a[k] for k in ("source_id", "destination_id", "quantity")}
            s["ledger"][op].update(phase="COMPLETED", compensations=[])
            s["dedup"][op] = "COMPLETED"
            self._mutate(name, a, {src: [old[0], left["quantity"]], dst: [old[1], right["quantity"]]})
            return {"applied": True, "duplicate_ignored": False, "revision": s["revision"]}
        op = a["operation_id"]
        if op not in s["ledger"]:
            return {"error": "UNKNOWN_OPERATION", "revision": revision}
        row = s["ledger"][op]
        if s["dedup"].get(op) == "COMPENSATED":
            return {"error": "ALREADY_RELEASED", "revision": revision}
        if name == "compensate_stock":
            src, dst = row["source_id"], row["destination_id"]
            left, right = s["records"][src], s["records"][dst]
            if left["protected"] or right["protected"]:
                return {"error": "PERMISSION_DENIED", "revision": revision}
            if right["quantity"] < a["destination_remove"]:
                return {"error": "INSUFFICIENT_STOCK", "revision": revision}
            old = [left["quantity"], right["quantity"]]
            left["quantity"] += a["source_restore"]
            right["quantity"] -= a["destination_remove"]
            row["compensations"].append({"source_restore": a["source_restore"], "destination_remove": a["destination_remove"]})
            self._mutate(name, a, {src: [old[0], left["quantity"]], dst: [old[1], right["quantity"]]})
            return {"compensated": True, "revision": s["revision"]}
        if name == "release_operation":
            row["phase"] = "COMPENSATED"
            s["dedup"][op] = "COMPENSATED"
            s["active_operations"] = {lane: value for lane, value in s["active_operations"].items() if value != op}
            self._mutate(name, a, {})
            return {"released": True, "revision": s["revision"]}
        raise ValueError("unregistered execution name")

    def execute(self, action, turn, available):
        name, a = action["tool"], action["arguments"]
        if name == "final":
            raise ValueError("final has no callable")
        before = self.snapshot()
        result = self._call(name, a)
        self._persist()
        event = {"event_id": f"e{len(self.events):03d}", "tool": name, "arguments": copy.deepcopy(a),
                 "generated_in_turn": turn, "call_index": 0, "generated_after_event_ids": list(available),
                 "status": "ERROR" if "error" in result else "SUCCESS", "result": result,
                 "state_before_execution_digest": digest(before), "state_after_execution_digest": digest(self.state),
                 "state_version_before": before["revision"], "state_version_after": self.state["revision"],
                 "state_before": before, "state_after": self.snapshot(), "callable_executed": True,
                 "side_effect_write": before != self.state}
        self.events.append(copy.deepcopy(event))
        return event


def protected_ok(task, initial, states):
    """Fault effects are baseline; do not forgive new damage later undone."""
    base = task["origin_state"]
    prefix = initial["audit"]
    previous_revision = initial["revision"]
    prior_debt = {sku: sum(r["quantity"] for r in initial["records"].values() if r["sku"] == sku) - total
                  for sku, total in base["stock_totals"].items()}
    for state in states:
        if state["audit"][:len(prefix)] != prefix or len(state["audit"]) < len(prefix):
            return False
        if type(state["revision"]) is not int or state["revision"] < previous_revision:
            return False
        violations = invariants(state, base)
        if any(v != "CONSERVATION" for v in violations):
            return False
        for sku, total in base["stock_totals"].items():
            debt = sum(r["quantity"] for r in state["records"].values() if r["sku"] == sku) - total
            if not prior_debt[sku] <= debt <= 0:
                return False
            prior_debt[sku] = debt
        prefix = state["audit"]
        previous_revision = state["revision"]
    return True
