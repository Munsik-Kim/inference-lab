"""Explicit allowlist for model-visible messages and public contracts only."""
import copy
from .common import CASE_ROOT, read_json, canonical


SYSTEM = (
    "Recover the inventory records and processing state to the checkpoint described in the incident history. "
    "Undo only the unwanted effects; preserve any transfer completed before that checkpoint. "
    "The intended remaining work will be performed later, not during recovery. "
    "Use public reads to inspect records, before/after incident history, the operation ledger and current revision. "
    "Compensate stock with explicit quantities and release the unwanted operation. "
    "Do not change unrelated records, protected fields or existing audit entries. "
    "Return exactly one JSON object with tool and arguments per response. "
    "After the last change, inspect a successful current inventory validation before final. "
    "An older validation is stale after any subsequent mutation. "
    "Final must list every registered record's current quantity, task_id, recovered and the observed validation_event_id. "
    "Validation checks public invariants, not whether the recovery matches the checkpoint."
)


def definitions():
    return read_json(CASE_ROOT / "specs/tools.json")


def grammar():
    return read_json(CASE_ROOT / "specs/grammar.json")


def initial_payload(task):
    public = task["public_request"]
    request = {key: copy.deepcopy(public[key]) for key in
               ("task_id", "scope_id", "record_ids", "incident_operation_id", "checkpoint_event_id", "request")}
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": canonical(request)}],
            "tools": definitions(), "output_schema": grammar()}


def append_result(messages, raw, event):
    result = {key: copy.deepcopy(event[key]) for key in ("event_id", "tool", "status", "result")}
    return [*messages, {"role": "assistant", "content": raw},
            {"role": "user", "content": canonical({"tool_result": result})}]


def append_guard(messages, raw):
    return [*messages, {"role": "assistant", "content": raw},
            {"role": "user", "content": canonical({"error": {"code": "FRESH_VALIDATION_REQUIRED"}})}]
