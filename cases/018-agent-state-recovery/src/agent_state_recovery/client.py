"""CPU fake client and a non-executable future adapter interface.

FakeClient sees only the model payload, never task/oracle/continuation objects.
Reported units and virtual time are SIMULATED, not tokenizer or server usage.
"""
import json
from typing import Protocol
from .common import canonical


class FutureModelAdapter(Protocol):
    def respond(self, payload: dict, timeout_seconds: float) -> dict: ...


class FakeClient:
    evidence_kind = "SYNTHETIC_CONTRACT_TEST"

    def __init__(self, mode):
        self.mode = mode
        self.step = 0

    def respond(self, payload, timeout_seconds):
        request = json.loads(payload["messages"][1]["content"])
        results = []
        for message in payload["messages"][2:]:
            if message["role"] == "user":
                value = json.loads(message["content"])
                if "tool_result" in value:
                    results.append(value["tool_result"])
        op = next((e["result"] for e in reversed(results) if e["tool"] == "read_operation"), None)
        records = next((e["result"]["records"] for e in reversed(results) if e["tool"] == "read_records"), [])
        validation = next((e for e in reversed(results) if e["tool"] == "validate_inventory"), None)
        rev = next((e["result"]["revision"] for e in reversed(results) if "revision" in e["result"]), 0)
        incident = request["incident_operation_id"]
        compensate = self.mode not in ("clean", "no_recovery", "repeated_final")
        steps = []
        if self.mode == "stale_validation":
            steps += ["validate"]
        if compensate:
            steps += ["history", "operation", "compensate"]
            if self.mode != "business_only":
                steps += ["release"]
        if self.mode == "repeated_final":
            steps = ["final"] * 20
        else:
            steps += ["records"]
            if self.mode == "no_final":
                pass
            elif self.mode in ("guard_retry", "guard_twice", "stale_validation"):
                steps += ["final"]
                if self.mode == "guard_twice":
                    steps += ["final"]
                if self.mode in ("guard_retry", "guard_twice"):
                    steps += ["validate", "final"]
            else:
                steps += ["validate", "final"]
        if self.step >= len(steps):
            raise StopIteration("script completed without final")
        step = steps[self.step]
        self.step += 1
        if step == "history":
            action = {"tool": "read_history", "arguments": {}}
        elif step == "operation":
            action = {"tool": "read_operation", "arguments": {"operation_id": incident}}
        elif step == "records":
            action = {"tool": "read_records", "arguments": {"record_ids": request["record_ids"]}}
        elif step == "validate":
            action = {"tool": "validate_inventory", "arguments": {"scope_id": request["scope_id"]}}
        elif step == "compensate":
            quantity = op["operation"]["quantity"]
            if self.mode == "wrong_business":
                quantity += 1
            action = {"tool": "compensate_stock", "arguments": {
                "operation_id": incident, "source_restore": quantity,
                "destination_remove": 0 if op["operation"]["phase"] == "DEBITED" else quantity,
                "expected_revision": rev}}
        elif step == "release":
            action = {"tool": "release_operation", "arguments": {"operation_id": incident, "expected_revision": rev}}
        else:
            action = {"tool": "final", "arguments": {"task_id": request["task_id"], "recovered": True,
                      "quantities": [{"record_id": r["record_id"], "quantity": r["quantity"]} for r in records] or
                                    [{"record_id": "unobserved", "quantity": 0}],
                      "validation_event_id": validation["event_id"] if validation else "not-observed"}}
        raw = canonical(action)
        return {"raw_text": raw, "status": "OK", "server_finish_reason": None,
                "actual_usage": None, "output_token_ids": None,
                "simulated": {"output_tokens": max(1, (len(raw.encode()) + 3) // 4), "elapsed_seconds": 0.01,
                              "unit_policy": "ceil(UTF8 bytes/4); NOT_TOKENIZER_USAGE"}}


class SequenceClient:
    """Explicit synthetic boundary inputs; never used as model measurements."""
    def __init__(self, values):
        self.values = iter(values)

    def respond(self, payload, timeout_seconds):
        item = next(self.values)
        if type(item) is dict and "raw_text" in item:
            return item
        raw = item if type(item) is str else canonical(item)
        return {"status": "OK", "raw_text": raw, "actual_usage": None, "output_token_ids": None,
                "server_finish_reason": None, "simulated": {"output_tokens": 1, "elapsed_seconds": 0.01,
                                                             "unit_policy": "boundary test units"}}
