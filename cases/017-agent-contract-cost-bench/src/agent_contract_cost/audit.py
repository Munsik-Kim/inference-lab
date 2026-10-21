"""Model-free replay plus separately written scalar/state endpoint checks.

Replaying stored text is not new inference. This module never reads a model,
tokenizer, endpoint or submitted summary. Input token hashes are checked against
saved integer IDs, not independently re-tokenized. Actual fixture replay and the
original scorer are one check; independent_endpoint is a separate implementation.
"""
from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import itertools
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from .contracts import parse_action
from .tools import Fixture, tool_definitions


ENDPOINT_FIELDS = ("VERIFIED_WORKFLOW_SUCCESS", "STATE_CORRECT", "FINAL_ANSWER_CORRECT",
                   "REQUIRED_ACTIONS_DONE", "OBSERVATION_ORDER_VALID",
                   "integrity_valid_throughout", "CLAIMED_DONE_WITH_WRONG_STATE",
                   "UNSUPPORTED_SUCCESS_CLAIM")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def strict_equal(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(strict_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(strict_equal(a, b) for a, b in zip(left, right))
    return left == right


def load_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def independent_desired(task):
    """Decimal half-up for A; explicit conservation/revision arithmetic for B/C."""
    state = copy.deepcopy(task["initial_state"])
    goal, family = task["goal"], task["family"]
    if family == "A":
        row = state["orders"][goal["order_id"]]
        gross = sum(Decimal(line["quantity"]) * Decimal(line["unit_price_minor"]) for line in row["lines"])
        amount = int((gross * (Decimal(10000) - Decimal(row["discount_bps"])) / Decimal(10000)).quantize(Decimal(1), rounding=ROUND_HALF_UP))
        row["total_minor"] = amount
        result = {"order_id": goal["order_id"], "total_minor": amount, "validated": True}
    elif family == "B":
        source = state["warehouses"][goal["source_id"]]
        destination = state["warehouses"][goal["destination_id"]]
        source["quantity"] = source["quantity"] - goal["quantity"]
        destination["quantity"] = destination["quantity"] + goal["quantity"]
        source["revision"] = source["revision"] + 1
        destination["revision"] = destination["revision"] + 1
        result = {"source_id": goal["source_id"], "destination_id": goal["destination_id"], "sku": goal["sku"],
                  "source_quantity": source["quantity"], "destination_quantity": destination["quantity"], "validated": True}
    elif family == "C":
        row = state["documents"][goal["document_id"]]
        row.update({key: copy.deepcopy(goal[key]) for key in ("status", "tags", "owner_id")})
        row["revision"] = row["revision"] + 1
        result = {"document_id": goal["document_id"], "revision": row["revision"], "status": row["status"],
                  "tags": copy.deepcopy(row["tags"]), "owner_id": row["owner_id"], "validated": True}
    else:
        raise ValueError("unknown task family")
    return state, {"tool": "final", "arguments": {"task_id": task["task_id"], "completed": True, "result": result}}


def independent_integrity(task, state):
    initial = copy.deepcopy(task["initial_state"])
    current = copy.deepcopy(state)
    family, goal = task["family"], task["goal"]
    table = {"A": "orders", "B": "warehouses", "C": "documents"}[family]
    targets = [goal["order_id"]] if family == "A" else [goal["source_id"], goal["destination_id"]] if family == "B" else [goal["document_id"]]
    fields = {"A": ("total_minor",), "B": ("quantity", "revision"), "C": ("status", "tags", "owner_id", "revision")}[family]
    try:
        for record_id in targets:
            if set(initial[table][record_id]) != set(current[table][record_id]):
                return False
            for field in fields:
                del initial[table][record_id][field]
                del current[table][record_id][field]
    except (KeyError, TypeError):
        return False
    return strict_equal(initial, current)


def _roles(task, event, desired_state):
    if event.get("status") != "SUCCESS":
        return []
    family, goal = task["family"], task["goal"]
    name, args, result = event["tool"], event["arguments"], event["result"]
    roles = []
    if family == "A":
        row = task["initial_state"]["orders"][goal["order_id"]]
        amount = desired_state["orders"][goal["order_id"]]["total_minor"]
        if name == "read_order" and args["order_id"] == goal["order_id"] and strict_equal(result.get("lines"), row["lines"]) and strict_equal(result.get("discount_bps"), row["discount_bps"]):
            roles.append("read")
        if name == "calculate_total" and sorted(map(canonical, args["lines"])) == sorted(map(canonical, row["lines"])) and strict_equal(args["discount_bps"], row["discount_bps"]) and strict_equal(result.get("total_minor"), amount):
            roles.append("calculation")
        if name == "write_order_total" and args["order_id"] == goal["order_id"] and strict_equal(args["total_minor"], amount):
            roles.append("write")
        if name == "validate_order" and args["order_id"] == goal["order_id"] and result.get("valid") is True:
            roles.append("validation")
    elif family == "B":
        initial = task["initial_state"]["warehouses"]
        if name == "read_stock" and args["sku"] == goal["sku"]:
            for role, key in (("source_read", "source_id"), ("destination_read", "destination_id")):
                if args["warehouse_id"] == goal[key] and all(strict_equal(result.get(field), initial[goal[key]][field]) for field in ("quantity", "revision", "sku")):
                    roles.append(role)
        if name == "transfer_stock" and all(strict_equal(args[key], goal[key]) for key in ("source_id", "destination_id", "sku", "quantity")) and strict_equal(args["source_revision"], initial[goal["source_id"]]["revision"]) and strict_equal(args["destination_revision"], initial[goal["destination_id"]]["revision"]) and result.get("transferred") is True:
            roles.append("transfer")
        if name == "validate_transfer" and all(strict_equal(args[key], goal[key]) for key in ("source_id", "destination_id", "sku")) and result.get("valid") is True:
            roles.append("validation")
    else:
        initial = task["initial_state"]["documents"][goal["document_id"]]
        if name == "read_document" and args["document_id"] == goal["document_id"] and strict_equal(result.get("revision"), initial["revision"]):
            roles.append("read")
        if name == "update_document" and all(strict_equal(args[key], goal[key]) for key in ("document_id", "status", "tags", "owner_id")) and strict_equal(args["expected_revision"], initial["revision"]) and result.get("updated") is True:
            roles.append("update")
        if name == "validate_document" and args["document_id"] == goal["document_id"] and result.get("valid") is True:
            roles.append("validation")
    return roles


def independent_endpoint(task, events, final_action, state, state_version, final_available_result_event_ids, terminal=None):
    """Independently determine state/final/observed-chain completion, no scorer import."""
    desired, expected_final = independent_desired(task)
    state_ok, final_ok = strict_equal(state, desired), strict_equal(final_action, expected_final)
    integrity = independent_integrity(task, state) and all(independent_integrity(task, event[side]) for event in events for side in ("state_before", "state_after")) and not any(event.get("integrity_violation_so_far", False) for event in events)
    obligations = {"A": ("read", "calculation", "write", "validation"), "B": ("source_read", "destination_read", "transfer", "validation"), "C": ("read", "update", "validation")}[task["family"]]
    groups = {key: [] for key in obligations}
    for index, event in enumerate(events):
        for role in _roles(task, event, desired):
            groups[role].append((index, event))
    executed, observed = False, False
    for combination in itertools.product(*(groups[key] for key in obligations)):
        indices = [item[0] for item in combination]
        if task["family"] == "B":
            ordered = indices[0] != indices[1] and max(indices[0:2]) < indices[2] < indices[3]
        else:
            ordered = all(a < b for a, b in zip(indices, indices[1:]))
        if not ordered:
            continue
        executed = True
        chain = {role: event for role, (_, event) in zip(obligations, combination)}
        dependencies = [("read", "calculation"), ("calculation", "write")] if task["family"] == "A" else [("source_read", "transfer"), ("destination_read", "transfer")] if task["family"] == "B" else [("read", "update")]
        valid = chain["validation"]
        available = all(chain[source]["event_id"] in chain[target]["generated_after_event_ids"] for source, target in dependencies)
        current_validation = valid["after_digest"] == digest(state) and valid["version_after"] == state_version
        if available and current_validation and valid["event_id"] in final_available_result_event_ids:
            observed = True
            break
    verified = bool(state_ok and final_ok and integrity and observed and terminal in (None, "FINAL", "VERIFIED_SUCCESS"))
    claimed = type(final_action) is dict and type(final_action.get("arguments")) is dict and final_action["arguments"].get("completed") is True
    return {"VERIFIED_WORKFLOW_SUCCESS": verified, "STATE_CORRECT": bool(state_ok and integrity),
            "FINAL_ANSWER_CORRECT": final_ok, "REQUIRED_ACTIONS_DONE": executed,
            "OBSERVATION_ORDER_VALID": observed, "integrity_valid_throughout": integrity,
            "CLAIMED_DONE_WITH_WRONG_STATE": bool(claimed and not (state_ok and integrity)),
            "UNSUPPORTED_SUCCESS_CLAIM": bool(claimed and not verified)}


def observation_view(event):
    return {key: copy.deepcopy(event[key]) for key in ("event_id", "tool", "status", "result")}


def independent_sse(lines):
    """Small separate content/token assembler for the registered single choice."""
    text, ids, prompt_ids, finish, stop, done, usage = "", [], None, None, None, False, None
    for item in lines:
        line = item if type(item) is str else item["line"]
        if not line.startswith("data:"):
            continue
        value = line[5:].strip()
        if value == "[DONE]":
            done = True
            continue
        obj = json.loads(value)
        if obj.get("usage") is not None:
            usage = obj["usage"]
        if obj.get("prompt_token_ids") is not None:
            if prompt_ids is not None and prompt_ids != obj["prompt_token_ids"]:
                raise ValueError("SSE prompt IDs changed")
            prompt_ids = obj["prompt_token_ids"]
        for choice in obj.get("choices", []):
            if choice.get("index") != 0:
                raise ValueError("unregistered SSE choice")
            delta = choice.get("delta") or {}
            if delta.get("tool_calls") or delta.get("function_call"):
                raise ValueError("native tool call mode is not registered")
            text += delta.get("content") or ""
            ids.extend(choice.get("token_ids") or [])
            if choice.get("finish_reason") is not None:
                finish, stop = choice["finish_reason"], choice.get("stop_reason")
    return {"raw_text": text, "output_token_ids": ids, "prompt_token_ids": prompt_ids,
            "finish_reason": finish, "stop_reason": stop, "stream_done": done, "usage": usage}


def audit_episode(task, episode, new_fixture_root):
    """Replay raw response actions into a new isolated fixture; return scalar row."""
    from .scoring import assess  # Replay comparator only; independent_endpoint above does not use it.
    identity = episode.get("identity", {})
    def field(name):
        if name in identity and name in episode and identity[name] != episode[name]:
            raise ValueError("conflicting episode identity: " + name)
        return episode.get(name, identity.get(name))
    errors = []
    def check(condition, label):
        if not condition:
            errors.append(label)
    check(field("task_id") == task["task_id"], "task identity mismatch")
    check(field("family") == task["family"], "family identity mismatch")
    fixture = Fixture(task, new_fixture_root)
    available, final, final_available = [], None, []
    counters = {key: 0 for key in ("model_request_attempts", "model_responses_received", "server_abort_requests", "generated_raw_objects", "parsed_tool_attempts", "contract_rejections", "callable_executions", "side_effect_writes", "final_attempts", "retry_triggered", "retry_consumed", "generated_output_tokens")}
    pending_retry = False
    accepted_actions = 0
    first_contract_error = None
    for index, request in enumerate(episode["requests"]):
        turn = request["generation_turn"]
        response = request["response"]
        previous_counters = dict(counters)
        if request.get("budget_before"):
            for key, count in previous_counters.items():
                check(request["budget_before"].get(key) == count, f"pre-request counter {key} {index}")
        counters["model_request_attempts"] += 1
        if pending_retry:
            counters["retry_consumed"] += 1
            pending_retry = False
        if "retry_consumed" in request:
            check(request["retry_consumed"] == counters["retry_consumed"] - previous_counters["retry_consumed"], f"retry request debit {index}")
        check(turn == index, f"turn order {index}")
        check(request["available_result_event_ids"] == available, f"observation IDs {index}")
        views = [observation_view(event) for event in fixture.events if event["event_id"] in available]
        check(request["observation_sha256"] == digest(views), f"observation hash {index}")
        check(request["state_digest"] == digest(fixture.snapshot()), f"state digest {index}")
        check(request["state_version"] == fixture.version, f"state version {index}")
        check(digest(request["input_token_ids"]) == request["input_token_sha256"], f"input token hash {index}")
        expected_cap = min(512, 4096 - previous_counters["generated_output_tokens"], 24576 - len(request["input_token_ids"]))
        check(request["request_plan"]["max_tokens"] == expected_cap, f"effective cap {index}")
        body = response.get("request_body", response.get("requestbody", {}))
        if body:
            check(body.get("messages") == request["messages"], f"actual messages {index}")
            check("tools" not in body and "tool_choice" not in body, f"native tool mode {index}")
            check(("structured_outputs" in body) == (field("policy") == "G"), f"policy grammar setting {index}")
            if "max_tokens" in body:
                check(body["max_tokens"] == expected_cap, f"actual request cap {index}")
        raw = response.get("raw_text", "")
        check(hashlib.sha256(raw.encode()).hexdigest() == response.get("raw_utf8_sha256"), f"raw string hash {index}")
        ids = response.get("output_token_ids") or []
        check(all(type(token) is int for token in ids), f"output token types {index}")
        check(len(ids) == response.get("output_tokens", len(ids)), f"output token count {index}")
        counters["generated_output_tokens"] += len(ids)
        counters["model_responses_received"] += int(response.get("model_responses_received", response.get("status") == "OK"))
        counters["server_abort_requests"] += int(response.get("server_abort_requests", 0))
        if response.get("raw_sse_lines"):
            rebuilt = independent_sse(response["raw_sse_lines"])
            for key in ("raw_text", "output_token_ids", "prompt_token_ids", "finish_reason", "stop_reason", "stream_done", "usage"):
                check(strict_equal(rebuilt[key], response.get(key)), f"SSE {key} {index}")
            if response.get("raw_sse_sha256"):
                check(response["raw_sse_sha256"] == digest([item["line"] if type(item) is dict else item for item in response["raw_sse_lines"]]), f"SSE line hash {index}")
        if response.get("status") != "OK":
            check(request.get("event") is None, f"failed transport executed event {index}")
            continue
        check(response.get("prompt_token_ids") == request["input_token_ids"], f"server input IDs {index}")
        usage = response.get("usage") or {}
        check(usage.get("completion_tokens") == len(ids), f"completion usage {index}")
        check(usage.get("prompt_tokens") == len(request["input_token_ids"]), f"prompt usage {index}")
        if response.get("output_token_sha256"):
            check(response["output_token_sha256"] == digest(ids), f"output token hash {index}")
        parsed = parse_action(raw, tool_definitions(task["family"]))
        check(parsed.as_dict() == request["parse"], f"saved parser result {index}")
        counters["parsed_tool_attempts"] += int(parsed.parsed_object)
        counters["generated_raw_objects"] += int(parsed.parsed_object)
        if not parsed.accepted:
            counters["contract_rejections"] += 1
            check(request.get("event") is None, f"contract reject executed event {index}")
            first_rejection = first_contract_error is None
            if first_rejection:
                first_contract_error = parsed.error.public()
                if field("policy") in ("N", "E"):
                    counters["retry_triggered"] += 1
            if "retry_triggered" in request:
                check(request["retry_triggered"] == counters["retry_triggered"] - previous_counters["retry_triggered"], f"retry trigger debit {index}")
            feedback = request.get("feedback")
            if feedback is not None:
                pending_retry = True
                check(field("policy") in ("N", "E"), f"retry in nonretry policy {index}")
                check(first_rejection, f"repeated retry feedback {index}")
                from .policies import NEUTRAL_FEEDBACK
                expected_feedback = NEUTRAL_FEEDBACK if field("policy") == "N" else json.dumps(parsed.error.public(), ensure_ascii=False, separators=(",", ":"))
                check(feedback == expected_feedback, f"feedback content {index}")
                if index + 1 < len(episode["requests"]):
                    next_messages = episode["requests"][index + 1]["messages"]
                    check(next_messages == request["messages"] + [{"role": "assistant", "content": raw}, {"role": "user", "content": feedback}], f"retry history {index}")
            continue
        accepted_actions += 1
        check(counters["parsed_tool_attempts"] <= 6, f"seventh parsed execution {index}")
        event = fixture.execute(parsed.action, turn, available)
        check(strict_equal(event, request.get("event")), f"fixture event {index}")
        counters["callable_executions"] += int(event["callable_executed"])
        counters["side_effect_writes"] += int(event["side_effect_write"])
        if parsed.action["tool"] == "final":
            counters["final_attempts"] += 1
            final, final_available = parsed.action, list(available)
            check(index == len(episode["requests"]) - 1, "continued after final")
        else:
            available.append(event["event_id"])
            if index + 1 < len(episode["requests"]):
                expected_messages = request["messages"] + [{"role": "assistant", "content": raw},
                                                          {"role": "user", "content": canonical(observation_view(event))}]
                check(episode["requests"][index + 1]["messages"] == expected_messages, f"tool result history {index}")
    check(strict_equal(fixture.events, episode["events"]), "episode event list")
    check(strict_equal(fixture.snapshot(), episode["final_state"]), "final state")
    check(fixture.version == episode["state_version"], "final state version")
    check(strict_equal(final, episode.get("final_action")), "final action")
    check(final_available == episode.get("final_available_result_event_ids", []), "final observation IDs")
    terminal = episode["terminal_reason"]
    original = assess(task, fixture.events, final, fixture.snapshot(), fixture.version, final_available, terminal)
    independent = independent_endpoint(task, fixture.events, final, fixture.snapshot(), fixture.version, final_available, terminal)
    reported = episode["reported_endpoint"]
    for key in ENDPOINT_FIELDS:
        check(strict_equal(original[key], independent[key]), "independent endpoint: " + key)
        if episode.get("observation_status") == "OBSERVED" or key != "VERIFIED_WORKFLOW_SUCCESS":
            check(strict_equal(reported[key], original[key]), "reported endpoint: " + key)
    saved_budget = episode.get("budget", {})
    for key, value in counters.items():
        if key in saved_budget:
            check(saved_budget[key] == value, "budget counter: " + key)
    check(counters["model_request_attempts"] <= 8 and counters["parsed_tool_attempts"] <= 6 and counters["generated_output_tokens"] <= 4096 and counters["retry_consumed"] <= 1, "registered episode budget")
    if field("policy") not in ("N", "E"):
        check(counters["retry_consumed"] == 0, "unregistered retry")
    observed = episode.get("observation_status") == "OBSERVED"
    return {"task_id": task["task_id"], "family": task["family"], "stage": field("stage"),
            "kv": field("kv"), "policy": field("policy"), "evidence_kind": episode.get("evidence_kind"),
            "attempt_id": identity.get("attempt_id", episode.get("attempt_id")), "audit_pass": not errors,
            "audit_errors": errors, "observation_status": episode.get("observation_status"),
            "verified": independent["VERIFIED_WORKFLOW_SUCCESS"] if observed else None,
            "state_correct": independent["STATE_CORRECT"] if observed else None,
            "final_correct": independent["FINAL_ANSWER_CORRECT"] if observed else None,
            "endpoint": independent, "reported_endpoint": reported, "counters": counters,
            "terminal_reason": terminal, "first_contract_error": first_contract_error,
            "accepted_actions": accepted_actions,
            "wall_seconds": episode.get("wall_seconds"), "source_sha256": digest(episode),
            "retokenization_status": "NOT_RUN_SAVED_IDS_HASH_CHECK_ONLY"}


def audit_run(tasks, runroot, newout):
    if isinstance(tasks, (str, Path)):
        tasks = load_json(tasks)
    if isinstance(tasks, dict):
        tasks = tasks["tasks"]
    lookup = {task["task_id"]: task for task in tasks}
    if len(lookup) != len(tasks):
        raise ValueError("duplicate task IDs")
    runroot, output = Path(runroot), Path(newout)
    output.mkdir(parents=True, exist_ok=False)
    (output / "fixtures").mkdir()
    files = sorted([*runroot.glob("cells/*.json"), *runroot.glob("cells/*.json.gz")])
    rows, identities = [], set()
    for index, path in enumerate(files):
        episode = load_json(path)
        task_id = episode.get("task_id", episode.get("identity", {}).get("task_id"))
        if task_id not in lookup:
            raise ValueError("cell refers to an unregistered task")
        row = audit_episode(lookup[task_id], episode, output / "fixtures" / f"cell-{index:04d}")
        key = (row["task_id"], row["kv"], row["policy"])
        if key in identities:
            raise ValueError("duplicate primary cell; attempts must be kept in separate ledger")
        identities.add(key)
        row["source_file"] = path.relative_to(runroot).as_posix()
        rows.append(row)
    write_json(output / "audited_rows.json", rows)
    columns = ["task_id", "family", "stage", "kv", "policy", "audit_pass", "observation_status", "verified", "state_correct", "final_correct", "terminal_reason", "wall_seconds", "source_file", "source_sha256"]
    with (output / "audited_rows.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    receipt = {"scope": "STORED_TRACE_REPLAY", "gpu_inference_performed": False,
               "cells": len(rows), "audit_pass": all(row["audit_pass"] for row in rows),
               "failed_cells": [row["source_file"] for row in rows if not row["audit_pass"]],
               "new_fixture_replays": len(rows), "independent_endpoint": "separate Decimal/state/DAG implementation",
               "input_tokens": "saved IDs/hash and server IDs; not independently re-tokenized",
               "gpu_weights_logits_verified": False}
    write_json(output / "audit_receipt.json", receipt)
    return {"receipt": receipt, "rows": rows}
