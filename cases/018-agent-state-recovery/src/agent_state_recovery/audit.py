"""Independent CPU transition/endpoint implementation for stored P0 raw traces.

Shares JSON parser, public input builder and serialization only. Does NOT import
world, runner, scoring, FakeClient or generator; stored booleans are comparison
targets, never inputs to recomputation. No model/tokenizer/network imports.
"""
import copy
import hashlib
from .common import digest, canonical, text_hash, read_json, write_new, fresh_dir, source_manifest, CASE_ROOT
from .contracts import parse, ContractError, is_parsed_object
from .public import initial_payload


class AuditMismatch(ValueError):
    pass


def equal(a, b, context):
    if canonical(a) != canonical(b):
        raise AuditMismatch(context)


def quantities(state):
    return {key: (row["quantity"], row["sku"], row["warehouse"]) for key, row in state["records"].items()}


def live_operations(state):
    live = sorted(k for k in state["ledger"] if state["dedup"].get(k) != "COMPENSATED")
    return [[(k, state["ledger"][k], state["dedup"].get(k)) for k in live], state["active_operations"]]


def public_violations(state, origin):
    errors = set()
    if state["scope_id"] != origin["scope_id"]:
        errors.add("SCOPE_ID")
    if sorted(state["records"]) != sorted(origin["records"]):
        return ["RECORD_MEMBERSHIP"]
    for key in origin["records"]:
        row, before = state["records"][key], origin["records"][key]
        immutable = {k: v for k, v in before.items() if k != "quantity"}
        if set(row) != set(before) or any(row.get(k) != v for k, v in immutable.items()):
            errors.add("PROTECTED_FIELD")
        if type(row.get("quantity")) is not int or row["quantity"] < 0:
            errors.add("NONNEGATIVE_INTEGER")
        if row.get("protected") and row["quantity"] != before["quantity"]:
            errors.add("PROTECTED_RECORD")
    totals = dict.fromkeys(origin["stock_totals"], 0)
    for row in state["records"].values():
        if row["sku"] in totals:
            totals[row["sku"]] += row["quantity"]
    if totals != origin["stock_totals"] or state["stock_totals"] != origin["stock_totals"]:
        errors.add("CONSERVATION")
    for lane, operation in state["active_operations"].items():
        if lane != "stock-lane" or operation not in state["ledger"] or state["dedup"].get(operation) != "ACTIVE":
            errors.add("ACTIVE_REFERENCE")
    for key, operation in state["ledger"].items():
        if any(operation[field] not in state["records"] for field in ("source_id", "destination_id")) or key not in state["dedup"]:
            errors.add("LEDGER_REFERENCE")
    return sorted(errors)


def protection(task, initial, trajectory):
    origin = task["origin_state"]
    previous = initial
    debts = {sku: sum(r["quantity"] for r in initial["records"].values() if r["sku"] == sku) - total
             for sku, total in origin["stock_totals"].items()}
    for state in trajectory:
        if (len(state["audit"]) < len(previous["audit"]) or state["audit"][:len(previous["audit"])] != previous["audit"]
                or type(state["revision"]) is not int or state["revision"] < previous["revision"]):
            return False
        if set(public_violations(state, origin)) - {"CONSERVATION"}:
            return False
        for sku, total in origin["stock_totals"].items():
            current = sum(row["quantity"] for row in state["records"].values() if row["sku"] == sku) - total
            if current > 0 or current < debts[sku]:
                return False
            debts[sku] = current
        previous = state
    return True


class ReferenceState:
    def __init__(self, task, state):
        self.task, self.s = task, copy.deepcopy(state)
        self.events = []

    def call(self, name, a):
        s = self.s
        version = s["revision"]
        def error(code):
            return {"error": code, "revision": version}
        if name == "read_records":
            ids = a["record_ids"]
            if len(set(ids)) != len(ids) or set(ids) - set(s["records"]):
                return error("UNKNOWN_OR_DUPLICATE_RECORD")
            return {"records": [{"record_id": key, **copy.deepcopy(s["records"][key])} for key in ids], "revision": version}
        if name == "read_history":
            return {"history": copy.deepcopy(s["audit"]), "revision": version}
        if name == "read_operation":
            key = a["operation_id"]
            if key not in s["ledger"]:
                return error("UNKNOWN_OPERATION")
            return {"operation_id": key, "operation": copy.deepcopy(s["ledger"][key]),
                    "dedup_status": s["dedup"].get(key), "active_operations": copy.deepcopy(s["active_operations"]), "revision": version}
        if name == "validate_inventory":
            if a["scope_id"] != s["scope_id"]:
                return error("UNKNOWN_SCOPE")
            violations = public_violations(s, self.task["origin_state"])
            return {"scope_id": a["scope_id"], "valid": len(violations) == 0, "violations": violations,
                    "state_version": version, "state_digest": digest(s)}
        if a["expected_revision"] != version:
            return error("REVISION_CONFLICT")
        changes = {}
        key = a["operation_id"]
        if name == "apply_transfer":
            if key in s["ledger"]:
                old = s["ledger"][key]
                if any(old[f] != a[f] for f in ("source_id", "destination_id", "quantity")):
                    return error("OPERATION_ID_CONFLICT")
                if s["dedup"].get(key) in ["COMPLETED", "COMPENSATED"]:
                    return {"applied": False, "duplicate_ignored": True, "revision": version}
            if s["active_operations"]:
                return error("ACTIVE_OPERATION")
            left, right = a["source_id"], a["destination_id"]
            if left not in s["records"] or right not in s["records"] or left == right:
                return error("UNKNOWN_OR_SAME_RECORD")
            r1, r2 = s["records"][left], s["records"][right]
            if r1["protected"] or r2["protected"] or r1["sku"] != r2["sku"]:
                return error("PERMISSION_OR_SKU")
            if r1["quantity"] < a["quantity"]:
                return error("INSUFFICIENT_STOCK")
            changes = {left: [r1["quantity"], r1["quantity"] - a["quantity"]], right: [r2["quantity"], r2["quantity"] + a["quantity"]]}
            for rid, (_, new_value) in changes.items():
                s["records"][rid]["quantity"] = new_value
            s["ledger"][key] = {"source_id": left, "destination_id": right, "quantity": a["quantity"],
                                "phase": "COMPLETED", "compensations": []}
            s["dedup"][key] = "COMPLETED"
            result = {"applied": True, "duplicate_ignored": False}
        else:
            if key not in s["ledger"]:
                return error("UNKNOWN_OPERATION")
            if s["dedup"].get(key) == "COMPENSATED":
                return error("ALREADY_RELEASED")
            entry = s["ledger"][key]
            if name == "compensate_stock":
                left, right = entry["source_id"], entry["destination_id"]
                r1, r2 = s["records"][left], s["records"][right]
                if r1["protected"] or r2["protected"]:
                    return error("PERMISSION_DENIED")
                if r2["quantity"] < a["destination_remove"]:
                    return error("INSUFFICIENT_STOCK")
                changes = {left: [r1["quantity"], r1["quantity"] + a["source_restore"]],
                           right: [r2["quantity"], r2["quantity"] - a["destination_remove"]]}
                for rid, (_, new_value) in changes.items():
                    s["records"][rid]["quantity"] = new_value
                entry["compensations"].append({"source_restore": a["source_restore"], "destination_remove": a["destination_remove"]})
                result = {"compensated": True}
            elif name == "release_operation":
                entry["phase"] = "COMPENSATED"
                s["dedup"][key] = "COMPENSATED"
                s["active_operations"] = {lane: op for lane, op in s["active_operations"].items() if op != key}
                result = {"released": True}
            else:
                raise AuditMismatch("unknown reference callable")
        s["revision"] = version + 1
        s["audit"].append({"audit_id": f"h{len(s['audit']):03d}", "kind": name, "arguments": copy.deepcopy(a), "changes": changes})
        return {**result, "revision": s["revision"]}

    def event(self, action, turn, available):
        before = copy.deepcopy(self.s)
        result = self.call(action["tool"], action["arguments"])
        event = {"event_id": f"e{len(self.events):03d}", "tool": action["tool"], "arguments": copy.deepcopy(action["arguments"]),
                 "generated_in_turn": turn, "call_index": 0, "generated_after_event_ids": list(available),
                 "status": "ERROR" if "error" in result else "SUCCESS", "result": result,
                 "state_before_execution_digest": digest(before), "state_after_execution_digest": digest(self.s),
                 "state_version_before": before["revision"], "state_version_after": self.s["revision"],
                 "state_before": before, "state_after": copy.deepcopy(self.s), "callable_executed": True,
                 "side_effect_write": canonical(before) != canonical(self.s)}
        self.events.append(event)
        return event


def verify_task(task):
    """Rebuild pre-fault checkpoint and fault arithmetic without generator."""
    origin = task["origin_state"]
    machine = ReferenceState(task, origin)
    if task["family"] == "B":
        operation = task["continuations"]["F2"][0]
        machine.call("apply_transfer", {**operation, "expected_revision": origin["revision"]})
    equal(machine.s, task["checkpoint"], "checkpoint construction")
    bad = task["public_request"]["incident_operation_id"]
    fault = task["error_state"]["ledger"][bad]
    reconstructed = copy.deepcopy(machine.s)
    reconstructed["records"][fault["source_id"]]["quantity"] -= fault["quantity"]
    if task["family"] != "C":
        reconstructed["records"][fault["destination_id"]]["quantity"] += fault["quantity"]
    reconstructed["ledger"][bad] = copy.deepcopy(fault)
    reconstructed["dedup"][bad] = "ACTIVE"
    reconstructed["active_operations"]["stock-lane"] = bad
    reconstructed["revision"] += 1
    event = task["fault_trace"][0]["action"]
    changed = {k: [task["checkpoint"]["records"][k]["quantity"], r["quantity"]]
               for k, r in reconstructed["records"].items() if r["quantity"] != task["checkpoint"]["records"][k]["quantity"]}
    equal(changed, event["changes"], "fault before/after quantities")
    reconstructed["audit"].append(copy.deepcopy(event))
    equal(reconstructed, task["error_state"], "fault reconstruction")
    equal(task["fault_trace"][0]["state_before"], task["checkpoint"], "fault prefix")
    equal(task["fault_trace"][0]["state_after"], reconstructed, "fault terminal")
    for key, expected in task["identity"].items():
        equal(digest(task[key]), expected, "input identity:" + key)


def current_validation(events, available, state, scope):
    match = None
    for e in events:
        r = e["result"]
        if (e["tool"] == "validate_inventory" and e["event_id"] in available and e["status"] == "SUCCESS"
                and r.get("scope_id") == scope and type(r.get("valid")) is bool and r["valid"]
                and r.get("state_version") == state["revision"] and r.get("state_digest") == digest(state)):
            match = e["event_id"]
    return match


def replay_continuation(task, initial, name):
    machine, available = ReferenceState(task, initial), []
    for params in task["continuations"][name]:
        read = machine.event({"tool": "read_records", "arguments": {"record_ids": [params["source_id"], params["destination_id"]]}}, len(machine.events), available)
        available.append(read["event_id"])
        event = machine.event({"tool": "apply_transfer", "arguments": {**params, "expected_revision": read["result"]["revision"]}}, len(machine.events), available)
        available.append(event["event_id"])
    result = machine.event({"tool": "read_records", "arguments": {"record_ids": task["public_request"]["record_ids"]}}, len(machine.events), available)
    available.append(result["event_id"])
    machine.event({"tool": "validate_inventory", "arguments": {"scope_id": task["public_request"]["scope_id"]}}, len(machine.events), available)
    return {"initial_snapshot_digest": digest(initial), "events": machine.events, "final_state": machine.s,
            "evidence_kind": "FIXED_CPU_CONTINUATION_NOT_MODEL_ACTIONS"}


def continuation_correct(task, actual, reference):
    states = [s for e in actual["events"] for s in (e["state_before"], e["state_after"])]
    if any(e["status"] != "SUCCESS" for e in actual["events"]) or actual["events"][-1]["result"].get("valid") is not True:
        return False
    effects = lambda branch: [(e["result"].get("applied"), e["result"].get("duplicate_ignored")) for e in branch["events"] if e["tool"] == "apply_transfer"]
    return (protection(task, actual["events"][0]["state_before"], states)
            and canonical(quantities(actual["final_state"])) == canonical(quantities(reference["final_state"]))
            and live_operations(actual["final_state"]) == live_operations(reference["final_state"])
            and effects(actual) == effects(reference))


def audit_episode(task, row):
    verify_task(task)
    start = task["checkpoint" if row["start"] == "clean" else "error_state"]
    equal(start, row["initial_state"], "initial state")
    equal(digest(start), row["initial_state_sha256"], "initial digest")
    machine, available = ReferenceState(task, start), []
    public = initial_payload(task)
    messages = public["messages"]
    equal(digest(public), row["initial_payload_sha256"], "initial public payload")
    final, final_ids, last_ids, reason = None, [], [], "UNKNOWN"
    counters = dict.fromkeys(("model_request_attempts", "model_responses_received", "server_abort_requests", "parsed_attempts",
                              "contract_rejections", "callable_executions", "side_effect_writes", "final_attempts", "guard_rejections", "guard_feedbacks"), 0)
    tokens, seconds = 0, 0.0
    live = row.get("evidence_kind") == "MODEL_EPISODE"
    input_token_total, output_observed, unknown_cost, declared = 0, 0, False, False
    limits = read_json(CASE_ROOT / "specs/protocol.json")["limits"]
    for turn, request in enumerate(row["requests"]):
        if reason != "UNKNOWN":
            raise AuditMismatch("generation after terminal")
        expected_payload = {**public, "messages": messages}
        equal(expected_payload, request["model_payload"], "request messages/tools/schema")
        equal(digest(expected_payload), request["payload_sha256"], "request hash")
        equal(request["available_result_event_ids"], available, "immutable observation event IDs")
        equal(request["observation_sha256"], digest(machine.events), "observation hash")
        equal(request["state_digest"], digest(machine.s), "pre-generation state digest")
        equal(request["state_version"], machine.s["revision"], "pre-generation revision")
        if live:
            from .live_client import wire_messages
            input_units = len(request["input_token_ids"])
            equal(digest(request["input_token_ids"]), request["input_token_sha256"], "actual input IDs hash")
            equal(wire_messages(expected_payload), request["wire_messages"], "wire message rendering")
            equal(request["engine_schema"], row["engine_schema"], "shared frozen engine grammar")
        else:
            input_units = (len(canonical(expected_payload).encode()) + 3) // 4
            equal(input_units, request["simulated_input_units"], "simulated input units")
        equal(min(limits["request_output_tokens"], limits["episode_output_tokens"] - tokens,
                  limits["context_tokens"] - input_units), request["effective_output_cap"], "effective simulated cap")
        equal(request["budget_before"], counters, "prefix counters")
        if counters["model_request_attempts"] >= limits["model_requests"] or counters["parsed_attempts"] >= limits["parsed_tool_or_final_attempts"]:
            raise AuditMismatch("generation after budget")
        last_ids = list(available)
        counters["model_request_attempts"] += 1
        response = request["response"]
        if response.get("status") != "OK":
            reason = "REQUEST_TIMEOUT" if response.get("status") == "REQUEST_TIMEOUT" else "ADAPTER_ERROR"
            if live:
                unknown_cost = True
                input_token_total += len(response.get("prompt_token_ids") or [])
                output_observed += len(response.get("output_token_ids", []))
                counters["model_responses_received"] += response.get("model_responses_received", 0)
            equal(counters, request["budget_after"], "failed request counters")
            continue
        counters["model_responses_received"] += 1
        if live:
            from .transport_audit import verify
            raw_transport = verify(response, request)
            units, elapsed = len(raw_transport["ids"]), response["elapsed_seconds"]
            input_token_total += len(raw_transport["prompt_ids"])
            output_observed += units
            if not 0 < request["request_timeout_seconds"] <= 180 or request["elapsed_episode_seconds_before_submission"] < 0:
                raise AuditMismatch("actual deadline parameters")
        else:
            units, elapsed = response["simulated"]["output_tokens"], response["simulated"]["elapsed_seconds"]
        if type(units) is not int or not 0 <= units <= request["effective_output_cap"] or elapsed < 0:
            reason = "ADAPTER_ERROR"
            equal(counters, request["budget_after"], "invalid response counters")
            continue
        tokens += units
        seconds += elapsed
        raw = response["raw_text"]
        equal(text_hash(raw), response["raw_utf8_sha256"], "raw UTF8 hash")
        episode_time = request["elapsed_episode_seconds_after_response"] if live else seconds
        if elapsed >= limits["request_seconds"] or episode_time >= limits["episode_seconds"]:
            reason = "REQUEST_TIMEOUT" if elapsed >= limits["request_seconds"] else "EPISODE_DEADLINE"
            equal(counters, request["budget_after"], "deadline counters")
            continue
        try:
            action = parse(raw, public["tools"])
        except ContractError as error:
            counters["parsed_attempts"] += int(is_parsed_object(raw))
            counters["contract_rejections"] += 1
            reason = "CONTRACT_FAILURE"
            equal(request["parse"], {"accepted": False, "error": error.public()}, "parse rejection")
        else:
            counters["parsed_attempts"] += 1
            equal(request["parse"], {"accepted": True, "action": action}, "raw parsed action")
            if action["tool"] == "final":
                declared |= action["arguments"]["recovered"] is True
                counters["final_attempts"] += 1
                fresh = current_validation(machine.events, available, machine.s, start["scope_id"])
                if row["policy"] == "R1" and fresh is None:
                    counters["guard_rejections"] += 1
                    equal(request["guard"]["code"], "FRESH_VALIDATION_REQUIRED", "public guard error")
                    equal(request["tool_event"], None, "guard must not create tool result")
                    if counters["guard_feedbacks"] == limits["guard_feedbacks"]:
                        reason = "GUARD_LIMIT"
                    else:
                        counters["guard_feedbacks"] += 1
                        messages = [*messages, {"role": "assistant", "content": raw},
                                    {"role": "user", "content": canonical({"error": {"code": "FRESH_VALIDATION_REQUIRED"}})}]
                else:
                    final, final_ids = action, list(available)
                    reason = "ACCEPTED_FINAL"
            else:
                event = machine.event(action, turn, available)
                equal(event, request["tool_event"], "independent callable/result/state replay")
                counters["callable_executions"] += 1
                counters["side_effect_writes"] += int(event["side_effect_write"])
                visible = {k: event[k] for k in ("event_id", "tool", "status", "result")}
                messages = [*messages, {"role": "assistant", "content": raw}, {"role": "user", "content": canonical({"tool_result": visible})}]
                available.append(event["event_id"])
        equal(counters, request["budget_after"], "post-response counters")
    if reason == "UNKNOWN":
        if counters["parsed_attempts"] >= limits["parsed_tool_or_final_attempts"]:
            reason = "TOOL_CALL_LIMIT"
        elif counters["model_request_attempts"] >= limits["model_requests"]:
            reason = "MODEL_CALL_LIMIT"
        elif tokens >= limits["episode_output_tokens"]:
            reason = "TOKEN_LIMIT"
        elif seconds >= limits["episode_seconds"]:
            reason = "EPISODE_DEADLINE"
        elif not live and (len(canonical({**public, "messages": messages}).encode()) + 3) // 4 >= limits["context_tokens"]:
            reason = "CONTEXT_LIMIT"
        elif live and row.get("preparation_failures"):
            reason = "ADAPTER_ERROR"
        elif live and row.get("terminal_preflight"):
            preflight = row["terminal_preflight"]
            if ("prepared" in preflight and len(preflight["prepared"]["token_ids"]) >= limits["context_tokens"]):
                reason = "CONTEXT_LIMIT"
            elif preflight.get("remaining_worker_seconds", 1) <= 0:
                reason = "WORKER_DEADLINE"
            elif preflight.get("elapsed_episode_seconds", 0) >= limits["episode_seconds"]:
                reason = "EPISODE_DEADLINE"
            else:
                raise AuditMismatch("unsupported preflight termination")
        else:
            reason = "CLIENT_EXHAUSTED"
    equal(machine.s, row["terminal_snapshot"], "terminal state")
    equal(digest(machine.s), row["terminal_snapshot_sha256"], "terminal frozen digest")
    equal(machine.events, row["events"], "complete tool event ledger")
    equal(final, row["final_action"], "accepted final")
    equal(final_ids, row["final_available_result_event_ids"], "final pre-observation IDs")
    equal(last_ids, row["last_generation_available_result_event_ids"], "last generation pre-observation IDs")
    equal(reason, row["terminal_reason"], "independent terminal branch")
    equal(counters, row["accounting"]["actual" if live else "synthetic"], "total request/attempt/callable counters")
    if live:
        equal(input_token_total if not unknown_cost else None, row["accounting"]["actual_input_tokens"], "actual input token sum")
        equal(output_observed if not unknown_cost else None, row["accounting"]["actual_output_tokens"], "actual output token sum")
    else:
        equal(tokens, row["accounting"]["simulated_output_tokens"], "simulated output units")
    components = {}
    for index, name in enumerate(("F1", "F2"), 1):
        actual = replay_continuation(task, machine.s, name)
        reference = replay_continuation(task, task["checkpoint"], name)
        equal(actual, row["continuations"][name]["actual"], "raw continuation actual:" + name)
        equal(reference, row["continuations"][name]["reference"], "raw continuation reference:" + name)
        components[f"C{index}"] = continuation_correct(task, actual, reference)
        equal(components[f"C{index}"], row["continuations"][name]["correct"], "continuation result:" + name)
    B = canonical(quantities(machine.s)) == canonical(quantities(task["checkpoint"]))
    P = protection(task, start, [start, *[s for e in machine.events for s in (e["state_before"], e["state_after"])], machine.s])
    fresh_id = current_validation(machine.events, final_ids if final else last_ids, machine.s, start["scope_id"])
    F = fresh_id is not None
    args = final["arguments"] if final else None
    q = args["quantities"] if args else []
    final_map = {r["record_id"]: r["quantity"] for r in q}
    final_ok = bool(args and args["task_id"] == task["task_id"] and args["recovered"] is True and len(q) == len(final_map)
                    and final_map == {k: r["quantity"] for k, r in machine.s["records"].items()}
                    and args["validation_event_id"] == fresh_id)
    Q = B and P and all(components.values())
    result = {"D": bool(declared), "B": B, "P": P, "F": F, **components,
              "Q": Q, "W": Q and reason == "ACCEPTED_FINAL" and final_ok and F,
              "FINAL_CONTENT_CORRECT": final_ok, "fresh_event_id": fresh_id,
              "F_context": "FINAL_PRE_GENERATION" if final else "LAST_GENERATION_DIAGNOSTIC"}
    for key, value in result.items():
        equal(value, row["reported_endpoint"][key], "independent metric:" + key)
    return {"episode_id": row["episode_id"], "status": "PASS", "independently_recomputed": result,
            "terminal_reason": reason, "requests_replayed": len(row["requests"]), "tool_events_replayed": len(machine.events),
            "reconstructed_terminal_snapshot": copy.deepcopy(machine.s)}


def verify_coverage(manifest):
    entries = manifest["episodes"]
    identities = [r["episode_id"] for r in entries]
    paths = [r["path"] for r in entries]
    if len(set(identities)) != len(entries) or len(set(paths)) != len(entries):
        raise AuditMismatch("duplicate episode identity/path")
    control_ids = {f"DEV-{family}{index:02d}_{mode}" for family in "ABC" for index in range(2)
                   for mode in ("clean", "no_recovery", "scripted_correct")}
    actual_controls = {r["episode_id"] for r in entries if r["kind"] == "CONTROL"}
    if control_ids != actual_controls or manifest["controls"] != 18:
        raise AuditMismatch("missing/extra DEV controls")
    diagnostic_ids = {"DEV-C00_diagnostic_business_only_R0", *{
        "DEV-A00_diagnostic_" + suffix for suffix in ("stale_validation_R0", "wrong_business_R1", "no_final_R0",
                                                      "guard_retry_R0", "guard_retry_R1", "guard_twice_R1", "repeated_final_R1")}}
    if ({r["episode_id"] for r in entries if r["kind"] == "DIAGNOSTIC"} != diagnostic_ids
            or manifest["diagnostics"] != 8 or len(entries) != 26):
        raise AuditMismatch("missing/extra diagnostic coverage")


def audit_run(run, output):
    run = __import__("pathlib").Path(run)
    output = __import__("pathlib").Path(output)
    if read_json(run / "run_manifest.json").get("phase") == "P1":
        from .p1_audit import audit_run as audit_p1
        return audit_p1(run, output)
    if output.exists():
        raise FileExistsError(output)
    replay_root = fresh_dir(output.parent / (output.stem + "-replayed-states"))
    manifest = read_json(run / "run_manifest.json")
    verify_coverage(manifest)
    frozen = read_json(run / "execution_freeze.json")
    equal(manifest["execution_source_sha256"], frozen["working_source_sha256"], "run vs execution freeze")
    equal(source_manifest(), frozen, "current source vs frozen CPU run")
    rows = []
    for entry in manifest["episodes"]:
        path = run / entry["path"]
        if not path.is_file() or path.is_symlink() or ".." in __import__("pathlib").Path(entry["path"]).parts:
            raise AuditMismatch("unsafe/missing episode path")
        record = read_json(path)
        equal(record["episode_id"], entry["episode_id"], "episode path/identity mapping")
        equal(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"], "stored trace file bytes")
        equal(digest(record), entry["canonical_sha256"], "stored trace canonical identity")
        task = read_json(run / "inputs" / (record["task_id"] + ".json"))
        verified = audit_episode(task, record)
        write_new(replay_root / (record["episode_id"] + ".json"), verified.pop("reconstructed_terminal_snapshot"))
        rows.append(verified)
    receipt = {"status": "PASS", "evidence_kind": "STORED_TRACE_REPLAY_OF_SYNTHETIC_CONTRACT_TESTS",
               "new_model_requests": 0, "rows": rows, "episodes": len(rows),
               "controls": manifest["controls"], "diagnostics": manifest["diagnostics"],
               "shared_with_runner": ["strict parser", "public input builder", "serialization/public schemas"],
               "independent": ["state transitions", "fault reconstruction", "event/observation checks", "protection", "continuations", "B/P/F/C/Q/W", "budget and terminal branch"],
               "actual_tokenizer_reencoding": "NOT_RUN", "external_human_or_machine_review": False}
    write_new(output, receipt)
    return receipt
