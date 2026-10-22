"""CPU episode engine with immutable generation observations and bounded guard."""
import copy
import time
from .common import CASE_ROOT, canonical, digest, read_json, text_hash
from .contracts import parse, ContractError, is_parsed_object
from .public import initial_payload, append_result, append_guard
from .world import World
from .scoring import assess, continuations, fresh_validation


def run_episode(task, client, root=None, *, policy="R0", start="error", episode_id=None, limits=None):
    if policy not in ("R0", "R1") or start not in ("clean", "error"):
        raise ValueError("unknown policy/start")
    limits = copy.deepcopy(limits or read_json(CASE_ROOT / "specs/protocol.json")["limits"])
    live = getattr(client, "is_live", False)
    started = time.perf_counter()
    initial = copy.deepcopy(task["checkpoint" if start == "clean" else "error_state"])
    world = World(task, initial, root)
    fixture_init_seconds = time.perf_counter() - started
    payload = initial_payload(task)
    messages = payload["messages"]
    events, available, requests = [], [], []
    counters = {k: 0 for k in ("model_request_attempts", "model_responses_received", "server_abort_requests",
                                "parsed_attempts", "contract_rejections", "callable_executions", "side_effect_writes",
                                "final_attempts", "guard_rejections", "guard_feedbacks")}
    simulated_tokens, virtual_seconds = 0, 0.0
    input_tokens_total, output_tokens_observed, unknown_response_cost = 0, 0, False
    final, final_available, last_available = None, [], []
    preparation_failures = []
    terminal_preflight = None
    reason = "UNKNOWN"
    while reason == "UNKNOWN":
        if live and client.remaining_seconds() <= 0:
            reason = "WORKER_DEADLINE"
            terminal_preflight = {"reason": reason, "remaining_worker_seconds": client.remaining_seconds()}
            break
        if counters["parsed_attempts"] >= limits["parsed_tool_or_final_attempts"]:
            reason = "TOOL_CALL_LIMIT"
            break
        if counters["model_request_attempts"] >= limits["model_requests"]:
            reason = "MODEL_CALL_LIMIT"
            break
        if simulated_tokens >= limits["episode_output_tokens"]:
            reason = "TOKEN_LIMIT"
            break
        elapsed_episode = time.perf_counter() - started if live else virtual_seconds
        if elapsed_episode >= limits["episode_seconds"]:
            reason = "EPISODE_DEADLINE"
            terminal_preflight = {"reason": reason, "elapsed_episode_seconds": elapsed_episode}
            break
        pre_available = tuple(available)
        if not live:
            last_available = list(pre_available)
        body = {"messages": copy.deepcopy(messages), "tools": copy.deepcopy(payload["tools"]),
                "output_schema": copy.deepcopy(payload["output_schema"])}
        try:
            prepared = client.prepare(body) if live else None
        except Exception as error:
            preparation_failures.append({"error_type": type(error).__name__, "error": str(error),
                                         "model_request_attempted": False})
            reason = "ADAPTER_ERROR"
            break
        input_units = len(prepared["token_ids"]) if live else (len(canonical(body).encode()) + 3) // 4
        remaining_context = limits["context_tokens"] - input_units
        if remaining_context <= 0:
            reason = "CONTEXT_LIMIT"
            if live:
                terminal_preflight = {"reason": reason, "prepared": prepared, "logical_payload": body}
            break
        cap = min(limits["request_output_tokens"], limits["episode_output_tokens"] - simulated_tokens, remaining_context)
        row = {"request_id": f"g{len(requests):03d}", "generated_in_turn": len(requests),
               "model_payload": body, "payload_sha256": digest(body),
               "available_result_event_ids": list(pre_available),
               "observation_sha256": digest([e for e in events if e["event_id"] in pre_available]),
               "state_digest": digest(world.state), "state_version": world.state["revision"],
               "budget_before": copy.deepcopy(counters), "effective_output_cap": cap,
               "input_token_ids": prepared["token_ids"] if live else None,
               "input_token_sha256": prepared["input_token_sha256"] if live else None, "actual_usage": None,
               "rendered_prompt": prepared["prompt_text"] if live else None,
               "wire_messages": prepared["messages"] if live else None,
               "engine_schema": client.schema if live else None,
               "simulated_input_units": None if live else input_units,
               "context_gate": "PINNED_LOCAL_TOKENIZER" if live else "NOT_RUN_REAL_TOKENIZATION", "response": None, "parse": None,
               "tool_event": None, "guard": None, "decision": None}
        try:
            if live:
                left = limits["episode_seconds"] - (time.perf_counter() - started)
                if left <= 0:
                    reason = "EPISODE_DEADLINE"
                    break
                last_available = list(pre_available)
                row["request_timeout_seconds"] = min(limits["request_seconds"], left, client.remaining_seconds())
                row["elapsed_episode_seconds_before_submission"] = time.perf_counter() - started
                response = client.respond(copy.deepcopy(body), row["request_timeout_seconds"], cap=cap,
                                          request_id=(episode_id or task["task_id"]) + f"-g{len(requests):03d}", prepared=prepared)
            else:
                response = client.respond(copy.deepcopy(body), min(limits["request_seconds"], limits["episode_seconds"] - virtual_seconds))
        except StopIteration:
            # No simulated generation occurred; the last actual generation remains the preceding row.
            last_available = requests[-1]["available_result_event_ids"] if requests else []
            reason = "CLIENT_EXHAUSTED"
            break
        except Exception as error:
            reason = "ADAPTER_ERROR"
            if live:
                unknown_response_cost = True
            row["response"] = {"status": "ADAPTER_ERROR", "error_type": type(error).__name__}
            counters["model_request_attempts"] += 1
            row["budget_after"] = copy.deepcopy(counters)
            requests.append(row)
            break
        counters["model_request_attempts"] += 1
        row["response"] = copy.deepcopy(response)
        if live:
            row["elapsed_episode_seconds_after_response"] = time.perf_counter() - started
        requests.append(row)
        if response.get("status") != "OK":
            reason = "REQUEST_TIMEOUT" if response.get("status") == "REQUEST_TIMEOUT" else "ADAPTER_ERROR"
            if live:
                unknown_response_cost = True
                output_tokens_observed += len(response.get("output_token_ids", []))
                input_tokens_total += len(response.get("prompt_token_ids") or [])
                counters["model_responses_received"] += response.get("model_responses_received", 0)
                row["actual_usage"] = response.get("usage")
            row["budget_after"] = copy.deepcopy(counters)
            break
        counters["model_responses_received"] += 1
        units = len(response["output_token_ids"]) if live else response["simulated"]["output_tokens"]
        elapsed = response["elapsed_seconds"] if live else response["simulated"]["elapsed_seconds"]
        if type(units) is not int or units < 0 or units > cap or elapsed < 0:
            reason = "ADAPTER_ERROR"
            row["budget_after"] = copy.deepcopy(counters)
            break
        simulated_tokens += units
        virtual_seconds += elapsed
        if live:
            input_tokens_total += len(response["prompt_token_ids"])
            output_tokens_observed += units
            row["actual_usage"] = response["usage"]
        raw = response["raw_text"]
        row["response"]["raw_utf8_sha256"] = text_hash(raw)
        if elapsed >= limits["request_seconds"]:
            reason = "REQUEST_TIMEOUT"
        elif (time.perf_counter() - started if live else virtual_seconds) >= limits["episode_seconds"]:
            reason = "EPISODE_DEADLINE"
        if reason != "UNKNOWN":
            row["decision"] = "NO_EXECUTION_TIME_LIMIT"
            row["budget_after"] = copy.deepcopy(counters)
            break
        try:
            action = parse(raw, payload["tools"])
        except ContractError as error:
            counters["parsed_attempts"] += int(is_parsed_object(raw))
            counters["contract_rejections"] += 1
            row["parse"] = {"accepted": False, "error": error.public()}
            row["decision"] = "CONTRACT_REJECTED"
            row["budget_after"] = copy.deepcopy(counters)
            reason = "CONTRACT_FAILURE"
            break
        counters["parsed_attempts"] += 1
        row["parse"] = {"accepted": True, "action": copy.deepcopy(action)}
        if action["tool"] == "final":
            counters["final_attempts"] += 1
            fresh_id = fresh_validation(events, pre_available, world.state, task["public_request"]["scope_id"])
            if policy == "R1" and fresh_id is None:
                counters["guard_rejections"] += 1
                row["guard"] = {"code": "FRESH_VALIDATION_REQUIRED", "checked": "public scope/digest/version/pre-generation event IDs only"}
                row["decision"] = "GUARD_REJECTED"
                if counters["guard_feedbacks"] >= limits["guard_feedbacks"]:
                    reason = "GUARD_LIMIT"
                else:
                    counters["guard_feedbacks"] += 1
                    messages = append_guard(messages, raw)
            else:
                final, final_available = copy.deepcopy(action), list(pre_available)
                row["decision"] = "FINAL_ACCEPTED"
                reason = "ACCEPTED_FINAL"
        else:
            event = world.execute(action, len(requests) - 1, pre_available)
            counters["callable_executions"] += 1
            counters["side_effect_writes"] += int(event["side_effect_write"])
            row["tool_event"] = copy.deepcopy(event)
            row["decision"] = "TOOL_RESULT_RETURNED"
            events.append(copy.deepcopy(event))
            messages = append_result(messages, raw, event)
            available.append(event["event_id"])
        row["budget_after"] = copy.deepcopy(counters)
    terminal = world.snapshot()
    snapshot_sha = digest(terminal)
    terminal_wall_seconds = time.perf_counter() - started
    continuation_started = time.perf_counter()
    branches = continuations(task, terminal)
    continuation_seconds = time.perf_counter() - continuation_started
    if digest(world.state) != snapshot_sha:
        raise AssertionError("continuation mutated frozen terminal snapshot")
    endpoint = assess(task, initial, terminal, events, final, final_available, last_available, reason, branches)
    endpoint["D"] = any(r.get("parse", {}).get("accepted") and r["parse"]["action"]["tool"] == "final"
                        and r["parse"]["action"]["arguments"]["recovered"] is True
                        for r in requests if r.get("parse"))
    return {"episode_id": episode_id or task["task_id"] + "_" + policy,
            "task_id": task["task_id"], "family": task["family"], "policy": policy, "start": start,
            "evidence_kind": "MODEL_EPISODE" if live else "SYNTHETIC_CONTRACT_TEST",
            "model_requests": counters["model_request_attempts"] if live else 0,
            "initial_state": initial, "initial_state_sha256": digest(initial),
            "initial_payload_sha256": digest(payload), "requests": requests, "events": events,
            "preparation_failures": preparation_failures,
            "terminal_preflight": terminal_preflight,
            "terminal_snapshot": terminal, "terminal_snapshot_sha256": snapshot_sha,
            "final_action": final, "final_available_result_event_ids": final_available,
            "last_generation_available_result_event_ids": last_available,
            "terminal_reason": reason, "reported_endpoint": endpoint, "continuations": branches,
            "accounting": {"actual" if live else "synthetic": counters,
                           "simulated_output_tokens": None if live else simulated_tokens,
                           "simulated_elapsed_seconds": None if live else virtual_seconds,
                           "actual_input_tokens": input_tokens_total if live and not unknown_response_cost else None,
                           "actual_output_tokens": output_tokens_observed if live and not unknown_response_cost else None,
                           "observed_input_tokens": input_tokens_total if live else None,
                           "observed_output_tokens": output_tokens_observed if live else None,
                           "unknown_response_cost": unknown_response_cost if live else None,
                           "episode_wall_seconds": terminal_wall_seconds if live else None,
                           "CPU_fixture_init_seconds": fixture_init_seconds if live else None,
                           "actual_worker_seconds": None,
                           "CPU_episode_seconds": time.perf_counter() - started,
                           "CPU_continuation_seconds": continuation_seconds}}
