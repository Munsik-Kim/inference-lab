"""One sequential policy episode with raw request/observation/attempt accounting.

The adapter is injected; importing this CPU path does not import Torch,
Transformers, vLLM, a tokenizer or a network client package.
"""
import asyncio
import copy
import hashlib
import time

from .client import build_payload
from .fixtures import digest
from .policies import PolicySession, retry_messages
from .prompts import append_result, initial_messages, observation_view
from .scoring import assess
from .schemas import server_schema
from .tools import Fixture, tool_definitions


UNOBSERVED_STATUSES = {"RAW_UNOBSERVABLE", "OOM", "SERVER_ERROR", "INFRA_ERROR", "INPUT_IDENTITY_MISMATCH", "SERVER_OUTPUT_BUDGET_VIOLATION"}


def _tokenize(adapter, messages):
    result = adapter.tokenize(copy.deepcopy(messages))
    if type(result) is not dict or type(result.get("token_ids")) is not list or any(type(token) is not int for token in result["token_ids"]) or type(result.get("prompt_text")) is not str:
        raise ValueError("adapter returned invalid tokenization")
    if result.get("input_token_sha256") != digest(result["token_ids"]):
        raise ValueError("adapter token hash is inconsistent")
    return result


def _response_failure(response):
    status = response.get("status", "INFRA_ERROR")
    if status == "REQUEST_TIMEOUT":
        return "REQUEST_TIMEOUT", "OBSERVED"
    if status == "OOM":
        return "OOM", "UNOBSERVED"
    if status == "SERVER_ERROR":
        return "SERVER_ERROR", "UNOBSERVED"
    return "INFRA_ERROR", "UNOBSERVED"


async def run_episode(task, policy, adapter, new_fixture_root, identity):
    """Run one registered task-policy-cache cell exactly once, without auto retry.

    ``adapter.tokenize(messages)`` is synchronous and uses the pinned local
    tokenizer. ``await adapter.send(payload, timeout)`` returns client.chat's raw
    record. Only the messages and public schema are passed to that adapter.
    """
    started = time.monotonic()
    identity = copy.deepcopy(identity)
    definitions = tool_definitions(task["family"])
    session = PolicySession(policy, definitions)
    fixture = Fixture(task, new_fixture_root)
    messages = initial_messages(task)
    initial_prompt_hash = digest(messages)
    initial_input_hash = None
    initial_prompt_tokens = None
    observations = []
    requests = []
    preparation_failures = []
    failures = []
    final_action = None
    final_available = []
    observation_status = "OBSERVED"
    while session.terminal_reason == "UNKNOWN":
        generation_turn = len(requests)
        available = [row["event_id"] for row in observations]
        state = fixture.snapshot()
        row = {"generation_turn": generation_turn, "messages": copy.deepcopy(messages), "available_result_event_ids": list(available), "observation_sha256": digest(observations), "state_digest": digest(state), "state_version": fixture.version, "budget_before": session.budget.ledger(), "input_token_ids": None, "input_token_sha256": None, "prompt_text": None, "request_plan": None, "response": None, "parse": None, "event": None, "feedback": None}
        try:
            tokenized = _tokenize(adapter, messages)
        except Exception as error:
            session.terminal_reason = "INFRA_ERROR"
            observation_status = "UNOBSERVED"
            row["local_error"] = {"type": type(error).__name__, "message": str(error)}
            # No network request was attempted; keep the failed input gate as a
            # separate preparation record instead of inventing a model response.
            failures.append({"generation_turn": generation_turn, "kind": "TOKENIZATION_GATE", "details": row["local_error"]})
            row["model_request_attempted"] = False
            preparation_failures.append(row)
            break
        row.update(input_token_ids=copy.deepcopy(tokenized["token_ids"]), input_token_sha256=tokenized["input_token_sha256"], prompt_text=tokenized["prompt_text"])
        if not requests and identity.get("expected_initial_input_token_sha256") not in (None, tokenized["input_token_sha256"]):
            session.terminal_reason = "INFRA_ERROR"
            observation_status = "UNOBSERVED"
            row["local_error"] = {"type": "FrozenInputMismatch", "message": "Actual initial IDs differ from sealed input"}
            row["model_request_attempted"] = False
            preparation_failures.append(row)
            failures.append({"generation_turn": 0, "kind": "FROZEN_INITIAL_INPUT_GATE", "details": row["local_error"]})
            break
        if initial_input_hash is None:
            initial_input_hash = tokenized["input_token_sha256"]
            initial_prompt_tokens = len(tokenized["token_ids"])
        elapsed = time.monotonic() - started
        bounds = session.budget.request_bounds(len(tokenized["token_ids"]), elapsed)
        if not bounds["allowed"]:
            session.terminal_reason = bounds["terminal_reason"]
            break
        plan = session.begin_request(len(tokenized["token_ids"]), elapsed)
        row["request_plan"] = plan
        row["model_request_attempted"] = True
        row["retry_consumed"] = session.budget.retry_consumed - row["budget_before"]["retry_consumed"]
        request_id = str(identity.get("episode_id", task["task_id"] + "-" + policy)) + f"-g{generation_turn}"
        payload = build_payload(messages, schema=server_schema(definitions) if policy == "G" else None, cap=plan["max_tokens"], request_id=request_id)
        try:
            # The transport owns the exact deadline and returns any partial SSE
            # evidence. A second identical timer would race its cleanup/receipt.
            response = await adapter.send(payload, plan["timeout_seconds"])
        except TimeoutError as error:
            response = {"status": "REQUEST_TIMEOUT", "model_request_attempts": 1, "model_responses_received": 0, "server_abort_requests": 0, "output_token_ids": [], "output_tokens": 0, "raw_text": "", "error_type": type(error).__name__, "drain_status": "NOT_CONFIRMED", "stream_close_on_error": True}
        except Exception as error:
            response = {"status": "INFRA_ERROR", "model_request_attempts": 1, "model_responses_received": 0, "server_abort_requests": 0, "output_token_ids": [], "output_tokens": 0, "raw_text": "", "error_type": type(error).__name__, "error": str(error)}
        if type(response) is not dict:
            response = {"status": "INFRA_ERROR", "model_request_attempts": 1, "model_responses_received": 0, "output_token_ids": [], "output_tokens": 0, "raw_text": "", "error_type": "InvalidAdapterResponse"}
        response.setdefault("request_body", copy.deepcopy(payload))
        response.setdefault("raw_text", "")
        if type(response["raw_text"]) is str:
            response.setdefault("raw_utf8_sha256", hashlib.sha256(response["raw_text"].encode("utf-8")).hexdigest())
        response["transport_status"] = response.get("status", "INFRA_ERROR")
        row["response"] = copy.deepcopy(response)
        requests.append(row)
        output_ids = response.get("output_token_ids", [])
        if type(output_ids) is not list or any(type(token) is not int for token in output_ids):
            output_ids = []
            response["status"] = "RAW_UNOBSERVABLE"
        count = len(output_ids)
        try:
            if response.get("model_responses_received", int(response.get("status") == "OK")) == 1:
                session.budget.finish_response(count)
            else:
                session.budget.fail_request(observed_output_tokens=count, abort_requested=bool(response.get("server_abort_requests", 0)))
        except ValueError as error:
            response["status"] = "SERVER_OUTPUT_BUDGET_VIOLATION"
            row["budget_error"] = str(error)
        if response.get("status") == "OK":
            # The request was observed even when its model behavior is later
            # rejected. A different server-rendered input invalidates pairing.
            if response.get("prompt_token_ids") != tokenized["token_ids"] or response.get("input_token_sha256") != tokenized["input_token_sha256"]:
                response["status"] = "INPUT_IDENTITY_MISMATCH"
            elif type(response.get("raw_text")) is not str or response.get("output_tokens", count) != count or response.get("raw_utf8_sha256") != hashlib.sha256(response["raw_text"].encode("utf-8")).hexdigest():
                response["status"] = "RAW_UNOBSERVABLE"
        row["response"] = copy.deepcopy(response)
        if response.get("status") != "OK":
            session.terminal_reason, observation_status = _response_failure(response)
            failures.append({"generation_turn": generation_turn, "kind": response.get("status", "INFRA_ERROR"), "details": {key: response.get(key) for key in ("http_status", "error_type", "error", "drain_status")}})
            break
        decision = session.consume_response(response["raw_text"])
        row["retry_triggered"] = session.budget.retry_triggered - row["budget_before"]["retry_triggered"]
        row["parse"] = decision.parsed.as_dict()
        if not decision.execute:
            failures.append({"generation_turn": generation_turn, "kind": "CONTRACT_FAILURE", "details": decision.parsed.error.public()})
            row["feedback"] = decision.feedback
            if decision.retry:
                messages = retry_messages(messages, response["raw_text"], policy, decision.parsed.error)
                continue
            break
        event = fixture.execute(decision.parsed.action, generation_turn, available)
        row["event"] = copy.deepcopy(event)
        if event["status"] == "CONTRACT_ERROR":
            session.terminal_reason = "INFRA_ERROR"
            observation_status = "UNOBSERVED"
            failures.append({"generation_turn": generation_turn, "kind": "PARSER_FIXTURE_CONTRACT_MISMATCH", "details": event["contract_error"]})
            break
        if event["callable_executed"]:
            session.budget.record_execution(side_effect=event["side_effect_write"])
        if event["status"] != "SUCCESS":
            failures.append({"generation_turn": generation_turn, "kind": "BUSINESS_EXECUTION_ERROR", "details": copy.deepcopy(event["result"])})
        if decision.parsed.action["tool"] == "final":
            final_action = copy.deepcopy(decision.parsed.action)
            final_available = list(available)
            endpoint = assess(task, fixture.events, final_action, fixture.snapshot(), fixture.version, final_available)
            session.after_execution(final=True, verified=endpoint["VERIFIED_WORKFLOW_SUCCESS"], state_correct=endpoint["STATE_CORRECT"])
            break
        # An actual callable result is returned regardless of business success;
        # parser rejections and host retry messages never create fake events.
        observations.append(observation_view(event))
        messages = append_result(messages, response["raw_text"], event)
        session.after_execution()
    final_state = fixture.snapshot()
    endpoint = assess(task, fixture.events, final_action, final_state, fixture.version, final_available, session.terminal_reason)
    replayable_endpoint = copy.deepcopy(endpoint)
    if observation_status == "UNOBSERVED":
        endpoint["VERIFIED_WORKFLOW_SUCCESS"] = None
    if not endpoint["VERIFIED_WORKFLOW_SUCCESS"] and final_action is not None:
        failures.append({"generation_turn": len(requests) - 1, "kind": "ENDPOINT_FAILURE", "details": {"unmet_obligations": endpoint["unmet_obligations"]}})
    return {"episode_id": identity.get("episode_id", task["task_id"] + "-" + str(identity.get("kv", "UNKNOWN")) + "-" + policy), "task_id": task["task_id"], "family": task["family"], "stage": identity.get("stage", task["split"]), "kv": identity.get("kv"), "policy": policy, "identity": identity, "evidence_kind": identity.get("evidence_kind", "MODEL_EPISODE"), "canonical_task_sha256": task["canonical_task_sha256"], "initial_prompt_hash": initial_prompt_hash, "initial_input_token_sha256": initial_input_hash, "initial_prompt_tokens": initial_prompt_tokens, "preparation_failures": preparation_failures, "requests": requests, "events": copy.deepcopy(fixture.events), "final_action": final_action, "final_available_result_event_ids": final_available, "final_state": final_state, "state_version": fixture.version, "terminal_reason": session.terminal_reason, "observation_status": observation_status, "reported_endpoint": endpoint, "independently_replayable_endpoint": replayable_endpoint, "budget": session.budget.ledger(), "first_failure": failures[0] if failures else None, "failures": failures, "last_available_result_event_ids": [row["event_id"] for row in observations], "last_observation_sha256": digest(observations), "requires_server_drain_check": any((row.get("response") or {}).get("drain_status") == "NOT_CONFIRMED" for row in requests), "wall_seconds": time.monotonic() - started}
