"""Post-hoc input-integrity checks, separate from frozen execution and scoring.

This stdlib CPU verifier does not generate HOLDOUT, re-tokenize messages, execute
the model, connect to a server, or change endpoint scores. It closes two explicit
coverage gaps in the original auditor: frozen first inputs and full G schemas.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE / "src"))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def verify_episode_inputs(task, frozen_input, episode, grammar, execution_source_sha256, protocol_sha256):
    """Check model-visible data only; no submitted success bool is consumed."""
    from agent_contract_cost.client import build_payload
    from agent_contract_cost.contracts import parse_action
    from agent_contract_cost.policies import NEUTRAL_FEEDBACK
    from agent_contract_cost.tools import tool_definitions

    errors = []
    def check(condition, label, turn=None):
        if not condition:
            errors.append({"check": label, "generation_turn": turn})
    identity = episode.get("identity", {})
    policy = episode.get("policy")
    check(policy in ("C0", "G", "N", "E"), "registered policy")
    check(episode.get("task_id") == task["task_id"] and identity.get("task_id") == task["task_id"], "registered task identity")
    check(identity.get("execution_source_sha256") == execution_source_sha256, "frozen execution source identity")
    check(identity.get("protocol_sha256") == protocol_sha256, "registered protocol identity")
    check(identity.get("expected_initial_input_token_sha256") == frozen_input["input_token_sha256"], "sealed initial input identity")
    check(identity.get("schema_sha256") == digest(grammar), "public schema identity")
    check(episode.get("canonical_task_sha256") == task["canonical_task_sha256"], "canonical task hash")
    check(episode.get("initial_prompt_hash") == frozen_input["messages_sha256"], "frozen initial messages hash")
    check(episode.get("initial_input_token_sha256") == frozen_input["input_token_sha256"], "frozen initial token hash")
    messages = copy.deepcopy(frozen_input["messages"])
    observations = []
    available = []
    definitions = tool_definitions(task["family"])
    retry_feedbacks = 0
    requests = episode["requests"]
    for index, row in enumerate(requests):
        response = row["response"]
        turn = row["generation_turn"]
        check(turn == index, "request order", turn)
        check(row["messages"] == messages, "history reconstructed from sealed first messages and real events/feedback", turn)
        check(row["available_result_event_ids"] == available, "pre-generation observation IDs", turn)
        check(row["observation_sha256"] == digest(observations), "public observation hash", turn)
        check(row["input_token_sha256"] == digest(row["input_token_ids"]), "saved input ID hash", turn)
        if index == 0:
            check(row["input_token_ids"] == frozen_input["token_ids"], "actual initial token IDs equal frozen IDs", turn)
            check(row["prompt_text"] == frozen_input["prompt_text"], "actual initial template text equals frozen text", turn)
        body = response.get("request_body")
        request_id = str(identity.get("episode_id", task["task_id"] + "-" + policy)) + f"-g{turn}"
        expected_body = build_payload(messages, schema=grammar if policy == "G" else None,
                                      cap=row["request_plan"]["max_tokens"], request_id=request_id)
        check(body == expected_body, "full actual HTTP body equals shared registered builder with public G schema only", turn)
        check(type(body) is dict and "tools" not in body and "tool_choice" not in body and "guided_json" not in body, "no native tool mode or legacy grammar", turn)
        if type(body) is dict:
            check(body.get("structured_outputs") == ({"json": grammar} if policy == "G" else None), "full public schema and no grammar in other policies", turn)
        if response.get("status") != "OK":
            check(row.get("event") is None and row.get("feedback") is None, "unobserved/transport-error response creates no execution or feedback", turn)
            check(index == len(requests) - 1, "no continuation after transport gate", turn)
            continue
        check(response.get("prompt_token_ids") == row["input_token_ids"], "server input IDs equal local saved IDs", turn)
        check(response.get("input_token_sha256") == row["input_token_sha256"], "server input hash equals local saved hash", turn)
        if response.get("prompt_text") is not None:
            check(response["prompt_text"] == row["prompt_text"], "server template text equals saved local template text", turn)
        raw = response["raw_text"]
        parsed = parse_action(raw, definitions)
        check(parsed.as_dict() == row["parse"], "same strict parser and public registry", turn)
        if parsed.accepted:
            event = row.get("event")
            check(event is not None, "accepted action has an event", turn)
            if event is None:
                continue
            check(event["tool"] == parsed.action["tool"] and event["arguments"] == parsed.action["arguments"], "executed action equals generated action without correction", turn)
            check(event["generated_after_event_ids"] == available, "execution event retains generation-time observations", turn)
            check(row.get("feedback") is None, "business execution result does not trigger policy-only retry", turn)
            if event["tool"] == "final":
                check(episode["final_available_result_event_ids"] == available, "final uses pre-generation validation observations", turn)
                check(index == len(requests) - 1, "no generation after final", turn)
            else:
                view = {key: copy.deepcopy(event[key]) for key in ("event_id", "tool", "status", "result")}
                observations.append(view)
                available.append(event["event_id"])
                messages.extend([{"role": "assistant", "content": raw}, {"role": "user", "content": canonical(view)}])
        else:
            check(row.get("event") is None, "rejected action creates no tool observation", turn)
            feedback = row.get("feedback")
            if feedback is not None:
                retry_feedbacks += 1
                check(policy in ("N", "E") and retry_feedbacks == 1, "feedback policy and one-retry limit", turn)
                expected = NEUTRAL_FEEDBACK if policy == "N" else json.dumps(parsed.error.public(), ensure_ascii=False, separators=(",", ":"))
                check(feedback == expected, "only registered neutral text or public code/path/expected feedback", turn)
                messages.extend([{"role": "assistant", "content": raw}, {"role": "user", "content": feedback}])
            else:
                check(index == len(requests) - 1, "no continuation without a registered retry", turn)
    return {"task_id": task["task_id"], "kv": episode["kv"], "policy": policy, "status": "PASS" if not errors else "FAIL", "requests_checked": len(requests), "errors": errors}


def verify_run(case, stage, output):
    from agent_contract_cost.prompts import initial_messages
    from agent_contract_cost.schemas import server_schema
    from agent_contract_cost.tools import tool_definitions

    case = case.resolve()
    output = output.resolve()
    if output.exists() or output.is_relative_to(case / "data") or output.is_relative_to(case / "src"):
        raise ValueError("Choose a new analysis output, not an input/source directory")
    output.mkdir(parents=True)
    label = "dev" if stage == "P1" else "holdout"
    runroot = case / "results" / ("p1_r1" if stage == "P1" else "p2_r1")
    input_path = case / "data" / (label + "_model_inputs.json")
    task_path = case / "data" / (label + "_tasks.json")
    manifest_path = case / "provenance" / (label + "_input_manifest.json")
    freeze_path = case / "provenance" / ("execution_freeze_" + stage.lower() + ".json")
    protocol_path = case / "specs/protocol.json"
    freeze, manifest = load(freeze_path), load(manifest_path)
    global_errors = []
    sources = []
    repo = case.parents[1]
    for entry in freeze["files"]:
        path = repo / entry["root_path"] if "root_path" in entry else case / entry["path"]
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            global_errors.append("Frozen source/input/identity changed: " + entry.get("path", entry.get("root_path")))
    if digest(freeze["files"]) != freeze["registration_bundle_sha256"] or digest(freeze["execution_source_files"]) != freeze["execution_source_sha256"]:
        global_errors.append("Freeze bundle/source digest is internally inconsistent")
    for name, path in (("tasks_sha256", task_path), ("model_inputs_sha256", input_path)):
        if sha(path) != manifest[name]:
            global_errors.append("Input manifest file hash differs: " + name)
    tasks = load(task_path)
    inputs = load(input_path)
    taskmap = {task["task_id"]: task for task in tasks}
    inputmap = {row["task_id"]: row for row in inputs}
    if len(taskmap) != len(tasks) or len(inputmap) != len(inputs) or set(taskmap) != set(inputmap):
        global_errors.append("Duplicate or mismatched registered tasks/inputs")
    grammars = {}
    for family in ("A", "B", "C"):
        definitions = tool_definitions(family)
        grammar = server_schema(definitions)
        grammars[family] = grammar
        if load(case / "specs" / ("tools_" + family + ".json")) != definitions or load(case / "specs" / ("grammar_" + family + ".json")) != grammar:
            global_errors.append("Frozen public definitions/schema differ: " + family)
    for task_id, task in taskmap.items():
        prepared = inputmap[task_id]
        public_task = {key: copy.deepcopy(task[key]) for key in ("task_id", "family", "prompt")}
        if initial_messages(public_task) != prepared["messages"]:
            global_errors.append("First messages differ from explicit public task allowlist: " + task_id)
        if digest(prepared["messages"]) != prepared["messages_sha256"] or digest(prepared["token_ids"]) != prepared["input_token_sha256"] or len(prepared["token_ids"]) != prepared["input_tokens"]:
            global_errors.append("Frozen message/token identities inconsistent: " + task_id)
        if hashlib.sha256(prepared["prompt_text"].encode()).hexdigest() != prepared["prompt_text_sha256"]:
            global_errors.append("Frozen rendered template text hash inconsistent: " + task_id)
        if digest(grammars[task["family"]]) != prepared["schema_sha256"]:
            global_errors.append("Frozen task schema identity differs: " + task_id)
    rows = []
    identities = set()
    for path in sorted((runroot / "cells").glob("*.json")):
        episode = load(path)
        task_id = episode["task_id"]
        if task_id not in taskmap:
            global_errors.append("Unregistered task in cell: " + path.name)
            continue
        key = (task_id, episode["kv"], episode["policy"])
        if key in identities:
            global_errors.append("Duplicate cell identity: " + repr(key))
        identities.add(key)
        row = verify_episode_inputs(taskmap[task_id], inputmap[task_id], episode, grammars[taskmap[task_id]["family"]], freeze["execution_source_sha256"], sha(protocol_path))
        row.update(source_file=str(path.relative_to(case)), source_sha256=sha(path))
        rows.append(row)
    expected = {(task_id, kv, policy) for task_id in taskmap for kv in ("K0", "K1") for policy in ("C0", "G", "N", "E")}
    missing = sorted(expected - identities)
    unexpected = sorted(identities - expected)
    if unexpected:
        global_errors.append("Unregistered matrix cell identities")
    status = "FAIL" if global_errors or any(row["status"] != "PASS" for row in rows) else ("PARTIAL" if missing else "PASS")
    for path in (task_path, input_path, manifest_path, freeze_path, protocol_path):
        sources.append({"path": str(path.relative_to(case)), "bytes": path.stat().st_size, "sha256": sha(path)})
    receipt = {"status": status, "stage": stage, "evidence_kind": "POSTHOC_INPUT_INTEGRITY_VERIFICATION", "new_model_requests": 0, "endpoint_scores_changed": False, "execution_source_or_inputs_changed": False, "planned_cells": len(expected), "checked_cells": len(rows), "checked_request_boundaries": sum(row["requests_checked"] for row in rows), "missing_cells": missing, "unexpected_cells": unexpected, "global_errors": global_errors, "failed_cells": [row["source_file"] for row in rows if row["status"] != "PASS"], "verification_source_sha256": sha(Path(__file__)), "execution_source_sha256": freeze["execution_source_sha256"], "source_files": sources, "checks": ["execution freeze and sealed input file identities", "initial messages reconstructed from task_id/prompt/public definitions allowlist", "initial token IDs and template text equal frozen inputs", "entire actual HTTP payload equals shared request builder", "full G schema equals registered public schema; other policies have no grammar", "all histories reconstructed from actual returned public tool observations and registered feedback", "server prompt IDs equal saved local IDs for observable responses"], "limitations": ["Saved ID hashes and frozen templates are checked without fresh tokenizer execution.", "This module does not replace independent fixture replay or endpoint scoring.", "No GPU logits, model weights, or counterfactual model behavior are reproduced."]}
    (output / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    if {"torch", "transformers", "vllm"} & set(sys.modules):
        raise RuntimeError("CPU input verification unexpectedly imported a model runtime")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--stage", choices=("P1", "P2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_run(args.case, args.stage, args.output)
    print(json.dumps({key: result[key] for key in ("status", "stage", "checked_cells", "checked_request_boundaries", "failed_cells", "global_errors")}, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else (2 if result["status"] == "PARTIAL" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
