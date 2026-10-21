"""Read-only annotations of tag ordering, idempotent writes and validation state.

Annotations do not relax the frozen list-sensitive endpoint, change scores, or
estimate success under a different policy. Incomplete runs are deliberately not
opened by the command-line entry point.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


CASE = Path(__file__).resolve().parents[1]
VALIDATORS = {"validate_order", "validate_transfer", "validate_document"}
WRITERS = {"write_order_total", "transfer_stock", "update_document"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def strict_equal(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return set(a) == set(b) and all(strict_equal(a[k], b[k]) for k in a)
    if type(a) is list:
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b))
    return a == b


def inspect_episode(task, episode):
    annotations = []
    prior_validations = []
    for request in episode["requests"]:
        turn = request["generation_turn"]
        parsed = request.get("parse")
        if task["family"] == "C" and parsed and parsed["accepted"]:
            action = parsed["action"]
            args = action["arguments"]
            received = args.get("tags") if action["tool"] == "update_document" else (args.get("result", {}).get("tags") if action["tool"] == "final" else None)
            requested = task["goal"]["tags"]
            if type(received) is list and received != requested and sorted(received) == sorted(requested):
                if action["tool"] == "update_document":
                    other_fields_match = all(strict_equal(args.get(key), task["goal"][key]) for key in ("document_id", "status", "owner_id"))
                else:
                    result = args["result"]
                    other_fields_match = all(strict_equal(result.get(key), task["goal"][key]) for key in ("document_id", "status", "owner_id"))
                annotations.append({"kind": "SAME_TAG_ITEMS_DIFFERENT_ARRAY_ORDER", "generation_turn": turn, "tool": action["tool"], "received_tags": copy.deepcopy(received), "requested_tags": copy.deepcopy(requested), "other_requested_metadata_fields_match": other_fields_match, "frozen_policy": "Tag arrays are ordered lists under strict_equal. Original endpoint remains unchanged.", "counterfactual_success": "NOT_EVALUATED"})
        event = request.get("event")
        if event is None or event["status"] != "SUCCESS":
            continue
        matching = [old for old in prior_validations if old["after_digest"] == event["before_digest"] and old["version_after"] == event["version_before"]]
        if event["tool"] in WRITERS:
            unchanged = strict_equal(event["state_before"], event["state_after"])
            if unchanged:
                observed = [old["event_id"] for old in matching if old["event_id"] in event["generated_after_event_ids"]]
                annotations.append({"kind": "SUCCESSFUL_IDEMPOTENT_WRITE", "generation_turn": turn, "tool": event["tool"], "event_id": event["event_id"], "current_validation_already_observed_ids": observed, "frozen_policy": "State version changes only when serialized state changes. An identical-value write is not a new state and does not invalidate a current validation.", "cost_counter_scope": "side_effect_writes counts state-changing writes, not physical JSON-file writes", "counterfactual_success": "NOT_EVALUATED"})
            elif prior_validations:
                annotations.append({"kind": "STATE_CHANGED_AFTER_PREVIOUS_VALIDATION", "generation_turn": turn, "tool": event["tool"], "event_id": event["event_id"], "earlier_validation_event_ids": [old["event_id"] for old in prior_validations], "frozen_policy": "Final requires a subsequently observed validation with the current state digest and version.", "counterfactual_success": "NOT_EVALUATED"})
        if event["tool"] in VALIDATORS and event["result"].get("valid") is True:
            same_action = [old["event_id"] for old in matching if old["tool"] == event["tool"] and strict_equal(old["arguments"], event["arguments"])]
            if same_action:
                annotations.append({"kind": "REPEATED_SUCCESSFUL_VALIDATION_SAME_STATE", "generation_turn": turn, "tool": event["tool"], "event_id": event["event_id"], "previous_same_state_validation_ids": same_action, "frozen_policy": "Extra validation is allowed and consumes the same parsed action/model/token budgets.", "counterfactual_success": "NOT_EVALUATED"})
            prior_validations.append(event)
    return annotations


def inspect_run(case, stage, output):
    case, output = case.resolve(), output.resolve()
    runroot = case / "results" / ("p1_r1" if stage == "P1" else "p2_r1")
    completion = load(runroot / "completion.json")
    if completion["status"] != "COMPLETED" or completion["missing"] or completion["completed"] != completion["planned"]:
        raise ValueError("Only a completed registered matrix may be annotated by this command")
    tasks_path = case / "data" / ("dev_tasks.json" if stage == "P1" else "holdout_tasks.json")
    tasks = {task["task_id"]: task for task in load(tasks_path)}
    rows, files = [], sorted((runroot / "cells").glob("*.json"))
    for path in files:
        episode = load(path)
        findings = inspect_episode(tasks[episode["task_id"]], episode)
        rows.append({"source_file": str(path.relative_to(case)), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "task_id": episode["task_id"], "kv": episode["kv"], "policy": episode["policy"], "annotations": findings})
    counts = {}
    affected = {}
    for row in rows:
        for finding in row["annotations"]:
            kind = finding["kind"]
            counts[kind] = counts.get(kind, 0) + 1
            affected.setdefault(kind, set()).add(row["source_file"])
    receipt = {"status": "COMPLETED_READ_ONLY_ANNOTATION", "stage": stage, "evidence_kind": "POSTHOC_CONTRACT_BOUNDARY_ANNOTATION", "cells_examined": len(rows), "endpoint_scores_changed": False, "new_model_requests": 0, "execution_source_changed": False, "annotation_counts": counts, "affected_episode_counts": {kind: len(paths) for kind, paths in affected.items()}, "tag_array_order_cases": len(affected.get("SAME_TAG_ITEMS_DIFFERENT_ARRAY_ORDER", set())), "idempotent_write_cases": len(affected.get("SUCCESSFUL_IDEMPOTENT_WRITE", set())), "repeat_validation_cases": len(affected.get("REPEATED_SUCCESSFUL_VALIDATION_SAME_STATE", set())), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "scope": "Only observed saved actions/events are annotated. No alternate parser, tag-set scorer, revised budget, regenerated response or recovery success is substituted."}
    output.mkdir(parents=True, exist_ok=False)
    (output / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--stage", choices=("P1", "P2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_run(args.case, args.stage, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
