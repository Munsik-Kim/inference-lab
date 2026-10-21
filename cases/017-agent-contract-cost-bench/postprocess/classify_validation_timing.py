"""Refine saved validation-timing annotations without reading success scores."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


CASE = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def is_target_validation(task, event):
    goal, args = task["goal"], event["arguments"]
    if task["family"] == "A":
        return event["tool"] == "validate_order" and args.get("order_id") == goal["order_id"]
    if task["family"] == "B":
        return event["tool"] == "validate_transfer" and all(args.get(key) == goal[key] for key in ("source_id", "destination_id", "sku"))
    return event["tool"] == "validate_document" and args.get("document_id") == goal["document_id"]


def classify(case, stage, annotation_root, output):
    runroot = case / "results" / ("p1_r1" if stage == "P1" else "p2_r1")
    completion = load(runroot / "completion.json")
    if completion["status"] != "COMPLETED" or completion["missing"]:
        raise ValueError("Only completed runs may be classified")
    tasks = {task["task_id"]: task for task in load(case / "data" / ("dev_tasks.json" if stage == "P1" else "holdout_tasks.json"))}
    rows = load(annotation_root / "rows.json")
    details = []
    for row in rows:
        if not any(item["kind"] == "STATE_CHANGED_AFTER_PREVIOUS_VALIDATION" for item in row["annotations"]):
            continue
        path = case / row["source_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["source_sha256"]:
            raise ValueError("Source episode changed after annotation")
        episode = load(path)
        task = tasks[episode["task_id"]]
        latest = [event["event_id"] for event in episode["events"]
            if event["status"] == "SUCCESS" and event["result"].get("valid") is True
            and is_target_validation(task, event)
            and event["after_digest"] == digest(episode["final_state"])
            and event["version_after"] == episode["state_version"]
            and event["event_id"] in episode["final_available_result_event_ids"]]
        classification = "REVALIDATED_CURRENT_STATE" if latest else ("FINAL_WITHOUT_OBSERVED_CURRENT_VALIDATION" if episode["final_action"] is not None else "NO_FINAL_ACTION")
        details.append({"source_file": row["source_file"], "source_sha256": row["source_sha256"],
            "task_id": task["task_id"], "family": task["family"], "kv": episode["kv"], "policy": episode["policy"],
            "classification": classification, "current_observed_target_validation_event_ids": latest,
            "final_state_version": episode["state_version"],
            "request_trace": [{"generation_turn": request["generation_turn"],
                "raw_text": request["response"]["raw_text"], "available_result_event_ids": request["available_result_event_ids"],
                "event_id": request["event"]["event_id"] if request["event"] else None,
                "event_version_before": request["event"]["version_before"] if request["event"] else None,
                "event_version_after": request["event"]["version_after"] if request["event"] else None}
                for request in episode["requests"]]})
    annotation_receipt = load(annotation_root / "receipt.json")
    summary = {"status": "READ_ONLY_ANNOTATION_REFINEMENT", "stage": stage, "new_model_requests": 0,
        "endpoint_scores_changed": False,
        "tag_array_order_only_cases": annotation_receipt["tag_array_order_cases"],
        "idempotent_write_cases": annotation_receipt["idempotent_write_cases"],
        "repeated_validation_same_state_cases": annotation_receipt["repeat_validation_cases"],
        "state_changed_after_previous_validation_cases": annotation_receipt["affected_episode_counts"].get("STATE_CHANGED_AFTER_PREVIOUS_VALIDATION", 0),
        "classifications": dict(collections.Counter(row["classification"] for row in details)),
        "by_family": dict(collections.Counter(row["family"] for row in details)),
        "by_policy": dict(collections.Counter(row["policy"] for row in details)),
        "scope": "Classification of recorded validation timing, not alternate-policy success or an additional performance denominator."}
    output.mkdir(parents=True, exist_ok=False)
    (output / "validation_timing_details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    (output / "validation_timing_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=CASE)
    parser.add_argument("--stage", choices=("P1", "P2"), required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(classify(args.case, args.stage, args.annotations, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
