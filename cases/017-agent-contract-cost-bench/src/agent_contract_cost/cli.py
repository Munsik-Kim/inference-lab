"""Model-free public verification entrypoints. Runtime execution is separate."""
import argparse
import json
import tempfile
from pathlib import Path

from .common import CASE, digest, read, utc, write_once


def validate(output):
    from .contracts import parse_action
    from .fixtures import gold_actions
    from .schemas import server_schema
    from .tools import tool_definitions
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted((CASE / "data").glob("*_tasks.json")):
        tasks = read(path)
        for task in tasks:
            defs = tool_definitions(task["family"])
            actions = gold_actions(task)
            if len(actions) > 6 or any(not parse_action(json.dumps(a), defs).accepted for a in actions):
                raise ValueError("A registered fixture has no budget-feasible strict gold path")
            rows.append({"task_id": task["task_id"], "split": task["split"], "gold_actions_including_final": len(actions),
                         "initial_state_sha256": task["initial_state_sha256"], "grammar_sha256": digest(server_schema(defs))})
    write_once(root / "validation.json", {"status": "PASS", "created_utc": utc(), "rows": rows,
                "evidence_kind": "SYNTHETIC_CONTRACT_TEST", "model_requests": 0})
    return {"status": "PASS", "tasks": len(rows), "model_requests": 0}


def oracle_check(output):
    from .audit import independent_endpoint
    from .fixtures import gold_actions
    from .scoring import assess
    from .tools import Fixture
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted((CASE / "data").glob("*_tasks.json")):
        for task in read(path):
            fixture = Fixture(task, root / "fixtures" / task["task_id"])
            available, final = [], None
            for turn, action in enumerate(gold_actions(task)):
                event = fixture.execute(action, turn, available)
                if action["tool"] == "final":
                    final = action
                    break
                available.append(event["event_id"])
            actual = assess(task, fixture.events, final, fixture.snapshot(), fixture.version, available)
            other = independent_endpoint(task, fixture.events, final, fixture.snapshot(), fixture.version, available)
            if not actual["VERIFIED_WORKFLOW_SUCCESS"] or not other["VERIFIED_WORKFLOW_SUCCESS"]:
                raise ValueError("Gold oracle disagreement")
            rows.append({"task_id": task["task_id"], "actions": len(fixture.events), "verified": True})
    write_once(root / "oracle_check.json", {"status": "PASS", "evidence_kind": "SYNTHETIC_CONTRACT_TEST", "rows": rows,
                "model_requests": 0, "independent_endpoint": "separate Decimal/state/DAG computation"})
    return {"status": "PASS", "tasks": len(rows), "model_requests": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Case017 CPU contracts and stored-response replay; no model/network downloads")
    parser.add_argument("command", choices=("validate", "oracle-check", "audit", "analyze"))
    parser.add_argument("--output", required=True, help="New owned output directory")
    parser.add_argument("--stage", choices=("P1", "P2"))
    args = parser.parse_args(argv)
    if args.command == "validate":
        result = validate(args.output)
    elif args.command == "oracle-check":
        result = oracle_check(args.output)
    else:
        if not args.stage:
            parser.error("audit/analyze require --stage")
        from .audit import audit_run
        tasks = CASE / "data" / ("dev_tasks.json" if args.stage == "P1" else "holdout_tasks.json")
        runroot = CASE / "results" / (args.stage.lower() + "_r1")
        if args.command == "audit":
            result = audit_run(tasks, runroot, args.output)["receipt"]
        else:
            from .analysis import analyze
            rows = read(runroot / "audit/audited_rows.json")
            result = analyze(rows, args.output, tasks=read(tasks), stage=args.stage)
            result = {k: result[k] for k in ("stage", "status", "planned_cells", "present_cells", "missing_or_unobserved_cells")}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
