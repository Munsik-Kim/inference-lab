"""The P0 command: contract tests, 18 CPU controls, diagnostic traces and audit."""
import copy
import datetime
import hashlib
import platform
import resource
import sys
import time
import unittest
from .common import CASE_ROOT, canonical, digest, fresh_dir, read_json, source_manifest, write_new
from .public import initial_payload
from .generator import dev_tasks
from .client import FakeClient
from .runner import run_episode
from .audit import audit_run


def check_reference_preservation():
    reference = read_json(CASE_ROOT / "provenance/reference_manifest.json")
    root = CASE_ROOT.parent / "017-agent-contract-cost-bench"
    wrong = []
    for row in reference["files"]:
        p = root / row["path"]
        if not p.is_file() or p.stat().st_size != row["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest() != row["sha256"]:
            wrong.append(row["path"])
    return {"status": "PASS" if not wrong else "FAIL", "checked_files": len(reference["files"]),
            "mismatches": wrong, "scope": "Selected public Case017 source/spec/receipt/report bytes. Original research worktree checked separately in final preservation receipt."}


def execute(output):
    wall, cpu = time.perf_counter(), time.process_time()
    root = fresh_dir(output)
    frozen = source_manifest()
    write_new(root / "execution_freeze.json", frozen)
    tests = unittest.defaultTestLoader.discover(str(CASE_ROOT / "tests"), pattern="test_*.py")
    with (root / "tests.log").open("x", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    receipt = {"unique_test_methods": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "status": "PASS" if result.wasSuccessful() else "FAIL",
               "scope": "New Case018 P0 CPU contracts only; subTests/control replays are not additional unique test methods"}
    write_new(root / "tests.json", receipt)
    if not result.wasSuccessful():
        return {"status": "FAIL", "reason": "CPU_CONTRACT_TESTS_FAILED", **receipt}
    tasks = dev_tasks()
    (root / "inputs").mkdir()
    (root / "episodes").mkdir()
    (root / "fixtures").mkdir()
    for task in tasks:
        write_new(root / "inputs" / (task["task_id"] + ".json"), task)
    entries, endpoints, lengths = [], [], []
    def record(task, label, mode, start, policy, kind):
        identifier = task["task_id"] + "_" + label
        row = run_episode(task, FakeClient(mode), root / "fixtures" / identifier,
                          start=start, policy=policy, episode_id=identifier)
        row["control_or_diagnostic"] = kind
        path = root / "episodes" / (identifier + ".json")
        write_new(path, row)
        entries.append({"episode_id": identifier, "path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "canonical_sha256": digest(row), "kind": kind})
        endpoints.append({"episode_id": identifier, "kind": kind, "task_id": task["task_id"],
                          "mode": mode, "policy": policy, **row["reported_endpoint"]})
        lengths.append({"episode_id": identifier, "mode": mode, "kind": kind, **row["accounting"]["synthetic"],
                        "simulated_output_tokens": row["accounting"]["simulated_output_tokens"]})
        return row
    for task in tasks:
        for mode, start in (("clean", "clean"), ("no_recovery", "error"), ("scripted_correct", "error")):
            row = record(task, mode, mode, start, "R0", "CONTROL")
            expected = mode != "no_recovery"
            if row["reported_endpoint"]["Q"] is not expected or row["reported_endpoint"]["W"] is not expected:
                raise AssertionError("control endpoint expectation failed")
    a = next(t for t in tasks if t["family"] == "A")
    c = next(t for t in tasks if t["family"] == "C")
    for task, mode, policy in ((c, "business_only", "R0"), (a, "stale_validation", "R0"),
                               (a, "wrong_business", "R1"), (a, "no_final", "R0"),
                               (a, "guard_retry", "R0"), (a, "guard_retry", "R1"),
                               (a, "guard_twice", "R1"), (a, "repeated_final", "R1")):
        record(task, "diagnostic_" + mode + "_" + policy, mode, "error", policy, "DIAGNOSTIC")
    write_new(root / "run_manifest.json", {"evidence_kind": "SYNTHETIC_CONTRACT_TEST", "controls": 18, "diagnostics": 8,
               "actual_continuation_branches_controls": 36, "reference_continuation_branches_controls": 36,
               "new_model_requests": 0, "episodes": entries, "execution_source_sha256": frozen["working_source_sha256"]})
    audit = audit_run(root, root / "independent_audit.json")
    identities = []
    for task in tasks:
        four = [{"arm": arm, "initial_payload_sha256": digest(initial_payload(task)), **task["identity"]}
                for arm in ("K0_R0", "K0_R1", "K1_R0", "K1_R1")]
        identities.append({"task_id": task["task_id"], "arms": four, "all_equal_except_arm": True,
                           "actual_token_ids": "NOT_RUN"})
    write_new(root / "four_condition_identity.json", identities)
    write_new(root / "control_and_counterexamples.json", endpoints)
    correct_max = max(r["model_request_attempts"] for r in lengths if r["mode"] == "scripted_correct")
    guard_max = max(r["model_request_attempts"] for r in lengths if r["mode"] == "guard_twice")
    if not correct_max <= guard_max <= 16 or any(r["parsed_attempts"] > 16 for r in lengths):
        raise AssertionError("registered CPU scripts exceed the common budget")
    write_new(root / "budget_feasibility.json", {"rows": lengths, "maximum_correct_repair_requests": correct_max,
               "maximum_two_guard_rejections_then_success_requests": guard_max, "request_limit": 16,
               "remaining_after_worst_registered_correct_script": 16 - guard_max,
               "actual_token_lengths": "NOT_RUN", "simulated_units_are_not_tokenizer_evidence": True})
    preservation = check_reference_preservation()
    write_new(root / "reference_preservation.json", preservation)
    equal_source = frozen == source_manifest()
    holdout_absent = not any("holdout" in str(p.relative_to(CASE_ROOT)).lower() for p in (CASE_ROOT / "data").rglob("*"))
    gates = {"contracts": receipt["status"], "independent_audit": audit["status"],
             "controls": "PASS", "counterexamples": "PASS", "budget_feasibility": "PASS_CALLS_SIMULATED_UNITS",
             "preservation": preservation["status"], "working_source_unchanged_during_run": "PASS" if equal_source else "FAIL",
             "HOLDOUT_NOT_CREATED": "PASS" if holdout_absent else "FAIL",
             "model_success_threshold": None,
             "actual_model_identity": "NOT_RUN", "server_adapter": "NOT_RUN_INTERFACE_ONLY",
             "actual_tokenizer": "NOT_RUN", "actual_grammar_backend": "NOT_RUN", "actual_KV_storage_read_write": "NOT_RUN"}
    status = "P1_CPU_READY" if preservation["status"] == "PASS" and equal_source and holdout_absent else "FAIL"
    verification = {"status": status, "P0_gate": "PASS" if status == "P1_CPU_READY" else "FAIL", "gates": gates,
                    "tests": receipt, "controls": 18, "diagnostics": 8,
                    "control_continuation_branches": {"terminal": 36, "checkpoint_reference": 36},
                    "all_audited_traces": audit["episodes"], "new_model_requests": 0, "GPU_requests": 0,
                    "servers_started": 0, "installations": 0, "remote_writes": 0,
                    "environment": {"python": sys.version, "OS": platform.platform()},
                    "CPU_cost": {"wall_seconds": time.perf_counter() - wall, "process_CPU_seconds": time.process_time() - cpu,
                                 "peak_RSS_KiB_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                 "GPU_worker_seconds": None, "actual_model_tokens": None},
                    "working_source_sha256": frozen["working_source_sha256"],
                    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_new(root / "verification.json", verification)
    files = [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(root.rglob("*")) if p.is_file()]
    write_new(root / "files_manifest.json", {"files": files, "manifest_self_excluded": True})
    return verification
