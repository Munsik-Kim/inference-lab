import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from agent_state_recovery.common import CASE_ROOT, canonical, digest, read_json
from agent_state_recovery.contracts import parse, ContractError
from agent_state_recovery.public import initial_payload, definitions, grammar
from agent_state_recovery.generator import dev_tasks, generate, check_p2_gate
from agent_state_recovery.client import FakeClient, SequenceClient
from agent_state_recovery.runner import run_episode
from agent_state_recovery.world import World, business, protected_ok
from agent_state_recovery.scoring import fresh_validation, run_continuation
from agent_state_recovery.audit import audit_episode, AuditMismatch, ReferenceState, protection, verify_task, verify_coverage


class P0Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = dev_tasks()
        cls.a = next(t for t in cls.tasks if t["family"] == "A")
        cls.c = next(t for t in cls.tasks if t["family"] == "C")

    def episode(self, mode, *, task=None, policy="R0", start="error", **kw):
        return run_episode(task or self.a, FakeClient(mode), policy=policy, start=start, **kw)

    def test_dev_generator_is_frozen_and_six_tasks(self):
        self.assertEqual(generate(), self.tasks)
        self.assertEqual(len({t["task_id"] for t in self.tasks}), 6)
        self.assertEqual([t["family"] for t in self.tasks], list("AABBCC"))

    def test_all_six_control_expectations_and_independent_audit(self):
        for task in self.tasks:
            for mode, start, expected in [("clean", "clean", True), ("no_recovery", "error", False), ("scripted_correct", "error", True)]:
                with self.subTest(task=task["task_id"], mode=mode):
                    row = self.episode(mode, task=task, start=start)
                    self.assertIs(row["reported_endpoint"]["Q"], expected)
                    self.assertIs(row["reported_endpoint"]["W"], expected)
                    self.assertEqual(audit_episode(task, row)["status"], "PASS")

    def test_fault_actually_changes_state_and_is_reachable(self):
        for task in self.tasks:
            self.assertNotEqual(task["checkpoint"], task["error_state"])
            self.assertNotEqual(business(task["checkpoint"]), business(task["error_state"]))
            verify_task(task)

    def test_false_declaration_is_not_quality(self):
        e = self.episode("no_recovery")["reported_endpoint"]
        self.assertTrue(e["D"])
        self.assertFalse(e["Q"])

    def test_business_only_repair_does_not_release_active_operation(self):
        row = self.episode("business_only", task=self.c)
        e = row["reported_endpoint"]
        self.assertTrue(e["B"])
        self.assertTrue(e["F"])
        self.assertFalse(e["C1"])
        self.assertFalse(e["C2"])
        self.assertFalse(e["Q"])
        self.assertTrue(row["terminal_snapshot"]["active_operations"])
        self.assertEqual(audit_episode(self.c, row)["status"], "PASS")

    def test_stale_validation_quality_and_workflow_are_distinct(self):
        row = self.episode("stale_validation")
        self.assertTrue(row["reported_endpoint"]["Q"])
        self.assertFalse(row["reported_endpoint"]["F"])
        self.assertFalse(row["reported_endpoint"]["W"])
        self.assertEqual(audit_episode(self.a, row)["status"], "PASS")

    def test_public_valid_true_does_not_mean_restored(self):
        row = self.episode("wrong_business", policy="R1")
        self.assertTrue(row["reported_endpoint"]["F"])
        self.assertFalse(row["reported_endpoint"]["B"])
        self.assertFalse(row["reported_endpoint"]["Q"])
        self.assertEqual(row["terminal_reason"], "ACCEPTED_FINAL")
        self.assertEqual(audit_episode(self.a, row)["status"], "PASS")

    def test_no_final_still_has_state_and_continuation_metrics(self):
        row = self.episode("no_final")
        self.assertTrue(row["reported_endpoint"]["Q"])
        self.assertFalse(row["reported_endpoint"]["W"])
        self.assertEqual(row["reported_endpoint"]["F_context"], "LAST_GENERATION_DIAGNOSTIC")
        self.assertEqual(audit_episode(self.a, row)["status"], "PASS")

    def test_current_validation_result_not_yet_delivered_is_not_observed(self):
        row = run_episode(self.a, SequenceClient([{"tool": "validate_inventory", "arguments": {"scope_id": self.a["public_request"]["scope_id"]}}]), start="clean")
        self.assertTrue(row["events"][0]["result"]["valid"])
        self.assertFalse(row["reported_endpoint"]["F"])
        self.assertEqual(audit_episode(self.a, row)["status"], "PASS")

    def test_protected_damage_even_if_restored_is_not_forgiven(self):
        initial = self.a["error_state"]
        damaged = copy.deepcopy(initial)
        key = next(k for k, r in damaged["records"].items() if r["protected"])
        damaged["records"][key]["quantity"] -= 1
        states = [initial, damaged, copy.deepcopy(initial)]
        self.assertFalse(protected_ok(self.a, initial, states))
        self.assertFalse(protection(self.a, initial, states))

    def test_protected_attribute_damage_then_restore_fails(self):
        s = self.a["error_state"]
        wrong = copy.deepcopy(s)
        wrong["records"][next(iter(wrong["records"]))]["owner"] = "changed"
        self.assertFalse(protected_ok(self.a, s, [s, wrong, s]))
        self.assertFalse(protection(self.a, s, [s, wrong, s]))

    def test_deleted_audit_prefix_is_new_damage(self):
        s = self.a["error_state"]
        wrong = copy.deepcopy(s)
        wrong["audit"].pop(0)
        self.assertFalse(protected_ok(self.a, s, [s, wrong]))
        self.assertFalse(protection(self.a, s, [s, wrong]))

    def test_new_negative_stock_cannot_be_hidden_by_terminal(self):
        s = self.a["error_state"]
        wrong = copy.deepcopy(s)
        wrong["records"][next(iter(wrong["records"]))]["quantity"] = -1
        self.assertFalse(protected_ok(self.a, s, [s, wrong, s]))
        self.assertFalse(protection(self.a, s, [s, wrong, s]))

    def test_legitimate_revision_and_append_only_history_are_accepted(self):
        row = self.episode("scripted_correct")
        self.assertGreater(row["terminal_snapshot"]["revision"], self.a["checkpoint"]["revision"])
        self.assertNotEqual(row["terminal_snapshot"], self.a["checkpoint"])
        self.assertTrue(row["reported_endpoint"]["W"])

    def test_r0_accepts_ineligible_final_r1_rejects_then_validates(self):
        r0 = self.episode("guard_retry")
        r1 = self.episode("guard_retry", policy="R1")
        self.assertEqual(r0["initial_payload_sha256"], r1["initial_payload_sha256"])
        self.assertEqual(r0["terminal_reason"], "ACCEPTED_FINAL")
        self.assertTrue(r0["reported_endpoint"]["Q"])
        self.assertFalse(r0["reported_endpoint"]["W"])
        self.assertEqual(r1["accounting"]["synthetic"]["guard_rejections"], 1)
        self.assertTrue(r1["reported_endpoint"]["W"])
        self.assertEqual(audit_episode(self.a, r1)["status"], "PASS")

    def test_two_guard_feedbacks_then_third_ineligible_final_stops(self):
        row = self.episode("repeated_final", policy="R1")
        counts = row["accounting"]["synthetic"]
        self.assertEqual((counts["model_request_attempts"], counts["parsed_attempts"], counts["guard_rejections"], counts["guard_feedbacks"]), (3, 3, 3, 2))
        self.assertEqual(counts["callable_executions"], 0)
        self.assertEqual(row["terminal_reason"], "GUARD_LIMIT")
        self.assertTrue(row["reported_endpoint"]["D"])
        self.assertEqual(audit_episode(self.a, row)["status"], "PASS")

    def test_two_guard_rejections_then_valid_final_fits_budget(self):
        row = self.episode("guard_twice", policy="R1")
        self.assertTrue(row["reported_endpoint"]["W"])
        self.assertEqual(len(row["requests"]), 9)
        self.assertEqual(row["accounting"]["synthetic"]["guard_feedbacks"], 2)
        self.assertEqual(audit_episode(self.a, row)["status"], "PASS")

    def test_same_task_all_four_conditions_have_identical_initial_inputs(self):
        for task in self.tasks:
            rows = [{"arm": arm, "public": digest(initial_payload(task)), **task["identity"]} for arm in ("K0_R0", "K0_R1", "K1_R0", "K1_R1")]
            self.assertTrue(all({k: v for k, v in r.items() if k != "arm"} == {k: v for k, v in rows[0].items() if k != "arm"} for r in rows))

    def test_payload_allowlist_no_oracle_target_script_or_internal_fields(self):
        task = copy.deepcopy(self.a)
        marker = "DO_NOT_EXPOSE_PRIVATE_ORACLE_78"
        task["oracle"] = {"secret": marker}
        task["checkpoint"]["secret"] = marker
        task["frozen_continuation"] = marker
        task["public_request"]["hidden_extra"] = marker
        payload = canonical(initial_payload(task))
        self.assertNotIn(marker, payload)
        self.assertNotIn('"continuations"', payload)
        self.assertNotIn('"oracle"', payload)

    def test_grammar_has_public_types_not_task_values_or_correct_next_action(self):
        raw = canonical(grammar())
        for task in self.tasks:
            for secret in (task["task_id"], task["public_request"]["incident_operation_id"], *task["public_request"]["record_ids"]):
                self.assertNotIn(secret, raw)
        self.assertEqual(len(grammar()["oneOf"]), len(definitions()))
        action = {"tool": "compensate_stock", "arguments": {"operation_id": "wrong-id", "source_restore": 999, "destination_remove": 998, "expected_revision": 4}}
        self.assertEqual(parse(canonical(action), definitions()), action)

    def test_strict_parser_rejects_duplicate_nan_bool_numbers_multiple_and_garbage(self):
        bad = ['{"tool":"read_history","tool":"read_history","arguments":{}}',
               '{"tool":"read_history","arguments":{},"x":NaN}',
               '{"tool":"compensate_stock","arguments":{"operation_id":"o","source_restore":true,"destination_remove":0,"expected_revision":1}}',
               '{"tool":"compensate_stock","arguments":{"operation_id":"o","source_restore":"2","destination_remove":0,"expected_revision":1}}',
               '{"tool":"read_history","arguments":{}} {}',
               '{"tool":"read_history","arguments":{}} prose', '[]', '```json\n{}\n```']
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                parse(raw, definitions())
        self.assertEqual(parse(' \n{"tool":"read_history","arguments":{}}\t', definitions())["tool"], "read_history")

    def test_parse_and_schema_rejections_consume_distinct_attempts(self):
        syntax = run_episode(self.a, SequenceClient(['{"tool":']))
        schema = run_episode(self.a, SequenceClient([{"tool": "not_registered", "arguments": {}}]))
        self.assertEqual(syntax["accounting"]["synthetic"]["parsed_attempts"], 0)
        self.assertEqual(schema["accounting"]["synthetic"]["parsed_attempts"], 1)
        for row in (syntax, schema):
            self.assertEqual(row["terminal_reason"], "CONTRACT_FAILURE")
            self.assertEqual(row["accounting"]["synthetic"]["callable_executions"], 0)
            self.assertEqual(audit_episode(self.a, row)["status"], "PASS")

    def test_sixteenth_nonfinal_executes_but_no_seventeenth_request(self):
        row = run_episode(self.a, SequenceClient([{"tool": "read_history", "arguments": {}}] * 20))
        self.assertEqual(len(row["requests"]), 16)
        self.assertEqual(len(row["events"]), 16)
        self.assertEqual(row["terminal_reason"], "TOOL_CALL_LIMIT")
        self.assertEqual(audit_episode(self.a, row)["status"], "PASS")

    def test_sixteenth_final_is_allowed(self):
        final = {"tool": "final", "arguments": {"task_id": self.a["task_id"], "recovered": False,
                 "quantities": [{"record_id": "wrong", "quantity": 1}], "validation_event_id": "none"}}
        row = run_episode(self.a, SequenceClient([{"tool": "read_history", "arguments": {}}] * 15 + [final]))
        self.assertEqual(row["terminal_reason"], "ACCEPTED_FINAL")
        self.assertEqual(len(row["requests"]), 16)
        self.assertEqual(audit_episode(self.a, row)["status"], "PASS")

    def test_model_token_context_and_deadline_limits_are_distinct(self):
        base = read_json(CASE_ROOT / "specs/protocol.json")["limits"]
        values = [{"tool": "read_history", "arguments": {}}] * 20
        for changes, expected in [({"model_requests": 2}, "MODEL_CALL_LIMIT"),
                                  ({"episode_output_tokens": 2}, "TOKEN_LIMIT"),
                                  ({"context_tokens": 1}, "CONTEXT_LIMIT"),
                                  ({"episode_seconds": 0.02}, "EPISODE_DEADLINE")]:
            with self.subTest(expected=expected):
                row = run_episode(self.a, SequenceClient(values), limits={**base, **changes})
                self.assertEqual(row["terminal_reason"], expected)

    def test_request_timeout_never_executes_a_late_response(self):
        raw = canonical({"tool": "read_history", "arguments": {}})
        row = run_episode(self.a, SequenceClient([{"raw_text": raw, "status": "OK", "simulated": {"output_tokens": 1, "elapsed_seconds": 180}}]))
        self.assertEqual(row["terminal_reason"], "REQUEST_TIMEOUT")
        self.assertEqual(len(row["events"]), 0)
        self.assertEqual(audit_episode(self.a, row)["status"], "PASS")

    def test_normal_tool_error_is_returned_and_consumes_same_budgets(self):
        row = run_episode(self.a, SequenceClient([{"tool": "read_records", "arguments": {"record_ids": ["../../unknown"]}}, {"tool": "read_history", "arguments": {}}]))
        self.assertEqual(row["events"][0]["status"], "ERROR")
        self.assertEqual(len(row["requests"]), 2)
        self.assertEqual(row["accounting"]["synthetic"]["parsed_attempts"], 2)
        self.assertEqual(row["accounting"]["synthetic"]["callable_executions"], 2)
        self.assertIn("UNKNOWN_OR_DUPLICATE_RECORD", row["requests"][1]["model_payload"]["messages"][-1]["content"])

    def test_wrong_revision_and_insufficient_stock_do_not_mutate(self):
        world = World(self.a, self.a["error_state"])
        before = digest(world.state)
        op = self.a["public_request"]["incident_operation_id"]
        for args in [dict(operation_id=op, source_restore=1, destination_remove=0, expected_revision=999),
                     dict(operation_id=op, source_restore=1, destination_remove=1000000, expected_revision=world.state["revision"])]:
            e = world.execute({"tool": "compensate_stock", "arguments": args}, 0, [])
            self.assertEqual(e["status"], "ERROR")
            self.assertEqual(digest(world.state), before)

    def test_duplicate_delivery_keeps_one_completed_effect(self):
        task = next(t for t in self.tasks if t["family"] == "B")
        world = World(task, task["checkpoint"])
        before = digest(world.state)
        op = task["continuations"]["F2"][0]
        e = world.execute({"tool": "apply_transfer", "arguments": {**op, "expected_revision": world.state["revision"]}}, 0, [])
        self.assertTrue(e["result"]["duplicate_ignored"])
        self.assertEqual(before, digest(world.state))

    def test_each_continuation_starts_from_independent_frozen_snapshot(self):
        row = self.episode("scripted_correct")
        frozen = row["terminal_snapshot_sha256"]
        for name in ("F1", "F2"):
            self.assertEqual(row["continuations"][name]["actual"]["initial_snapshot_digest"], frozen)
            branch = row["continuations"][name]["actual"]
            for event in branch["events"]:
                if event["tool"] == "apply_transfer":
                    self.assertEqual(event["arguments"]["expected_revision"], event["state_before"]["revision"])
        self.assertEqual(digest(row["terminal_snapshot"]), frozen)

    def test_immutable_generation_observation_and_event_id_tampering_rejected(self):
        row = self.episode("scripted_correct")
        changed = copy.deepcopy(row)
        changed["requests"][-1]["available_result_event_ids"][-1] = "wrong-event-same-state"
        with self.assertRaises(AuditMismatch):
            audit_episode(self.a, changed)
        e = row["events"][-1]
        self.assertNotIn(e["event_id"], e["generated_after_event_ids"])

    def test_audit_detects_tampered_raw_state_and_stored_boolean(self):
        row = self.episode("scripted_correct")
        for kind in ("raw", "state", "boolean"):
            changed = copy.deepcopy(row)
            if kind == "raw":
                changed["requests"][0]["response"]["raw_text"] += " }"
            elif kind == "state":
                changed["terminal_snapshot"]["revision"] += 1
            else:
                changed["reported_endpoint"]["Q"] = False
            with self.subTest(kind=kind), self.assertRaises(AuditMismatch):
                audit_episode(self.a, changed)

    def test_audit_rejects_missing_or_duplicate_control_manifest(self):
        controls = [{"episode_id": f"DEV-{f}{i:02d}_{m}", "path": f"episodes/DEV-{f}{i:02d}_{m}.json", "kind": "CONTROL"}
                    for f in "ABC" for i in range(2) for m in ("clean", "no_recovery", "scripted_correct")]
        with self.assertRaises(AuditMismatch):
            verify_coverage({"controls": 18, "diagnostics": 8, "episodes": controls[:-1]})
        with self.assertRaises(AuditMismatch):
            verify_coverage({"controls": 18, "diagnostics": 8, "episodes": controls + [controls[0]]})

    def test_no_holdout_payload_and_gate_fails_closed_without_success_threshold(self):
        protocol = read_json(CASE_ROOT / "specs/protocol.json")
        self.assertIsNone(protocol["gates"]["model_success_threshold"])
        with self.assertRaises(ValueError):
            generate("HOLDOUT")
        complete = dict.fromkeys(protocol["gates"]["P1"], True)
        complete.update(completed_episode_count=24, execution_source_sha256="test-identity", model_successes=0)
        self.assertTrue(check_p2_gate(complete))
        for key in protocol["gates"]["P1"]:
            bad = {**complete, key: False}
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_p2_gate(bad)
        self.assertFalse(any("holdout" in str(p.relative_to(CASE_ROOT)).lower() for p in (CASE_ROOT / "data").rglob("*")))

    def test_hand_calculated_two_sides_and_partial_debit(self):
        # Independent arithmetic illustration, not a model observation or generator call.
        self.assertEqual((20 - 3, 5 + 3), (17, 8))
        self.assertEqual((17 + 3, 8 - 3), (20, 5))
        # Duplicate: preserve original 17/8; compensate only the second 3.
        self.assertEqual((14 + 3, 11 - 3), (17, 8))
        # Partial: undo source debit only; adding 3 to the destination would be wrong.
        self.assertEqual((17 + 3, 5 - 0), (20, 5))
        for task in self.tasks:
            bad = task["error_state"]["ledger"][task["public_request"]["incident_operation_id"]]
            src, dst, qty = bad["source_id"], bad["destination_id"], bad["quantity"]
            actual = task["error_state"]["records"]
            ref = task["checkpoint"]["records"]
            self.assertEqual(actual[src]["quantity"] + qty, ref[src]["quantity"])
            self.assertEqual(actual[dst]["quantity"] - (0 if task["family"] == "C" else qty), ref[dst]["quantity"])

    def test_fixture_root_and_symlink_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "existing").mkdir()
            (root / "link").symlink_to(root / "existing", target_is_directory=True)
            with self.assertRaises(ValueError):
                World(self.a, self.a["error_state"], root / "link" / "child")
            with self.assertRaises(FileExistsError):
                World(self.a, self.a["error_state"], root / "existing")
            w = World(self.a, self.a["error_state"], root / "owned")
            (w.root / "state.json").unlink()
            (w.root / "state.json").symlink_to(root / "do-not-touch")
            with self.assertRaises(ValueError):
                w.execute({"tool": "read_history", "arguments": {}}, 0, [])
            self.assertFalse((root / "do-not-touch").exists())

    def test_cpu_imports_and_help_need_no_torch_sdk_tokenizer_or_network(self):
        script = '''import builtins,runpy,socket,sys
original=builtins.__import__
def guarded(name,*a,**k):
 if name.split('.')[0] in {'torch','transformers','vllm','tokenizers','huggingface_hub','httpx','requests'}: raise AssertionError('forbidden SDK '+name)
 return original(name,*a,**k)
builtins.__import__=guarded
def deny(*a,**k): raise AssertionError('network forbidden')
socket.socket=deny
sys.argv=['agent_state_recovery','--help']
runpy.run_module('agent_state_recovery',run_name='__main__')
'''
        env = {**os.environ, "PYTHONPATH": str(CASE_ROOT / "src"), "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
        r = subprocess.run([sys.executable, "-B", "-c", script], cwd=CASE_ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("audit", r.stdout)


if __name__ == "__main__":
    unittest.main()
