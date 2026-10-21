"""Synthetic CPU audit fixtures; none are model measurements or recovery results."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from agent_contract_cost.analysis import analyze
from agent_contract_cost.audit import (audit_episode, audit_run, digest, independent_desired,
                                       independent_endpoint, independent_sse, observation_view)
from agent_contract_cost.contracts import grammar_schema, parse_action
from agent_contract_cost.fixtures import generate_tasks, gold_actions, oracle
from agent_contract_cost.scoring import assess
from agent_contract_cost.prompts import initial_messages
from agent_contract_cost.tools import Fixture, tool_definitions


def synthetic_episode(task, root, actions=None):
    """Scripted raw records with actual local fixture execution, explicitly labeled."""
    actions = gold_actions(task) if actions is None else actions
    fixture = Fixture(task, root)
    messages = initial_messages(task)
    requests, available, final, final_available = [], [], None, []
    for turn, action in enumerate(actions):
        text = json.dumps(action, ensure_ascii=False)
        ids = [101 + turn]
        raw_ids = [201 + turn]
        request = {"generation_turn": turn, "messages": copy.deepcopy(messages),
                   "available_result_event_ids": list(available),
                   "observation_sha256": digest([observation_view(event) for event in fixture.events if event["event_id"] in available]),
                   "state_digest": digest(fixture.snapshot()), "state_version": fixture.version,
                   "input_token_ids": ids, "input_token_sha256": digest(ids),
                   "prompt_text": "SYNTHETIC_NOT_TOKENIZED", "budget_before": {},
                   "request_plan": {"max_tokens": 512}, "feedback": None}
        response = {"status": "OK", "raw_text": text, "raw_utf8_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "output_token_ids": raw_ids, "output_tokens": len(raw_ids), "prompt_token_ids": ids,
                    "usage": {"completion_tokens": 1, "prompt_tokens": 1}, "finish_reason": "stop", "stop_reason": None,
                    "model_responses_received": 1, "server_abort_requests": 0,
                    "request_body": {"messages": copy.deepcopy(messages), "structured_outputs": {"json": grammar_schema(tool_definitions(task["family"]))}}}
        parsed = parse_action(text, tool_definitions(task["family"]))
        request["parse"] = parsed.as_dict()
        event = fixture.execute(action, turn, available)
        request["event"], request["response"] = event, response
        requests.append(request)
        if action["tool"] == "final":
            final, final_available = copy.deepcopy(action), list(available)
        else:
            available.append(event["event_id"])
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": json.dumps(observation_view(event), ensure_ascii=False, sort_keys=True, separators=(",", ":"))}]
    endpoint = assess(task, fixture.events, final, fixture.snapshot(), fixture.version, final_available, "VERIFIED_SUCCESS")
    count = len(requests)
    budget = {"model_request_attempts": count, "model_responses_received": count,
              "server_abort_requests": 0, "generated_raw_objects": count, "parsed_tool_attempts": count,
              "contract_rejections": 0, "callable_executions": count - int(final is not None),
              "side_effect_writes": sum(event["side_effect_write"] for event in fixture.events),
              "final_attempts": int(final is not None), "retry_triggered": 0, "retry_consumed": 0, "generated_output_tokens": count}
    return {"identity": {"stage": "P1", "kv": "K0", "policy": "G", "task_id": task["task_id"], "family": task["family"], "attempt_id": "synthetic-test-only"},
            "task_id": task["task_id"], "stage": "P1", "kv": "K0", "policy": "G", "family": task["family"],
            "evidence_kind": "SYNTHETIC_CONTRACT_TEST", "requests": requests, "events": fixture.events,
            "final_action": final, "final_available_result_event_ids": final_available,
            "final_state": fixture.snapshot(), "state_version": fixture.version,
            "terminal_reason": "VERIFIED_SUCCESS", "observation_status": "OBSERVED", "reported_endpoint": endpoint,
            "budget": budget, "wall_seconds": 1.0}


class IndependentAuditTests(unittest.TestCase):
    def test_separate_decimal_oracle_matches_all_dev_expected_states(self):
        for task in generate_tasks("DEV"):
            desired, final = independent_desired(task)
            self.assertEqual(desired, oracle(task)["state"])
            self.assertEqual(final["arguments"], oracle(task)["final_arguments"])
        task = copy.deepcopy(generate_tasks("DEV")[0])
        row = task["initial_state"]["orders"][task["goal"]["order_id"]]
        row["lines"], row["discount_bps"] = [{"quantity": 1, "unit_price_minor": 1}], 5000
        self.assertEqual(independent_desired(task)[0]["orders"][task["goal"]["order_id"]]["total_minor"], 1)

    def test_all_family_gold_raw_replay_and_independent_endpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            for index, task in enumerate(generate_tasks("DEV")):
                episode = synthetic_episode(task, Path(temp) / f"original-{index}")
                row = audit_episode(task, episode, Path(temp) / f"replay-{index}")
                self.assertTrue(row["audit_pass"], row["audit_errors"])
                self.assertTrue(row["verified"])
                self.assertEqual(row["retokenization_status"], "NOT_RUN_SAVED_IDS_HASH_CHECK_ONLY")

    def test_raw_hash_or_reported_success_cannot_override_recomputed_raw(self):
        task = generate_tasks("DEV")[0]
        with tempfile.TemporaryDirectory() as temp:
            episode = synthetic_episode(task, Path(temp) / "original")
            episode["requests"][0]["response"]["raw_utf8_sha256"] = "0" * 64
            episode["reported_endpoint"]["VERIFIED_WORKFLOW_SUCCESS"] = False
            row = audit_episode(task, episode, Path(temp) / "replay")
            self.assertFalse(row["audit_pass"])
            self.assertTrue(row["verified"])
            self.assertIn("raw string hash 0", row["audit_errors"])
            self.assertIn("reported endpoint: VERIFIED_WORKFLOW_SUCCESS", row["audit_errors"])

    def test_observation_id_mismatch_is_rejected_even_same_state(self):
        task = generate_tasks("DEV")[0]
        with tempfile.TemporaryDirectory() as temp:
            episode = synthetic_episode(task, Path(temp) / "original")
            episode["requests"][1]["available_result_event_ids"] = ["a-different-result"]
            row = audit_episode(task, episode, Path(temp) / "replay")
            self.assertFalse(row["audit_pass"])
            self.assertIn("observation IDs 1", row["audit_errors"])

    def test_correct_final_without_executed_work_or_without_observed_validation_fails(self):
        task = generate_tasks("DEV")[0]
        desired, final = independent_desired(task)
        endpoint = independent_endpoint(task, [], final, desired, 1, [], "VERIFIED_SUCCESS")
        self.assertTrue(endpoint["STATE_CORRECT"])
        self.assertTrue(endpoint["FINAL_ANSWER_CORRECT"])
        self.assertFalse(endpoint["VERIFIED_WORKFLOW_SUCCESS"])
        with tempfile.TemporaryDirectory() as temp:
            episode = synthetic_episode(task, Path(temp) / "original")
            endpoint = independent_endpoint(task, episode["events"], episode["final_action"], episode["final_state"], episode["state_version"], [], "VERIFIED_SUCCESS")
            self.assertFalse(endpoint["OBSERVATION_ORDER_VALID"])

    def test_schema_valid_wrong_amount_and_unsupported_claim_remain_failure(self):
        task = generate_tasks("DEV")[0]
        actions = gold_actions(task)
        actions[2]["arguments"]["total_minor"] += 1
        actions[-1]["arguments"]["result"]["total_minor"] += 1
        with tempfile.TemporaryDirectory() as temp:
            episode = synthetic_episode(task, Path(temp) / "original", actions)
            row = audit_episode(task, episode, Path(temp) / "replay")
            self.assertTrue(row["audit_pass"], row["audit_errors"])
            self.assertFalse(row["verified"])
            self.assertTrue(row["endpoint"]["CLAIMED_DONE_WITH_WRONG_STATE"])

    def test_protected_field_change_then_restore_still_fails_independent_integrity(self):
        task = generate_tasks("DEV")[0]
        with tempfile.TemporaryDirectory() as temp:
            episode = synthetic_episode(task, Path(temp) / "original")
            event = episode["events"][0]
            control = next(key for key in task["initial_state"]["orders"] if key != task["goal"]["order_id"])
            event["state_after"]["orders"][control]["protected_customer"] = "tampered then restored"
            endpoint = independent_endpoint(task, episode["events"], episode["final_action"], episode["final_state"], episode["state_version"], episode["final_available_result_event_ids"], "VERIFIED_SUCCESS")
            self.assertFalse(endpoint["integrity_valid_throughout"])
            self.assertFalse(endpoint["VERIFIED_WORKFLOW_SUCCESS"])

    def test_independent_sse_fragments_do_not_duplicate_role_usage_or_done(self):
        payloads = [{"choices": [{"index": 0, "delta": {"role": "assistant"}}], "prompt_token_ids": [1, 2]},
                    {"choices": [{"index": 0, "delta": {"content": '{"to'}, "token_ids": [3]}]},
                    {"choices": [{"index": 0, "delta": {"content": 'ol":"x"}'}, "token_ids": [4], "finish_reason": "stop", "stop_reason": 9}]},
                    {"choices": [], "usage": {"completion_tokens": 2, "prompt_tokens": 2}}]
        lines = ["data: " + json.dumps(value) for value in payloads] + ["data: [DONE]"]
        result = independent_sse(lines)
        self.assertEqual(result["raw_text"], '{"tool":"x"}')
        self.assertEqual(result["output_token_ids"], [3, 4])
        self.assertEqual(result["prompt_token_ids"], [1, 2])
        self.assertEqual(result["finish_reason"], "stop")
        self.assertTrue(result["stream_done"])

    def test_audit_run_uses_fresh_roots_and_detects_duplicate_cells(self):
        task = generate_tasks("DEV")[0]
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            episode = synthetic_episode(task, temp / "original")
            run = temp / "run"
            (run / "cells").mkdir(parents=True)
            (run / "cells" / "one.json").write_text(json.dumps(episode))
            result = audit_run([task], run, temp / "audit")
            self.assertTrue(result["receipt"]["audit_pass"])
            with self.assertRaises(FileExistsError):
                audit_run([task], run, temp / "audit")
            (run / "cells" / "two.json").write_text(json.dumps(episode))
            with self.assertRaisesRegex(ValueError, "duplicate primary"):
                audit_run([task], run, temp / "duplicate-audit")

    def test_analysis_preserves_exact_pair_losses_recovery_and_shared_seed(self):
        tasks = [{"task_id": "A", "family": "A"}, {"task_id": "B", "family": "B"}, {"task_id": "C", "family": "C"}]
        rows = []
        for task in tasks:
            for kv in ("K0", "K1"):
                for policy in ("C0", "G", "N", "E"):
                    success = policy != "C0" if task["family"] == "A" else policy == "C0" if task["family"] == "B" else True
                    rows.append({**task, "stage": "P1", "kv": kv, "policy": policy, "evidence_kind": "SYNTHETIC_CONTRACT_TEST",
                                 "audit_pass": True, "observation_status": "OBSERVED", "verified": success, "state_correct": success,
                                 "final_correct": success, "terminal_reason": "VERIFIED_SUCCESS" if success else "INCORRECT_STATE",
                                 "counters": {"generated_output_tokens": 20, "model_request_attempts": 2}, "wall_seconds": 1})
        with tempfile.TemporaryDirectory() as temp:
            a = analyze(rows, Path(temp) / "a", tasks=tasks, bootstrap_count=200, seed=7, include_synthetic=True)
            b = analyze(rows, Path(temp) / "b", tasks=tasks, bootstrap_count=200, seed=7, include_synthetic=True)
            self.assertEqual(a["comparisons"], b["comparisons"])
            comparison = next(row for row in a["comparisons"] if row["comparison_id"] == "K0:G-C0")
            self.assertEqual(comparison["counts"], {"recovery": 1, "loss": 1, "both_success": 1})
            self.assertEqual(comparison["effect_fraction"], 0)
            self.assertEqual(comparison["ci_95_fraction"], [0, 0])
            self.assertTrue(comparison["ci_zero_width_is_not_equivalence"])
            self.assertEqual(a["status"], "COMPLETE")

    def test_missing_not_imputed_and_zero_success_cost_is_undefined(self):
        tasks = [{"task_id": "A", "family": "A"}]
        rows = [{**tasks[0], "stage": "P1", "kv": "K0", "policy": "C0", "evidence_kind": "SYNTHETIC_CONTRACT_TEST",
                 "audit_pass": True, "observation_status": "OBSERVED", "verified": False, "state_correct": False, "final_correct": False,
                 "terminal_reason": "CONTRACT_FAILURE", "counters": {"generated_output_tokens": 10}, "wall_seconds": 2}]
        with tempfile.TemporaryDirectory() as temp:
            summary = analyze(rows, Path(temp) / "a", tasks=tasks, bootstrap_count=10, include_synthetic=True)
            self.assertEqual(summary["status"], "PARTIAL")
            self.assertEqual(summary["missing_or_unobserved_cells"], 7)
            baseline = summary["groups"][0]
            self.assertIsNone(baseline["tokens_per_verified_success"])
            comparison = next(row for row in summary["comparisons"] if row["comparison_id"] == "K0:G-C0")
            self.assertIsNone(comparison["effect_fraction"])
            self.assertEqual((comparison["missing_effect_lower"], comparison["missing_effect_upper"]), (0, 1))

    def test_synthetic_rows_are_not_accepted_as_model_measurements_by_default(self):
        tasks = [{"task_id": "A", "family": "A"}]
        row = {**tasks[0], "stage": "P1", "kv": "K0", "policy": "C0", "evidence_kind": "SYNTHETIC_CONTRACT_TEST"}
        with tempfile.TemporaryDirectory() as temp:
            summary = analyze([row], Path(temp) / "a", tasks=tasks, bootstrap_count=10)
            self.assertEqual(summary["present_cells"], 0)
            self.assertEqual(summary["missing_or_unobserved_cells"], 8)
            self.assertEqual(len(summary["excluded"]), 1)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)


if __name__ == "__main__":
    unittest.main()
