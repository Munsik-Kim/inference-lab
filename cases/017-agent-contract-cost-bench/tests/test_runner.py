"""SYNTHETIC_CONTRACT_TEST: scripted adapters are not model measurements."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_contract_cost.audit import audit_episode
from agent_contract_cost.fixtures import canonical, digest, generate_tasks, gold_actions
from agent_contract_cost.policies import NEUTRAL_FEEDBACK
from agent_contract_cost.prompts import initial_messages, observation_view
from agent_contract_cost.runner import run_episode


class ScriptedAdapter:
    def __init__(self, responses, *, input_mismatch=False, context_overflow=False):
        self.responses = copy.deepcopy(responses)
        self.calls = []
        self.input_mismatch = input_mismatch
        self.context_overflow = context_overflow

    def tokenize(self, messages):
        text = canonical(messages)
        ids = list(range(25000)) if self.context_overflow else list(text.encode("utf-8"))
        return {"token_ids": ids, "input_token_sha256": digest(ids), "prompt_text": text}

    async def send(self, payload, timeout):
        self.calls.append({"payload": copy.deepcopy(payload), "timeout": timeout})
        if not self.responses:
            raise AssertionError("script exhausted: runner requested an unregistered extra response")
        item = self.responses.pop(0)
        if item == "TIMEOUT":
            raise TimeoutError("synthetic request timeout")
        raw = canonical(item) if type(item) is dict else item
        prompt_ids = self.tokenize(payload["messages"])["token_ids"]
        if self.input_mismatch:
            prompt_ids = prompt_ids + [999999]
        output_ids = list(raw.encode("utf-8"))
        return {"status": "OK", "model_request_attempts": 1, "model_responses_received": 1, "server_abort_requests": 0, "request_body": copy.deepcopy(payload), "raw_text": raw, "raw_utf8_sha256": hashlib.sha256(raw.encode()).hexdigest(), "raw_token_text": raw, "output_token_ids": output_ids, "output_tokens": len(output_ids), "prompt_token_ids": prompt_ids, "input_token_sha256": digest(prompt_ids), "output_token_sha256": digest(output_ids), "finish_reason": "stop", "stop_reason": None, "server_finish_reason": "stop", "server_stop_reason": None, "stream_done": True, "raw_sse_lines": [], "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(output_ids)}, "elapsed_seconds": 0.001}


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tasks = generate_tasks("DEV")
        self.counter = 0

    def tearDown(self):
        self.temp.cleanup()

    def task(self, family="A"):
        return copy.deepcopy(next(task for task in self.tasks if task["family"] == family))

    async def run_script(self, task, policy, adapter):
        self.counter += 1
        return await run_episode(task, policy, adapter, self.root / f"fixture-{self.counter}", {"episode_id": f"synthetic-{self.counter}", "kv": "K0", "stage": "CPU", "evidence_kind": "SYNTHETIC_CONTRACT_TEST"})

    def check_audit(self, task, episode):
        self.counter += 1
        audited = audit_episode(task, episode, self.root / f"audit-{self.counter}")
        self.assertTrue(audited["audit_pass"], audited["audit_errors"])
        return audited

    async def test_gold_three_families_complete_and_independent_audit_matches(self):
        for family in "ABC":
            task = self.task(family)
            adapter = ScriptedAdapter(gold_actions(task))
            result = await self.run_script(task, "C0", adapter)
            self.assertTrue(result["reported_endpoint"]["VERIFIED_WORKFLOW_SUCCESS"])
            self.assertEqual(result["terminal_reason"], "VERIFIED_SUCCESS")
            self.assertEqual(result["budget"]["model_request_attempts"], len(gold_actions(task)))
            self.assertEqual(result["budget"]["final_attempts"], 1)
            self.assertEqual(result["budget"]["callable_executions"], len(gold_actions(task)) - 1)
            self.assertEqual(result["final_available_result_event_ids"], result["requests"][-1]["available_result_event_ids"])
            self.assertNotIn(result["events"][-1]["event_id"], result["final_available_result_event_ids"])
            self.assertTrue(self.check_audit(task, result)["verified"])

    async def test_final_json_alone_is_not_workflow_success(self):
        task = self.task()
        result = await self.run_script(task, "C0", ScriptedAdapter([gold_actions(task)[-1]]))
        self.assertFalse(result["reported_endpoint"]["VERIFIED_WORKFLOW_SUCCESS"])
        self.assertTrue(result["reported_endpoint"]["FINAL_ANSWER_CORRECT"])
        self.assertEqual(result["budget"]["callable_executions"], 0)
        self.assertEqual(result["terminal_reason"], "INCORRECT_STATE")
        self.check_audit(task, result)

    async def test_neutral_and_error_retry_once_keep_observation_empty(self):
        task = self.task()
        for policy in ("N", "E"):
            result = await self.run_script(task, policy, ScriptedAdapter(["{broken"] + gold_actions(task)))
            first, retry = result["requests"][:2]
            self.assertEqual(first["available_result_event_ids"], [])
            self.assertEqual(retry["available_result_event_ids"], [])
            self.assertEqual(first["observation_sha256"], retry["observation_sha256"])
            self.assertIsNone(first["event"])
            self.assertEqual(retry["messages"][-2], {"role": "assistant", "content": "{broken"})
            feedback = retry["messages"][-1]["content"]
            if policy == "N":
                self.assertEqual(feedback, NEUTRAL_FEEDBACK)
            else:
                self.assertEqual(set(json.loads(feedback)), {"code", "path", "expected"})
            self.assertEqual(result["budget"]["retry_triggered"], 1)
            self.assertEqual(result["budget"]["retry_consumed"], 1)
            self.assertEqual(result["budget"]["model_request_attempts"], 6)
            self.assertEqual(result["budget"]["parsed_tool_attempts"], 5)
            self.assertTrue(result["reported_endpoint"]["VERIFIED_WORKFLOW_SUCCESS"])
            self.check_audit(task, result)

    async def test_second_contract_failure_stops_without_extra_retry(self):
        task = self.task()
        for policy in ("N", "E"):
            adapter = ScriptedAdapter(["{broken", "still broken", *gold_actions(task)])
            result = await self.run_script(task, policy, adapter)
            self.assertEqual(len(adapter.calls), 2)
            self.assertEqual(result["terminal_reason"], "CONTRACT_FAILURE")
            self.assertEqual(result["budget"]["retry_consumed"], 1)
            self.assertEqual(result["events"], [])
            self.check_audit(task, result)

    async def test_unknown_name_consumes_parsed_budget_without_callable(self):
        task = self.task()
        result = await self.run_script(task, "C0", ScriptedAdapter([{"tool": "readOrder", "arguments": {"order_id": task["goal"]["order_id"]}}]))
        self.assertEqual(result["terminal_reason"], "CONTRACT_FAILURE")
        self.assertEqual(result["budget"]["parsed_tool_attempts"], 1)
        self.assertEqual(result["budget"]["contract_rejections"], 1)
        self.assertEqual(result["budget"]["callable_executions"], 0)
        self.assertEqual(result["events"], [])
        self.check_audit(task, result)

    async def test_server_input_mismatch_is_unobserved_and_not_model_failure(self):
        task = self.task()
        result = await self.run_script(task, "C0", ScriptedAdapter(gold_actions(task), input_mismatch=True))
        self.assertEqual(result["observation_status"], "UNOBSERVED")
        self.assertIsNone(result["reported_endpoint"]["VERIFIED_WORKFLOW_SUCCESS"])
        self.assertEqual(result["requests"][0]["response"]["status"], "INPUT_IDENTITY_MISMATCH")
        self.assertEqual(result["requests"][0]["response"]["transport_status"], "OK")
        self.assertEqual(result["events"], [])
        self.assertEqual(result["budget"]["model_responses_received"], 1)
        self.check_audit(task, result)

    async def test_sixth_nonfinal_stops_under_explicit_final_attempt_policy(self):
        task = self.task()
        read = gold_actions(task)[0]
        adapter = ScriptedAdapter([read] * 6 + [gold_actions(task)[-1]])
        result = await self.run_script(task, "C0", adapter)
        self.assertEqual(len(adapter.calls), 6)
        self.assertEqual(result["terminal_reason"], "TOOL_CALL_LIMIT")
        self.assertEqual(result["budget"]["final_attempts"], 0)
        self.assertTrue(result["budget"]["final_opportunity_blocked_by_registered_attempt_budget"])
        self.assertEqual(result["budget"]["model_request_attempts"], 6)
        self.check_audit(task, result)

    async def test_request_timeout_debits_attempt_but_not_received_response(self):
        task = self.task()
        result = await self.run_script(task, "C0", ScriptedAdapter(["TIMEOUT"]))
        self.assertEqual(result["terminal_reason"], "REQUEST_TIMEOUT")
        self.assertEqual(result["budget"]["model_request_attempts"], 1)
        self.assertEqual(result["budget"]["model_responses_received"], 0)
        self.assertEqual(result["budget"]["generated_output_tokens"], 0)
        self.assertTrue(result["requires_server_drain_check"])
        self.assertFalse(result["reported_endpoint"]["VERIFIED_WORKFLOW_SUCCESS"])
        self.check_audit(task, result)

    async def test_business_error_is_real_observation_and_does_not_trigger_retry(self):
        task = self.task()
        unknown_id = {"tool": "read_order", "arguments": {"order_id": "unknown-but-schema-valid-id"}}
        result = await self.run_script(task, "E", ScriptedAdapter([unknown_id] + gold_actions(task)))
        self.assertEqual(result["events"][0]["status"], "ERROR")
        self.assertEqual(result["requests"][1]["available_result_event_ids"], [result["events"][0]["event_id"]])
        self.assertEqual(result["budget"]["retry_triggered"], 0)
        self.assertEqual(result["budget"]["parsed_tool_attempts"], 6)
        self.assertTrue(result["reported_endpoint"]["VERIFIED_WORKFLOW_SUCCESS"])
        self.check_audit(task, result)

    async def test_shared_initial_input_only_g_has_structured_outputs(self):
        task = self.task()
        first_rows = []
        for policy in ("C0", "G", "N", "E"):
            result = await self.run_script(task, policy, ScriptedAdapter(gold_actions(task)))
            first_rows.append(result["requests"][0])
            self.assertEqual("structured_outputs" in first_rows[-1]["response"]["request_body"], policy == "G")
            self.assertNotIn("tools", first_rows[-1]["response"]["request_body"])
            self.check_audit(task, result)
        self.assertTrue(all(row["messages"] == first_rows[0]["messages"] for row in first_rows))
        self.assertEqual(len({row["input_token_sha256"] for row in first_rows}), 1)

    async def test_context_limit_prevents_http_without_fake_request(self):
        task = self.task()
        adapter = ScriptedAdapter(gold_actions(task), context_overflow=True)
        result = await self.run_script(task, "C0", adapter)
        self.assertEqual(result["terminal_reason"], "CONTEXT_LIMIT")
        self.assertEqual(result["requests"], [])
        self.assertEqual(result["budget"]["model_request_attempts"], 0)
        self.assertEqual(adapter.calls, [])
        self.check_audit(task, result)

    async def test_prompt_and_observation_allowlist_exclude_audit_data(self):
        task = self.task()
        original = initial_messages(task)
        task["oracle"] = "PRIVATE_ORACLE_SENTINEL"
        task["frozen_continuation"] = "PRIVATE_OTHER_MODEL_RESPONSE"
        task["initial_state"]["private_field"] = "PRIVATE_INITIAL_STATE"
        task["goal"]["hidden_answer"] = "PRIVATE_EXPECTED_AMOUNT"
        self.assertEqual(initial_messages(task), original)
        event = {"event_id": "e0", "tool": "read_order", "status": "SUCCESS", "result": {"quantity": 7}, "state_after": "PRIVATE_STATE", "oracle": "PRIVATE_GOLD"}
        self.assertEqual(set(observation_view(event)), {"event_id", "tool", "status", "result"})


if __name__ == "__main__":
    unittest.main()
