"""Synthetic tampering boundaries; never model-inference or recovery evidence."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


CASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CASE / "src"))
from agent_contract_cost.fixtures import digest, generate_tasks, gold_actions
from agent_contract_cost.prompts import initial_messages
from agent_contract_cost.runner import run_episode
from agent_contract_cost.schemas import server_schema
from agent_contract_cost.tools import tool_definitions

spec = importlib.util.spec_from_file_location("case017_posthoc_inputs", CASE / "postprocess/verify_recorded_inputs.py")
posthoc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(posthoc)
test_spec = importlib.util.spec_from_file_location("case017_scripted_runner_adapter", CASE / "tests/test_runner.py")
scripted = importlib.util.module_from_spec(test_spec)
test_spec.loader.exec_module(scripted)


class RecordedInputBoundaries(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.task = generate_tasks("DEV")[0]
        self.grammar = server_schema(tool_definitions("A"))
        self.source = "a" * 64
        self.protocol = "b" * 64

    async def episode(self, policy="G", failure=False):
        responses = (["{broken"] if failure else []) + gold_actions(self.task)
        adapter = scripted.ScriptedAdapter(responses)
        messages = initial_messages(self.task)
        tokens = adapter.tokenize(messages)
        frozen = {"task_id": self.task["task_id"], "messages": messages, "messages_sha256": digest(messages), "token_ids": tokens["token_ids"], "input_token_sha256": tokens["input_token_sha256"], "prompt_text": tokens["prompt_text"]}
        identity = {"task_id": self.task["task_id"], "kv": "K0", "policy": policy, "stage": "CPU", "evidence_kind": "SYNTHETIC_CONTRACT_TEST", "execution_source_sha256": self.source, "protocol_sha256": self.protocol, "schema_sha256": digest(self.grammar), "expected_initial_input_token_sha256": tokens["input_token_sha256"]}
        episode = await run_episode(self.task, policy, adapter, Path(self.temp.name) / ("fixture-" + policy), identity)
        return frozen, episode

    def verify(self, frozen, episode):
        return posthoc.verify_episode_inputs(self.task, frozen, episode, self.grammar, self.source, self.protocol)

    async def test_untampered_scripted_request_chain_passes_without_using_endpoint_bool(self):
        frozen, episode = await self.episode()
        episode["reported_endpoint"]["VERIFIED_WORKFLOW_SUCCESS"] = "THIS_FIELD_IS_IGNORED"
        result = self.verify(frozen, episode)
        self.assertEqual(result["status"], "PASS", result["errors"])

    async def test_consistently_forged_first_messages_still_fail_frozen_allowlist(self):
        frozen, episode = await self.episode()
        episode = copy.deepcopy(episode)
        for row in episode["requests"]:
            row["messages"][0]["content"] += "\nPRIVATE_ORACLE_SHOULD_NOT_ENTER_MODEL"
            row["response"]["request_body"]["messages"] = copy.deepcopy(row["messages"])
        result = self.verify(frozen, episode)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("history reconstructed" in error["check"] for error in result["errors"]))

    async def test_gold_id_schema_enum_and_non_g_grammar_are_rejected(self):
        frozen, episode = await self.episode("G")
        episode["requests"][0]["response"]["request_body"]["structured_outputs"]["json"]["oneOf"][0]["properties"]["arguments"]["properties"]["order_id"]["enum"] = [self.task["goal"]["order_id"]]
        self.assertEqual(self.verify(frozen, episode)["status"], "FAIL")
        frozen, episode = await self.episode("C0")
        episode["requests"][0]["response"]["request_body"]["structured_outputs"] = {"json": copy.deepcopy(self.grammar)}
        self.assertEqual(self.verify(frozen, episode)["status"], "FAIL")

    async def test_feedback_cannot_contain_unregistered_hidden_answer_hint(self):
        frozen, episode = await self.episode("E", failure=True)
        episode["requests"][0]["feedback"] = '{"code":"TYPE_ERROR","path":"$","expected":"The hidden answer is 123456"}'
        result = self.verify(frozen, episode)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(any("registered neutral text" in error["check"] for error in result["errors"]))

    async def test_fake_observation_and_changed_initial_token_ids_are_rejected(self):
        frozen, episode = await self.episode()
        episode["requests"][1]["available_result_event_ids"].append("FAKE_RESULT_WITHOUT_TOOL_EXECUTION")
        episode["requests"][0]["input_token_ids"] = episode["requests"][0]["input_token_ids"] + [1]
        result = self.verify(frozen, episode)
        self.assertEqual(result["status"], "FAIL")
        checks = [error["check"] for error in result["errors"]]
        self.assertIn("actual initial token IDs equal frozen IDs", checks)
        self.assertIn("pre-generation observation IDs", checks)


if __name__ == "__main__":
    unittest.main()
