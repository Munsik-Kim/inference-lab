"""Synthetic annotation boundaries; frozen endpoint expectations are unchanged."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

CASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CASE / "src"))
from agent_contract_cost.fixtures import generate_tasks, gold_actions
from agent_contract_cost.runner import run_episode

spec = importlib.util.spec_from_file_location("case017_contract_annotations", CASE / "postprocess/inspect_contract_boundaries.py")
annotations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annotations)
adapter_spec = importlib.util.spec_from_file_location("case017_annotation_scripted_adapter", CASE / "tests/test_runner.py")
scripted = importlib.util.module_from_spec(adapter_spec)
adapter_spec.loader.exec_module(scripted)


class ContractAnnotationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tasks = generate_tasks("DEV")
        self.count = 0

    async def run_actions(self, task, actions):
        self.count += 1
        return await run_episode(task, "C0", scripted.ScriptedAdapter(actions), Path(self.temp.name) / f"fixture-{self.count}", {"kv": "K0", "evidence_kind": "SYNTHETIC_CONTRACT_TEST"})

    async def test_tag_order_annotation_does_not_replace_strict_list_endpoint(self):
        task = next(task for task in self.tasks if task["family"] == "C")
        actions = gold_actions(task)
        actions[1]["arguments"]["tags"].reverse()
        episode = await self.run_actions(task, actions)
        found = annotations.inspect_episode(task, episode)
        self.assertTrue(any(row["kind"] == "SAME_TAG_ITEMS_DIFFERENT_ARRAY_ORDER" for row in found))
        self.assertFalse(episode["reported_endpoint"]["VERIFIED_WORKFLOW_SUCCESS"])
        self.assertTrue(all(row["counterfactual_success"] == "NOT_EVALUATED" for row in found))

    async def test_idempotent_write_after_validation_preserves_actual_current_validation(self):
        task = self.tasks[0]
        actions = gold_actions(task)
        actions = actions[:-1] + [copy.deepcopy(actions[2]), actions[-1]]
        episode = await self.run_actions(task, actions)
        found = annotations.inspect_episode(task, episode)
        match = next(row for row in found if row["kind"] == "SUCCESSFUL_IDEMPOTENT_WRITE")
        self.assertTrue(match["current_validation_already_observed_ids"])
        self.assertTrue(episode["reported_endpoint"]["VERIFIED_WORKFLOW_SUCCESS"])
        self.assertEqual(episode["budget"]["side_effect_writes"], 1)

    async def test_changed_state_and_repeated_validation_have_distinct_annotations(self):
        task = self.tasks[0]
        gold = gold_actions(task)
        wrong = copy.deepcopy(gold[2])
        wrong["arguments"]["total_minor"] += 1
        stale = await self.run_actions(task, gold[:-1] + [wrong, gold[-1]])
        self.assertTrue(any(row["kind"] == "STATE_CHANGED_AFTER_PREVIOUS_VALIDATION" for row in annotations.inspect_episode(task, stale)))
        self.assertFalse(stale["reported_endpoint"]["VERIFIED_WORKFLOW_SUCCESS"])
        repeated = await self.run_actions(task, gold[:-1] + [copy.deepcopy(gold[3]), gold[-1]])
        self.assertTrue(any(row["kind"] == "REPEATED_SUCCESSFUL_VALIDATION_SAME_STATE" for row in annotations.inspect_episode(task, repeated)))
        self.assertTrue(repeated["reported_endpoint"]["VERIFIED_WORKFLOW_SUCCESS"])


if __name__ == "__main__":
    unittest.main()
