"""Synthetic contract tests only: no measured model responses or GPU imports."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_contract_cost.fixtures import digest, generate_tasks, gold_actions, oracle, total_minor
from agent_contract_cost.scoring import assess
from agent_contract_cost.tools import Fixture, schema_error, tool_definitions


class FixtureContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tasks = generate_tasks("DEV")
        self.counter = 0

    def tearDown(self):
        self.temp.cleanup()

    def task(self, family="A"):
        return copy.deepcopy(next(task for task in self.tasks if task["family"] == family))

    def fixture(self, task):
        self.counter += 1
        return Fixture(task, self.root / f"fixture-{self.counter}")

    def replay(self, task, actions=None, omit_observations=()):
        fixture = self.fixture(task)
        final = None
        final_available = []
        for turn, action in enumerate(gold_actions(task) if actions is None else actions):
            available = [] if turn in omit_observations else [event["event_id"] for event in fixture.events]
            event = fixture.execute(action, turn, available)
            if action["tool"] == "final":
                final = action
                final_available = available
        verdict = assess(task, fixture.events, final, fixture.snapshot(), fixture.version, final_available)
        return fixture, verdict

    def test_dev_generation_and_manifest_are_deterministic(self):
        self.assertEqual(self.tasks, generate_tasks("DEV"))
        self.assertEqual([sum(task["family"] == family for task in self.tasks) for family in "ABC"], [3, 3, 2])
        self.assertEqual(len({task["task_id"] for task in self.tasks}), 8)
        self.assertEqual(len({task["initial_state_sha256"] for task in self.tasks}), 8)
        for task in self.tasks:
            self.assertEqual(task["initial_state_sha256"], digest(task["initial_state"]))
            self.assertEqual(task["oracle_sha256"], digest(oracle(task)))
        self.assertEqual(len(generate_tasks("COST")), 6)
        with self.assertRaises(ValueError):
            generate_tasks("RANDOM")

    def test_all_dev_gold_paths_execute_including_final_within_six(self):
        for task in self.tasks:
            with self.subTest(task=task["task_id"]):
                self.assertLessEqual(len(gold_actions(task)), 6)
                fixture, verdict = self.replay(task)
                self.assertTrue(verdict["VERIFIED_WORKFLOW_SUCCESS"], verdict)
                self.assertFalse(fixture.events[-1]["side_effect_write"])
                self.assertFalse(fixture.events[-1]["callable_executed"])
                self.assertEqual(fixture.events[0]["version_before"], 0)

    def test_public_schemas_exclude_fixture_specific_answers(self):
        for task in self.tasks:
            serialized = json.dumps(tool_definitions(task["family"]), sort_keys=True)
            self.assertNotIn(task["task_id"], serialized)
            for value in task["goal"].values():
                if type(value) is str and value.startswith(("R-", "X-", "SKU-", "owner-")):
                    self.assertNotIn(value, serialized)
            self.assertNotIn("oracle", serialized)

    def test_discount_integer_rounding_has_exact_half_up(self):
        self.assertEqual(total_minor([{"quantity": 1, "unit_price_minor": 101}], 5000), 51)
        self.assertEqual(total_minor([{"quantity": 3, "unit_price_minor": 199}, {"quantity": 2, "unit_price_minor": 101}], 1250), (799 * 8750 + 5000) // 10000)

    def test_unknown_names_ids_and_path_strings_do_not_modify_files(self):
        task = self.task()
        fixture = self.fixture(task)
        initial = fixture.snapshot()
        unknown_name = fixture.execute({"tool": "readOrder", "arguments": {"order_id": task["goal"]["order_id"]}}, 0, [])
        self.assertEqual(unknown_name["status"], "CONTRACT_ERROR")
        traversal = fixture.execute({"tool": "write_order_total", "arguments": {"order_id": "../../outside", "total_minor": 1}}, 1, [unknown_name["event_id"]])
        self.assertEqual(traversal["result"]["error"], "UNKNOWN_RECORD")
        self.assertEqual(initial, fixture.snapshot())
        self.assertFalse((self.root / "outside").exists())
        link = self.root / "linked"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            Fixture(task, link / "new")

    def test_schema_distinguishes_bool_integer_and_numeric_string(self):
        schema = next(row["parameters"] for row in tool_definitions("A") if row["name"] == "write_order_total")
        for wrong in (True, "123", 1.0, None):
            self.assertEqual(schema_error({"order_id": "unknown", "total_minor": wrong}, schema)["code"], "TYPE_ERROR")
        self.assertIsNone(schema_error({"order_id": "unknown", "total_minor": 123}, schema))

    def test_wrong_total_is_stored_and_public_validation_does_not_correct_gold(self):
        task = self.task()
        actions = gold_actions(task)
        actions[2]["arguments"]["total_minor"] += 1
        fixture, verdict = self.replay(task, actions)
        self.assertTrue(fixture.events[3]["result"]["valid"])
        self.assertEqual(fixture.snapshot()["orders"][task["goal"]["order_id"]]["total_minor"], actions[2]["arguments"]["total_minor"])
        self.assertFalse(verdict["STATE_CORRECT"])
        self.assertFalse(verdict["VERIFIED_WORKFLOW_SUCCESS"])

    def test_correct_guessed_values_without_read_observation_do_not_verify(self):
        for turn in (1, 2):
            fixture, verdict = self.replay(self.task(), omit_observations=(turn,))
            self.assertTrue(verdict["STATE_CORRECT"])
            self.assertTrue(verdict["FINAL_ANSWER_CORRECT"])
            self.assertFalse(verdict["OBSERVATION_ORDER_VALID"])
            self.assertFalse(verdict["VERIFIED_WORKFLOW_SUCCESS"])

    def test_atomic_transfer_failure_never_partially_updates(self):
        task = self.task("B")
        for field, value in (("quantity", 10**8), ("destination_revision", 10**8)):
            fixture = self.fixture(task)
            action = gold_actions(task)[2]
            action["arguments"][field] = value
            event = fixture.execute(action, 0, [])
            self.assertEqual(event["status"], "ERROR")
            self.assertEqual(fixture.snapshot(), task["initial_state"])
            self.assertFalse(event["side_effect_write"])
            self.assertTrue(event["callable_executed"])

    def test_wrong_transfer_quantity_runs_but_independent_oracle_rejects(self):
        task = self.task("B")
        actions = gold_actions(task)
        actions[2]["arguments"]["quantity"] += 1
        fixture, verdict = self.replay(task, actions)
        self.assertTrue(fixture.events[2]["result"]["transferred"])
        self.assertTrue(fixture.events[3]["result"]["valid"])
        self.assertFalse(verdict["VERIFIED_WORKFLOW_SUCCESS"])

    def test_metadata_update_keeps_wrong_public_values_without_correction(self):
        task = self.task("C")
        actions = gold_actions(task)
        actions[1]["arguments"]["owner_id"] = "wrong-but-schema-valid-owner"
        fixture, verdict = self.replay(task, actions)
        self.assertTrue(fixture.events[2]["result"]["valid"])
        self.assertEqual(fixture.snapshot()["documents"][task["goal"]["document_id"]]["owner_id"], "wrong-but-schema-valid-owner")
        self.assertFalse(verdict["STATE_CORRECT"])
        stale = self.fixture(task)
        bad_revision = gold_actions(task)[1]
        bad_revision["arguments"]["expected_revision"] += 1
        self.assertEqual(stale.execute(bad_revision, 0, [])["result"]["error"], "REVISION_CONFLICT")
        self.assertEqual(stale.snapshot(), task["initial_state"])

    def test_final_alone_never_passes_even_when_file_and_json_are_correct(self):
        task = self.task()
        fixture = self.fixture(task)
        fixture._write(oracle(task)["state"])
        final = gold_actions(task)[-1]
        fixture.execute(final, 0, [])
        verdict = assess(task, fixture.events, final, fixture.snapshot(), fixture.version, [])
        self.assertTrue(verdict["STATE_CORRECT"])
        self.assertTrue(verdict["FINAL_ANSWER_CORRECT"])
        self.assertFalse(verdict["REQUIRED_ACTIONS_DONE"])
        self.assertFalse(verdict["VERIFIED_WORKFLOW_SUCCESS"])

    def test_stale_validation_cannot_support_a_later_correct_state(self):
        task = self.task()
        actions = gold_actions(task)
        wrong_write = copy.deepcopy(actions[2])
        wrong_write["arguments"]["total_minor"] += 1
        actions = actions[:-1] + [wrong_write, copy.deepcopy(actions[2]), actions[-1]]
        fixture, verdict = self.replay(task, actions)
        self.assertTrue(verdict["STATE_CORRECT"])
        self.assertTrue(verdict["FINAL_ANSWER_CORRECT"])
        self.assertFalse(verdict["VERIFIED_WORKFLOW_SUCCESS"])
        self.assertGreater(fixture.version, fixture.events[3]["version_after"])

    def test_unauthorized_control_change_remains_violation_after_restore(self):
        task = self.task()
        fixture = self.fixture(task)
        changed = fixture.snapshot()
        control = next(key for key in changed["orders"] if key != task["goal"]["order_id"])
        changed["orders"][control]["total_minor"] += 1
        fixture._write(changed)
        fixture.snapshot()
        fixture._write(task["initial_state"])
        fixture.snapshot()
        self.assertTrue(fixture.integrity_violation)
        available = []
        for turn, action in enumerate(gold_actions(task)):
            event = fixture.execute(action, turn, available)
            if action["tool"] == "final":
                final_available = available
            available = available + [event["event_id"]]
        verdict = assess(task, fixture.events, gold_actions(task)[-1], fixture.snapshot(), fixture.version, final_available)
        self.assertFalse(verdict["integrity_valid_throughout"])
        self.assertFalse(verdict["VERIFIED_WORKFLOW_SUCCESS"])

    def test_equivalent_read_and_line_orders_are_permitted(self):
        task_b = self.task("B")
        actions_b = gold_actions(task_b)
        actions_b[0], actions_b[1] = actions_b[1], actions_b[0]
        self.assertTrue(self.replay(task_b, actions_b)[1]["VERIFIED_WORKFLOW_SUCCESS"])
        task_a = self.task()
        actions_a = gold_actions(task_a)
        actions_a[1]["arguments"]["lines"].reverse()
        self.assertTrue(self.replay(task_a, actions_a)[1]["VERIFIED_WORKFLOW_SUCCESS"])

    def test_final_bool_extra_keys_and_unobserved_validation_fail(self):
        task = self.task()
        final_index = len(gold_actions(task)) - 1
        self.assertFalse(self.replay(task, omit_observations=(final_index,))[1]["VERIFIED_WORKFLOW_SUCCESS"])
        for change in ("bool", "extra"):
            actions = gold_actions(task)
            if change == "bool":
                actions[-1]["arguments"]["result"]["validated"] = 1
            else:
                actions[-1]["arguments"]["result"]["unexpected"] = "extra"
            fixture, verdict = self.replay(task, actions)
            self.assertEqual(fixture.events[-1]["status"], "CONTRACT_ERROR")
            self.assertFalse(verdict["FINAL_ANSWER_CORRECT"])
            self.assertFalse(verdict["VERIFIED_WORKFLOW_SUCCESS"])

    def test_observation_ids_are_real_prior_generation_events(self):
        task = self.task()
        fixture = self.fixture(task)
        action = gold_actions(task)[0]
        with self.assertRaises(ValueError):
            fixture.execute(action, 0, ["future-event"])
        event = fixture.execute(action, 0, [])
        with self.assertRaises(ValueError):
            fixture.execute(gold_actions(task)[1], 0, [event["event_id"]])
        self.assertEqual(len(fixture.events), 1)

    def test_replay_in_new_owned_root_reconstructs_identical_events_and_verdict(self):
        for family in "ABC":
            task = self.task(family)
            original, verdict = self.replay(task)
            replayed = self.fixture(task)
            for event in original.events:
                new = replayed.execute({"tool": event["tool"], "arguments": event["arguments"]}, event["generated_in_turn"], event["generated_after_event_ids"])
                self.assertEqual(new, event)
            self.assertEqual(replayed.snapshot(), original.snapshot())
            final = gold_actions(task)[-1]
            self.assertEqual(assess(task, replayed.events, final, replayed.snapshot(), replayed.version, original.events[-1]["generated_after_event_ids"]), verdict)


if __name__ == "__main__":
    unittest.main()
