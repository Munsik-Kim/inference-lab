"""Synthetic CPU registration gates; no models, servers or HOLDOUT generation."""
import asyncio
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_contract_cost import experiment, prepare
from agent_contract_cost.common import digest, sha_file
from agent_contract_cost.fixtures import generate_tasks, gold_actions
from agent_contract_cost.runner import run_episode
from test_runner import ScriptedAdapter


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.case = self.repo / "cases/017-agent-contract-cost-bench"
        self.case.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def make_freezable_tree(self):
        names = ["__init__.py", "common.py", "contracts.py", "policies.py", "fixtures.py", "tools.py", "scoring.py",
                 "schemas.py", "prompts.py", "runner.py", "client.py", "server_manager.py", "p0_inspector.py", "prepare.py", "experiment.py"]
        for name in names:
            path = self.case / "src/agent_contract_cost" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# SYNTHETIC_CONTRACT_TEST registration file " + name + "\n")
        originals = ["followups/nvfp4-technical-r1/src/nvfp4_technical/__init__.py",
                     "followups/nvfp4-technical-r1/src/nvfp4_technical/worker.py",
                     "src/kv_agent_bench/__init__.py", "src/kv_agent_bench/profiler_extension.py", "src/kv_agent_bench/worker_inspection.py"]
        for name in originals:
            path = self.repo / "cases/016-kv-agent-bench" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# synthetic upstream registration fixture\n")
        for name in ("model_identity.json", "environment_identity.json", "grammar_cpu_validation_r2.json"):
            write(self.case / "provenance" / name, {"scope": "SYNTHETIC_CONTRACT_TEST"})
        write(self.case / "specs/protocol.json", {"synthetic": True})
        write(self.case / "data/dev_tasks.json", [{"synthetic": True}])

    def make_runtime_tree(self):
        python = self.root / "isolated/bin/python"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"synthetic interpreter identity - never executable")
        site = python.parent.parent / "lib/python3.12/site-packages"
        source = site / "vllm/example.py"
        source.parent.mkdir(parents=True)
        source.write_text("# synthetic installed source\n")
        metadata = site / "vllm-0.synthetic.dist-info/METADATA"
        metadata.parent.mkdir()
        metadata.write_text("Name: synthetic-vllm\n")
        runtime = {"python": str(python), "snapshot": "SYNTHETIC_LOCAL_SNAPSHOT", "mode": "GRAPH", "pool_bytes": 4294967296}
        environment = {"source_files": [{"relative_package_path": "vllm/example.py", "sha256": sha_file(source)}],
                       "metadata": [{"distribution": metadata.parent.name, "metadata_sha256": sha_file(metadata)}]}
        write(self.case / "provenance/environment_identity.json", environment)
        write(self.case / "specs/runtime_public.json", {
            "private_runtime_configuration_sha256": digest(runtime), "python_executable_sha256": sha_file(python),
            "environment_identity_sha256": sha_file(self.case / "provenance/environment_identity.json")})
        return runtime, python, source, metadata

    def test_freeze_includes_actual_upstream_chain_and_detects_source_tamper(self):
        self.make_freezable_tree()
        with patch.object(prepare, "CASE", self.case), patch.object(prepare, "REPO", self.repo), patch.object(prepare.subprocess, "check_output", return_value="synthetic-base\n"):
            frozen = prepare.execution_freeze("synthetic_r1")
            upstream = [row for row in frozen["files"] if "root_path" in row]
            self.assertEqual(len(upstream), 5)
            prepare.check_freeze(frozen)
            (self.repo / upstream[-1]["root_path"]).write_text("# changed synthetic installed observer\n")
            with self.assertRaisesRegex(ValueError, "Frozen execution dependency changed"):
                prepare.check_freeze(frozen)

    def test_registration_bundle_and_execution_source_are_distinct_identities(self):
        self.make_freezable_tree()
        with patch.object(prepare, "CASE", self.case), patch.object(prepare, "REPO", self.repo), patch.object(prepare.subprocess, "check_output", return_value="synthetic-base\n"):
            first = prepare.execution_freeze("synthetic_r1")
            write(self.case / "data/additional_synthetic_inputs.json", [{"new": "registered input, not model output"}])
            second = prepare.execution_freeze("synthetic_r2")
            self.assertEqual(first["execution_source_sha256"], second["execution_source_sha256"])
            self.assertNotEqual(first["registration_bundle_sha256"], second["registration_bundle_sha256"])
            before = (self.case / "provenance/synthetic_r1.json").read_bytes()
            with self.assertRaises(FileExistsError):
                prepare.execution_freeze("synthetic_r1")
            self.assertEqual(before, (self.case / "provenance/synthetic_r1.json").read_bytes())

    def test_runtime_gate_detects_config_interpreter_package_and_metadata_tamper(self):
        runtime, python, source, metadata = self.make_runtime_tree()
        with patch.object(experiment, "CASE", self.case):
            experiment.check_runtime(runtime)
            changed = {**runtime, "pool_bytes": 3221225472}
            with self.assertRaisesRegex(ValueError, "runtime differs"):
                experiment.check_runtime(changed)
            for path, expected in ((python, "interpreter changed"), (source, "runtime source changed"), (metadata, "metadata changed")):
                original = path.read_bytes()
                path.write_bytes(original + b"tampered\n")
                with self.assertRaisesRegex(ValueError, expected):
                    experiment.check_runtime(runtime)
                path.write_bytes(original)
            experiment.check_runtime(runtime)

    def test_unclosed_worker_attempt_is_not_silently_forgotten(self):
        private = self.root / "private"
        private.mkdir()
        journal = private / "worker_journal.jsonl"
        journal.write_text(json.dumps({"event": "START", "stage": "P0"}) + "\n")
        with self.assertRaisesRegex(ValueError, "Unclosed previous worker attempt"):
            experiment.gpu_journal({"private_root": str(private)})
        with journal.open("a") as stream:
            stream.write(json.dumps({"event": "END", "stage": "P0", "worker_wall_seconds": 12.5}) + "\n")
        _, total = experiment.gpu_journal({"private_root": str(private)})
        self.assertEqual(total, 12.5)

    def test_p2_requires_completed_p1_and_valid_seal_before_any_server(self):
        write(self.case / "results/p0_r1/gates.json", {"P1_allowed": True})
        write(self.case / "specs/protocol.json", {})
        for label in ("execution_freeze_p1", "execution_freeze_p2"):
            write(self.case / "provenance" / (label + ".json"), {"files": []})
        with patch.object(experiment, "CASE", self.case), patch.object(prepare, "check_freeze"), patch.object(experiment, "check_runtime"), patch("agent_contract_cost.server_manager.OwnedServer") as server:
            with self.assertRaises(FileNotFoundError):
                asyncio.run(experiment.stage({}, "P2"))
            write(self.case / "results/p1_r1/completion.json", {"status": "PARTIAL", "completed": 63})
            with self.assertRaisesRegex(ValueError, "P1 matrix is incomplete"):
                asyncio.run(experiment.stage({}, "P2"))
            write(self.case / "results/p1_r1/completion.json", {"status": "COMPLETED", "completed": 64, "planned": 64, "missing": []})
            with self.assertRaises(FileNotFoundError):
                asyncio.run(experiment.stage({}, "P2"))
            write(self.case / "provenance/holdout_seal.json", {"outputs_observed": 1, "policy_or_scorer_hashes_verified_unchanged": True})
            with self.assertRaisesRegex(ValueError, "HOLDOUT seal invalid"):
                asyncio.run(experiment.stage({}, "P2"))
            server.assert_not_called()

    def test_p2_seal_checks_completion_and_both_registered_input_hashes(self):
        completion = self.case / "results/p1_r1/completion.json"
        sealpath = self.case / "provenance/holdout_seal.json"
        taskpath = self.case / "data/holdout_tasks.json"
        inputpath = self.case / "data/holdout_model_inputs.json"
        write(completion, {"status": "COMPLETED", "completed": 64, "planned": 64, "missing": []})
        write(self.case / "provenance/execution_freeze_p1.json", {"files": []})
        # These are placeholder gate files, not generated HOLDOUT tasks/inputs.
        write(taskpath, {"evidence_kind": "SYNTHETIC_CONTRACT_TEST", "placeholder": "tasks"})
        write(inputpath, {"evidence_kind": "SYNTHETIC_CONTRACT_TEST", "placeholder": "inputs"})
        seal = {"outputs_observed": 0, "policy_or_scorer_hashes_verified_unchanged": True,
                "p1_completion_sha256": sha_file(completion), "holdout_tasks_sha256": sha_file(taskpath),
                "holdout_model_inputs_sha256": sha_file(inputpath)}
        write(sealpath, seal)
        with patch.object(experiment, "CASE", self.case), patch.object(prepare, "check_freeze"):
            self.assertEqual(experiment.check_p2_gate(), seal)
            for path, text in ((taskpath, "Sealed HOLDOUT inputs changed"), (inputpath, "Sealed HOLDOUT inputs changed")):
                original = path.read_bytes()
                path.write_bytes(original + b" \n")
                with self.assertRaisesRegex(ValueError, text):
                    experiment.check_p2_gate()
                path.write_bytes(original)
            updated = json.loads(completion.read_text())
            updated["new_unsealed_field"] = "different completion receipt"
            write(completion, updated)
            with self.assertRaisesRegex(ValueError, "P1 completion differs"):
                experiment.check_p2_gate()

    def test_input_registration_resume_checks_bytes_and_does_not_overwrite(self):
        with patch.object(prepare, "CASE", self.case):
            tasks, records = prepare.prepare_split("DEV")
            path = self.case / "data/dev_tasks.json"
            original = path.read_bytes()
            self.assertEqual(prepare.prepare_split("DEV"), (tasks, records))
            self.assertEqual(original, path.read_bytes())
            tampered = copy.deepcopy(tasks)
            tampered[0]["prompt"] += " changed synthetic prompt"
            write(path, tampered)
            with self.assertRaisesRegex(ValueError, "Cannot resume changed"):
                prepare.prepare_split("DEV")
            self.assertEqual(json.loads(path.read_text()), tampered)


class FrozenInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_expected_initial_hash_mismatch_performs_zero_http_requests(self):
        task = generate_tasks("DEV")[0]
        adapter = ScriptedAdapter(gold_actions(task))
        with tempfile.TemporaryDirectory() as temp:
            episode = await run_episode(task, "C0", adapter, Path(temp) / "fixture", {
                "stage": "CPU", "kv": "K0", "evidence_kind": "SYNTHETIC_CONTRACT_TEST",
                "expected_initial_input_token_sha256": "0" * 64})
        self.assertEqual(adapter.calls, [])
        self.assertEqual(episode["requests"], [])
        self.assertEqual(episode["budget"]["model_request_attempts"], 0)
        self.assertEqual(episode["terminal_reason"], "INFRA_ERROR")
        self.assertEqual(episode["observation_status"], "UNOBSERVED")
        self.assertEqual(episode["preparation_failures"][0]["local_error"]["type"], "FrozenInputMismatch")


if __name__ == "__main__":
    unittest.main()
