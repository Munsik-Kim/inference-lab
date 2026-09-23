"""Timing orchestration tests; no model inference or timing measurements run."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import benchmark_isolated as benchmark


class IsolatedBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ckda-isolated-tests-")
        self.root = Path(self.temporary.name)
        self.policy = json.loads((benchmark.ROOT/"configs/execution_policy.json").read_text())
        self.policy_path = self.root/"execution_policy.json"
        self.freeze(self.policy_path,self.policy)
        self.rows = [self.fixture(seed) for seed in range(3)]

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def freeze(path,value):
        benchmark.write_json(path,value)
        path.with_suffix(".sha256").write_text(benchmark.sha256(path)+"\n")

    def fixture(self,seed):
        directory = self.root/f"primary{seed}"
        directory.mkdir()
        checkpoint = self.root/f"final{seed}.pt"
        checkpoint.write_bytes(f"checkpoint{seed}".encode())
        protocol = dict(arms=["NATIVE_FP32"])
        runtime = dict(cpu_threads=2,benchmark=dict(sequences=16,length=128,repeats=3,seed=2002))
        self.freeze(directory/"protocol.json",protocol)
        self.freeze(directory/"evaluation_runtime.json",runtime)
        source = {}
        for name in ("codec/learned.py","codec/online.py","codec/packed.py","codec/groups.py",
                     "codec/survival.py","scripts/evaluate_learned.py"):
            target = directory/"frozen-source"/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text("# fixture, never executed\n")
            source[name] = benchmark.sha256(target)
        for filename in ("token_coefficients.json","token_coefficients.bin","calibration.npz"):
            (directory/filename).write_bytes(filename.encode())
        (directory/"codecs").mkdir()
        (directory/"DEV").mkdir()
        config = directory/"codecs/NATIVE_FP32.json"
        config.write_text("{}")
        benchmark.write_json(directory/"DEV/NATIVE_FP32.json",dict(config_sha256=benchmark.sha256(config),
                             basis_sha256=benchmark.hashlib.sha256(b"").hexdigest()))
        manifest = dict(phase="FROZEN_PRIMARY",checkpoint_training_status="TRAINING_COMPLETE",model_seed=seed,
                        checkpoint_sha256=benchmark.sha256(checkpoint),protocol_sha256=benchmark.sha256(directory/"protocol.json"),
                        runtime_sha256=benchmark.sha256(directory/"evaluation_runtime.json"),
                        frozen_source_directory="frozen-source",source_sha256=source,
                        token_table_config_sha256=benchmark.sha256(directory/"token_coefficients.json"),
                        token_table_data_sha256=benchmark.sha256(directory/"token_coefficients.bin"),
                        calibration_sha256=benchmark.sha256(directory/"calibration.npz"),test_accessed=True)
        benchmark.write_json(directory/"manifest.json",manifest)
        self.freeze(directory/"index.json",dict(manifest=manifest,splits={"DEV":{"NATIVE_FP32":{}},"TEST":{"NATIVE_FP32":{}}}))
        benchmark.write_json(directory/"timing.json",{})
        return directory,checkpoint

    def inspect(self,seed=0):
        return benchmark.inspect_evaluation(*self.rows[seed])

    def test_completed_identity_and_three_seed_policy(self):
        inspections = [self.inspect(seed) for seed in range(3)]
        benchmark.validate_policy(self.policy,inspections)
        self.assertEqual(inspections[0]["primary_index_sha256"],benchmark.sha256(self.rows[0][0]/"index.json"))
        with self.assertRaisesRegex(ValueError,"distinct trained seeds"):
            benchmark.validate_policy(self.policy,[inspections[0]]*3)
        changed = copy.deepcopy(inspections)
        changed[1]["runtime"]["benchmark"]["repeats"] = 4
        with self.assertRaisesRegex(ValueError,"Runtime benchmark"):
            benchmark.validate_policy(self.policy,changed)

    def test_incomplete_primary_blocks_before_output_or_child(self):
        (self.rows[2][0]/"index.json").unlink()
        output = self.root/"isolated"
        with patch.object(benchmark.subprocess,"run") as child:
            with self.assertRaises(FileNotFoundError):
                benchmark.main(self.arguments(output))
            child.assert_not_called()
        self.assertFalse(output.exists())

    def test_source_checkpoint_and_codec_tampering_rejected(self):
        mutations = [(self.rows[0][0]/"frozen-source/codec/packed.py","Frozen source"),
                     (self.rows[0][1],"Checkpoint hash"),
                     (self.rows[0][0]/"codecs/NATIVE_FP32.json","Serialized codec"),
                     (self.rows[0][0]/"token_coefficients.bin","Fitted artifact")]
        for path,message in mutations:
            original = path.read_bytes()
            path.write_bytes(original+b"changed")
            with self.assertRaisesRegex(ValueError,message):
                self.inspect()
            path.write_bytes(original)

    def arguments(self,output):
        return ["--upstream",str(self.root/"unused-upstream"),"--evaluations",*[str(row[0]) for row in self.rows],
                "--checkpoints",*[str(row[1]) for row in self.rows],"--policy",str(self.policy_path),"--output",str(output)]

    def test_three_serial_fresh_process_calls_no_measurements(self):
        output = self.root/"isolated"
        observed = []
        def fake_child(command,**kwargs):
            seed = int(command[-1])
            observed.append(seed)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"],"")
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"],"2")
            self.assertIn("--child-plan",command)
            benchmark.write_json(output/f"seed{seed}.json",dict(mock=True))
            return benchmark.subprocess.CompletedProcess(command,0)
        with patch.object(benchmark.subprocess,"run",side_effect=fake_child) as child:
            self.assertEqual(benchmark.main(self.arguments(output)),0)
            self.assertEqual(child.call_count,3)
        self.assertEqual(observed,[0,1,2])
        index,_ = benchmark.frozen_json(output/"index.json")
        self.assertEqual(index["status"],"COMPLETE")
        self.assertEqual(set(index["seeds"]),{"0","1","2"})

    def test_failed_measurement_is_not_retried(self):
        output = self.root/"failed"
        failure = benchmark.subprocess.CompletedProcess(["mock"],7)
        with patch.object(benchmark.subprocess,"run",return_value=failure) as child:
            with self.assertRaisesRegex(RuntimeError,"no timing retry"):
                benchmark.main(self.arguments(output))
            child.assert_called_once()
        status = json.loads((output/"status.json").read_text())
        self.assertEqual(status["status"],"FAILED_NO_RETRY")
        self.assertFalse((output/"index.json").exists())


if __name__ == "__main__":
    unittest.main()
