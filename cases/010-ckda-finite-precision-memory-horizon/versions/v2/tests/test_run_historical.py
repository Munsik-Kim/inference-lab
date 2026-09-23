"""Historical orchestration checks; synthetic arithmetic, no model inference."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from source import run_historical as runner
from source.online_v2 import OnlineAdapter, PackedCodec, _V1Adapter

HELPER = Path(__file__).with_name("historical_synthetic_worker.py")
spec = importlib.util.spec_from_file_location("historical_synthetic_test", HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class HistoricalContracts(unittest.TestCase):
    def test_fixed_design_and_boundary_clipping(self):
        self.assertEqual(runner.HISTORICAL_ARMS, ("NATIVE_FP32", "UNIFORM_8", "UNIFORM_5", "LOWRANK_4_8_R2"))
        self.assertEqual(runner.INVALID_CASES, {0: ("UNIFORM_2", "UNIFORM_3"), 1: ("UNIFORM_2",), 2: ("UNIFORM_2",)})
        self.assertEqual(runner.boundaries(2048), [1, 17, 63, 256, 1024, 2048])
        self.assertEqual(runner.boundaries(64), [1, 17, 63, 64])
        self.assertEqual(runner.boundaries(1), [1])
        for value in [0, -1, True, 2.5]:
            with self.assertRaises(ValueError): runner.boundaries(value)

    def test_earliest_invalid_write_then_lowest_original_index(self):
        values = [None] * 512
        values[2], values[10], values[40] = 90, 8, 8
        ids = [str(i) for i in range(512)]
        found = runner.select_invalid_stream({"execution": {"first_invalid_write_position": values}}, ids)
        self.assertEqual((found["original_row"], found["sample_id"], found["original_first_invalid_write"]), (10, "10", 8))
        self.assertEqual((found["original_batch_size"], found["replay_batch_size"]), (512, 1))
        for bad in [[None] * 512, [None] * 16, [False] + [None] * 511, [2049] + [None] * 511]:
            with self.assertRaises(ValueError):
                runner.select_invalid_stream({"execution": {"first_invalid_write_position": bad}}, ids)

    def test_first_prediction_difference_orders_by_write_not_row(self):
        first = np.zeros((3, 6), np.int8)
        second = first.copy()
        second[0, 4], second[2, 1], second[1, 1] = 2, 3, -1
        diff = runner.first_prediction_difference(first, second)
        self.assertEqual((diff["row"], diff["write_index"], diff["candidate"]), (1, 1, -1))
        self.assertIsNone(runner.first_prediction_difference(first, first))
        self.assertEqual(runner.first_prediction_difference(first, second[:, :4])["kind"], "shape")
        second = first.copy(); second[1, 0] = 2
        self.assertIsNone(runner.first_prediction_difference(first, second)["group_token_index"])

    def test_comparing_version_body_offsets_is_explicit(self):
        body = np.arange(10, dtype=np.uint8).reshape(2, 5)
        old = SimpleNamespace(payload=np.concatenate((np.zeros((2, 8), np.uint8), body), 1))
        new = SimpleNamespace(payload=np.concatenate((np.ones((2, 17), np.uint8), body), 1))
        self.assertTrue(runner.payload_comparison(old, new, reference_header=8, candidate_header=17)["match"])
        self.assertFalse(runner.payload_comparison(old, new)["match"])
        new.payload[1, 19] += 1
        diff = runner.payload_comparison(old, new, reference_header=8, candidate_header=17)["first_difference"]
        self.assertEqual((diff["row"], diff["byte_in_compared_region"]), (1, 2))
        self.assertIn("not a per-write", diff["time_scope"])

    def test_chunk_calls_keep_batch_one_bos_and_terminal_absorption(self):
        tokens = np.ones((3, 64), np.int64)
        tokens[0, 16] = 5
        adapter = OnlineAdapter(PackedCodec((1, 2, 2), 4, stochastic=True))
        seeds = np.array([5, 100, 203], np.uint64)
        full, expected, _ = helper.synthetic_sequence_call(None, {}, tokens, adapter, seeds)
        calls = []

        def track(*args, **kwargs):
            calls.append((args[2].shape, kwargs["include_bos"], kwargs["initial"]))
            return helper.synthetic_sequence_call(*args, **kwargs)

        split, state, report = runner.chunked_call(track, None, {}, tokens, adapter, seeds, v2=True)
        np.testing.assert_array_equal(full, split)
        np.testing.assert_array_equal(expected.payload, state.payload)
        self.assertEqual([entry[0][0] for entry in calls], [3] * 4)
        self.assertEqual([entry[1] for entry in calls], [True, False, False, False])
        self.assertIsNone(calls[0][2])
        self.assertTrue(all(entry[2] is not None for entry in calls[1:]))
        self.assertEqual(report["counts"]["active_update_attempts"] + report["counts"]["terminal_noop_steps"], 3 * 65)
        self.assertTrue(np.all(split[0, 17:] == -1))

    def test_worker_input_hash_shape_dtype_and_extra_array_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.npz"
            def plan():
                return {"input_file": str(path), "input_file_sha256": runner.common.file_sha(path), "input_shape": [2, 3]}
            np.savez(path, tokens=np.zeros((2, 3), np.uint8), gold=np.zeros((2, 3), np.uint8))
            retained = plan()
            self.assertEqual(runner.verify_input_files(retained)[0].dtype, np.int64)
            path.write_bytes(path.read_bytes() + b"extra")
            with self.assertRaisesRegex(ValueError, "bytes changed"): runner.verify_input_files(retained)
            for arrays in [dict(tokens=np.zeros((2, 3), np.int64), gold=np.zeros((2, 3), np.uint8)),
                           dict(tokens=np.zeros((2, 3), np.uint8), gold=np.full((2, 3), 6, np.uint8)),
                           dict(tokens=np.zeros((2, 3), np.uint8), gold=np.zeros((2, 3), np.uint8), unexpected=np.zeros(1))]:
                np.savez(path, **arrays)
                with self.assertRaises(ValueError): runner.verify_input_files(plan())

    def test_actual_fresh_process_runtime_checkpoints_and_public_receipt(self):
        tokens = np.ones((2, 64), np.int64)
        tokens[0, 16] = 5
        gold = np.zeros(tokens.shape, np.uint8)
        codec = PackedCodec((1, 2, 2), 4, stochastic=True)
        old, new = _V1Adapter(codec), OnlineAdapter(codec)
        v1 = SimpleNamespace(sequence_call=helper.synthetic_sequence_call,
                             stream_seeds=lambda seed, batch: np.arange(batch, dtype=np.uint64) + seed)
        real_run = subprocess.run
        launches = []

        def synthetic_child(command, **kwargs):
            launches.append(command)
            replaced = [sys.executable, "-B", str(HELPER), command[-1]]
            return real_run(replaced, **kwargs)

        with tempfile.TemporaryDirectory() as directory, patch.object(runner.subprocess, "run", side_effect=synthetic_child):
            public, private = Path(directory) / "public", Path(directory) / "private"
            result = runner.run_cell(model=None, table={}, v1=v1, v2_call=helper.synthetic_sequence_call,
                old=old, new=new, tokens=tokens, gold=gold, ids=["synthetic-a", "synthetic-b"], seed=0,
                input_seed=99, name="SYNTHETIC_SR4_FAULT", cohort="SYNTHETIC_TEST_ONLY", checkpoint="unused-model",
                upstream="unused-upstream", public=public, private=private,
                protocol_hash=runner.common.file_sha(runner.common.ROOT / "protocol_v2.json"))
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(len(launches), 1)
            self.assertEqual(result["fresh_processes"], 1)
            self.assertTrue(result["fresh_serialized_boundary_trace_match"])
            self.assertFalse(result["reviewer_probe_executed"])
            self.assertEqual(result["fresh_primary_sample_count"], 0)
            self.assertEqual(result["first_terminal_write"], [17, None])
            self.assertEqual(result["legacy_error"]["stage"], "v1_full")
            self.assertEqual({path.name for path in public.iterdir()}, {"receipt.json", "predictions.npz"})
            raw = (public / "receipt.json").read_text()
            self.assertNotIn(directory, raw)
            self.assertNotIn("unused-model", raw)
            for end in [1, 17, 63, 64]:
                self.assertTrue((private / "parent-checkpoints" / f"cut-{end:04d}" / "runtime.bin").is_file())
            child = json.loads((private / "fresh-worker" / "receipt.json").read_text())
            self.assertEqual(child["counts"]["terminal_noop_steps"], 47)
            with np.load(public / "predictions.npz", allow_pickle=False) as data:
                self.assertEqual(data["v2_fresh"].shape, (2, 65))
                np.testing.assert_array_equal(data["v2_fresh"], data["v2_full"])

    def test_deadline_and_no_private_payload_inside_public(self):
        self.assertEqual(runner.deadline_value("1000"), 1000)
        self.assertEqual(runner.deadline_value("1970-01-01T00:00:01Z"), 1)
        with self.assertRaises(ValueError): runner.deadline_value("2026-01-01T00:00:00")
        for value in ['inf', '-inf', 'nan']:
            with self.assertRaises(ValueError): runner.deadline_value(value)
        with self.assertRaises(runner.DeadlineReached): runner.check_deadline(0)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "outside public"):
                runner.main(["--seed", "0", "--checkpoint", "unused", "--upstream", "unused", "--v1-case", "unused",
                             "--output", directory, "--private-output", str(Path(directory) / "private")])

    def test_healthy_legacy_serialization_matches_v2_body(self):
        tokens = np.ones((2, 18), np.int64)
        codec = PackedCodec((1, 2, 2), 8, stochastic=True)
        v1 = SimpleNamespace(sequence_call=helper.synthetic_sequence_call,
                             stream_seeds=lambda seed, batch: np.arange(batch, dtype=np.uint64) + seed)
        real_run = subprocess.run
        def synthetic_child(command, **kwargs):
            return real_run([sys.executable, '-B', str(HELPER), command[-1]], **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.object(runner.subprocess, 'run', side_effect=synthetic_child):
            public, private = Path(directory) / 'public', Path(directory) / 'private'
            result = runner.run_cell(model=None, table={}, v1=v1, v2_call=helper.synthetic_sequence_call,
                old=_V1Adapter(codec), new=OnlineAdapter(codec), tokens=tokens, gold=np.zeros(tokens.shape, np.uint8),
                ids=['synthetic-a', 'synthetic-b'], seed=0, input_seed=99, name='SYNTHETIC_SR8_HEALTHY',
                cohort='SYNTHETIC_TEST_ONLY', checkpoint='unused-model', upstream='unused-upstream',
                public=public, private=private, protocol_hash=runner.common.file_sha(runner.common.ROOT / 'protocol_v2.json'))
            self.assertIsNone(result['legacy_error'])
            for key in ['v1_full_vs_split', 'v1_vs_v2_full_body']:
                self.assertTrue(result['comparisons'][key]['predictions_match'])
                self.assertTrue(result['comparisons'][key]['payload']['match'])
            self.assertEqual(result['serialized_at_all_boundaries'], ['v1', 'v2'])
            for end in [1, 17, 18]:
                folder = private / 'legacy-checkpoints' / f'cut-{end:04d}'
                self.assertEqual({p.name for p in folder.iterdir()}, {'codec.json', 'basis.bin', 'runtime.bin', 'predictions.npy'})

    def test_legacy_programming_and_corruption_errors_are_not_numeric_results(self):
        for message in ['expected [batch,1,1,1]', 'payload has nonfinite native values',
                        'RNG arguments require stochastic=True', 'arbitrary implementation failure']:
            self.assertFalse(runner.legacy_numeric_error(ValueError(message)))
        self.assertFalse(runner.legacy_numeric_error(RuntimeError('bug')))
        self.assertTrue(runner.legacy_numeric_error(ValueError('per-head scale exceeds finite FP32 range')))
        self.assertTrue(runner.legacy_numeric_error(FloatingPointError('numeric failure')))


if __name__ == "__main__":
    unittest.main()
