"""Synthetic short CPU cohorts only; no frozen scientific cohort is run."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from source.evaluation import sequence_call
from source.online_v2 import OnlineAdapter, NativeFloatCodec, PackedCodec
from source.precision import PrecisionDiagnostic, MODES
from source.reference import learned, v1
from source.run_precision import (ScalarHistory, aggregate_scalars, execute_precision,
                                  masked, save_cell, stable_l2, stable_mse)
from source.run_trace import execute_trace, trace_values


class DiagnosticUtilityTests(unittest.TestCase):
    def test_stable_norm_and_finite_population(self):
        value = np.array([[3e200, 4e200], [0., 0.]])
        np.testing.assert_allclose(stable_l2(value), [5e200, 0], rtol=1e-14)
        np.testing.assert_allclose(stable_mse(np.array([[1e154, 1e154]])), [1e308])
        out = masked([0, 3, np.inf], [False, True, True])
        self.assertTrue(np.isnan(out[0]) and out[1] == 3 and np.isnan(out[2]))

    def test_before_after_first_failure_and_valid_counts(self):
        predictions = np.array([[0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.int8)
        gold = np.zeros((2, 3), dtype=np.uint8)
        arrays = {"norm": np.array([[1., 2., 3., 4.], [10., 20., np.nan, 40.]])}
        summary = aggregate_scalars(arrays, predictions, gold)
        np.testing.assert_array_equal(summary["norm__before_first_failure__valid_count"], [0, 2, 0, 1])
        np.testing.assert_array_equal(summary["norm__at_or_after_first_failure__valid_count"], [0, 0, 1, 1])
        self.assertEqual(summary["norm__at_or_after_first_failure__mean"][2], 3.)
        self.assertTrue(np.isnan(summary["norm__before_first_failure__mean"][2]))
        self.assertTrue(summary["at_or_after_first_failure"][0, 3])

    def test_history_rejects_state_tensors(self):
        history = ScalarHistory(2)
        with self.assertRaisesRegex(ValueError, "one scalar"):
            history.append({"bad": np.zeros((2, 12, 16, 16))})

    def test_terminal_placeholder_excluded_from_all_trace_metrics(self):
        adapter = OnlineAdapter(PackedCodec((1, 2, 2), bits=4))
        state = adapter.initial(np.zeros((2, 1, 2, 2), dtype=np.float32))

        def update(x):
            result = np.ones_like(x)
            result[0] = np.inf
            return result

        state, represented, _ = adapter.step(state, update, diagnostics=False)
        values = trace_values(adapter, state, represented, update(represented),
            native=np.ones_like(represented), shadow=np.ones_like(represented, dtype=np.float64),
            native_active=np.ones(2, bool), shadow_active=np.ones(2, bool),
            logits=np.zeros((2, 6)), target=np.zeros(2, np.int64))
        self.assertFalse(values["active"][0])
        for name in ("state_norm", "residual_norm", "native_state_error_mse", "self_writeback_mse", "state_scale_maximum"):
            self.assertTrue(np.isnan(values[name][0]), name)
            self.assertTrue(np.isfinite(values[name][1]), name)


class DiagnosticExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(1907)
        upstream_location = os.environ.get("CKDA_UPSTREAM")
        if not upstream_location:
            raise unittest.SkipTest("Set CKDA_UPSTREAM to the pinned upstream checkout")
        upstream = learned.load_upstream(upstream_location)
        cls.model = learned.create_model(upstream, backend="naive_recurrent", device="cpu").eval()
        cls.table = v1.token_table(cls.model)
        cls.tokens = np.array([[0, 1, 2, 3], [4, 5, 0, 1]], dtype=np.int64)
        cls.gold = np.zeros_like(cls.tokens, dtype=np.uint8)
        identity = {"model_seed": 0, "checkpoint_sha256": "0"*64,
                    "verified_checkpoint_sha256": {str(s): str(s)*64 for s in range(3)},
                    "token_table_sha256": hashlib.sha256(v1.table_bytes(cls.table)[1]).hexdigest()}
        cls.diagnostic = PrecisionDiagnostic(cls.model, cls.table, identity)

    def arms(self):
        shape = (12, 16, 16)
        basis = np.broadcast_to(np.eye(16, 2, dtype=np.float32), (12, 16, 2)).copy()
        mapping = np.full((12, 16), 5, dtype=np.uint8)
        mapping.flat[:3] = 6
        return {
            "NATIVE_FP32": OnlineAdapter(NativeFloatCodec(shape)),
            "UNIFORM_8": OnlineAdapter(PackedCodec(shape, bits=8)),
            "UNIFORM_5": OnlineAdapter(PackedCodec(shape, bits=5)),
            "LOWRANK_4_8_R2": OnlineAdapter(PackedCodec(shape, bits=4), PackedCodec((12, 2, 16), bits=8), basis),
            "MIXED_5_6_BUDGET": OnlineAdapter(PackedCodec(shape, bits=5, mixed_bits=mapping)),
        }

    def test_trace_has_no_effect_on_candidate_predictions_or_cache(self):
        adapters = self.arms()
        records, shadows = execute_trace(self.model, self.table, self.tokens, self.gold, adapters)
        self.assertFalse(shadows["passed_to_candidate"])
        for name, adapter in adapters.items():
            reference_pred, reference_state, _ = sequence_call(self.model, self.table, self.tokens, adapter)
            np.testing.assert_array_equal(records[name]["predictions"], reference_pred)
            np.testing.assert_array_equal(records[name]["final_state"].payload, reference_state.payload)
            self.assertEqual(records[name]["execution"]["completed_writes"], 5)
            self.assertEqual(records[name]["scalars"]["state_norm"].shape, (2, 5))
        np.testing.assert_array_equal(records["NATIVE_FP32"]["scalars"]["native_state_error_mse"], 0)

    def test_precision_D00_matches_original_evaluation_and_saves_no_intervals(self):
        records, traces = execute_precision(self.diagnostic, self.tokens, self.gold)
        reference_pred, _, _ = sequence_call(self.model, self.table, self.tokens, self.arms()["NATIVE_FP32"])
        np.testing.assert_array_equal(records["D00"]["predictions"], reference_pred)
        self.assertEqual(traces["D11"]["readout"]["rms_mean_square"], "torch.float64")
        with tempfile.TemporaryDirectory() as directory:
            result = save_cell(Path(directory)/"D00", records["D00"], self.gold,
                               {"arm": "D00", "model_seed": 0, "cohort": "synthetic-unit-test"})
            self.assertIsNone(result["family_size"])
            self.assertTrue(all(row["p_upper"] is None for row in result["grid_rows"]))
            with np.load(Path(directory)/"D00/scalar_trajectories.npz", allow_pickle=False) as data:
                self.assertTrue(all(value.ndim == 2 for value in data.values()))
            json.dumps(result, allow_nan=False)
            with self.assertRaises(FileExistsError):
                save_cell(Path(directory)/"D00", records["D00"], self.gold, {})

    def test_deadline_retains_only_observed_prefix(self):
        records, _ = execute_precision(self.diagnostic, self.tokens, self.gold, deadline=time.monotonic()-1)
        for mode in MODES:
            self.assertEqual(records[mode]["predictions"].shape, (2, 0))
            self.assertEqual(records[mode]["execution"]["status"], "NOT_COMPLETE")
        with tempfile.TemporaryDirectory() as directory:
            saved = save_cell(Path(directory)/"partial", records["D00"], self.gold, {"arm": "D00"})
            self.assertIsNone(saved["metrics"])
            self.assertEqual(saved["status"], "NOT_COMPLETE")


if __name__ == "__main__":
    unittest.main()
