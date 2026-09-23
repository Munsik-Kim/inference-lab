"""Phase A causal targets, toy geometry, frozen controls, and byte accounting."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codec.groups import frozen_sequences, make_group
from codec.toy import (ARM_NAMES, CONDITION_NAMES, SPLITS, affine_transition, calibrate,
                       evaluate_arm, make_adapter, make_conditions, protocol,
                       quantized_operators, readout, run_protocol, symbolic_capacity_control,
                       _step_finite_rows)
from codec.online import OnlineAdapter
from codec.packed import NativeFloatCodec


class ToyGeometryTests(unittest.TestCase):
    def test_exact_group_gold_agrees_with_causal_real_operators(self):
        for condition in make_conditions():
            with self.subTest(condition=condition.name):
                batch = frozen_sequences(condition.group_name, "TEST", 7, 99, seed=3001)
                state = condition.initial(7)
                for step in range(1, 100):
                    state = affine_transition(condition.operators(batch.tokens[:, step - 1], step))(state)
                    prediction = readout(state, condition.prototypes(step))
                    np.testing.assert_array_equal(prediction, batch.gold[:, step - 1])
                    np.testing.assert_allclose(state[:, 0, :, 0], condition.prototypes(step)[prediction], atol=3e-13)

    def test_signed_control_operators_are_integer_lattice_permutations(self):
        condition = make_conditions()[0]
        self.assertEqual(set(np.unique(condition.base_operators)), {-1., 0., 1.})
        np.testing.assert_array_equal(np.abs(condition.base_operators).sum(axis=-1), 1)
        np.testing.assert_array_equal(np.abs(condition.base_operators).sum(axis=-2), 1)
        self.assertEqual(set(np.unique(condition.base_prototypes)), {-7., -3., -1., 1., 3., 7.})
        calibration = calibrate(condition)
        batch = frozen_sequences("S4", "TEST", 8, 64, seed=3001)
        for arm in ("uniform_4", "stochastic_4"):
            result, correct, _ = evaluate_arm(condition, arm, calibration, batch, 1, [64])
            self.assertTrue(correct.all())
            self.assertEqual(result["diagnostics"]["state_mse_per_coordinate"], 0)

    def test_readout_ties_are_smallest_id_and_nonfinite_fails(self):
        prototypes = np.asarray([[1., 0.], [-1., 0.]])
        states = np.asarray([[0., 0.], [np.nan, 0.], [np.inf, 0.]])[:, None, :, None]
        np.testing.assert_array_equal(readout(states, prototypes), [0, -1, -1])

    def test_known_plane_schedule_moves_and_switches_prototypes(self):
        conditions = {condition.name: condition for condition in make_conditions()}
        switching = conditions["c31_switching_plane"]
        np.testing.assert_allclose(switching.prototypes(31)[:, 2:], 0, atol=1e-15)
        np.testing.assert_allclose(switching.prototypes(32)[:, :2], 0, atol=1e-15)
        np.testing.assert_allclose(switching.prototypes(64)[:, 2:], 0, atol=1e-15)
        moving = conditions["c31_moving_plane"]
        self.assertGreater(np.linalg.norm(moving.prototypes(25)[:, 2:]), 1)

    def test_noncommuting_s4_rotated_operator_order(self):
        condition = next(c for c in make_conditions() if c.name == "s4_rotated")
        group = make_group("S4")
        first, second = 1, 2
        self.assertNotEqual(group.multiplication[first, second], group.multiplication[second, first])
        np.testing.assert_allclose(condition.base_operators[second] @ condition.base_operators[first],
                                   condition.base_operators[group.multiplication[second, first]], atol=1e-15)


class ToyProtocolTests(unittest.TestCase):
    def test_numerical_row_failure_preserves_healthy_rows_and_old_packed_bytes(self):
        adapter = OnlineAdapter(NativeFloatCodec((1, 2, 1), "float32"))
        initial = np.asarray([[3e38, 1], [1, 2]], dtype=np.float32)[:, None, :, None]
        state = adapter.initial(initial)
        original_bytes = state.payload.copy()
        active = np.ones(2, dtype=bool)
        operators = np.asarray([[[2., 0.], [0., 1.]], np.eye(2)])
        state, output, _, failed = _step_finite_rows(adapter, state, operators, active)
        self.assertEqual(failed, [(0, "transition_output")])
        np.testing.assert_array_equal(active, [False, True])
        self.assertTrue(np.isnan(output[0]).all())
        np.testing.assert_array_equal(output[1], initial[1])
        np.testing.assert_array_equal(state.payload[0], original_bytes[0])
        state, output, _, failed = _step_finite_rows(adapter, state, operators, active)
        self.assertEqual(failed, [])
        self.assertTrue(np.isnan(output[0]).all())
        np.testing.assert_array_equal(output[1], initial[1])
        self.assertEqual(int.from_bytes(state.payload[1, :8].tobytes(), "little"), 2)
        self.assertEqual(int.from_bytes(state.payload[0, :8].tobytes(), "little"), 0)

    def test_shape_or_programming_errors_are_not_scored_as_numerical_failures(self):
        from unittest.mock import patch
        adapter = OnlineAdapter(NativeFloatCodec((1, 2, 1), "float32"))
        state = adapter.initial(np.ones((2, 1, 2, 1)))
        with patch.object(adapter, "step", side_effect=ValueError("bad operator shape")):
            with self.assertRaisesRegex(ValueError, "bad operator shape"):
                _step_finite_rows(adapter, state, np.broadcast_to(np.eye(2), (2, 2, 2)), np.ones(2, dtype=bool))

    def test_primary_protocol_prespecifies_full_family(self):
        frozen = protocol()
        self.assertEqual(len(ARM_NAMES), 25)
        self.assertEqual(len(CONDITION_NAMES), 5)
        self.assertEqual(frozen["family_size"], 875)
        self.assertEqual(frozen["splits"]["TEST"], {"n": 512, "horizon": 2048, "seed": 3001})
        self.assertEqual([SPLITS[s]["seed"] for s in ("CAL", "DEV", "TEST")], [1001, 2001, 3001])
        self.assertTrue(frozen["primary"])
        self.assertFalse(frozen["update_gold_access"])
        self.assertTrue(all(f"uniform_{bits}" in ARM_NAMES for bits in range(2, 17)))

    def test_calibration_is_deterministic_orthonormal_and_only_cal(self):
        for condition in make_conditions():
            with self.subTest(condition=condition.name):
                fitted = calibrate(condition)
                self.assertEqual(fitted, calibrate(condition))
                self.assertEqual(fitted["calibration"], SPLITS["CAL"])
                basis = np.asarray(fitted["basis"])
                np.testing.assert_allclose(basis.T @ basis, np.eye(fitted["rank"]), atol=1e-14)
                self.assertEqual(fitted["rank"], 1 if condition.dimension == 2 else 2)
                self.assertEqual(fitted["mixed_4_8"].count(8), 1)
                self.assertEqual(fitted["mixed_6_8"].count(8), condition.dimension // 2)

    def test_every_arm_runs_and_ledgers_match_actual_shared_bytes(self):
        condition = next(c for c in make_conditions() if c.name == "c31_rotation_2d")
        fitted = calibrate(condition)
        batch = frozen_sequences("C31", "TEST", 4, 8, seed=3001)
        for arm in ARM_NAMES:
            with self.subTest(arm=arm):
                result, correct, shared = evaluate_arm(condition, arm, fitted, batch, 25, [8])
                self.assertEqual(result["ledger"]["total_shared_bytes"], len(shared))
                self.assertEqual(correct.shape, (4, 8))
                ledger = result["ledger"]
                self.assertEqual(ledger["batch_payload_nbytes"], 4 * ledger["stream_payload_bytes"])
                self.assertEqual(ledger["amortized_bytes_per_stream"]["16"],
                                 ledger["stream_payload_bytes"] + len(shared) / 16)
                json.dumps(result, allow_nan=False)

    def test_symbolic_control_counts_table_decoder_and_cursor(self):
        condition = make_conditions()[1]
        batch = frozen_sequences("C31", "TEST", 8, 64, seed=3001)
        result, shared = symbolic_capacity_control(condition, batch)
        self.assertEqual(result["tau"], [None] * 8)
        self.assertEqual(result["ledger"]["stream_payload_bytes"], 9)
        self.assertEqual(result["ledger"]["multiplication_table_bytes"], 31 * 31)
        self.assertEqual(result["ledger"]["total_shared_bytes"], len(shared))
        self.assertNotIn("infinite", result["interpretation"].split("no infinite")[0])

    def test_coefficient_quantization_is_distinct_from_state_quantization(self):
        condition = make_conditions()[1]
        fixed = quantized_operators(condition, 64)
        self.assertEqual(fixed.codes.dtype, np.int8)
        self.assertEqual(fixed.scales.dtype, np.float32)
        self.assertFalse(fixed.dynamic)
        self.assertGreater(np.max(np.abs(fixed.at(np.arange(31), 1) - condition.base_operators)), 0)
        adapter = make_adapter(condition, "coefficient_int8_fp32", calibrate(condition))
        self.assertEqual(adapter.state_codec.dtype, np.dtype("<f4"))
        dynamic = quantized_operators(make_conditions()[3], 8)
        self.assertTrue(dynamic.dynamic)
        self.assertEqual(dynamic.codes.shape, (8, 31, 4, 4))

    def test_frozen_artifacts_precede_test_and_correctness_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            observed = []
            def check_frozen(key, result):
                self.assertTrue((output / "protocol-frozen.json").exists())
                self.assertTrue((output / "calibration-identity.json").exists())
                observed.append(key)
            manifest = run_protocol(output, smoke=True, progress=check_frozen)
            self.assertFalse(manifest["primary"])
            self.assertEqual(manifest["completed_arms"], 125)
            self.assertEqual(manifest["failed_arms"], [])
            self.assertEqual(len(observed), 125)
            frozen_bytes = (output / "protocol-frozen.json").read_bytes().rstrip(b"\n")
            self.assertEqual(hashlib.sha256(frozen_bytes).hexdigest(), manifest["protocol_sha256"])
            self.assertEqual(len((output / "results.jsonl").read_text().splitlines()), 130)
            one = np.load(next((output / "correctness").glob("*.npz")))
            restored = np.unpackbits(one["packed"], axis=1, count=int(one["shape"][1]), bitorder="little").astype(bool)
            self.assertEqual(restored.shape, (8, 64))
            with self.assertRaises(FileExistsError):
                run_protocol(output, smoke=True)


if __name__ == "__main__":
    unittest.main()
