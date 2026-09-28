"""Synthetic scalar contracts for the saved-record publication audit."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "posthoc/analyze.py"
spec = importlib.util.spec_from_file_location("case011_publication_posthoc", SCRIPT)
posthoc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(posthoc)


class PosthocTests(unittest.TestCase):
    def test_first_error_not_erased_by_recovery(self):
        g = np.zeros((2, 4), dtype=np.int8)
        p = np.array([[0, 1, 0, 0], [0, 0, 0, 0]], dtype=np.int8)
        self.assertEqual(posthoc.lengths(p, g).tolist(), [1, 4])

    def test_invalid_is_error_and_censor_is_full_length(self):
        g = np.zeros((3, 3), dtype=np.int8)
        p = np.array([[-1, 0, 0], [0, 0, -1], [0, 0, 0]], dtype=np.int8)
        self.assertEqual(posthoc.lengths(p, g).tolist(), [0, 2, 3])

    def test_rejects_noninteger_and_invalid_labels(self):
        g = np.zeros((1, 2), dtype=np.int8)
        with self.assertRaises(ValueError):
            posthoc.lengths(g.astype(float), g)
        with self.assertRaises(ValueError):
            posthoc.lengths(np.array([[6, 0]], dtype=np.int8), g)

    def test_horizon_is_failure_through_token(self):
        l = np.array([1] + [4]*19)
        self.assertEqual(posthoc.empirical_horizon(l, 4), 4)
        self.assertEqual(posthoc.empirical_horizon(np.array([1, 1] + [4]*18), 4), 1)

    def test_transition_partition_wrong_to_wrong_is_explicit(self):
        g = np.zeros((1, 5), dtype=np.int8)
        a = np.array([[0, 0, 1, 1, 1]], dtype=np.int8)
        b = np.array([[0, 1, 0, 1, 2]], dtype=np.int8)
        r = posthoc.transitions(a, b, g)
        for k in ("both_correct", "original_only_correct", "mixed_only_correct", "both_wrong_same_answer", "wrong_to_wrong_different_answer"):
            self.assertEqual(r[k], 1)
        self.assertEqual(r["all_answer_disagreement"], 3)
        self.assertEqual(r["both_wrong_total"], 2)

    def test_paired_bootstrap_constant_difference(self):
        r = posthoc.paired_interval(np.full(20, -4), 71101, repetitions=100)
        self.assertEqual(r["mean_delta_tokens"], -4)
        self.assertEqual(r["interval_tokens"], [-4, -4])

    def test_paired_bootstrap_determinism_and_nonfinite_rejection(self):
        d = np.array([-1, 0, 2, 5])
        self.assertEqual(posthoc.paired_interval(d, 42, repetitions=100), posthoc.paired_interval(d, 42, repetitions=100))
        with self.assertRaises(ValueError):
            posthoc.paired_interval(np.array([1, np.nan]), 42)

    def test_interaction_keeps_joint_pairing(self):
        native = np.array([100, -100, 10, -10])
        int8 = native + 2
        r = posthoc.paired_interval(int8-native, 1, repetitions=100)
        self.assertEqual(r["interval_tokens"], [2, 2])

    def test_pairing_rejects_id_order_and_gold_mismatch(self):
        a = {"sample_ids": np.array(["a", "b"]), "gold": np.array([[0], [1]]),
             "input_hash": np.array("x"), "checkpoint_sha256": np.array("y")}
        b = {k:v.copy() for k,v in a.items()}
        posthoc.check_pairing(a, b)
        b["sample_ids"] = b["sample_ids"][::-1]
        with self.assertRaises(ValueError):
            posthoc.check_pairing(a, b)
        b = {k:v.copy() for k,v in a.items()}
        b["gold"][0, 0] = 5
        with self.assertRaises(ValueError):
            posthoc.check_pairing(a, b)

    def test_shift_counts_sum_to_all_sequences(self):
        r = posthoc.difference_distribution(np.array([-8, -2, 0, 0, 4, 8]))
        self.assertEqual((r["longer"], r["shorter"], r["same"]), (2, 2, 2))
        self.assertEqual(r["median_tokens"], 0)
        self.assertEqual(r["n_sequences"], 6)

    def test_strict_json_rejects_nonfinite(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/"bad.json"
            p.write_text('{"score": Infinity}')
            with self.assertRaises(ValueError):
                posthoc.read_json(p)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "output exists"):
                posthoc.analyze(SCRIPT.parents[2], Path(folder))

    def test_saved_summary_denominators_and_protocol(self):
        data = posthoc.read_json(SCRIPT.parent / "data/summary.json")
        self.assertEqual((data["new_model_forwards"], data["new_fits"]), (0, 0))
        self.assertEqual((data["actual_recurrent_rollouts"], data["logical_readout_conditions"]), (6, 18))
        self.assertEqual(len(data["solver"]["candidates"]), 18)
        self.assertEqual(len(data["solver"]["selected"]), 6)
        for row in data["bands"]:
            self.assertEqual(row["denominator_tokens"], 1024*(row["last_group_token"]-row["first_group_token"]+1))
            for head in ("original", "mixed"):
                self.assertEqual(row[head]["first_error_denominator"], 1024)
                self.assertFalse(row[head]["first_error_is_conditional_hazard"])
        self.assertEqual(data["conditional_margin"]["status"], "UNAVAILABLE_NOT_RECORDED")
        for row in data["interaction"]:
            self.assertEqual(row["bootstrap_seed"], 71101+row["checkpoint_seed"])
            self.assertEqual(row["confidence"], .95)


if __name__ == "__main__":
    unittest.main()
