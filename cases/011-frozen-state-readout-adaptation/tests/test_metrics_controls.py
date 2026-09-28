"""Synthetic contracts only; these values are not measured CKDA results."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

CASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CASE))
from source import controls, metrics

spec = importlib.util.spec_from_file_location("independent_case011_audit", CASE / "analysis" / "audit.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


class MetricsContracts(unittest.TestCase):
    def setUp(self):
        self.gold = np.zeros((3, 4), dtype=np.int8)
        self.pred = np.asarray([[0, 1, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 0]], dtype=np.int8)

    def test_first_error_recovery_censor(self):
        result = metrics.summarize_head(self.pred, self.gold)
        self.assertEqual(result["tau"], [2, None, 1])
        self.assertEqual(result["rmst0"], 5/3)
        self.assertEqual(result["sequences_with_correct_token_after_first_error"], 2)
        self.assertEqual(result["correct_tokens_after_first_error"], 5)
        self.assertEqual(result["survival_by_token"], [2/3, 1/3, 1/3, 1/3])

    def test_no_failure_is_not_infinite_horizon(self):
        result = metrics.summarize_head(self.gold, self.gold)
        self.assertEqual(result["empirical_t005"], 4)
        self.assertTrue(result["empirical_horizon_at_observation_limit"])
        self.assertIsNone(result["confidence_supported_horizon"])
        self.assertEqual(result["right_censored_count"], 3)

    def test_rmst_excludes_first_failure(self):
        p = np.full_like(self.gold, -1)
        self.assertEqual(metrics.summarize_head(p, self.gold)["rmst0"], 0)

    def test_horizon_every_token_not_sparse_grid(self):
        g = np.zeros((20, 20), dtype=np.int8)
        p = g.copy()
        p[:2, 12] = 1
        self.assertEqual(metrics.summarize_head(p, g)["empirical_t005"], 12)

    def test_shape_labels_and_head_identity_rejected(self):
        for bad in (self.pred.astype(float), self.pred[:, :-1], np.full_like(self.pred, 6)):
            with self.assertRaises(ValueError):
                metrics.summarize_head(bad, self.gold)
        with self.assertRaises(ValueError):
            metrics.summarize_predictions(np.stack([self.pred]*3), self.gold, ["x", "x", "z"])

    def test_bootstrap_pairing_perfect_constant_shift(self):
        result = metrics.paired_bootstrap_rmst(np.full(32, 3), seed=63001)
        self.assertEqual(result["interval_tokens"], [3, 3])
        self.assertEqual(result["confidence"], 1-.05/3)

    def test_bootstrap_matches_independent_sequential_draws(self):
        arrays = np.asarray([[0, 2, 4, 8], [0, 0, 0, 0], [1, 1, 5, 9]])
        actual = metrics.paired_bootstrap_rmst(arrays[2]-arrays[0], seed=63002)
        independent = auditor.independent_primary(arrays, 1)
        self.assertEqual(actual["interval_tokens"], independent["interval_tokens"])

    def test_logits_scores_stable_and_invalid_separate(self):
        logits = np.asarray([[1000, 1000, 1000, 1000, 1000, 1000], [np.nan, 0, 0, 0, 0, 0]])
        result = metrics.score_logits(logits, np.asarray([0, 0]))
        self.assertAlmostEqual(result["gold_ce"][0], np.log(6), places=12)
        self.assertEqual(result["gold_margin"][0], 0)
        self.assertFalse(result["finite"][1])
        self.assertTrue(np.isnan(result["gold_ce"][1]))

    def test_independent_audit_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            predictions = np.stack([self.pred, self.gold, self.gold])
            np.savez(temp/"pred.npz", predictions=predictions, gold=self.gold,
                     head_names=np.asarray(["ORIGINAL", "SHORT_REFIT", "MIXED_REFIT"]),
                     sample_ids=np.asarray(["a", "b", "c"]), checkpoint_seed=np.asarray(0),
                     input_hash=np.asarray("a"*64), checkpoint_sha256=np.asarray("b"*64),
                     storage=np.asarray("UNIFORM_8"))
            summary = metrics.summarize_predictions(predictions, self.gold)
            (temp/"summary.json").write_text(json.dumps(summary, allow_nan=False))
            result = auditor.audit(temp/"pred.npz", temp/"summary.json", checkpoint_seed=0)
            self.assertEqual(result["status"], "PASS")
            summary["ORIGINAL"]["rmst0"] += 1
            (temp/"summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "rmst0"):
                auditor.audit(temp/"pred.npz", temp/"summary.json", checkpoint_seed=0)

    def test_independent_audit_duplicate_sample_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"pred.npz"
            np.savez(path, predictions=np.stack([self.pred]*3), gold=self.gold,
                     head_names=np.asarray(["ORIGINAL", "SHORT_REFIT", "MIXED_REFIT"]),
                     sample_ids=np.asarray(["a", "a", "c"]), checkpoint_seed=np.asarray(0),
                     input_hash=np.asarray("a"*64), checkpoint_sha256=np.asarray("b"*64),
                     storage=np.asarray("UNIFORM_8"))
            with self.assertRaisesRegex(ValueError, "sample IDs"):
                auditor.audit(path, Path(temp)/"absent.json", checkpoint_seed=0)

    def test_cross_cell_pairing_and_storage_checkpoint_rejection(self):
        receipts = []
        for seed in range(3):
            for storage in ("NATIVE_FP32", "UNIFORM_8"):
                receipts.append({"status": "PASS", "checkpoint_seed": seed, "n_sequences": 1024,
                                 "max_group_tokens": 2048, "identity": {
                                     "storage": storage, "checkpoint_sha256": str(seed)*64,
                                     "input_hash": "a"*64, "sample_ids_sha256": "b"*64,
                                     "gold_sha256": "c"*64}})
        self.assertEqual(auditor.check_pairing(receipts)["rollout_cells"], 6)
        receipts[-1]["identity"]["checkpoint_sha256"] = "d"*64
        with self.assertRaisesRegex(ValueError, "frozen checkpoint"):
            auditor.check_pairing(receipts)

    def test_cross_cell_input_and_missing_cell_rejection(self):
        with self.assertRaisesRegex(ValueError, "six"):
            auditor.check_pairing([])


class ControlContracts(unittest.TestCase):
    def test_position_bands_and_extrapolation_frozen(self):
        self.assertEqual(controls.position_bands(np.asarray([1,32,33,64,65,128,129,256,2048])).tolist(),
                         [0,0,1,1,2,2,3,3,3])

    def test_input_only_fit_and_lowest_label_tie(self):
        counts = np.zeros((4,6,6), dtype=np.int64)
        counts[0,2,4] = 3
        predictor = controls.InputOnlyPredictor(counts)
        self.assertEqual(predictor.predict(np.asarray([2,3]), np.asarray([1,1])).tolist(), [4,0])
        self.assertEqual(predictor.to_dict()["fit_rows"], 3)

    def test_input_only_fit_denominator_and_probability_sums(self):
        t = np.tile(np.arange(6, dtype=np.int8), (2,1))
        predictor = controls.InputOnlyPredictor.fit(t, t)
        self.assertEqual(predictor.counts.sum(), 12)
        np.testing.assert_allclose(predictor.probabilities.sum(axis=-1), 1)

    def test_fit_cannot_use_long_test_positions(self):
        with self.assertRaises(ValueError):
            controls.InputOnlyPredictor.fit(np.zeros((1,257), dtype=np.int8), np.zeros((1,257), dtype=np.int8))
        with self.assertRaises(ValueError):
            controls.position_bands(np.asarray([0]))

    def test_shuffle_keeps_current_token_has_no_bucket_fixed_points(self):
        t = np.asarray([0,0,0,1,1,2,3,3,4,4,4], dtype=np.int8)
        permutation, ledger = controls.shuffle_permutation(t, 128)
        np.testing.assert_array_equal(t, t[permutation])
        self.assertEqual(sorted(permutation.tolist()), list(range(len(t))))
        self.assertEqual(ledger["singleton_unchanged_rows"], 1)
        self.assertEqual(ledger["fixed_point_rows"], 1)
        self.assertEqual(permutation[5], 5)

    def test_shuffle_repeatable_no_label_argument(self):
        t = np.asarray([0]*8+[1]*8, dtype=np.int8)
        first, _ = controls.shuffle_permutation(t, 3)
        second, _ = controls.shuffle_permutation(t, 3)
        np.testing.assert_array_equal(first, second)
        self.assertNotEqual(first.tolist(), controls.shuffle_permutation(t, 4)[0].tolist())

    def test_shuffle_singleton_coverage_explicit(self):
        t = np.arange(6, dtype=np.int8)
        p, ledger = controls.shuffle_permutation(t, 1)
        np.testing.assert_array_equal(p, np.arange(6))
        self.assertEqual(ledger["shuffled_rows"], 0)
        self.assertEqual(ledger["singleton_unchanged_rows"], 6)

    def test_shuffle_invalid_bos_or_labels_rejected(self):
        for token, pos in (([0,1],0), ([0,6],1), ([0,-1],1)):
            with self.assertRaises(ValueError):
                controls.shuffle_permutation(np.asarray(token), pos)


if __name__ == "__main__":
    unittest.main()
