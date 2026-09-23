"""CPU-only checks for exact target direction and sequence-level inference."""

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codec.groups import SPLIT_SEEDS, frozen_sequences, make_group
from codec.survival import (
    clopper_pearson_upper,
    correctness_from_labels,
    first_failure_from_labels,
    first_failure_times,
    paired_bootstrap,
    summarize_by_model_seed,
    summarize_first_failures,
)


class ExactGroupTests(unittest.TestCase):
    def test_identity_inverses_and_every_matrix_product(self):
        for name, order in [("S3", 6), ("S4", 24), ("C31", 31)]:
            group = make_group(name)
            self.assertEqual(group.order, order)
            np.testing.assert_array_equal(group.multiplication[0], np.arange(order))
            np.testing.assert_array_equal(group.multiplication[:, 0], np.arange(order))
            np.testing.assert_array_equal(group.multiplication[np.arange(order), group.inverses], 0)
            np.testing.assert_array_equal(group.multiplication[group.inverses, np.arange(order)], 0)
            products = group.matrices[:, None] @ group.matrices[None, :]
            np.testing.assert_array_equal(products, group.matrices[group.multiplication])
            np.testing.assert_array_equal(group.matrices.sum(axis=-1), 1)
            np.testing.assert_array_equal(group.matrices.sum(axis=-2), 1)

    def test_noncommuting_left_multiplication_direction(self):
        group = make_group("S3")
        a = group.elements.index((1, 0, 2))
        b = group.elements.index((0, 2, 1))
        self.assertNotEqual(group.multiplication[a, b], group.multiplication[b, a])
        gold = group.trace(np.asarray([a, b], dtype=np.int64))
        self.assertEqual(gold.tolist(), [a, int(group.multiplication[b, a])])
        np.testing.assert_array_equal(group.matrices[gold[-1]], group.matrices[b] @ group.matrices[a])

    def test_known_initial_and_bos_are_explicit(self):
        group = make_group("S3")
        tokens = np.asarray([[1, 2], [2, 1]], dtype=np.int64)
        initial = np.asarray([3, 4], dtype=np.int64)
        without_bos = group.trace(tokens, initial=initial)
        with_bos = group.trace(tokens, initial=initial, include_bos=True)
        np.testing.assert_array_equal(with_bos[:, 0], initial)
        np.testing.assert_array_equal(with_bos[:, 1:], without_bos)
        np.testing.assert_array_equal(without_bos[:, 0], group.multiplication[tokens[:, 0], initial])
        predicted = with_bos.copy()
        predicted[:, 0] = -1  # Known BOS is not a primary prediction.
        self.assertEqual(first_failure_from_labels(predicted, with_bos, 6, include_bos=True), [None, None])
        self.assertEqual(first_failure_from_labels(predicted, with_bos, 6), [1, 1])

    def test_frozen_sequences_are_full_group_and_prefix_stable(self):
        short = frozen_sequences("S3", "CAL", 2, 17)
        long = frozen_sequences("S3", "CAL", 4, 400)
        np.testing.assert_array_equal(short.tokens, long.tokens[:2, :17])
        np.testing.assert_array_equal(short.gold, long.gold[:2, :17])
        self.assertEqual(short.sequence_ids, long.sequence_ids[:2])
        self.assertEqual(set(long.tokens.ravel()), set(range(6)))
        self.assertFalse(long.tokens.flags.writeable)
        self.assertFalse(long.gold.flags.writeable)
        self.assertEqual(len(set(SPLIT_SEEDS.values())), 3)

    def test_splits_have_disjoint_streams_even_with_supplied_seed(self):
        batches = [frozen_sequences("S3", split, 4, 100, seed=77) for split in SPLIT_SEEDS]
        for i in range(3):
            for j in range(i):
                self.assertFalse(np.array_equal(batches[i].tokens, batches[j].tokens))
                self.assertTrue(set(batches[i].sequence_ids).isdisjoint(batches[j].sequence_ids))

    def test_invalid_group_inputs(self):
        group = make_group("S3")
        for tokens in [[-1], [6], [np.nan], [1.0], [True]]:
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                group.trace(tokens)
        with self.assertRaises(ValueError):
            group.trace(np.asarray([1]), initial=np.nan)
        with self.assertRaises(ValueError):
            frozen_sequences(group, "TEST", 2, 8, seed=-1)


class FirstFailureTests(unittest.TestCase):
    def test_recovery_does_not_restore_survival(self):
        correct = np.asarray([[True, False, True, True], [True, True, True, True],
                              [False, True, True, True], [True, True, True, False]])
        taus = first_failure_times(correct)
        self.assertEqual(taus, [2, None, 1, 4])
        summary = summarize_first_failures(taus, 4, horizons=[1, 2, 4], correct=correct)
        self.assertEqual([r["failure_probability"] for r in summary["horizons"]], [0.25, 0.5, 0.75])
        self.assertEqual([r["survival_probability"] for r in summary["horizons"]], [0.75, 0.5, 0.25])
        self.assertEqual(summary["restricted_mean_failure_free_length"], 2.0)
        self.assertEqual(summary["step_accuracy"]["overall"], 13 / 16)
        self.assertEqual(summary["step_accuracy"]["at_horizon"][-1]["accuracy"], 0.75)
        self.assertEqual(summary["right_censored_sequences"], 1)
        self.assertEqual(summary["observed_failures"], 3)
        json.dumps(summary, allow_nan=False)

    def test_failure_at_final_step_is_not_censoring(self):
        summary = summarize_first_failures([4, None], 4, horizons=[4])
        self.assertEqual(summary["tau"], [4, None])
        self.assertEqual(summary["restricted_mean_failure_free_length"], 3.5)
        self.assertEqual(summary["horizons"][0]["failure_probability"], 0.5)
        self.assertEqual(summary["right_censored_sequences"], 1)

    def test_bos_exclusion_starts_primary_time_at_one(self):
        correct = np.asarray([[False, True, False, True], [False, True, True, True]])
        self.assertEqual(first_failure_times(correct, include_bos=True), [2, None])
        result = summarize_first_failures([2, None], 3, horizons=[3], correct=correct, include_bos=True)
        self.assertEqual(result["step_accuracy"]["overall"], 5 / 6)

    def test_nonfinite_predictions_fail_and_nonfinite_gold_is_rejected(self):
        gold = np.asarray([[0, 1, 2, 3, 4, 5]])
        predicted = np.asarray([[0, np.nan, np.inf, -1, 4.5, 5]])
        np.testing.assert_array_equal(correctness_from_labels(predicted, gold, 6), [[True, False, False, False, False, True]])
        self.assertEqual(first_failure_from_labels(predicted, gold, 6), [2])
        for bad_gold in [[[np.nan]], [[np.inf]], [[-1]], [[0.5]], [[6]]]:
            with self.subTest(gold=bad_gold), self.assertRaises(ValueError):
                correctness_from_labels([[0]], bad_gold, 6)
        for malformed in [[[True, np.nan]], [[1, 0]], [True, False]]:
            with self.assertRaises(ValueError):
                first_failure_times(malformed)

    def test_censoring_must_be_none_not_out_of_window_tau(self):
        for taus in [[0], [-1], [5], [np.inf], [np.nan], [4.0], [True]]:
            with self.subTest(taus=taus), self.assertRaises(ValueError):
                summarize_first_failures(taus, 4, horizons=[4])

    def test_first_failures_and_correctness_cannot_disagree(self):
        with self.assertRaises(ValueError):
            summarize_first_failures([None], 4, horizons=[4], correct=np.asarray([[True, False, True, True]]))

    def test_zero_events_are_finite_censored_horizons(self):
        summary = summarize_first_failures([None] * 512, 2048)
        self.assertEqual(summary["right_censored_sequences"], 512)
        self.assertEqual(summary["restricted_mean_failure_free_length"], 2048)
        for epsilon in summary["empirical_T_epsilon"]:
            self.assertEqual(epsilon["horizon"], 2048)
            self.assertEqual(epsilon["relation"], ">=")
            self.assertTrue(epsilon["right_censored_at_max_horizon"])
        self.assertEqual(summary["confidence"]["family_size"], 7)
        self.assertTrue(summary["is_primary_n"])
        self.assertLess(summary["horizons"][-1]["failure_upper_bound"], 0.01)
        json.dumps(summary, allow_nan=False)

    def test_empirical_maximum_does_not_claim_population_horizon(self):
        summary = summarize_first_failures([None], 32, horizons=[32])
        empirical = summary["empirical_T_epsilon"][0]
        confidence = summary["confidence_supported_T_epsilon_lower_bound"][0]
        self.assertEqual(empirical["horizon"], 32)
        self.assertEqual(empirical["quantity"], "empirical_sample_T_epsilon")
        self.assertIn("population claim requires confidence", empirical["interpretation"])
        self.assertIsNone(confidence["horizon"])
        self.assertEqual(confidence["quantity"], "population_T_epsilon_lower_bound")

    def test_exact_one_sided_binomial_bounds(self):
        bound = clopper_pearson_upper(0, 512)
        self.assertAlmostEqual(bound, 1 - 0.05 ** (1 / 512), places=15)
        self.assertAlmostEqual(bound, 0.00583395560056, places=13)
        self.assertEqual(clopper_pearson_upper(512, 512), 1)
        bound = clopper_pearson_upper(1, 3)
        self.assertAlmostEqual((1 - bound) ** 3 + 3 * bound * (1 - bound) ** 2, 0.05, places=13)
        bound = clopper_pearson_upper(1, 3, alpha=0.9)
        self.assertAlmostEqual((1 - bound) ** 3 + 3 * bound * (1 - bound) ** 2, 0.9, places=13)
        self.assertGreater(clopper_pearson_upper(1, 512), clopper_pearson_upper(0, 512))
        self.assertGreater(clopper_pearson_upper(0, 512, alpha=0.01), clopper_pearson_upper(0, 512))

    def test_family_correction_controls_selected_maximum(self):
        corrected = summarize_first_failures([None] * 512, 2048, family_size=7 * 12 * 4)
        supported = {row["epsilon"]: row for row in corrected["confidence_supported_T_epsilon_lower_bound"]}
        self.assertEqual(supported[0.05]["horizon"], 2048)
        self.assertIsNone(supported[0.01]["horizon"])
        with self.assertRaises(ValueError):
            summarize_first_failures([None] * 512, 2048, family_size=1)

    def test_duplicate_sequence_ids_cannot_inflate_sample_size(self):
        with self.assertRaisesRegex(ValueError, "duplicate sequence IDs"):
            summarize_first_failures([None, None], 4, horizons=[4], sequence_ids=["seq0", "seq0"])
        strata = summarize_by_model_seed({
            11: {"taus": [None, 2], "sequence_ids": ["seq0", "seq1"]},
            12: {"taus": [None, None], "sequence_ids": ["seq0", "seq1"]},
        }, max_horizon=4, horizons=[4], family_size=2)
        self.assertIsNone(strata["pooled_sequence_n"])
        self.assertEqual(strata["model_seed_summaries"]["11"]["n_sequences"], 2)
        self.assertEqual(strata["model_seed_summaries"]["12"]["n_sequences"], 2)
        self.assertEqual(strata["model_seed_summaries"]["11"]["model_seed"], 11)

    def test_horizon_and_confidence_validation(self):
        for horizons in [[4, 2], [2, 2, 4], [0, 4], [5], [2]]:
            with self.assertRaises(ValueError):
                summarize_first_failures([None], 4, horizons=horizons)
        for alpha in [0, 1, math.nan, math.inf]:
            with self.assertRaises(ValueError):
                clopper_pearson_upper(0, 1, alpha)


class PairedBootstrapTests(unittest.TestCase):
    def test_pairing_uses_ids_and_is_reproducible(self):
        arguments = dict(sequence_ids_a=["a", "b", "c"], sequence_ids_b=["c", "a", "b"],
                         max_horizon=4, horizons=[2, 4], seed=101, n_bootstrap=100)
        result = paired_bootstrap([1, None, 3], [None, 2, None], **arguments)
        self.assertEqual(result, paired_bootstrap([1, None, 3], [None, 2, None], **arguments))
        self.assertAlmostEqual(result["delta_survival"][-1]["estimate"], 1 / 3)
        self.assertAlmostEqual(result["delta_restricted_mean_failure_free_length"]["estimate"], 1)
        self.assertEqual(result["n_sequences"], 3)
        json.dumps(result, allow_nan=False)

    def test_identical_arms_have_zero_paired_uncertainty(self):
        result = paired_bootstrap([1, None, 3], [1, None, 3], sequence_ids_a=["a", "b", "c"],
                                  sequence_ids_b=["a", "b", "c"], max_horizon=4, seed=1, n_bootstrap=30)
        self.assertEqual(result["delta_restricted_mean_failure_free_length"], {"estimate": 0.0, "lower": 0.0, "upper": 0.0})

    def test_unpaired_or_duplicated_ids_are_rejected(self):
        for ids_b in [["a", "c"], ["a", "a"], None]:
            with self.assertRaises(ValueError):
                paired_bootstrap([1, None], [None, None], sequence_ids_a=["a", "b"],
                                 sequence_ids_b=ids_b, max_horizon=4, seed=1, n_bootstrap=10)


if __name__ == "__main__":
    unittest.main()
