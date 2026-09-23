"""Independent small statistical checks, with no model/runtime import."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

PATH = Path(__file__).resolve().parents[1] / "analysis/v1_reanalysis.py"
SPEC = importlib.util.spec_from_file_location("historical_reanalysis", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HistoricalStatisticsTests(unittest.TestCase):
    def test_censoring_and_failure_boundary(self):
        np.testing.assert_array_equal(MODULE.failure_free_lengths([1, 3, None], 4), [0, 2, 4])
        self.assertEqual(MODULE.empirical_token_horizon([1, 3, None], 4, 1 / 3), 2)
        self.assertEqual(MODULE.empirical_token_horizon([1, 3, None], 4, 0), 0)
        self.assertEqual(MODULE.empirical_token_horizon([None] * 3, 4, .05), 4)

    def test_identity_pairing_reorders_baseline(self):
        c = dict(model_seed=0, max_horizon=9, sequence_ids=["a", "b"], tau=[5, 8])
        b = dict(model_seed=0, max_horizon=9, sequence_ids=["b", "a"], tau=[6, 2])
        np.testing.assert_array_equal(MODULE.paired_difference(c, b), [3, 2])
        b["sequence_ids"] = ["a", "a"]
        with self.assertRaises(ValueError):
            MODULE.paired_difference(c, b)

    def test_constant_paired_effect_has_degenerate_interval(self):
        result = MODULE.paired_bootstrap(np.full(7, 3), model_seed=0, repeats=211)
        self.assertEqual((result["rmst_delta_tokens"], result["pointwise_ci95_low"], result["pointwise_ci95_high"]), (3, 3, 3))

    def test_reproducibility_and_swapping_sign(self):
        x = np.asarray([-9, 0, 2, 4, 7])
        a = MODULE.paired_bootstrap(x, model_seed=1, repeats=1001)
        b = MODULE.paired_bootstrap(x, model_seed=1, repeats=1001)
        reverse = MODULE.paired_bootstrap(-x, model_seed=1, repeats=1001)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a["pointwise_ci95_low"], -reverse["pointwise_ci95_high"])
        self.assertAlmostEqual(a["pointwise_ci95_high"], -reverse["pointwise_ci95_low"])

    def test_invalid_tau_and_mismatched_pair_fail(self):
        for values in ([0], [5], [1.5], [True]):
            with self.assertRaises(ValueError):
                MODULE.failure_free_lengths(values, 4)
        c = dict(model_seed=0, max_horizon=4, sequence_ids=["a"], tau=[None])
        b = dict(model_seed=1, max_horizon=4, sequence_ids=["a"], tau=[None])
        with self.assertRaises(ValueError):
            MODULE.paired_difference(c, b)

    def test_public_receipt_rejects_private_paths(self):
        self.assertEqual(MODULE.safe_public_json({"file": "results/receipt.json"}), {"file": "results/receipt.json"})
        with self.assertRaises(ValueError):
            MODULE.safe_public_json({"file": "/home/person/checkpoint.pt"})


if __name__ == "__main__":
    unittest.main()
