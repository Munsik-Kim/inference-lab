"""Aggregation fixtures are synthetic tests, never measured CKDA results."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

CASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("case011_aggregate_tests", CASE / "analysis" / "aggregate.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)


def fixture(root, updates=None):
    directory = Path(root) / "results" / "fresh" / "seed0" / "UNIFORM_8"
    directory.mkdir(parents=True)
    gold = np.zeros((3, 32), dtype=np.uint8)
    pred = np.zeros((3, 3, 32), dtype=np.int8)
    pred[0, 0, 5] = -1
    count = np.count_nonzero(pred != -1, axis=1)
    values = {"predictions": pred, "gold": gold,
              "head_names": np.asarray(aggregate.HEADS), "sample_ids": np.asarray(["a", "b", "c"]),
              "input_hash": np.asarray("a"*64), "checkpoint_sha256": np.asarray("b"*64),
              "checkpoint_seed": np.asarray(0), "storage": np.asarray("UNIFORM_8"),
              "shuffled_predictions": pred, "input_only_predictions": gold.astype(np.int8),
              "ce_sum": count.astype(float)*2, "gold_margin_sum": count.astype(float)*-1,
              "top_margin_sum": count.astype(float), "score_valid_count": count}
    if updates:
        values.update(updates)
    np.savez(directory/"predictions.npz", **values)
    summary = aggregate.metrics.summarize_predictions(pred, gold)
    (directory/"summary.json").write_text(json.dumps(summary, allow_nan=False))
    (directory/"receipt.json").write_text(json.dumps({"scope": "synthetic_test"}))
    return values


class AggregateContracts(unittest.TestCase):
    def test_reads_source_without_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            path = Path(tmp)/"results/fresh/seed0/UNIFORM_8/predictions.npz"
            before = aggregate.sha(path)
            cell = aggregate.load_cell(Path(tmp), 0, "UNIFORM_8", expected_shape=(3,32))
            self.assertEqual(cell["summary"]["ORIGINAL"]["tau"], [6,None,None])
            self.assertEqual(before, aggregate.sha(path))

    def test_incomplete_full_test_cell_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            with self.assertRaisesRegex(ValueError, "incomplete TEST"):
                aggregate.load_cell(Path(tmp), 0, "UNIFORM_8")

    def test_score_denominator_matches_invalids(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp, {"score_valid_count": np.full((3,32), 3, dtype=np.int64)})
            with self.assertRaisesRegex(ValueError, "finite predictions"):
                aggregate.load_cell(Path(tmp), 0, "UNIFORM_8", expected_shape=(3,32))

    def test_nonfinite_score_sum_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            sums = np.ones((3,32), dtype=float)
            sums[0,0] = np.nan
            fixture(tmp, {"ce_sum": sums})
            with self.assertRaisesRegex(ValueError, "score sums"):
                aggregate.load_cell(Path(tmp), 0, "UNIFORM_8", expected_shape=(3,32))

    def test_score_range_uses_summed_valid_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture(tmp)
            cell = aggregate.load_cell(Path(tmp), 0, "UNIFORM_8", expected_shape=(3,32))
            rows = aggregate.score_rows(cell, 0, "UNIFORM_8")
            row = rows[0]
            self.assertEqual(row["valid_score_observations"], 95)
            self.assertEqual(row["invalid_score_observations"], 1)
            self.assertEqual(row["total_token_observations"], 96)
            self.assertEqual(row["gold_ce_nats"], 2)
            self.assertEqual(row["gold_ce_sum_nats"], 190)
            self.assertEqual(row["gold_margin"], -1)
            self.assertEqual(row["token_accuracy"], 95/96)

    def test_secondary_directions_and_seed_indices_fixed(self):
        self.assertEqual([r[0] for r in aggregate.SECONDARY], list(range(5)))
        for pair in aggregate.SECONDARY[2:]:
            self.assertEqual(pair[2][0], "UNIFORM_8")
            self.assertEqual(pair[3][0], "NATIVE_FP32")
            self.assertEqual(pair[2][1], pair[3][1])

    def test_existing_aggregate_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"already"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                aggregate.collect(Path(tmp), Path(tmp)/"absent.json", output)

    def test_no_valid_score_is_null_not_zero_mean(self):
        cell = {"gold": np.zeros((2,32), dtype=np.uint8),
                "predictions": np.full((3,2,32), -1, dtype=np.int8),
                "score_valid_count": np.zeros((3,32), dtype=np.int64),
                "ce_sum": np.zeros((3,32)), "gold_margin_sum": np.zeros((3,32)),
                "top_margin_sum": np.zeros((3,32))}
        row = aggregate.score_rows(cell, 0, "UNIFORM_8")[0]
        self.assertIsNone(row["gold_ce_nats"])
        self.assertEqual(row["invalid_score_observations"], 64)
        self.assertEqual(row["token_accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
