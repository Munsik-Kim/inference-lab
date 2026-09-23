"""Read-only reporting contracts: real byte caps, missing horizons, timing provenance."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_results.py"
SPEC = importlib.util.spec_from_file_location("case010_summary", SCRIPT)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def record(arm, stream, shared, horizon, rmst, seed=0):
    return {"domain": "learned", "condition": "learned_s3", "model_seed": seed, "arm": arm,
            "evaluation_version": "unit-test-fixture", "per_stream_bytes": stream, "shared_bytes": shared,
            "total_bytes": {str(n): shared + n * stream for n in (1, 16, 128)},
            "amortized_bytes": {str(n): stream + shared / n for n in (1, 16, 128)},
            "empirical_horizon": horizon, "supported_horizon": horizon,
            "rmst": rmst, "final_failure_probability": .5, "timing": None}


def benchmark(seconds):
    return {"config": {"sequences": 16, "length": 128, "repeats": 3, "seed": 2002},
            "cpu_threads": 2, "scope": "unit-test complete-call scope",
            "arms": {"NATIVE_FP32": {"median_seconds": seconds, "all_seconds": [seconds] * 3,
                                       "milliseconds_per_group_token": seconds * 1000 / 2048, "all_complete": True}}}


class BudgetComparisonTests(unittest.TestCase):
    def test_actual_shared_amortization_changes_feasible_frontier(self):
        arms = [record("UNIFORM_4", 100, 1000, 32, 90), record("UNIFORM_8", 200, 0, 128, 200),
                record("MIXED_4_8", 130, 60, 64, 120), record("STOCHASTIC_4", 140, 10, 64, 125),
                record("NATIVE_FP32", 210, 0, 64, 130), record("FULL_RESIDUAL_4_4", 150, 100, 64, 150)]
        rows = [row for row in report.budget_rows(arms) if row["candidate"] == "FULL_RESIDUAL_4_4"]
        one = next(row for row in rows if row["N_streams"] == 1)
        many = next(row for row in rows if row["N_streams"] == 128)
        self.assertEqual(one["candidate_total_byte_cap"], 250)
        self.assertEqual(one["best_TEST_supported_baseline"], "UNIFORM_8")
        self.assertTrue(one["native_fp32_feasible"])
        self.assertNotIn("UNIFORM_4", one["feasible_baselines"])
        self.assertEqual(many["candidate_total_byte_cap"], 19300)
        self.assertEqual(many["best_TEST_supported_baseline"], "STOCHASTIC_4")
        self.assertFalse(many["native_fp32_feasible"])
        self.assertNotIn("UNIFORM_8", many["feasible_baselines"])
        self.assertIn("MIXED_4_8", many["feasible_baselines"])
        self.assertIn("not DEV-selected", one["baseline_selection_scope"])
        self.assertIn("not a test", one["horizon_comparison_scope"])

    def test_no_qualifying_horizon_is_none_not_numeric_zero(self):
        arms = [record("UNIFORM_4", 100, 0, None, 5), record("FULL_RESIDUAL_4_4", 200, 0, None, 9)]
        rows = report.budget_rows(arms)
        for row in rows:
            self.assertIsNone(row["candidate_supported_T05"])
            self.assertEqual(row["candidate_supported_T05_status"], "no qualifying grid horizon (<32)")
            self.assertIsNone(row["best_TEST_supported_T05"])

    def test_seeds_do_not_share_frontier_winners(self):
        rows = report.budget_rows([record("UNIFORM_4", 10, 0, 2048, 2048, seed=1),
                                   record("UNIFORM_4", 10, 0, 32, 50, seed=0),
                                   record("FULL_RESIDUAL_4_4", 20, 0, 64, 60, seed=0)])
        target = next(row for row in rows if row["candidate"] == "FULL_RESIDUAL_4_4" and row["N_streams"] == 128)
        self.assertEqual(target["best_TEST_supported_T05"], 32)

    def test_secondary_and_residual_arms_cannot_enter_reference_frontier(self):
        for name in ("FULL_RESIDUAL_4_4", "LOWRANK_4_8_R2", "UNTRANSPORTED_4_4", "COEFFICIENT_INT8_FP32"):
            self.assertFalse(report.baseline(name))
        for name in ("NATIVE_FP32", "UNIFORM_16", "MIXED_6_8", "STOCHASTIC_4"):
            self.assertTrue(report.baseline(name))

    def test_absorbed_numerical_failures_remain_complete_primary_records(self):
        for status in ("COMPLETE", "NONFINITE_ROWS_ABSORBED", "NONFINITE_LOGITS_WITH_FINITE_STATE"):
            summary = {"max_horizon": 2048, "execution": {"status": status, "completed_writes": 2049}}
            self.assertEqual(report.validate_learned_execution(summary, "UNIFORM_2")["status"], status)
            summary["execution"]["completed_writes"] = 2048
            with self.assertRaisesRegex(ValueError, "incomplete"):
                report.validate_learned_execution(summary, "UNIFORM_2")


class TimingProvenanceTests(unittest.TestCase):
    def test_embedded_cost_is_retained_but_not_silently_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            embedded = path / "timing.json"
            embedded.write_text(json.dumps(benchmark(3)))
            before = embedded.read_bytes()
            sources = report.timing_sources(path, {"model_seed": 0, "checkpoint_sha256": "abc"})
            timing = report.arm_timings(sources, "NATIVE_FP32")
            fixture = record("NATIVE_FP32", 10, 0, 32, 50)
            fixture["timing"] = timing
            self.assertEqual(timing["primary_kind"], "NOT_MEASURED")
            self.assertIsNone(report.timing_value(fixture))
            self.assertEqual(report.timing_value(fixture, "embedded"), 3000 / 2048)
            self.assertEqual(embedded.read_bytes(), before)

    def test_isolated_override_keeps_both_sources_and_verifies_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "timing.json").write_text(json.dumps(benchmark(3)))
            isolated = path / "separate.json"
            raw = {"model_seed": 0, "checkpoint_sha256": "abc", "isolated": True, "benchmark": benchmark(1)}
            isolated.write_text(json.dumps(raw))
            sources = report.timing_sources(path, {"model_seed": 0, "checkpoint_sha256": "abc"}, isolated)
            timing = report.arm_timings(sources, "NATIVE_FP32")
            self.assertEqual(set(sources), {"embedded", "isolated"})
            self.assertEqual(timing["primary_kind"], "isolated")
            self.assertEqual(timing["isolated"]["median_seconds"], 1)
            self.assertEqual(timing["embedded"]["median_seconds"], 3)
            self.assertEqual(sources["isolated"]["metadata"]["file_sha256"], report.sha(isolated))
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                report.timing_sources(path, {"model_seed": 0, "checkpoint_sha256": "different"}, isolated)
            with self.assertRaisesRegex(ValueError, "model seed"):
                report.timing_sources(path, {"model_seed": 1, "checkpoint_sha256": "abc"}, isolated)

    def test_named_isolated_detection_and_protocol_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            isolated = path / "timing-isolated.json"
            isolated.write_text(json.dumps(benchmark(1)))
            sources = report.timing_sources(path, {"model_seed": 0, "checkpoint_sha256": "abc"})
            self.assertEqual(sources["isolated"]["metadata"]["selection"], "named timing-isolated.json")
            wrong = copy.deepcopy(benchmark(1))
            wrong["config"]["seed"] = 3001
            isolated.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValueError, "fixed DEV2002"):
                report.timing_sources(path, {"model_seed": 0, "checkpoint_sha256": "abc"})

    def test_embedded_file_cannot_be_relabeled_and_duplicate_flags_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            embedded = path / "timing.json"
            embedded.write_text(json.dumps(benchmark(1)))
            with self.assertRaisesRegex(ValueError, "relabeled"):
                report.timing_sources(path, {"model_seed": 0, "checkpoint_sha256": "abc"}, embedded)
        with self.assertRaises(ValueError):
            report.parse_isolated_arguments(["0=a", "0=b"])
        with self.assertRaises(ValueError):
            report.parse_isolated_arguments(["not-a-seed-file"])

    def test_dev_smoke_is_excluded_from_primary_reporting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text(json.dumps({"phase": "DEV_SMOKE"}))
            with self.assertRaisesRegex(ValueError, "DEV/SMOKE"):
                report.read_learned(path)


class NativeReferenceTests(unittest.TestCase):
    def fixture(self):
        n, t = 512, 33
        return {"domain": "learned", "arm": "NATIVE_FP32", "model_seed": 2,
                "source_directory_name": "unit-fixture", "source_json_name": "TEST/NATIVE_FP32.json", "source_json_sha256": "abc",
                "summary": {"n_sequences": n, "bos": {"accuracy": 1.0}, "confidence": {"family_size": 546},
                            "horizons": [{"horizon": t, "first_failures": 100, "survival_probability": 1 - 100/n, "failure_upper_bound": .3}],
                            "token_counts_at_horizons": [{"horizon": t, "prefix_correct": n*t - 100, "prefix_total": n*t,
                                                          "final_quarter_correct": n*9 - 50, "final_quarter_total": n*9}]}}

    def test_counts_ceil_quarter_bos_and_provenance(self):
        row = report.native_reference_rows([self.fixture()])[0]
        self.assertEqual(row["all_token_total"], 512 * 33)
        self.assertEqual(row["final_quarter_total"], 512 * 9)
        self.assertEqual(row["all_token_accuracy"], (512 * 33 - 100) / (512 * 33))
        self.assertEqual(row["bos_accuracy"], 1)
        self.assertTrue(row["bos_excluded_from_scored_steps"])
        self.assertEqual(row["source_json_sha256"], "abc")
        self.assertEqual(row["confidence_family_size"], 546)

    def test_wrong_denominators_or_out_of_range_counts_raise(self):
        for key, value in (("prefix_total", 1), ("final_quarter_total", 512 * 8), ("prefix_correct", -1), ("final_quarter_correct", 10**9)):
            fixture = self.fixture()
            fixture["summary"]["token_counts_at_horizons"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                report.native_reference_rows([fixture])


if __name__ == "__main__":
    unittest.main()
