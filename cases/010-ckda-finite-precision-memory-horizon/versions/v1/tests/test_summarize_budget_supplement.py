"""Fictional small-sample reporting tests; no retained research outcomes used."""
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_budget_supplement.py"
SPEC = importlib.util.spec_from_file_location("case010_supplement_summary", SCRIPT)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)

GRID = [1, 2, 3, 4, 5, 6, 8]


def policy():
    return dict(model_seeds=[0, 1, 2], test=dict(group="S3", split="TEST", seed=12345,
        sequences=4, length=8, token_sha256=report.sha(b"fictional tokens"), gold_sha256=report.sha(b"fictional gold")),
        evaluation=dict(horizons=GRID, alpha=.05, epsilons=[.05, .01], family_size=630))


def summary(seed=0, taus=(None, 2, 8, None), stream=10, shared=3, family=546):
    result = report.summarize_first_failures(taus, 8, horizons=GRID,
        sequence_ids=[f"fictional-sequence-{i}" for i in range(len(taus))], family_size=family, model_seed=seed)
    result.update(execution=dict(status="COMPLETE", completed_writes=9),
        ledger=dict(per_stream_persistent_bytes=stream, shared_bytes=shared,
                    total_bytes={str(n): shared + n * stream for n in report.STREAM_COUNTS}))
    return result


def record(arm, stream=10, shared=3, seed=0, supported=None, empirical=None, rmst=4):
    result = report.reexpress(summary(seed=seed, stream=stream, shared=shared), arm, seed, 546, policy())
    result.update(source_kind="original", source_json=f"fictional/{arm}.json", source_json_sha256=report.sha(arm.encode()),
                  supported_T05=supported, empirical_T05=empirical, rmst=rmst)
    return result


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return report.sha(raw)


def make_runs(root):
    """N=4, length=8 fictional artifacts with the same serialization structure."""
    rules = policy()
    rules.update(original_protocol_sha256=report.sha((json.dumps(dict(arms=list(report.ORIGINAL_ARMS)), sort_keys=True) + "\n").encode()),
                 runtime_sha256=report.sha(b"{}\n"), seed_identities=[])
    inputs = dict(group="S3", split="TEST", seed=12345, sequences=4, group_length=8,
                  sequence_ids=[f"fictional-sequence-{i}" for i in range(4)],
                  token_sha256=rules["test"]["token_sha256"], gold_sha256=rules["test"]["gold_sha256"])
    input_digest = report.sha((json.dumps(inputs, sort_keys=True) + "\n").encode())
    for seed in rules["model_seeds"]:
        rules["seed_identities"].append(dict(model_seed=seed, checkpoint_sha256=report.sha(f"fictional checkpoint {seed}".encode()),
            calibration_sha256=report.sha(b"fictional calibration"), token_table_config_sha256=report.sha(b"fictional config"),
            token_table_data_sha256=report.sha(b"fictional table"), source_sha256={"fictional.py": report.sha(b"fictional source")},
            test_inputs_file_sha256=input_digest))
    policy_digest = report.sha((json.dumps(rules, sort_keys=True) + "\n").encode())
    originals, supplements = [], []
    for seed, identity in enumerate(rules["seed_identities"]):
        original = root / f"original-{seed}"
        extra = root / f"supplement-{seed}"
        originals.append(original)
        supplements.append(extra)
        original_manifest = dict(identity, phase="FROZEN_PRIMARY", protocol_sha256=rules["original_protocol_sha256"],
                                 runtime_sha256=rules["runtime_sha256"], model_checkpoint_bytes=123)
        original_manifest.pop("test_inputs_file_sha256")
        for folder, supplement in ((original, False), (extra, True)):
            write(folder / "protocol.json", dict(arms=list(report.ORIGINAL_ARMS)))
            (folder / "evaluation_runtime.json").write_bytes(b"{}\n")
            write(folder / "TEST_inputs.json", inputs)
            if supplement:
                manifest = dict(model_seed=seed, checkpoint_sha256=identity["checkpoint_sha256"], phase=report.PHASE,
                    status="COMPLETE", policy_sha256=policy_digest, same_test=True, independent_confirmation=False,
                    complete_call_timing="NOT_MEASURED", primary_index_sha256=report.sha((original / "index.json").read_bytes()),
                    primary_manifest_sha256=report.sha((original / "manifest.json").read_bytes()))
                write(folder / "budget_supplement_v1.json", rules)
                write(folder / "timing.json", dict(status="NOT_MEASURED"))
            else:
                manifest = original_manifest
            write(folder / "manifest.json", manifest)
            arm_index = {}
            for i, arm in enumerate(report.SUPPLEMENT_ARMS if supplement else report.ORIGINAL_ARMS):
                taus = (None, None, None, None) if i % 3 == 0 else (None, 2 + seed, 8, None)
                row = summary(seed=seed, taus=taus, stream=10 + i, family=630 if supplement else 546)
                if supplement:
                    row.update(phase=report.PHASE, supplement_policy_sha256=policy_digest, independent_confirmation=False)
                relative = f"TEST/{arm}.json"
                digest = write(folder / relative, row)
                arm_index[arm] = dict(file=relative, sha256=digest, execution="COMPLETE",
                    final_survival=row["horizons"][-1]["survival_probability"], rmst=row["restricted_mean_failure_free_length"])
            index = dict(manifest=manifest, family_size=630 if supplement else 546, splits=dict(TEST=arm_index))
            if supplement:
                index["status"] = "COMPLETE"
            digest = write(folder / "index.json", index)
            (folder / "index.sha256").write_text(digest + "\n")
    return originals, supplements, rules, policy_digest


class ReexpressionTests(unittest.TestCase):
    def test_joint_bound_changes_but_raw_events_censoring_and_accuracy_do_not(self):
        retained = summary()
        retained["step_accuracy"] = {"overall": .75, "interpretation": "fictional recovery accuracy"}
        before = copy.deepcopy(retained)
        derived = report.reexpress(retained, "UNIFORM_4", 0, 546, policy())["summary"]
        self.assertEqual(retained, before)
        self.assertEqual(derived["tau"], [None, 2, 8, None])
        self.assertEqual(derived["right_censored_sequences"], 2)
        self.assertEqual(derived["observed_failures"], 2)
        self.assertEqual(derived["restricted_mean_failure_free_length"], 6)
        self.assertEqual(derived["step_accuracy"], retained["step_accuracy"])
        self.assertEqual(derived["horizons"][-1]["first_failures"], 2)
        self.assertGreater(derived["horizons"][0]["failure_upper_bound"], retained["horizons"][0]["failure_upper_bound"])
        self.assertEqual(derived["confidence"]["family_size"], 630)

    def test_zero_event_bound_uses_joint_family_and_none_is_not_zero(self):
        result = report.reexpress(summary(taus=(None,) * 4), "NATIVE_FP32", 0, 546, policy())
        self.assertAlmostEqual(result["summary"]["horizons"][-1]["failure_upper_bound"],
                               1 - (.05 / 630) ** (1 / 4), places=14)
        self.assertIsNone(result["supported_T05"])
        self.assertEqual(result["empirical_T05"], 8)
        self.assertIn("not a measured zero", report.horizon_status(None, GRID))
        self.assertIn("population claim requires", report.horizon_status(8, GRID))

    def test_duplicate_ids_partial_execution_and_inconsistent_tau_rejected(self):
        changes = [("sequence_ids", ["duplicate"] * 4),
                   ("execution", dict(status="COMPLETE", completed_writes=8)),
                   ("tau", [1, 2, 8, None]), ("n_sequences", 8)]
        for key, value in changes:
            bad = summary()
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                report.reexpress(bad, "UNIFORM_4", 0, 546, policy())

    def test_absorbed_rows_are_retained_and_wrong_byte_sum_is_rejected(self):
        raw = summary()
        raw["execution"]["status"] = "NONFINITE_ROWS_ABSORBED"
        self.assertEqual(report.reexpress(raw, "UNIFORM_2", 0, 546, policy())["summary"]["n_sequences"], 4)
        raw["ledger"]["total_bytes"]["128"] += 1
        with self.assertRaisesRegex(ValueError, "ledger"):
            report.reexpress(raw, "UNIFORM_2", 0, 546, policy())


class FrontierTests(unittest.TestCase):
    def test_actual_shared_bytes_change_feasibility_and_include_added_mixed(self):
        records = [record("UNIFORM_4", 100, 1000, supported=1, empirical=1, rmst=1),
                   record("NATIVE_FP32", 210, 0, supported=4, empirical=4, rmst=4),
                   record("MIXED_4_8_TOP2", 130, 60, supported=3, empirical=3, rmst=3),
                   record("STOCHASTIC_4", 140, 10, supported=2, empirical=2, rmst=2),
                   record("FULL_RESIDUAL_4_4", 150, 100, supported=4, empirical=4, rmst=4)]
        rows = [row for row in report.budget_rows(records) if row["candidate"] == "FULL_RESIDUAL_4_4"]
        one, many = rows[0], rows[-1]
        self.assertEqual(one["candidate_total_byte_cap"], 250)
        self.assertEqual(one["best_TEST_supported_T05_baseline"], "NATIVE_FP32")
        self.assertTrue(one["native_fp32_feasible"])
        self.assertNotIn("UNIFORM_4", one["feasible_baselines"])
        self.assertEqual(many["best_TEST_supported_T05_baseline"], "MIXED_4_8_TOP2")
        self.assertFalse(many["native_fp32_feasible"])
        self.assertIn("STOCHASTIC_4", many["feasible_baselines"])
        self.assertIn("not DEV-selected", one["baseline_selection_scope"])
        self.assertEqual(len(report.budget_rows(records)), 3 * len(records))

    def test_seed_strata_and_no_qualifying_baseline(self):
        records = [record("UNIFORM_4", seed=0), record("UNIFORM_4", seed=1, supported=8, empirical=8),
                   record("LOWRANK_4_8_R1", stream=20, seed=0)]
        rows = [r for r in report.budget_rows(records) if r["candidate"] == "LOWRANK_4_8_R1"]
        for row in rows:
            self.assertIsNone(row["best_TEST_supported_T05"])
            self.assertIsNone(row["best_TEST_supported_T05_baseline"])
            self.assertIn("no qualifying", row["best_TEST_supported_T05_status"])
        self.assertEqual({row["model_seed"] for row in rows}, {0})

    def test_explicit_ties_and_residual_exclusion(self):
        for name in ("FULL_RESIDUAL_4_4", "LOWRANK_4_8_R2", "UNTRANSPORTED_4_4"):
            self.assertFalse(report.is_baseline(name))
        records = [record("UNIFORM_8", stream=10, supported=4, rmst=3),
                   record("MIXED_4_8_TOP2", stream=10, supported=4, rmst=3),
                   record("UNIFORM_16", stream=11, supported=4, rmst=3)]
        self.assertEqual(report.best_feasible(records, "supported_T05", 128)["arm"], "MIXED_4_8_TOP2")


class ArtifactTests(unittest.TestCase):
    def test_fictional_complete_family_outputs_and_original_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            originals, extras, rules, digest = make_runs(root)
            before = {p: report.sha(p.read_bytes()) for p in root.rglob("*") if p.is_file()}
            sources = report.Sources()
            records, metadata = report.combine(originals, extras, rules, digest, sources)
            output = root / "derived"
            report.write_outputs(output, records, metadata, digest, sources)
            self.assertEqual(before, {p: report.sha(p.read_bytes()) for p in before})
            with (output / "all_arms.csv").open() as stream:
                arms = list(csv.DictReader(stream))
            with (output / "horizon_curves.csv").open() as stream:
                curves = list(csv.DictReader(stream))
            with (output / "budget_comparisons.csv").open() as stream:
                budgets = list(csv.DictReader(stream))
            self.assertEqual((len(arms), len(curves), len(budgets)), (90, 630, 270))
            self.assertEqual({row["n_sequences"] for row in arms}, {"4"})
            self.assertEqual({row["joint_family_size"] for row in arms}, {"630"})
            document = json.loads((output / "summary.json").read_text())
            self.assertFalse(document["independent_confirmation"])
            self.assertEqual(document["phase"], report.PHASE)
            self.assertEqual(document["additional_complete_call_timing"], "NOT_MEASURED")
            receipt = json.loads((output / "source_hashes.json").read_text())
            self.assertIn("exact file bytes", receipt["hash_kind"])
            self.assertTrue(all(not row["file"].startswith("/") for row in receipt["sources"]))
            for row in receipt["outputs"]:
                self.assertEqual(row["sha256"], report.sha((output / row["file"]).read_bytes()))
            with self.assertRaises(FileExistsError):
                report.write_outputs(output, records, metadata, digest, sources)

    def test_changed_result_hash_missing_arm_and_duplicate_seed_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            originals, extras, rules, digest = make_runs(Path(temporary))
            path = originals[0] / "TEST/UNIFORM_4.json"
            raw = path.read_bytes()
            path.write_bytes(raw + b" ")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                report.combine(originals, extras, rules, digest, report.Sources())
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "duplicate model-seed"):
                report.combine([originals[0], originals[0], originals[2]], extras, rules, digest, report.Sources())
            path.unlink()
            with self.assertRaisesRegex(ValueError, "incomplete arm family"):
                report.combine(originals, extras, rules, digest, report.Sources())

    def test_supplement_must_reference_exact_primary_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            originals, extras, rules, digest = make_runs(Path(temporary))
            folder = extras[0]
            manifest = json.loads((folder / "manifest.json").read_text())
            manifest["primary_index_sha256"] = report.sha(b"different fictional attempt")
            write(folder / "manifest.json", manifest)
            index = json.loads((folder / "index.json").read_text())
            index["manifest"] = manifest
            index_digest = write(folder / "index.json", index)
            (folder / "index.sha256").write_text(index_digest + "\n")
            with self.assertRaisesRegex(ValueError, "different primary result"):
                report.combine(originals, extras, rules, digest, report.Sources())


if __name__ == "__main__":
    unittest.main()
