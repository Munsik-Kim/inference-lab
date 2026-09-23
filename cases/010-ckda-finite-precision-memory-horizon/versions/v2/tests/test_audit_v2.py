"""Small independent audit fixtures; no model/runtime/metrics imports."""
import importlib.util
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("independent_audit_v2", Path(__file__).resolve().parents[1] / "analysis/audit_v2.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class IndependentAuditTests(unittest.TestCase):
    @staticmethod
    def fixture():
        pred = np.array([[0, 0, 0, 0, 0], [0, 0, -1, -1, -1]], np.int8)
        gold = np.zeros((2, 4), np.uint8)
        packed = np.array([[15], [1]], np.uint8)
        ledger = {"per_stream_persistent_bytes": 21, "state_code_bytes": 4, "residual_code_bytes": 0,
                  "scale_bytes": 0, "rng_bytes": 0, "cursor_bytes": 8, "terminal_code_bytes": 1,
                  "first_terminal_write_bytes": 8, "header_bytes": 17, "gauge_bytes": 0, "padding_bytes": 0,
                  "shared_bytes": 100, "shared_config_bytes": 50, "shared_basis_bytes": 0,
                  "shared_codec_bytes": 50, "shared_token_table_bytes": 40, "shared_runtime_metadata_bytes": 10,
                  "payload_tensor_storage_bytes": 42, "serialized_payload_bytes": 42,
                  "total_bytes": {str(n): 100 + n * 21 for n in (1, 16, 128)}}
        summary = {"schema": "case010-failure-aware-v2-cell-v1", "N": 2, "T": 4, "tau": [None, 2], "RMST0": 2.5,
                   "model_seed": 0, "arm": "NATIVE_FP32", "cohort": "diagnostic", "family_size": None,
                   "empirical_T05_all_tokens": 1, "empirical_T01_all_tokens": 1,
                   "supported_T05_grid": None, "supported_T01_grid": None, "token_accuracy": .625,
                   "final_quarter_accuracy": .5, "bos_accuracy": 1., "bos_unscored_in_tau": True,
                   "first_failure_censored": 1, "invalid_prediction_tokens": 3,
                   "terminal_codes": [0, 2], "first_terminal_write": [None, 2], "terminal_count": 1,
                   "grid_rows": [{"horizon": 4, "failures": 1, "F": .5, "survival": .5, "p_upper": None}],
                   "execution": {"status": "COMPLETE_WITH_TERMINAL_STREAMS", "completed_writes": 5,
                                 "active_update_attempts": 8, "terminal_noop_steps": 2}, "ledger": ledger}
        return summary, pred, gold, packed

    def test_bos_is_unscored_but_first_real_invalid_is_failure(self):
        pred = np.array([[-1, 0, 0, 0, 0], [0, -1, 0, 0, 0], [0, 0, 0, 2, 0]], dtype=np.int8)
        gold = np.zeros((3, 4), dtype=np.uint8)
        r = AUDIT.reconstruct(pred, gold)
        self.assertEqual(r["tau"], [None, 1, 3])
        self.assertEqual(r["RMST0"], 2)
        self.assertEqual(r["failure_counts"], [1, 1, 2, 2])
        self.assertAlmostEqual(r["token_accuracy"], 10 / 12)
        self.assertEqual(r["final_quarter_accuracy"], 1)

    def test_first_failure_remains_absorbing_after_label_recovery(self):
        pred = np.array([[0, 0, 1, 0, 0]], dtype=np.int8)
        r = AUDIT.reconstruct(pred, np.zeros((1, 4), np.uint8))
        self.assertEqual(r["tau"], [2])
        self.assertEqual(r["failure_counts"], [0, 1, 1, 1])
        self.assertEqual(r["empirical"][.05], 1)

    def test_terminal_at_bos_and_late_write(self):
        predictions = np.array([[-1, -1, -1], [2, 1, -1], [-1, 1, 2]], dtype=np.int8)
        counts = AUDIT.check_terminal(predictions, [2, 3, 0], [0, 2, None])
        self.assertEqual(counts["0"], 1)
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.check_terminal(predictions, [2, 3, 0], [0, 1, None])
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.check_terminal(predictions, [2, 3, 0], [0, 2, 0])

    def test_packed_correctness_and_padding_cannot_hide_errors(self):
        correct = np.array([[True, False, True]], dtype=bool)
        AUDIT.check_packed(np.array([[5]], dtype=np.uint8), correct)
        for byte in (7, 133):
            with self.assertRaises(AUDIT.AuditError):
                AUDIT.check_packed(np.array([[byte]], dtype=np.uint8), correct)

    def test_binomial_sum_matches_exact_small_case(self):
        p = .37
        expected = sum(math.comb(12, k) * p**k * (1 - p)**(12 - k) for k in range(5))
        self.assertAlmostEqual(math.exp(AUDIT.log_binomial_cdf(4, 12, p)), expected, places=13)
        AUDIT.check_cp_upper(.5, 0, 4, .0625)
        AUDIT.check_cp_upper(1., 4, 4, .0625)
        # At p=0.5 the exact n=3, k=1 CDF is 1/2.
        AUDIT.check_cp_upper(.5, 1, 3, .5)
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.check_cp_upper(.6, 1, 3, .5)

    def test_binomial_large_n_stays_finite(self):
        # Symmetry at p=.5 for odd n yields exactly one half.
        self.assertAlmostEqual(AUDIT.log_binomial_cdf(511, 1023, .5), math.log(.5), places=10)

    def test_invalid_labels_and_dtypes_rejected(self):
        for predictions in (np.zeros((2, 4), np.float32), np.full((2, 4), -2, np.int8)):
            with self.assertRaises(AUDIT.AuditError):
                AUDIT.reconstruct(predictions, np.zeros((2, 3), np.uint8))

    def test_missing_cells_are_not_an_accepted_intersection(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AUDIT.AuditError):
                AUDIT.audit_results(directory)

    def test_full_diagnostic_scalar_fixture_and_silent_denominator_change(self):
        summary, pred, gold, packed = self.fixture()
        result = AUDIT.check_summary(summary, pred, gold, packed, expected_fresh=False)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["N"], 2)
        summary["N"] = 1
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.check_summary(summary, pred, gold, packed, expected_fresh=False)

    def test_ledger_detects_missing_terminal_header_or_shared_bytes(self):
        summary, *_ = self.fixture()
        for key in ("terminal_code_bytes", "shared_token_table_bytes", "serialized_payload_bytes"):
            ledger = copy.deepcopy(summary["ledger"])
            ledger[key] -= 1
            with self.assertRaises(AUDIT.AuditError):
                AUDIT.check_ledger(ledger, 2)

    def test_no_qualifying_token_is_none_and_no_diagnostic_family_claim(self):
        pred = np.array([[0, -1, -1]], dtype=np.int8)
        self.assertIsNone(AUDIT.reconstruct(pred, np.zeros((1, 2), np.uint8))["empirical"][.05])
        summary, pred, gold, packed = self.fixture()
        summary["family_size"] = 195
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.check_summary(summary, pred, gold, packed, expected_fresh=False)

    def test_explicit_partial_status_never_passes_empty_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inputs").mkdir()
            (root / "inputs/manifest.json").write_text("{}")
            result = AUDIT.audit_results(root, allow_partial=True)
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertEqual(len(result["missing_cells"]), 15)

    def test_actual_shared_bytes_and_initial_header(self):
        summary, *_ = self.fixture()
        config = {"format": "ckda-online-failure-aware-v2", "header_bytes": 17,
                  "bos_policy": "present_write0", "cursor_semantics": "next_zero_based_write",
                  "state": {"format": "ckda-native-v1", "dtype": "<f4", "shape": [1, 1, 1]},
                  "residual": None, "basis_shape": None, "basis_sha256": hashlib.sha256(b"").hexdigest()}
        blobs = {"initial_payload": bytes(9) + ((1 << 64) - 1).to_bytes(8, "little") + bytes(4),
                 "codec_config": json.dumps(config).encode(), "basis": b"", "token_config": b"{}", "token_data": bytes(4),
                 "runtime_config": json.dumps({"invalid_prediction": -1, "terminal_output": "always_invalid",
                                                "bos_policy": "present_write0"}).encode()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary["shared_artifacts"] = {}
            for name, raw in blobs.items():
                path = root / (name + ".bin")
                path.write_bytes(raw)
                summary["shared_artifacts"][name] = {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            ledger = summary["ledger"]
            ledger["shared_config_bytes"] = len(blobs["codec_config"])
            ledger["shared_token_table_bytes"] = len(blobs["token_config"]) + len(blobs["token_data"])
            ledger["shared_runtime_metadata_bytes"] = len(blobs["runtime_config"])
            self.assertEqual(AUDIT.check_shared_artifacts(summary, root)["status"], "PASS")
            altered = bytearray(blobs["initial_payload"])
            altered[0] = 1
            (root / "initial_payload.bin").write_bytes(altered)
            summary["shared_artifacts"]["initial_payload"]["sha256"] = hashlib.sha256(altered).hexdigest()
            with self.assertRaises(AUDIT.AuditError):
                AUDIT.check_shared_artifacts(summary, root)

    def test_shared_artifact_path_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in ("../anything", "/anything", "C:/anything", "some\\file"):
                with self.assertRaises(AUDIT.AuditError):
                    AUDIT.safe_artifact(directory, path)

    def test_frozen_gold_and_sample_identity_are_checked(self):
        summary, _, gold, _ = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inputs").mkdir()
            np.save(root / "inputs/gold.npy", gold)
            ids_path = root / "inputs/ids.json"
            ids_path.write_text('["sequence-a", "sequence-b"]')
            item = {"sequences": 2, "group_tokens": 4, "gold_file": "inputs/gold.npy",
                    "gold_file_sha256": AUDIT.sha(root / "inputs/gold.npy"),
                    "sample_ids_file": "inputs/ids.json", "sample_ids_sha256": AUDIT.sha(ids_path)}
            manifest = root / "inputs/manifest.json"
            manifest.write_text(json.dumps({"cohorts": {"diagnostic": item}}))
            summary.update(input_manifest_sha256=AUDIT.sha(manifest), sample_ids_sha256=AUDIT.sha(ids_path))
            AUDIT.check_input_identity(summary, gold, root)
            changed_gold = gold.copy()
            changed_gold[0, 0] = 1
            with self.assertRaises(AUDIT.AuditError):
                AUDIT.check_input_identity(summary, changed_gold, root)
            summary["sample_ids_sha256"] = "0" * 64
            with self.assertRaises(AUDIT.AuditError):
                AUDIT.check_input_identity(summary, gold, root)


if __name__ == "__main__":
    unittest.main()
