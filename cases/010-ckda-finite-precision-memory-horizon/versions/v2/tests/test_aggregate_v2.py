"""Aggregation gates and fixed paired-seed behavior, using synthetic records."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("fresh_aggregate", Path(__file__).resolve().parents[1] / "analysis/aggregate_v2.py")
AGGREGATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATE)


class AggregateTests(unittest.TestCase):
    @staticmethod
    def fixture():
        protocol = {"checkpoint_sha256": {str(s): str(s) * 64 for s in AGGREGATE.SEEDS}, "fresh": {
            "N": 1024, "T": 2048, "simultaneous_family": 195, "model_seeds": list(AGGREGATE.SEEDS),
            "arms": list(AGGREGATE.ARMS), "pairs": [list(p) for p in AGGREGATE.PAIRS],
            "paired_bootstrap_repeats": 5000, "paired_bootstrap_seed_rule": "60101+100*model_seed+pair_index",
            "confidence_grid": [32], "alpha": .05, "epsilon_primary": .05}}
        cells = []
        for seed in AGGREGATE.SEEDS:
            for arm in AGGREGATE.ARMS:
                ledger = {"per_stream_persistent_bytes": 21, "header_bytes": 17, "shared_bytes": 100,
                          "shared_codec_bytes": 50, "shared_token_table_bytes": 40, "shared_runtime_metadata_bytes": 10,
                          "total_bytes": {str(n): 100 + 21 * n for n in (1, 16, 128)}}
                cell = {"schema": "case010-failure-aware-v2-cell-v1", "kind": "FRESH_TEST_FIXED_CHECKPOINTS",
                        "model_seed": seed, "arm": arm, "cohort": "fresh", "N": 1024, "T": 2048, "family_size": 195,
                        "checkpoint_sha256": str(seed) * 64, "protocol_sha256": "p", "input_manifest_sha256": "i",
                        "sample_ids_sha256": "s", "tokens_sha256": "t", "execution": {"status": "COMPLETE"},
                        "grid_rows": [{"horizon": 32, "F": 0}], "alpha": .05, "epsilon_primary": .05,
                        "RMST0": 10., "ledger": ledger}
                cell.update({k: 0 for k in ("empirical_T05_all_tokens", "empirical_T01_all_tokens", "supported_T05_grid",
                    "supported_T01_grid", "token_accuracy", "final_quarter_accuracy", "bos_accuracy",
                    "first_failure_censored", "terminal_count", "invalid_prediction_tokens")})
                cells.append(cell)
        return protocol, cells

    def validate(self, protocol, cells):
        AGGREGATE.validate_cell_identities(cells, protocol, protocol_sha256="p", input_manifest_sha256="i",
                                           sample_ids_sha256="s", tokens_sha256="t")

    def test_complete_identity_set_and_no_missing_or_duplicate_cells(self):
        protocol, cells = self.fixture()
        self.validate(protocol, cells)
        for altered in (cells[:-1], cells[:-1] + [cells[0]]):
            with self.assertRaises(ValueError):
                self.validate(protocol, altered)

    def test_checkpoint_input_and_status_mismatches_rejected(self):
        protocol, cells = self.fixture()
        for field in ("checkpoint_sha256", "sample_ids_sha256", "protocol_sha256", "tokens_sha256"):
            changed = copy.deepcopy(cells)
            changed[0][field] = "changed"
            with self.assertRaises(ValueError):
                self.validate(protocol, changed)
        cells[0]["execution"]["status"] = "PARTIAL_EXECUTION"
        with self.assertRaises(ValueError):
            self.validate(protocol, cells)

    def test_fixed_pair_seed_schedule_and_45_byte_rows(self):
        protocol, cells = self.fixture()
        calls = []

        def paired(a, b, *, bootstrap_seed, repeats):
            calls.append((a["model_seed"], a["arm"], b["arm"], bootstrap_seed, repeats))
            return {"model_seed": a["model_seed"], "candidate": a["arm"], "baseline": b["arm"],
                    "bootstrap_seed": bootstrap_seed, "bootstrap_repeats": repeats, "checkpoint_pooling": False,
                    "N": a["N"], "RMST0_delta": 0., "pointwise_95_CI": [0., 0.],
                    "positive_difference_sequences": 0, "negative_difference_sequences": 0,
                    "interval_scope": "pointwise"}

        fresh, pairs, byte_rows, details = AGGREGATE.build_tables(cells, protocol, paired)
        self.assertEqual((len(fresh), len(pairs), len(byte_rows), len(details)), (15, 9, 45, 9))
        self.assertEqual([c[3] for c in calls], [60101, 60102, 60103, 60201, 60202, 60203, 60301, 60302, 60303])
        self.assertTrue(all(c[4] == 5000 for c in calls))
        self.assertEqual(byte_rows[-1]["total_bytes"], 2788)


if __name__ == "__main__":
    unittest.main()
