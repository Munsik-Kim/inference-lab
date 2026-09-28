"""Synthetic contracts only; these fixtures are not measured CKDA results."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source.fitting import equal_band_ce, fit_head, select_candidate
from source.head_patch import apply_patch, load_patch, save_patch


class SmallFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("unchanged_buffer", torch.arange(4.0))
        self.mlp = nn.Sequential(nn.Linear(48, 192), nn.GELU(), nn.Linear(192, 6))

    def forward(self, x):
        return self.mlp(x)


class FittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def fixture(self):
        g = torch.Generator().manual_seed(4711)
        phi = torch.randn(24, 192, generator=g)
        gold = torch.arange(24) % 6
        w = torch.randn(6, 192, generator=g) * .01
        b = torch.zeros(6)
        return phi, gold, w, b

    def test_deterministic_fit_preserves_inputs_and_shape(self):
        values = self.fixture()
        before = [t.clone() for t in values]
        a = fit_head(*values, .01)
        b = fit_head(*values, .01)
        self.assertTrue(torch.equal(a["weight"], b["weight"]))
        self.assertTrue(torch.equal(a["bias"], b["bias"]))
        self.assertEqual(a["solver"]["objective_trace"], b["solver"]["objective_trace"])
        self.assertEqual(a["weight"].dtype, torch.float32)
        self.assertEqual(a["weight"].shape, (6, 192))
        self.assertLess(a["solver"]["final_objective"], a["solver"]["objective_trace"][0]["objective"])
        self.assertLessEqual(a["solver"]["iterations"], 200)
        self.assertLessEqual(a["solver"]["objective_evaluations"], 1000)
        self.assertEqual(a["solver"]["trainable_parameter_count"], 1158)
        json.dumps(a["solver"], allow_nan=False)
        for current, original in zip(values, before):
            self.assertTrue(torch.equal(current, original))

    def test_centered_penalty_is_zero_at_initial_head(self):
        phi, gold, w, b = self.fixture()
        result = fit_head(phi, gold, w, b, 1.0)
        first = result["solver"]["objective_trace"][0]
        expected = float(F.cross_entropy(F.linear(phi.double(), w.double(), b.double()), gold))
        self.assertEqual(first["centered_l2"], 0.0)
        self.assertEqual(first["objective"], expected)

    def test_hard_evaluation_limit_rejects_candidate(self):
        phi, gold, w, b = self.fixture()
        result = fit_head(phi, gold, w, b, .01, max_evaluations=1)
        self.assertEqual(result["solver"]["objective_evaluations"], 1)
        self.assertFalse(result["solver"]["eligible_for_dev"])
        self.assertTrue(torch.equal(result["weight"], w))
        self.assertTrue(torch.equal(result["bias"], b))
        with self.assertRaisesRegex(ValueError, "BLOCKED_FITTING"):
            select_candidate([result], phi, gold, torch.tensor([1, 33, 65, 129] * 6))

    def test_iteration_limit_is_reported_without_claiming_convergence(self):
        result = fit_head(*self.fixture(), .01, max_iter=1)
        self.assertEqual(result["solver"]["status"], "MAX_ITER_NOT_CONVERGED")
        self.assertTrue(result["solver"]["eligible_for_dev"])
        self.assertFalse(result["solver"]["gradient_tolerance_met"])

    def test_nonfinite_features_labels_and_shape_rejected(self):
        phi, gold, w, b = self.fixture()
        for invalid in (float("nan"), float("inf")):
            bad = phi.clone(); bad[0, 0] = invalid
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                fit_head(bad, gold, w, b, .01)
        with self.assertRaises(ValueError):
            fit_head(phi[:, :-1], gold, w, b, .01)
        with self.assertRaises(ValueError):
            fit_head(phi, gold.float() + .1, w, b, .01)
        with self.assertRaises(ValueError):
            fit_head(phi, gold + 6, w, b, .01)
        with self.assertRaises(ValueError):
            fit_head(phi, gold, w, b, .02)

    def test_dev_equal_band_weighting_and_fp32_execution(self):
        phi, gold, w, b = self.fixture()
        positions = torch.tensor([1] * 18 + [33] * 2 + [65] * 2 + [129] * 2)
        report = equal_band_ce(phi.double(), gold, positions, w.double(), b.double())
        losses = F.cross_entropy(F.linear(phi, w, b).double(), gold, reduction="none")
        expected = sum(float(losses[positions == p].mean()) for p in (1, 33, 65, 129)) / 4
        self.assertAlmostEqual(report["equal_band_ce"], expected, places=14)
        self.assertNotAlmostEqual(report["equal_band_ce"], float(losses.mean()), places=8)
        self.assertEqual(report["logit_dtype"], "float32")

    def test_dev_missing_bos_or_bad_band_rejected(self):
        phi, gold, w, b = self.fixture()
        for positions in (torch.ones(24), torch.zeros(24), torch.full((24,), 257)):
            with self.assertRaises(ValueError):
                equal_band_ce(phi, gold, positions, w, b)

    def test_tie_prefers_large_lambda_order_independent(self):
        phi, gold, w, b = self.fixture()
        candidates = [{"weight": w, "bias": b, "solver": {"regularization": x}}
                      for x in (.01, 1., .0001)]
        p = torch.tensor([1, 33, 65, 129] * 6)
        self.assertEqual(select_candidate(candidates, phi, gold, p)["regularization"], 1.)
        self.assertEqual(select_candidate(candidates[::-1], phi, gold, p)["regularization"], 1.)
        with self.assertRaises(ValueError):
            select_candidate(candidates[:1] * 2, phi, gold, p)


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "candidate"
        self.w = torch.arange(1152, dtype=torch.float32).reshape(6, 192) / 1152
        self.b = torch.arange(6, dtype=torch.float32) / 6
        self.meta = {
            "evidence_kind": "synthetic_test", "base_checkpoint_sha256": "a" * 64,
            "fit_input_sha256": "b" * 64, "dev_input_sha256": "c" * 64,
            "code_sha256": "d" * 64, "feature_boundary": "test-GELU-final-v1",
            "regularization": .01, "solver": {"status": "synthetic_fixture"},
        }

    def create(self):
        return save_patch(self.path, self.w, self.b, self.meta)

    def rewrite_manifest(self, change):
        path = self.path / "manifest.json"
        obj = json.loads(path.read_text())
        change(obj)
        path.write_text(json.dumps(obj))

    def test_exact_serialization_byte_ledger_no_overwrite(self):
        receipt = self.create()
        self.assertEqual(receipt["payload_bytes"], 4632)
        self.assertEqual(receipt["total_file_bytes"], 4632 + receipt["manifest_bytes"])
        result = load_patch(self.path, expected_base_sha256="a" * 64)
        self.assertTrue(torch.equal(result["tensors"]["mlp.2.weight"], self.w))
        self.assertTrue(torch.equal(result["tensors"]["mlp.2.bias"], self.b))
        with self.assertRaises(FileExistsError):
            self.create()

    def test_wrong_base_and_missing_extra_files_rejected(self):
        self.create()
        with self.assertRaisesRegex(ValueError, "base checkpoint"):
            load_patch(self.path, expected_base_sha256="f" * 64)
        (self.path / "unexpected").write_text("x")
        with self.assertRaisesRegex(ValueError, "inventory"):
            load_patch(self.path, expected_base_sha256="a" * 64)
        (self.path / "unexpected").unlink()
        (self.path / "mlp.2.bias.f32le").unlink()
        with self.assertRaisesRegex(ValueError, "inventory"):
            load_patch(self.path, expected_base_sha256="a" * 64)

    def test_payload_checksum_and_length_rejected(self):
        self.create()
        p = self.path / "mlp.2.weight.f32le"
        p.write_bytes(p.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "length/checksum"):
            load_patch(self.path, expected_base_sha256="a" * 64)

    def test_shape_dtype_names_and_traversal_rejected(self):
        for key, value in (("shape", [192, 6]), ("dtype", "float64"),
                           ("name", "mlp.0.weight"), ("file", "../elsewhere")):
            with self.subTest(key=key):
                path = self.root / key
                save_patch(path, self.w, self.b, self.meta)
                m = path / "manifest.json"; obj = json.loads(m.read_text())
                obj["tensors"][0][key] = value; m.write_text(json.dumps(obj))
                with self.assertRaises(ValueError):
                    load_patch(path, expected_base_sha256="a" * 64)

    def test_input_shape_dtype_and_nonfinite_rejected_before_writing(self):
        for w in (self.w.double(), self.w.T, self.w * float("nan")):
            with self.assertRaises(ValueError):
                save_patch(self.path, w, self.b, self.meta)
            self.assertFalse(self.path.exists())

    def test_apply_changes_only_two_tensors_preserves_parameter_identity(self):
        self.create()
        model = SmallFixture()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        ids = {k: id(v) for k, v in model.named_parameters()}
        report = apply_patch(model, self.path, expected_base_sha256="a" * 64)
        self.assertEqual(report["changed_tensors"], ["mlp.2.bias", "mlp.2.weight"])
        self.assertEqual(ids, {k: id(v) for k, v in model.named_parameters()})
        for name, value in model.state_dict().items():
            if name not in report["changed_tensors"]:
                self.assertTrue(torch.equal(value, before[name]))
        x = torch.arange(48, dtype=torch.float32).reshape(1, 48) / 48
        phi = model.mlp[1](model.mlp[0](x))
        self.assertTrue(torch.equal(model(x), F.linear(phi, self.w, self.b)))

    def test_in_memory_tampering_rejected(self):
        self.create()
        patch = load_patch(self.path, expected_base_sha256="a" * 64)
        patch["tensors"]["mlp.2.weight"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            apply_patch(SmallFixture(), patch, expected_base_sha256="a" * 64)

    def test_fresh_process_load_and_logits_identical(self):
        self.create()
        source_root = str(Path(__file__).resolve().parents[1])
        script = """
import json, sys, torch
sys.path.insert(0, sys.argv[1])
from source.head_patch import load_patch
p=load_patch(sys.argv[2], expected_base_sha256='a'*64)
x=torch.arange(192,dtype=torch.float32).reshape(1,192)/192
y=torch.nn.functional.linear(x,p['tensors']['mlp.2.weight'],p['tensors']['mlp.2.bias'])
print(json.dumps({'logits':y.tolist(),'labels':y.argmax(-1).tolist(),'finite':bool(torch.isfinite(y).all())},allow_nan=False))
"""
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run([sys.executable, "-B", "-c", script, source_root, str(self.path)],
                                   check=True, text=True, capture_output=True, cwd=self.root, env=env)
        output = json.loads(completed.stdout)
        x = torch.arange(192, dtype=torch.float32).reshape(1, 192) / 192
        y = F.linear(x, self.w, self.b)
        self.assertEqual(output["logits"], y.tolist())
        self.assertEqual(output["labels"], y.argmax(-1).tolist())
        self.assertTrue(output["finite"])

    def test_duplicate_json_keys_and_symlink_rejected(self):
        self.create()
        m = self.path / "manifest.json"
        text = m.read_text()
        m.write_text(text.replace('"schema":', '"schema":"duplicate","schema":', 1))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_patch(self.path, expected_base_sha256="a" * 64)
        m.write_text(text)
        b = self.path / "mlp.2.bias.f32le"
        original = b.read_bytes(); b.unlink()
        external = self.root / "body"; external.write_bytes(original); b.symlink_to(external)
        with self.assertRaisesRegex(ValueError, "inventory"):
            load_patch(self.path, expected_base_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
