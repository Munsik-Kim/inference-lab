"""Small synthetic contracts for reload tooling; no real checkpoint is loaded."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source.reload_check import checkpoint_path, collect_smoke, patch_path


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(), torch.nn.Linear(192, 6))
        for parameter in self.parameters():
            parameter.requires_grad_(False)


class FakeRollout:
    """Fixture only: integer token accumulator with an unrelated final head."""
    def __init__(self, model, table, storage, *, model_seed, batch_size):
        self.model = model
        self.values = np.zeros(batch_size, dtype=np.int64)
        self.cursor = 0
        self.table_sha256 = "a" * 64

    def step(self, ids):
        self.values += ids
        self.cursor += 1
        phi = torch.from_numpy(np.repeat(self.values[:, None], 192, axis=1).astype(np.float32))
        logits = self.model.mlp[2](phi)
        return SimpleNamespace(phi=phi, logits_original=logits, status={
            "active": np.ones(len(ids), dtype=bool), "finite_features": np.ones(len(ids), dtype=bool),
            "cursor": np.full(len(ids), self.cursor, dtype=np.uint64),
            "terminal_code": np.zeros(len(ids), dtype=np.uint8)})

    def export_bytes(self):
        return self.values.tobytes() + self.cursor.to_bytes(8, "little")

    def assert_frozen(self):
        if any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise AssertionError("fixture model has a gradient")


class ReloadToolTests(unittest.TestCase):
    def test_detached_head_and_actual_replacement_have_same_cache(self):
        torch.set_num_threads(2)
        model = FakeModel()
        w = torch.arange(1152, dtype=torch.float32).reshape(6, 192) / 1152
        b = torch.arange(6, dtype=torch.float32) / 6
        candidate = {"tensors": {"mlp.2.weight": w, "mlp.2.bias": b}}
        tokens = np.tile(np.arange(32, dtype=np.uint8) % 6, (8, 1))
        with patch("source.reload_check.FrozenRollout", FakeRollout):
            detached = collect_smoke(model, {}, "NATIVE_FP32", 0, tokens, {"TEST": candidate})
            with torch.no_grad():
                model.mlp[2].weight.copy_(w); model.mlp[2].bias.copy_(b)
            actual = collect_smoke(model, {}, "NATIVE_FP32", 0, tokens)
        self.assertEqual(len(actual["cache_boundary_sha256"]), 33)
        self.assertEqual(actual["cache_boundary_sha256"], detached["cache_boundary_sha256"])
        self.assertEqual(actual["final_payload"], detached["final_payload"])
        self.assertEqual(actual["feature_sha256"], detached["feature_sha256"])
        self.assertTrue(np.array_equal(actual["logits"]["ACTUAL"], detached["logits"]["TEST"]))
        self.assertTrue(np.array_equal(actual["labels"]["ACTUAL"], detached["labels"]["TEST"]))
        self.assertTrue(actual["no_gradients"])

    def test_smoke_shape_contract(self):
        with self.assertRaisesRegex(ValueError, "8x32"):
            collect_smoke(FakeModel(), {}, "NATIVE_FP32", 0, np.zeros((1, 32)))

    def test_nonfinite_smoke_cannot_pass(self):
        model = FakeModel()
        with torch.no_grad():
            model.mlp[2].weight.fill_(float("nan"))
        with patch("source.reload_check.FrozenRollout", FakeRollout):
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                collect_smoke(model, {}, "NATIVE_FP32", 0, np.zeros((8, 32), dtype=np.uint8))

    def test_checkpoint_location_must_be_unambiguous(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                checkpoint_path(root, 0)
            target = root / "training-seed0/final.pt"; target.parent.mkdir(); target.write_bytes(b"fixture")
            self.assertEqual(checkpoint_path(root, 0), target)
            (root / "seed0.pt").write_bytes(b"second fixture")
            with self.assertRaises(ValueError):
                checkpoint_path(root, 0)

    def test_selected_patch_location_must_be_unambiguous(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                patch_path(root, 0, "SHORT_REFIT")
            target = root / "seed0-SHORT_REFIT"; target.mkdir()
            self.assertEqual(patch_path(root, 0, "SHORT_REFIT"), target)
            (root / "seed0/SHORT_REFIT").mkdir(parents=True)
            with self.assertRaises(ValueError):
                patch_path(root, 0, "SHORT_REFIT")


if __name__ == "__main__":
    unittest.main()
