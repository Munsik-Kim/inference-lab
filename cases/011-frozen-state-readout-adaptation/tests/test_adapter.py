"""Synthetic contracts only; no learned model performance is asserted here."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from source.adapter import FrozenRollout, read_features, parameter_manifest
from source.references import load_references, load_storage


class SyntheticNorm(torch.nn.Module):
    def forward(self, value, gate):
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * gate.sigmoid()


class SyntheticLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 12
        self.head_k_dim = self.head_v_dim = 16
        self.o_norm = SyntheticNorm()
        self.o_proj = torch.nn.Linear(192, 48, bias=False)


class SyntheticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(111)
            self.layer = SyntheticLayer()
            self.norm = torch.nn.LayerNorm(48)
            self.mlp = torch.nn.Sequential(torch.nn.Linear(48, 192), torch.nn.GELU(), torch.nn.Linear(192, 6))


def synthetic_table():
    rng = np.random.default_rng(119)
    table = {name: rng.normal(size=(7, 12, 16)).astype(np.float32) * 0.05
             for name in ("q", "k", "v", "g")}
    table["alpha"] = np.full((7, 12, 16), 0.97, dtype=np.float32)
    table["beta"] = np.full((7, 12), 0.5, dtype=np.float32)
    table["e"] = rng.normal(size=(7, 48)).astype(np.float32)
    return table


class AdapterContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.ref = load_references()

    def make(self, storage="NATIVE_FP32", count=3):
        return FrozenRollout(SyntheticModel(), synthetic_table(), storage, batch_size=count)

    def test_feature_boundary_bitwise_original(self):
        roll = self.make()
        for ids in (np.full(3, 6), np.arange(3), np.arange(3) + 2):
            out = roll.step(ids)
            expected = self.ref.v1.logits_from_numpy(roll.model, roll.adapter.represented(roll.state),
                                                    self.ref.v1.gather(roll.table, ids))
            self.assertTrue(torch.equal(expected, out.logits_original))
            self.assertTrue(torch.equal(roll.model.mlp[2](out.phi), expected))
            self.assertFalse(out.phi.requires_grad)

    def test_frozen_parameters_no_gradients(self):
        roll = self.make()
        before = parameter_manifest(roll.model)
        roll.step(np.full(3, 6))
        roll.step(np.arange(3))
        roll.assert_frozen()
        self.assertEqual(before, parameter_manifest(roll.model))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in roll.model.parameters()))

    def test_changed_hidden_parameter_detected(self):
        roll = self.make()
        with torch.no_grad():
            roll.model.mlp[0].weight[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "mlp.0.weight"):
            roll.assert_frozen()

    def test_wrong_head_shape_rejected(self):
        model = SyntheticModel()
        model.mlp[2] = torch.nn.Linear(192, 7)
        with self.assertRaisesRegex(ValueError, "192"):
            FrozenRollout(model, synthetic_table(), "NATIVE_FP32")

    def test_storage_reuses_frozen_config(self):
        for name, expected_bytes in (("NATIVE_FP32", 12305), ("UNIFORM_8", 3137)):
            roll = self.make(name)
            self.assertEqual(roll.adapter.bytes_per_stream, expected_bytes)
            self.assertEqual(roll.adapter.config_bytes, load_storage(name).config_bytes)
        with self.assertRaisesRegex(ValueError, "permits only"):
            self.make("UNIFORM_5")

    def test_no_gold_position_or_head_in_step_api(self):
        roll = self.make()
        with self.assertRaises(TypeError):
            roll.step(np.zeros(3, dtype=np.int64), gold=np.zeros(3))
        with self.assertRaises(TypeError):
            roll.step(np.zeros(3, dtype=np.int64), position=3)

    def test_readout_changes_do_not_change_cache(self):
        left, right = self.make("UNIFORM_8"), self.make("UNIFORM_8")
        with torch.no_grad():
            right.model.mlp[2].weight.neg_()
            right.model.mlp[2].bias.add_(3)
        different = False
        for ids in (np.full(3, 6), np.arange(3), np.arange(3) + 2):
            first, second = left.step(ids), right.step(ids)
            self.assertEqual(left.export_bytes(), right.export_bytes())
            self.assertTrue(torch.equal(first.phi, second.phi))
            different |= not torch.equal(first.logits_original, second.logits_original)
        self.assertTrue(different)

    def test_split_resume_predictions_and_bytes(self):
        rng = np.random.default_rng(121)
        tokens = rng.integers(0, 6, (3, 13))
        for storage in ("NATIVE_FP32", "UNIFORM_8"):
            full = self.make(storage)
            outputs = [full.step(np.full(3, 6)).predictions_original]
            for ids in tokens.T:
                outputs.append(full.step(ids).predictions_original)
            for cut in (0, 1, 5, 12):
                split = self.make(storage)
                got = [split.step(np.full(3, 6)).predictions_original]
                for ids in tokens[:, :cut].T:
                    got.append(split.step(ids).predictions_original)
                restored = self.make(storage).import_bytes(split.export_bytes(), 3)
                for ids in tokens[:, cut:].T:
                    got.append(restored.step(ids).predictions_original)
                np.testing.assert_array_equal(got, outputs)
                self.assertEqual(full.export_bytes(), restored.export_bytes())

    def test_terminal_absorption_and_mixed_batch(self):
        roll = self.make()
        roll.step(np.full(3, 6))
        def faulty(value):
            updated = value.copy()
            updated[0] = np.nan
            return updated
        roll.state, _, _ = roll.adapter.step(roll.state, faulty, diagnostics=False)
        saved = roll.export_bytes()
        restored = self.make().import_bytes(saved, 3)
        for _ in range(4):
            a, b = roll.step(np.arange(3)), restored.step(np.arange(3))
            self.assertEqual(a.predictions_original[0], -1)
            self.assertTrue(a.status["active"][1:].all())
            self.assertEqual(roll.export_bytes(), restored.export_bytes())
            self.assertEqual(roll.state.payload[0, 17:].tobytes(), np.frombuffer(saved, dtype=np.uint8).reshape(3, -1)[0, 17:].tobytes())

    def test_finite_state_invalid_readout_continues(self):
        roll = self.make()
        with torch.no_grad():
            roll.model.mlp[2].bias[0] = float("nan")
        out = roll.step(np.full(3, 6))
        self.assertTrue(out.status["active"].all())
        self.assertTrue((out.predictions_original == -1).all())
        with torch.no_grad():
            roll.model.mlp[2].bias[0] = 0
        out = roll.step(np.arange(3))
        self.assertTrue(out.status["active"].all())
        self.assertTrue((out.predictions_original >= 0).all())

    def test_plain_wrong_prediction_does_not_terminate(self):
        roll = self.make()
        out = roll.step(np.full(3, 6))
        arbitrary_opposing_gold = (out.predictions_original + 1) % 6
        self.assertFalse(np.any(arbitrary_opposing_gold == out.predictions_original))
        self.assertTrue(roll.step(np.arange(3)).status["active"].all())

    def test_malformed_tokens_and_resume_rejected(self):
        roll = self.make()
        for ids in ([-1, 0, 1], [7, 0, 1], [1.0, 2.0, 3.0], [[1, 2, 3]]):
            with self.assertRaises(ValueError):
                roll.step(ids)
        with self.assertRaises(ValueError):
            roll.step([1, 2])
        raw = roll.export_bytes()
        for bad in (raw[:-1], raw + b"x"):
            with self.assertRaises(ValueError):
                roll.import_bytes(bad, 3)

    def test_import_namespace_preserves_unrelated_codec(self):
        code = '''
import sys, types
sys.path.insert(0, sys.argv[1])
sentinel=types.ModuleType('codec'); sentinel.marker='unrelated'
sys.modules['codec']=sentinel
before=list(sys.path)
from source.references import load_references
r=load_references()
assert sys.modules['codec'] is sentinel
assert sys.path == before
assert r.packed.__name__.startswith('_diova_case011_case010_')
'''
        proc = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_fresh_process_suffix_bytes(self):
        roll = self.make("UNIFORM_8")
        roll.step(np.full(3, 6))
        roll.step(np.arange(3))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "cache.bin").write_bytes(roll.export_bytes())
            code = '''
import sys, hashlib, json
from pathlib import Path
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, str(Path(sys.argv[1])/'tests'))
from test_adapter import AdapterContracts
import torch, numpy as np
torch.set_num_threads(2)
a=AdapterContracts().make('UNIFORM_8')
p=Path(sys.argv[2]); a.import_bytes((p/'cache.bin').read_bytes(),3)
out=a.step(np.array([5,4,3]))
(p/'result.json').write_text(json.dumps({'pred':out.predictions_original.tolist(), 'bytes':hashlib.sha256(a.export_bytes()).hexdigest()}))
'''
            proc = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT), str(path)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            out = roll.step(np.array([5, 4, 3]))
            expected = {"pred": out.predictions_original.tolist(), "bytes": hashlib.sha256(roll.export_bytes()).hexdigest()}
            self.assertEqual(json.loads((path / "result.json").read_text()), expected)


if __name__ == "__main__":
    unittest.main()
