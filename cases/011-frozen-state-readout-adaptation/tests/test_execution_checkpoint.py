"""Synthetic evaluation checkpoint corruption and all-terminal runner contracts."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from source.adapter import FrozenRollout
from source.check_execution_checkpoint import validate
from source import run_eval
from source.data import file_sha
from test_adapter import SyntheticModel, synthetic_table


class ExecutionCheckpointContracts(unittest.TestCase):
    def snapshot(self, parent):
        rollout = FrozenRollout(SyntheticModel(), synthetic_table(), 'UNIFORM_8', batch_size=3)
        for ids in (np.full(3, 6), np.arange(3), np.array([3, 4, 5])):
            rollout.step(ids)
        pred = np.zeros((3, 3, 8), dtype=np.int8)
        scores = {k: np.zeros((3, 8), dtype=np.int64 if k == 'score_valid_count' else np.float64)
                  for k in ('ce_sum', 'gold_margin_sum', 'top_margin_sum', 'score_valid_count')}
        scores['score_valid_count'][:] = 3
        identity = {'fixture': 'synthetic_test'}
        row = run_eval.save_boundary(parent, rollout, pred, pred.copy(), scores, 2, identity)
        return rollout, parent / 'offset-0002', identity, row

    def test_valid_boundary_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rollout, path, identity, row = self.snapshot(root)
            got = validate(path, storage='UNIFORM_8', model_seed=0, n_sequences=3, max_tokens=8, identity=identity)
            self.assertEqual(got['state_sha256'], row['state_sha256'])
            self.assertEqual((path / 'state.bin').read_bytes(), rollout.export_bytes())
            with self.assertRaisesRegex(ValueError, 'replace'):
                self.snapshot(root)

    def test_rehashed_wrong_cursor_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, path, identity, _ = self.snapshot(Path(tmp))
            raw = bytearray((path / 'state.bin').read_bytes())
            raw[:8] = (2).to_bytes(8, 'little')
            (path / 'state.bin').write_bytes(raw)
            receipt = json.loads((path / 'receipt.json').read_text())
            receipt['files']['state.bin'] = file_sha(path / 'state.bin')
            (path / 'receipt.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'cursor'):
                validate(path, storage='UNIFORM_8', model_seed=0, n_sequences=3, max_tokens=8, identity=identity)

    def test_rehashed_missing_prediction_prefix_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, path, identity, _ = self.snapshot(Path(tmp))
            with np.load(path / 'outputs.npz', allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            arrays['predictions'] = arrays['predictions'][:, :, :1]
            np.savez_compressed(path / 'outputs.npz', **arrays)
            receipt = json.loads((path / 'receipt.json').read_text())
            receipt['files']['outputs.npz'] = file_sha(path / 'outputs.npz')
            (path / 'receipt.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'prefix'):
                validate(path, storage='UNIFORM_8', model_seed=0, n_sequences=3, max_tokens=8, identity=identity)

    def test_extra_file_and_wrong_identity_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, path, identity, _ = self.snapshot(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'identity'):
                validate(path, storage='UNIFORM_8', model_seed=0, n_sequences=3, max_tokens=8, identity={'fixture': 'other'})
            (path / 'extra.bin').write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError, 'inventory'):
                validate(path, storage='UNIFORM_8', model_seed=0, n_sequences=3, max_tokens=8, identity=identity)

    def test_all_terminal_cell_never_becomes_a_valid_readout(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'sample_ids.json').write_text(json.dumps(['a', 'b', 'c']))
            model = SyntheticModel(); table = synthetic_table()
            expected_hash = 'a' * 64
            selected = {'heads': [{'model_seed': 0, 'condition': h, 'patch_directory': h,
                                   'manifest_sha256': expected_hash} for h in run_eval.HEADS[1:]]}
            patched = {'tensors': {'mlp.2.weight': model.mlp[2].weight.detach().clone(),
                                  'mlp.2.bias': model.mlp[2].bias.detach().clone()}}
            def terminal_rollout(*args, **kwargs):
                rollout = FrozenRollout(*args, **kwargs)
                def faulty(x): return np.full_like(x, np.nan)
                rollout.state, _, _ = rollout.adapter.step(rollout.state, faulty, diagnostics=False)
                return rollout
            args = SimpleNamespace(private=str(root / 'private'), checkpoint_root=str(root), upstream=str(root))
            identity = {'tokens_sha256': expected_hash, 'sample_ids_path': 'sample_ids.json'}
            control = SimpleNamespace(predict=lambda tokens: np.zeros(tokens.shape, dtype=np.int8))
            tokens = np.array([[1, 2], [2, 3], [3, 4]], dtype=np.uint8)
            with patch.object(run_eval, 'ROOT', root), patch.object(run_eval, 'file_sha', return_value=expected_hash), \
                 patch.object(run_eval, 'load_verified_model', return_value=(model, table)), \
                 patch.object(run_eval, 'load_patch', return_value=patched), \
                 patch.object(run_eval, 'FrozenRollout', side_effect=terminal_rollout):
                run_eval.cell(args, 0, 'UNIFORM_8', tokens, np.zeros_like(tokens), identity, selected, control)
            out = root / 'results/fresh/seed0/UNIFORM_8'
            with np.load(out / 'predictions.npz', allow_pickle=False) as raw:
                self.assertTrue((raw['predictions'] == -1).all())
                self.assertTrue((raw['shuffled_predictions'] == -1).all())
                self.assertTrue((raw['score_valid_count'] == 0).all())
                self.assertTrue((raw['terminal_code'] != 0).all())
            self.assertEqual(json.loads((out / 'receipt.json').read_text())['terminal_count'], 3)


if __name__ == '__main__':
    unittest.main()
