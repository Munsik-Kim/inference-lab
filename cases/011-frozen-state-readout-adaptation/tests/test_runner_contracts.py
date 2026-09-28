"""Input/feature-control integration contracts, using only synthetic or SMOKE data."""
from itertools import product
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from source import data
from source.adapter import FrozenRollout
from source.controls import InputOnlyPredictor, shuffle_permutation
from source.references import load_references
from test_adapter import SyntheticModel, synthetic_table


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunnerContracts(unittest.TestCase):
    def fixture(self, root):
        folder = root / 'inputs' / 'smoke'
        folder.mkdir(parents=True)
        tokens = np.array([[1, 2, 3], [3, 2, 1]], dtype=np.uint8)
        gold = data.gold_trace(tokens)
        np.save(folder / 'tokens.npy', tokens, allow_pickle=False)
        np.save(folder / 'gold.npy', gold, allow_pickle=False)
        (folder / 'sample_ids.json').write_text(json.dumps(['fixture:0', 'fixture:1']))
        item = {'seed': 1, 'split_tag': 202, 'sequences': 2, 'group_tokens': 3,
            'tokens_path': 'inputs/smoke/tokens.npy', 'gold_path': 'inputs/smoke/gold.npy',
            'tokens_sha256': file_sha(folder / 'tokens.npy'), 'gold_sha256': file_sha(folder / 'gold.npy'),
            'sample_ids_path': 'inputs/smoke/sample_ids.json', 'sample_ids_sha256': file_sha(folder / 'sample_ids.json'),
            'per_sequence_token_sha256': [hashlib.sha256(row.tobytes()).hexdigest() for row in tokens]}
        manifest = {'schema': 'case011-inputs-v1', 'cohorts': {'smoke': item}}
        self.store(root, manifest)
        return folder, manifest

    def store(self, root, manifest):
        (root / 'inputs/manifest.json').write_text(json.dumps(manifest))

    def test_independent_gold_all_three_token_products(self):
        tokens = np.array(list(product(range(6), repeat=3)), dtype=np.uint8)
        actual = data.gold_trace(tokens)
        expected = load_references().groups.make_group('S3').trace(tokens)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[:, 0], tokens[:, 0])
        # This pair is noncommuting, making a reversed product observable.
        self.assertNotEqual(int(data.gold_trace(np.array([[1, 2]], dtype=np.uint8))[0, -1]),
                            int(data.gold_trace(np.array([[2, 1]], dtype=np.uint8))[0, -1]))

    def test_smoke_identity_and_bos_excluded(self):
        tokens, gold, item = data.load('smoke')
        self.assertEqual(tokens.shape, (8, 32))
        self.assertEqual(gold.shape, tokens.shape)
        self.assertTrue((tokens <= 5).all())
        self.assertTrue((gold <= 5).all())
        self.assertEqual(item['sequences'], 8)

    def test_split_plan_and_equal_feature_row_counts(self):
        manifest = json.loads((ROOT / 'inputs/manifest.json').read_text())
        cohorts = manifest['cohorts']
        self.assertEqual(set(cohorts), {'smoke', 'fit', 'dev', 'fresh'})
        self.assertEqual(len({c['seed'] for c in cohorts.values()}), 4)
        self.assertEqual((cohorts['fit']['sequences'], cohorts['fit']['group_tokens']), (512, 256))
        self.assertEqual((cohorts['dev']['sequences'], cohorts['dev']['group_tokens']), (128, 256))
        self.assertEqual((cohorts['fresh']['sequences'], cohorts['fresh']['group_tokens']), (1024, 2048))
        self.assertFalse(manifest['prior_seed_scan_matches'])
        self.assertTrue(all(row['duplicates'] == 0 for row in manifest['complete_sequence_duplicate_checks']))
        positions_path = ROOT / 'inputs/mixed_positions.npy'
        self.assertEqual(file_sha(positions_path), manifest['mixed_positions_sha256'])
        positions = np.load(positions_path, allow_pickle=False)
        self.assertEqual(positions.shape, (512, 32))
        for low, high in data.BANDS:
            np.testing.assert_array_equal(((positions >= low) & (positions <= high)).sum(1), np.full(512, 8))
        self.assertTrue(all(len(set(row.tolist())) == 32 for row in positions))
        self.assertEqual(manifest['short_rows'], positions.size)
        self.assertEqual(manifest['mixed_rows'], positions.size)

    def test_changed_tokens_rejected_without_rehash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder, _ = self.fixture(root)
            tokens = np.load(folder / 'tokens.npy'); tokens[0, 0] = 5
            np.save(folder / 'tokens.npy', tokens, allow_pickle=False)
            with patch.object(data, 'ROOT', root), self.assertRaises(ValueError):
                data.load('smoke')

    def test_independent_gold_rejects_rehashed_wrong_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder, manifest = self.fixture(root)
            gold = np.load(folder / 'gold.npy'); gold[0, 1] = (gold[0, 1] + 1) % 6
            np.save(folder / 'gold.npy', gold, allow_pickle=False)
            manifest['cohorts']['smoke']['gold_sha256'] = file_sha(folder / 'gold.npy')
            self.store(root, manifest)
            with patch.object(data, 'ROOT', root), self.assertRaisesRegex(ValueError, 'gold'):
                data.load('smoke')

    def test_declared_shape_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, manifest = self.fixture(root)
            manifest['cohorts']['smoke']['sequences'] = 3
            self.store(root, manifest)
            with patch.object(data, 'ROOT', root), self.assertRaises(ValueError):
                data.load('smoke')

    def test_id_bytes_and_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder, manifest = self.fixture(root)
            ids_path = folder / 'sample_ids.json'
            ids_path.write_text(json.dumps(['fixture:0', 'fixture:0']))
            with patch.object(data, 'ROOT', root), self.assertRaises(ValueError):
                data.load('smoke')
            manifest['cohorts']['smoke']['sample_ids_sha256'] = file_sha(ids_path)
            self.store(root, manifest)
            with patch.object(data, 'ROOT', root), self.assertRaises(ValueError):
                data.load('smoke')

    def test_per_sequence_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); _, manifest = self.fixture(root)
            manifest['cohorts']['smoke']['per_sequence_token_sha256'][0] = '0' * 64
            self.store(root, manifest)
            with patch.object(data, 'ROOT', root), self.assertRaises(ValueError):
                data.load('smoke')

    def test_control_reads_cannot_mutate_the_persistent_cache(self):
        torch.set_num_threads(2)
        current = np.array([0, 0, 1, 1, 2, 2, 3, 4], dtype=np.int64)
        rollout = FrozenRollout(SyntheticModel(), synthetic_table(), 'UNIFORM_8', batch_size=8)
        rollout.step(np.full(8, 6))
        out = rollout.step(current)
        before = rollout.export_bytes()
        order, receipt = shuffle_permutation(current, 1)
        shuffled_phi = out.phi[torch.from_numpy(order)]
        original_logits = rollout.model.mlp[2](out.phi)
        control_logits = rollout.model.mlp[2](shuffled_phi)
        self.assertEqual(tuple(control_logits.shape), (8, 6))
        self.assertEqual(before, rollout.export_bytes())
        self.assertTrue(torch.equal(out.logits_original, original_logits))
        np.testing.assert_array_equal(current, current[order])
        self.assertEqual(receipt['singleton_unchanged_rows'], 2)
        self.assertEqual(receipt['shuffled_rows'], 6)
        rollout.assert_frozen()

    def test_input_only_requires_neither_phi_nor_future_gold(self):
        train_tokens = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.uint8)
        train_gold = data.gold_trace(train_tokens)
        predictor = InputOnlyPredictor.fit(train_tokens, train_gold, positions=np.array([1, 33, 65, 129]))
        self.assertEqual(predictor.to_dict()['fit_rows'], 8)
        current = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)
        np.testing.assert_array_equal(predictor.predict(current, np.full(6, 256)),
                                      predictor.predict(current, np.full(6, 2048)))
        with self.assertRaises(TypeError):
            predictor.predict(current, np.full(6, 256), gold=current)
        with self.assertRaises(TypeError):
            shuffle_permutation(current, 10, labels=current)


if __name__ == '__main__':
    unittest.main()
