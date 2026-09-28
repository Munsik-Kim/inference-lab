"""Independent strict audit of an evaluation checkpoint before a future resume.

Checks serialized runtime cursor against saved prediction/score prefix, exact
inventory and identity, without executing a model or touching original files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source.data import ROOT, file_sha
from source.references import load_references, load_storage

SCORES = ("ce_sum", "gold_margin_sum", "top_margin_sum", "score_valid_count")


def expected_identity(seed, storage):
    refs = load_references()
    manifest = json.loads((ROOT / 'inputs/manifest.json').read_text())
    return {'seed': seed, 'storage': storage, 'checkpoint_sha256': refs.common.CHECKPOINTS[seed],
            'input_hash': manifest['cohorts']['fresh']['tokens_sha256'],
            'selected_heads_sha256': file_sha(ROOT / 'configs/selected_heads.json'),
            'pretest_freeze_sha256': file_sha(ROOT / 'configs/pretest_freeze.json')}


def validate(directory, *, storage, model_seed, n_sequences, max_tokens,
             identity=None, case010_root=None):
    path = Path(directory)
    if not path.is_dir() or path.is_symlink():
        raise ValueError('Expected a completed real execution checkpoint directory')
    wanted = {'receipt.json', 'state.bin', 'outputs.npz'}
    if {p.name for p in path.iterdir()} != wanted or any(p.is_symlink() or not p.is_file() for p in path.iterdir()):
        raise ValueError('Execution checkpoint inventory mismatch')
    receipt = json.loads((path / 'receipt.json').read_text())
    if set(receipt) != {'offset', 'identity', 'files'}:
        raise ValueError('Unknown execution receipt schema')
    offset = receipt['offset']
    if type(offset) is not int or not 1 <= offset <= max_tokens:
        raise ValueError('Saved offset outside declared evaluation range')
    if path.name != f'offset-{offset:04d}':
        raise ValueError('Offset directory name and receipt differ')
    if type(n_sequences) is not int or n_sequences < 1:
        raise ValueError('Expected positive sequence count')
    expected = expected_identity(model_seed, storage) if identity is None else identity
    if receipt['identity'] != expected:
        raise ValueError('Execution identity differs from the frozen input/head/model identity')
    if set(receipt['files']) != {'state.bin', 'outputs.npz'}:
        raise ValueError('Execution file checksum inventory mismatch')
    for name, digest in receipt['files'].items():
        if file_sha(path / name) != digest:
            raise ValueError('Execution file checksum mismatch: ' + name)
    adapter = load_storage(storage, model_seed, case010_root)
    raw = (path / 'state.bin').read_bytes()
    state = adapter.from_bytes(raw, batch_size=n_sequences)
    status = adapter.terminal_info(state)
    if not np.all(status['cursor'] == offset + 1):
        raise ValueError('Saved runtime cursor does not equal scored prefix plus BOS')
    with np.load(path / 'outputs.npz', allow_pickle=False) as arrays:
        wanted_arrays = {'predictions', 'shuffled_predictions', *SCORES}
        if set(arrays.files) != wanted_arrays:
            raise ValueError('Saved output array inventory mismatch')
        for name in ('predictions', 'shuffled_predictions'):
            pred = arrays[name]
            if pred.dtype != np.int8 or pred.shape != (3, n_sequences, offset) or np.any(pred < -1) or np.any(pred > 5):
                raise ValueError('Saved prediction prefix shape/dtype/labels mismatch: ' + name)
            for row in np.flatnonzero(~status['active']):
                first_scored = max(1, int(status['first_terminal_write'][row]))
                if np.any(pred[:, row, first_scored - 1:] != -1):
                    raise ValueError('A terminal stream has valid predictions after its absorbing failure')
        valid = arrays['score_valid_count']
        if valid.dtype != np.int64 or valid.shape != (3, offset) or np.any(valid < 0) or np.any(valid > n_sequences):
            raise ValueError('Invalid score denominators or saved prefix shape')
        for name in SCORES[:-1]:
            value = arrays[name]
            if value.dtype != np.float64 or value.shape != (3, offset) or not np.isfinite(value).all():
                raise ValueError('Nonfinite or malformed score prefix: ' + name)
            if np.any(value[valid == 0] != 0):
                raise ValueError('Zero-denominator score cells must contain zero summed observations')
    return {'status': 'PASS', 'offset': offset, 'n_sequences': n_sequences,
            'state_bytes': len(raw), 'state_sha256': hashlib.sha256(raw).hexdigest(),
            'receipt_sha256': file_sha(path / 'receipt.json'),
            'terminal_streams': int((~status['active']).sum()),
            'cursor_matches_BOS_and_prefix': True,
            'scope': 'Independent state/prefix/identity audit; no model execution'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--private-eval-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite execution-checkpoint audit')
    rows = []
    for seed in range(3):
        for storage in ('NATIVE_FP32', 'UNIFORM_8'):
            folder = args.private_eval_root / f'seed{seed}' / storage
            for path in sorted(folder.glob('offset-*')):
                if path.name.endswith('.partial'):
                    continue
                row = validate(path, storage=storage, model_seed=seed, n_sequences=1024, max_tokens=2048)
                row.update(model_seed=seed, storage=storage)
                rows.append(row)
    if not rows:
        raise ValueError('No completed evaluation checkpoints found')
    args.output.write_text(json.dumps({'status': 'PASS', 'checkpoints': rows}, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'PASS', 'checkpoint_count': len(rows)}))


if __name__ == '__main__':
    main()
