"""Replay the original Case007 selector on CALIBRATION Q, without a model."""
from __future__ import annotations
import argparse
import importlib.util
import math
from pathlib import Path
import numpy as np
from common import ROOT, C7, checked_read, json_text, new_output, require, sha, source_manifest

def validate(q, groups, budget: int) -> np.ndarray:
    require(budget in (4, 8) and type(budget) is int, 'Budget must be the frozen 4 or 8')
    q = np.asarray(q, dtype=np.float64)
    require(q.shape == (16, 16) and np.isfinite(q).all(), 'Q must be finite 16x16')
    # The recorded matrix is symmetric. No correction/symmetrization is applied.
    require(np.array_equal(q, q.T), 'Q symmetry mismatch')
    require(len(groups) == 16 and all(len(g) == 192 for g in groups), 'Wrong grouping')
    require([x for g in groups for x in g] == list(range(3072)), 'Groups overlap or change channel order')
    return q

def replay(root: Path = ROOT) -> dict:
    manifest = source_manifest(root)
    selection = checked_read(root, C7+'/results/raw/selection.json', manifest)
    protocol = checked_read(root, C7+'/configs/protocol.json', manifest)
    source = root/C7/'src/core.py'
    require(sha(source) == manifest['files'][C7+'/src/core.py'], 'Changed selector source')
    require(selection['heldout_accessed'] is False, 'Selection used held-out data')
    require(len(set(selection['calibration_ids'])) == 96, 'Missing/duplicate calibration IDs')
    require(selection['protocol_sha256'] == sha(root/C7/'configs/protocol.json'), 'Protocol mismatch')
    spec = importlib.util.spec_from_file_location('showcase_frozen_case007_core', source)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    results = {}
    for budget in (4, 8):
        q = validate(selection['Q'], protocol['groups'], budget)
        for method in ('INDEPENDENT', 'PAIRWISE'):
            key = f'{method}_{budget}'
            actual = core.select(q, budget, method)
            expected = selection['selections'][key]
            require(actual['removed'] == expected['removed'], 'Removed-group mismatch: '+key)
            require(actual['enumerated'] == math.comb(16, budget), 'Enumeration mismatch')
            require(actual['objective'] == expected['objective'], 'Objective mismatch: '+key)
            results[key] = {**actual, 'matches_recorded': True,
                            'objective_definition': 'sum diagonal Q' if method == 'INDEPENDENT' else 'sum selected Q submatrix'}
    return {'status': 'PASS', 'evidence_kind': 'CPU_SELECTOR_REPLAY',
            'model_forwards': 0, 'selection_data': '96 CALIBRATION prompts only',
            'source_revision': manifest['evidence_revision'], 'selection_sha256': sha(root/C7/'results/raw/selection.json'),
            'tie_policy': 'minimum objective, then lexicographically smallest removal tuple', 'results': results}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True, help='New JSON outside repository')
    args = parser.parse_args()
    out = new_output(args.repo, args.output)
    result = replay(args.repo)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json_text(result)+'\n')
    print(json_text(result))

if __name__ == '__main__':
    main()
