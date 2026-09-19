"""Split-separated local error levels from preserved norm statistics; CPU only."""
from pathlib import Path
import argparse
import json
import numpy as np


def main(case: Path, output: Path):
    if output.exists():
        raise ValueError('A new output file is required')
    raw = case / 'results/raw/R'
    load = lambda name: json.loads((raw / name).read_text())
    cal, dev, held = load('calibration_readout.json'), load('development.json'), load('model_records.json')
    eta = str(load('selection.json')['eta'])
    result = {'scope': 'Planned local mean/tail description; prompts are independent units', 'splits': {}}
    for split, expected in [('calibration', 128), ('development', 48), ('heldout', 192)]:
        result['splits'][split] = {}
        for structure in ['I25', 'P25', 'S50']:
            for repaired in [False, True]:
                arm = structure + ('-R' if repaired else '')
                if split == 'calibration':
                    rows = [(r['id'], r['local']['repaired' if repaired else 'uncorrected'])
                            for r in cal if r['structure'] == structure]
                elif split == 'development':
                    rows = [(r['id'], r['local'][structure][eta if repaired else 'uncorrected']) for r in dev]
                else:
                    rows = [(r['id'], r['local']) for r in held if r['arm'] == arm]
                if len(rows) != expected or len({i for i, _ in rows}) != expected:
                    raise ValueError('Duplicate or missing prompt')
                relative = np.array([v['relative_error'] for _, v in rows], dtype=np.float64)
                rms = np.array([v['absolute_rms'] for _, v in rows], dtype=np.float64)
                if not np.isfinite(relative).all() or not np.isfinite(rms).all():
                    raise ValueError('Invalid local output statistics')
                result['splits'][split][arm] = {
                    'n_prompts': expected, 'mean_relative': float(relative.mean()),
                    'median_relative': float(np.median(relative)), 'p95_relative': float(np.quantile(relative, .95)),
                    'mean_absolute_rms': float(rms.mean()), 'worst_prompt': rows[int(relative.argmax())][0],
                    'maximum_relative': float(relative.max()), 'unit': 'fraction of native teacher Frobenius norm'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.case, args.output)
