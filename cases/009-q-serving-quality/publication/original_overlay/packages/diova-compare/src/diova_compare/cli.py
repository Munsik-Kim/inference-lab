"""Console interface with fresh output directories."""
import argparse
import csv
import json
from pathlib import Path
from .core import compare, load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    v = sub.add_parser('validate'); v.add_argument('--input', type=Path, required=True)
    c = sub.add_parser('compare')
    c.add_argument('--baseline', type=Path, required=True); c.add_argument('--candidate', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True); c.add_argument('--intersection', action='store_true')
    c.add_argument('--replicates', type=int, default=2000); c.add_argument('--seed', type=int, default=909220)
    a = p.parse_args()
    try:
        if a.command == 'validate':
            print(json.dumps({'status': 'PASS', 'records': len(load(a.input))})); return
        result = compare(load(a.baseline), load(a.candidate), a.intersection, a.replicates, a.seed)
        a.output.mkdir(parents=True, exist_ok=False)
        (a.output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        fields = sorted(set().union(*(r.keys() for r in result['pairs'])))
        with (a.output / 'pairs.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(result['pairs'])
        lines = ['# Paired comparison', '', 'Paired items within each task. Intervals are pointwise descriptive; separate benchmarks are not averaged.', '']
        for task, r in result['tasks'].items():
            lines += [f'## {task}', '', f"Pairs: {r['n_pairs']}; evidence kinds: {', '.join(sorted({p['evidence_kind'] for p in result['pairs'] if p['task'] + '@' + p['task_version'] == task}))}.", '', '```json', json.dumps(r, indent=2), '```', '']
        (a.output / 'report.md').write_text('\n'.join(lines))
        print(json.dumps({'status': 'PASS', 'n_pairs': result['n_pairs'], 'output': str(a.output)}))
    except (ValueError, OSError, KeyError, TypeError) as e:
        p.exit(2, 'diova-compare: ' + str(e) + '\n')


if __name__ == '__main__':
    main()
