"""Reproduce measured tables and figures in a separate output directory.

Publication-only renderer; not part of the frozen measurement code. Editorial
README/ANALYSIS prose is maintained separately. No result or decision is written.
"""
import argparse
import json
from pathlib import Path

CASE = Path(__file__).resolve().parents[1]


def dev_table(result):
    lines = ['| Setting | Median error | p95 error | Complete speedup [95% CI] | DEV outcome |',
             '| --- | --- | --- | --- | --- |']
    labels = {'B': 'B: fused BF16', 'A_PUBLIC': 'A_PUBLIC: FP8, fp32+fp16',
              'V1': 'V1: FP8, fp32', 'V2': 'V2: FP8, fp32+fp32', 'V4': 'V4: FP16 PV, fp32'}
    for row in result['rows']:
        if row['speedup'] is None:
            lines.append('| V3: FP8, fp32, smooth V | Not measured in DEV | Not measured in DEV | Not measured in DEV | Excluded at validity gate |')
            continue
        outcome = 'Control' if row['config_id'] == 'B' else 'Error limits exceeded' if row['speedup'] >= 1.5 else 'Error and speed limits missed'
        lines.append(f"| {labels[row['config_id']]} | {100*row['numerical']['median']:.3f}% | {100*row['numerical']['p95']:.3f}% | {row['speedup']:.3f}× [{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}] | {outcome} |")
    return '\n'.join(lines) + '\n'


def render(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    (output / 'dev_table.md').write_text(dev_table(result))
    # Retain the original post-measurement plot layout and every measured point.
    for metric, limit in [('median', 1), ('p95', 3)]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for row in result['rows']:
            if row['speedup'] is None:
                continue
            cid = row['config_id']; x = row['speedup']; y = 100 * row['numerical'][metric]
            ax.scatter(x, y, s=45)
            offset = {'V1': (8, 12), 'V2': (-22, -20)}.get(cid, (8, 8))
            ax.annotate('Anchor' if cid == 'A_PUBLIC' else cid, (x, y), xytext=offset, textcoords='offset points')
        ax.axvline(1.5, color='gray', linestyle='--', label='DEV speedup gate 1.50×')
        ax.axhline(limit, color='red', linestyle='--', label=f'Local {metric} error limit {limit}%')
        ax.set(xlim=(.9, 2.4), ylim=(0, 2.85 if metric == 'median' else 5.5),
               xlabel='Paired complete-call wall speedup vs BF16', ylabel=f'{metric} relative output error (%)',
               title='DEV: 8 documents, L4096, layer 13, RTX 5080')
        ax.legend(fontsize=9, loc='upper left'); fig.tight_layout()
        fig.savefig(output / f'dev_{metric}_pareto.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(CASE):
        parser.error('Choose an output directory outside the case to preserve recorded files')
    result = json.loads((CASE / 'results/dev_summary.json').read_text())
    assert result['final_decision'] == 'STOP_DEV_SCREEN'
    render(result, args.output)


if __name__ == '__main__':
    main()
