"""Independent CPU recalculation of AI quality labels and paired denominators.

Does not import the study's quality() or the supplemental calculate() function.
This checks arithmetic and identity, not the correctness of the AI's judgments.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
S = ('A_time', 'B_time', 'C_time')
VALID = {'satisfied', 'not_satisfied', 'uncertain'}


def read(path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recompute(records, annotations, items):
    """Independent scalar implementation, also usable by synthetic contracts."""
    source = {r['job_id']: r for r in records}
    item_map = {r['image_id']: r for r in items}
    labels = {r['image_id']: r for r in annotations['rows']}
    if len(source) != len(records) or len(item_map) != len(items) or len(labels) != len(annotations['rows']):
        raise ValueError('Duplicate image IDs')
    if set(source) != set(item_map) or set(source) != set(labels):
        raise ValueError('Missing or unknown image ID')
    scores = {}
    for iid, item in item_map.items():
        row = labels[iid]
        if row['image_sha256'] != item['image_sha256'] or source[iid]['image_sha256'] != item['image_sha256']:
            raise ValueError('Image hash mismatch')
        if set(row['values']) != {c['id'] for c in item['constraints']} or not row['values']:
            raise ValueError('Constraint coverage mismatch')
        if not set(row['values'].values()) <= VALID:
            raise ValueError('Invalid annotation value')
        vs = list(row['values'].values())
        scores[iid] = (int(all(v == 'satisfied' for v in vs)), vs.count('satisfied') / len(vs), int('not_satisfied' not in vs))
    mapping = {}
    for r in records:
        key = (r['job']['prompt_id'], r['job']['seed_label'], r['job']['setting']['id'])
        if key in mapping:
            raise ValueError('Duplicate prompt/seed/setting')
        if r['status'] != 'SUCCESS':
            raise ValueError('Failed output cannot be treated as a successful image')
        mapping[key] = r['job_id']
    prompts = sorted({k[0] for k in mapping})
    seeds = sorted({k[1] for k in mapping})
    if len(mapping) != len(prompts) * len(seeds) * len(S):
        raise ValueError('Incomplete paired settings')
    pairs = Counter({'both_pass': 0, 'A_only': 0, 'C_only': 0, 'neither': 0})
    differences = []
    gains, losses = Counter(), Counter()
    for prompt in prompts:
        per_seed = []
        for seed in seeds:
            a, c = mapping[prompt, seed, S[0]], mapping[prompt, seed, S[2]]
            x, y = scores[a][0], scores[c][0]
            pairs[('neither', 'C_only', 'A_only', 'both_pass')[2*x+y]] += 1
            per_seed.append(y-x)
            for key in labels[a]['values']:
                va = labels[a]['values'][key] == 'satisfied'
                vc = labels[c]['values'][key] == 'satisfied'
                k = prompt + '/' + key
                gains[k] += int(vc and not va)
                losses[k] += int(va and not vc)
        differences.append(math.fsum(per_seed) / len(seeds))
    rng = np.random.default_rng(72501)
    bootstrap = []
    # A separate loop gives the same frozen RNG draws as the original vectorized
    # implementation, without treating images or seeds as independent clusters.
    for _ in range(5000):
        ids = rng.integers(0, len(prompts), len(prompts))
        bootstrap.append(math.fsum(differences[i] for i in ids) / len(prompts))
    settings, categories = {}, {}
    for s in S:
        ids = [iid for key, iid in mapping.items() if key[2] == s]
        counts = Counter(v for iid in ids for v in labels[iid]['values'].values())
        settings[s] = {
            'images': len(ids), 'passed_images': sum(scores[i][0] for i in ids),
            'all_constraint_pass_rate': math.fsum(scores[i][0] for i in ids) / len(ids),
            'mean_constraint_fraction': math.fsum(scores[i][1] for i in ids) / len(ids),
            'all_uncertain_as_pass_rate': math.fsum(scores[i][2] for i in ids) / len(ids),
            'constraint_labels': dict(counts),
            'uncertain_images': sum('uncertain' in labels[i]['values'].values() for i in ids),
        }
    for cat in sorted({i['category'] for i in items}):
        categories[cat] = {}
        for s in S:
            ids = [iid for key, iid in mapping.items() if key[2] == s and item_map[iid]['category'] == cat]
            categories[cat][s] = {'n': len(ids), 'all_constraint_pass_rate': sum(scores[i][0] for i in ids) / len(ids)}
    return {
        'settings': settings, 'categories': categories, 'paired': dict(pairs),
        'constraint_gains': dict(gains), 'constraint_losses': dict(losses),
        'uncertain_constraint_count': sum(list(r['values'].values()).count('uncertain') for r in annotations['rows']),
        'primary': {'estimate': math.fsum(differences) / len(prompts),
                    'interval': np.quantile(bootstrap, [.025, .975], method='linear').tolist()},
    }


def audit(root=ROOT):
    root = Path(root)
    folder = root / 'publication/assessment_v1'
    summary = read(folder / 'summary.json')
    protocol = read(folder / 'protocol.json')
    annotations = read(folder / 'annotations.json')
    if summary['status'] != 'MODEL_ASSISTED_EVALUATION_COMPLETE' or summary['human_evaluation'] != 'NOT_RUN':
        raise ValueError('Assessment/human status mismatch')
    if summary['evaluator'] != annotations['evaluator'] or annotations['evaluator'] != protocol['evaluator']:
        raise ValueError('Evaluator metadata mismatch')
    if annotations['evaluator']['type'] != 'model-assisted' or annotations['evaluator']['human_raters'] != 0:
        raise ValueError('AI assessment cannot be reported as human labels')
    for path, expected in summary['source_hashes'].items():
        p = Path(path)
        if p.is_absolute() or '..' in p.parts or sha(root / p) != expected:
            raise ValueError('Assessment source identity mismatch: ' + path)
    if sha(folder / 'protocol.json') != annotations['assessment_protocol_sha256']:
        raise ValueError('Protocol link mismatch')
    if sha(folder / 'input_manifest.json') != protocol['input_manifest_sha256']:
        raise ValueError('Input manifest hash mismatch')
    items = read(folder / 'input_manifest.json')['items']
    original_items = [i for i in read(root / 'analysis/annotation_items.json')['items'] if i['split'] == 'MAIN']
    original_map = {i['image_id']: i for i in original_items}
    if len(items) != 192 or len(annotations['rows']) != 192:
        raise ValueError('Not all 192 MAIN images assessed')
    for n, (item, annotation) in enumerate(zip(items, annotations['rows']), 1):
        if {k:v for k,v in item.items() if k != 'assessment_index'} != original_map.get(item['image_id']):
            raise ValueError('Original checklist mismatch')
        if item['assessment_index'] != n or annotation['assessment_index'] != n or annotation['image_id'] != item['image_id']:
            raise ValueError('Actual assessment order mismatch')
        if not annotation['inspected_original_pixels'] or not annotation['visible_evidence']:
            raise ValueError('Missing inspection evidence')
        if sha(root / item['path']) != item['image_sha256']:
            raise ValueError('PNG hash mismatch')
    records = []
    for path in sorted((root / 'results/attempts').glob('*/attempt-*/record.json')):
        r = read(path)
        if r['job']['phase'] == 'main':
            records.append(r)
    calculated = recompute(records, annotations, items)
    expected = summary['quality']
    for key in ('settings', 'categories', 'paired', 'constraint_gains', 'constraint_losses', 'uncertain_constraint_count'):
        if calculated[key] != expected[key]:
            raise ValueError('Independent score mismatch: ' + key)
    for x, y in zip([calculated['primary']['estimate']] + calculated['primary']['interval'], [expected['primary']['estimate']] + expected['primary']['interval']):
        if not math.isclose(x, y, rel_tol=0, abs_tol=1e-12):
            raise ValueError('Independent paired bootstrap mismatch')
    if expected['primary']['prompt_clusters'] != 16 or protocol['primary']['template_family_count'] != 4:
        raise ValueError('Input uncertainty denominator changed')
    if read(root / 'analysis/summary.json')['quality']['primary'] is not None:
        raise ValueError('Historical pending summary was overwritten')
    return {'status': 'PASS', 'assessment': summary['status'], 'scored_main_images': len(records),
            'prompt_seed_pairs': 64, 'prompt_clusters': 16, 'human_raters': 0,
            'uncertain_constraints': calculated['uncertain_constraint_count'],
            'paired': calculated['paired'], 'primary': calculated['primary'],
            'annotations_sha256': sha(folder / 'annotations.json'), 'new_generations': 0,
            'scope': 'Independent arithmetic audit; no independent human or visual re-rating'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
