"""Score the saved MAIN images using the supplemental, single-AI annotations.

This reads the original study and never writes analysis/summary.json. Run the
independent audit.py after regenerating this supplement.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.analyze import quality

PROTOCOL_SHA256 = '4f755d9b88ceb2e6d3bc99c2c0306419b8309a88e1e434ff5dd35c68346642e6'
ANNOTATIONS_SHA256 = '12367bad474de2923739c4702b50e5ea67d0d258b19e768ec45ef002f66e85e6'
SETTINGS = ('A_time', 'B_time', 'C_time')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def calculate(root=ROOT):
    root = Path(root)
    folder = root / 'publication/assessment_v1'
    if sha(folder / 'protocol.json') != PROTOCOL_SHA256:
        raise ValueError('Frozen supplemental protocol identity changed')
    if sha(folder / 'annotations.json') != ANNOTATIONS_SHA256:
        raise ValueError('Completed annotation identity changed; corrections need a new version')
    protocol = read(folder / 'protocol.json')
    manifest = read(folder / 'input_manifest.json')
    annotations = read(folder / 'annotations.json')
    if sha(folder / 'input_manifest.json') != protocol['input_manifest_sha256']:
        raise ValueError('Masked assessment inventory changed')
    if annotations['assessment_protocol_sha256'] != PROTOCOL_SHA256:
        raise ValueError('Annotation/protocol mismatch')
    if annotations['evaluator'] != protocol['evaluator']:
        raise ValueError('Evaluator identity mismatch')
    all_rows = read(root / 'analysis/record_index.json')['rows']
    main = [r for r in all_rows if r['job']['phase'] == 'main']
    expected_items = [i for i in read(root / 'analysis/annotation_items.json')['items'] if i['split'] == 'MAIN']
    original_items = {i['image_id']: i for i in expected_items}
    items = manifest['items']
    if len(main) != 192 or len(items) != 192 or len(annotations['rows']) != 192:
        raise ValueError('MAIN assessment requires every planned image')
    if {i['image_id'] for i in items} != set(original_items):
        raise ValueError('Assessment changed the MAIN cohort')
    for index, (item, annotation) in enumerate(zip(items, annotations['rows']), 1):
        if {k: v for k, v in item.items() if k != 'assessment_index'} != original_items[item['image_id']]:
            raise ValueError('Assessment changed an original prompt/checklist/image')
        if annotation['image_id'] != item['image_id'] or item['assessment_index'] != index or annotation['assessment_index'] != index:
            raise ValueError('Assessment order or identity mismatch')
        if annotation.get('inspected_original_pixels') is not True or not annotation.get('visible_evidence'):
            raise ValueError('Missing actual inspection evidence')
        if sha(root / item['path']) != item['image_sha256']:
            raise ValueError('Original PNG identity mismatch')
    rubric_hash = sha(root / 'configs/rubric.json')
    q = quality(main, annotations, items, rubric_hash)
    if not q['coverage']['complete']:
        raise ValueError('Incomplete MAIN annotations')
    label_lookup = {r['image_id']: r for r in annotations['rows']}
    for setting in SETTINGS:
        rr = [r for r in main if r['job']['setting']['id'] == setting]
        counts = Counter(v for r in rr for v in label_lookup[r['job_id']]['values'].values())
        q['settings'][setting].update({
            'passed_images': sum(q['per_image'][r['job_id']]['pass'] for r in rr),
            'constraint_labels': dict(sorted(counts.items())),
            'uncertain_images': sum('uncertain' in label_lookup[r['job_id']]['values'].values() for r in rr),
        })
    for image_id, data in q['per_image'].items():
        data['visible_evidence'] = label_lookup[image_id]['visible_evidence']
        data['assessment_index'] = label_lookup[image_id]['assessment_index']
    paths = [
        'configs/protocol.json', 'configs/prompts.json', 'configs/rubric.json',
        'configs/main_settings.json', 'analysis/record_index.json',
        'analysis/annotation_items.json', 'analysis/summary.json',
        'analysis/analyze.py', 'source/contracts.py',
        'publication/assessment_v1/protocol.json',
        'publication/assessment_v1/input_manifest.json',
        'publication/assessment_v1/annotations.json',
        'publication/assessment_v1/analyze.py',
    ]
    return {
        'schema': 'case012-model-assisted-summary-v1',
        'status': 'MODEL_ASSISTED_EVALUATION_COMPLETE',
        'scope': 'Supplemental scoring of saved MAIN PNGs; original human-pending summary preserved',
        'evaluator': annotations['evaluator'], 'human_evaluation': 'NOT_RUN',
        'new_generations': 0, 'new_training': 0,
        'source_hashes': {p: sha(root / p) for p in paths},
        'quality': q,
        'timing': read(root / 'analysis/summary.json')['timing'],
        'limitations': {
            'time_matching': 'DEV +/-5% target missed; not an exact equal-time result',
            'masking': protocol['masking'],
            'uncertainty': protocol['primary']['interpretation'],
            'sampling': '16 prompts, four noise seeds each, four related template families',
            'scoring_scope': 'Explicit frozen constraints only; not every prose requirement, aesthetics or official GenEval',
        },
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, help='New external JSON; never overwrites an existing file')
    args = parser.parse_args()
    summary = calculate(args.root)
    if args.output:
        if args.output.exists():
            raise ValueError('Output must be a new file')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n')
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
