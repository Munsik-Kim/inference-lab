"""Recalculate the user's MAIN-only JSON; never overwrite the original study."""
import argparse
import importlib.util
import json
from pathlib import Path
from common import ROOT, read, require, sha
from case012_pages import C12, annotation_data, load_case012


def score(root, annotations):
    data = load_case012(root); inputs = annotation_data(root, data)
    require(annotations.get('schema') == 'case012-annotations-v1'
            and annotations.get('scope') == 'MAIN_ONLY'
            and annotations.get('rubric_sha256') == inputs['rubric_sha256']
            and annotations.get('image_set_sha256') == inputs['image_set_sha256'], 'Annotation input identity mismatch')
    require(annotations.get('evaluator', {}).get('type') == 'human', 'Human labels required; AI scores are separate')
    require(isinstance(annotations.get('rows'), list) and isinstance(annotations.get('draft_rows'), list), 'Rows/drafts required')
    expected = {i['image_id']: i for i in inputs['items']}; seen = set()
    for rows, complete in [(annotations['rows'], True), (annotations['draft_rows'], False)]:
        for row in rows:
            image_id = row.get('image_id')
            require(image_id in expected and image_id not in seen, 'Unknown/duplicate annotation image')
            seen.add(image_id); item = expected[image_id]; values = row.get('values')
            require(row.get('image_sha256') == item['image_sha256'], 'Annotation image hash mismatch')
            require(isinstance(values, dict), 'Constraint dictionary required')
            keys = {c['id'] for c in item['constraints']}
            require(set(values) <= keys and all(v in ('satisfied','not_satisfied','uncertain') for v in values.values()), 'Unknown constraint/value')
            require(set(values) == keys if complete else 0 < len(values) < len(keys), 'Complete/draft constraint coverage mismatch')
    done = {r['image_id'] for r in annotations['rows']}; missing = set(expected)-done
    require(set(annotations.get('incomplete_image_ids', [])) == missing
            and len(annotations.get('incomplete_image_ids', [])) == len(missing), 'Annotation completeness mismatch')
    require(annotations.get('status') == ('PARTIAL' if missing else 'COMPLETE'), 'Annotation status mismatch')
    cursor = annotations.get('cursor')
    require(type(cursor) is int and 0 <= cursor < len(inputs['items'])
            and annotations.get('current_image_id') == inputs['items'][cursor]['image_id'], 'Annotation cursor mismatch')
    # Import the frozen model-free function; its CLI build() is never called.
    path = root/C12/'analysis/analyze.py'
    spec = importlib.util.spec_from_file_location('case012_saved_quality', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    records = [r for r in read(root/C12/'analysis/record_index.json')['rows'] if r['job']['phase'] == 'main']
    original_items = [i for i in read(root/C12/'analysis/annotation_items.json')['items'] if i['split'] == 'MAIN']
    quality = module.quality(records, annotations, original_items, inputs['rubric_sha256'])
    return {'schema':'case012-user-recalculation-v1', 'evidence_kind':'USER_HUMAN_ANNOTATIONS',
            'quality':quality, 'draft_images':len(annotations['draft_rows']),
            'image_set_sha256':inputs['image_set_sha256'],
            'source_hashes':data['sources'], 'new_generations':0,
            'published_AI_scores_modified':False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=ROOT)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New JSON file outside the repository')
    a = p.parse_args(); root = a.repo.resolve(); out = a.output.resolve()
    require(not out.is_relative_to(root) and not out.exists(), 'Output must be a new external file')
    result = score(root, read(a.annotations))
    result['annotation_file_sha256'] = sha(a.annotations)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f: f.write(text)
    print(json.dumps({'status':result['quality']['status'], 'coverage':result['quality']['coverage'], 'output':str(out)}))


if __name__ == '__main__': main()
