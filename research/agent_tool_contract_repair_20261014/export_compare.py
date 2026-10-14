"""Export checked new episodes to the existing diova-compare schema."""
import argparse
import hashlib
import json
from pathlib import Path
from analyze import first_object


def main(result):
    if json.loads((result / 'verification.json').read_text())['status'] != 'PASS':
        raise ValueError('independent audit required')
    runtime = json.loads((result / 'runtime_manifest.json').read_text())
    reg = json.loads((result / 'preregistration.json').read_text())
    target = result / 'compare_inputs'; target.mkdir(exist_ok=False)
    task_version = 'original8-mitigation-dev-v1:' + hashlib.sha256((result / 'input_manifest.json').read_bytes()).hexdigest()
    for method, conditions in reg['methods'].items():
        paths = [p for p in (result / 'workers' / method).glob('attempt*') if (p / 'receipt.json').exists()]
        if len(paths) != 1:
            raise ValueError('ambiguous successful attempt')
        rows = [json.loads(s) for s in (paths[0] / 'episodes.jsonl').read_text().splitlines()]
        for condition in conditions:
            exported = []
            for row in rows:
                if row['condition'] != condition:
                    continue
                response = [e for e in row['events'] if 'output_token_ids' in e][-1]
                parsed = first_object(response['text'])
                canonical = lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                am = 'RTN' if method == 'RTN' else 'D16'
                artifact = runtime['artifacts'][am]['manifest_sha256'] if method != 'BF16' else runtime['model']['revision']
                exported.append({'schema_version': 1, 'sample_id': row['id'],
                    'task': 'case013.tool-contract.original8.FREE', 'task_version': task_version,
                    'prompt_hash': row['first_input_hash'], 'gold': [canonical(row['gold'])],
                    'gold_definition': 'first-json-and-fixture-actions-v1',
                    'model_id': runtime['model']['id'] + '@' + runtime['model']['revision'],
                    'artifact_id': method + ':external-artifact-identity=' + artifact + ':decoder=' + condition,
                    'output_type': 'extracted_answer', 'answer': canonical(parsed) if parsed is not None else '__PARSE_FAILURE__',
                    'correct': row['success'], 'evidence_kind': 'measurement', 'process_exit_code': 0})
            with (target / (method + '-' + condition + '.jsonl')).open('x') as f:
                for row in exported:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
    (target / 'SCOPE.md').write_text('Same eight previously observed DEV tasks. Initial prompts are paired; subsequent histories differ by the registered decoder/feedback intervention. Extracted-answer correctness retains actual fixture action requirements. Any CLI bootstrap is descriptive, not held-out superiority or noninferiority. Source events and action evidence remain in workers/. External artifact identity is not a newly hashed base weight file.\n')
    print(json.dumps({'status': 'EXPORTED', 'files': sum(1 for _ in target.glob('*.jsonl')), 'records': sum(1 for p in target.glob('*.jsonl') for _ in p.open())}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--result', type=Path, required=True)
    main(p.parse_args().result)
