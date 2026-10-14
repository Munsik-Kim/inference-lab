"""Independent stdlib readback: tool state, task denominators and paired counts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from prepare import inventory, write


def read(p):
    return json.loads(Path(p).read_text())


def first_object(text):
    decoder = json.JSONDecoder()
    for i, c in enumerate(text):
        if c == '{':
            try:
                x, _ = decoder.raw_decode(text[i:])
                if isinstance(x, dict):
                    return x
            except ValueError:
                pass
    return None


def score(row, task, schema):
    config = dict(task['fixture']['config'])
    success_validate = False
    names = set(schema)
    errors = unknown = 0
    pending = []
    responses = []
    for event in row['events']:
        if 'output_token_ids' in event:
            if pending:
                # A hard tool budget may leave the remaining requested calls
                # unexecuted. Only this declared limit allows partial execution.
                if row['tool_calls'] != 6:
                    raise ValueError('response calls missing tool events')
                pending = []
            responses.append(event)
            if len(event['output_token_ids']) != event['output_tokens']:
                raise ValueError('output token count mismatch')
            for chunk in event['raw_text'].split('<tool_call>')[1:]:
                x = first_object(chunk.split('</tool_call>')[0])
                if x is not None:
                    pending.append(x)
            continue
        if 'tool' not in event:
            raise ValueError('unknown event type')
        if not pending:
            raise ValueError('tool was executed without a model call')
        obj = pending.pop(0)
        name, args, answer = event['tool'], event['arguments'], event['result']
        if obj.get('name') != name or obj.get('arguments') != args:
            raise ValueError('tool name/args silently corrected')
        bad = name not in names or not isinstance(args, dict)
        if not bad:
            bad = set(args) != set(schema[name]['required'])
        if not bad and name == 'calculate':
            bad = any(type(args[k]) not in (int, float) or not math.isfinite(args[k]) for k in ('a', 'b'))
        if not bad and name == 'apply_fixture_patch':
            bad = type(args['enabled']) is not bool or type(args['limit']) is not int
        unknown += name not in names
        if bad:
            errors += 1
            if 'error' not in answer:
                raise ValueError('invalid tool falsely executed')
            if row['condition'] in ('FEEDBACK', 'BOTH'):
                if name not in names and (answer.get('error') != 'UNKNOWN_TOOL_NAME' or answer.get('allowed_tool_names') != sorted(names)):
                    raise ValueError('feedback registry mismatch')
            continue
        if name == 'read_document':
            expected = task['fixture']['document']
        elif name == 'read_record':
            expected = task['fixture']['record']
        elif name == 'query_table':
            expected = task['fixture']['updates']
        elif name == 'calculate':
            expected = {'product': args['a'] * args['b']}
        elif name == 'apply_fixture_patch':
            config = dict(args); expected = {'applied': True, 'config': config}
        else:
            expected = {'valid': config == task['fixture']['expected_config']}
            success_validate |= bool(expected['valid'])
        if expected != answer:
            raise ValueError('incorrect or fabricated tool result')
    if config != row['final_config'] or errors != row['tool_errors'] or unknown != row['unregistered_name_events']:
        raise ValueError('state/error denominator mismatch')
    if len(responses) != row['model_calls'] or sum(e['input_tokens'] for e in responses) != row['total_input_tokens'] or sum(e['output_tokens'] for e in responses) != row['total_output_tokens']:
        raise ValueError('request/token denominator mismatch')
    if row['condition'] in ('ENUM', 'BOTH') and unknown:
        raise ValueError('enum name contract not met')
    last = responses[-1]
    final = first_object(last['text'])
    final_has_calls = any(first_object(s.split('</tool_call>')[0]) is not None for s in last['raw_text'].split('<tool_call>')[1:])
    successful = bool(final and not final_has_calls and all(final.get(k) == v for k, v in task['gold'].items()))
    if task['family'] == 'fixture_patch':
        successful = successful and config == task['fixture']['expected_config'] and success_validate
    if row['status'] in ('TIMEOUT', 'CONTEXT_LIMIT', 'CALL_LIMIT', 'TOOL_LIMIT', 'PARSE_ERROR'):
        successful = False
    if successful != row['success']:
        raise ValueError('original success contract mismatch: ' + row['id'])
    return successful


def run(result, input_directory=None, output=None, skip_external_preservation=False):
    reg = read(result / 'preregistration.json'); runtime = read(result / 'runtime_manifest.json')
    original = Path(runtime['original_result_root'])
    source_inputs = Path(input_directory) if input_directory else original / 'inputs'
    tasks = {t['id']: t for t in read(source_inputs / 'agent_tasks.json')}
    schema = {t['function']['name']: t['function']['parameters'] for t in read(source_inputs / 'tool_schema.json')}
    records = []; receipts = []
    for method, conditions in reg['methods'].items():
        dirs = [p for p in (result / 'workers' / method).glob('attempt*') if (p / 'receipt.json').exists()]
        if len(dirs) != 1:
            raise ValueError('need exactly one successful worker for ' + method)
        out = dirs[0]; receipts.append(read(out / 'receipt.json'))
        rows = [json.loads(line) for line in (out / 'episodes.jsonl').read_text().splitlines()]
        expected = {(method, c, t) for c in conditions for t in tasks}
        actual = [(r['method'], r['condition'], r['id']) for r in rows]
        if len(set(actual)) != len(actual) or set(actual) != expected:
            raise ValueError('missing/duplicate/cohort mismatch')
        for row in rows:
            score(row, tasks[row['id']], schema)
        records += rows
    summaries = []
    pairs = []
    for method, conditions in reg['methods'].items():
        baseline = {r['id']: r for r in records if r['method'] == method and r['condition'] == 'BASELINE'}
        for condition in conditions:
            selected = [r for r in records if r['method'] == method and r['condition'] == condition]
            if any(r['first_input_hash'] != baseline[r['id']]['first_input_hash'] for r in selected):
                raise ValueError('paired initial input mismatch')
            summaries.append({'method': method, 'condition': condition, 'successes': sum(r['success'] for r in selected),
                'tasks': len(selected), 'unregistered_name_events': sum(r['unregistered_name_events'] for r in selected),
                'tool_errors': sum(r['tool_errors'] for r in selected), 'parse_errors': sum(r['parse_errors'] for r in selected),
                'model_requests': sum(r['model_calls'] for r in selected),
                'forced_enum_changes': sum(t['forced_change'] for r in selected for e in r['events'] if 'enum_trace' in e for t in e['enum_trace']),
                'failed_task_ids': [r['id'] for r in selected if not r['success']],
                'original_first_output_matches': sum(r['first_output_equals_original'] is True for r in selected)})
            if condition == 'BASELINE':
                continue
            counts = {'both_success': 0, 'baseline_only': 0, 'candidate_only': 0, 'both_failed': 0}
            for r in selected:
                a, b = baseline[r['id']]['success'], r['success']
                counts['both_success' if a and b else 'baseline_only' if a else 'candidate_only' if b else 'both_failed'] += 1
            pairs.append({'method': method, 'baseline': 'BASELINE', 'candidate': condition, 'n_tasks': len(selected), **counts})
    protected = read(result / 'protected_before.json')
    identity = read(result / 'identity.json'); repo = Path(identity['repo']); research = Path(identity['research'])
    current = {}
    if not skip_external_preservation:
        current = {'cases': inventory(repo / 'cases'), 'original_r2': inventory(original, private=True),
            'prior_followup': inventory(research / 'results/diova_agent_followup_r1', private=True),
            'original_source': inventory(research / 'source'),
            'frozen_followup_source': inventory(repo / 'research/agent_stability_20261013')}
        changed = {k: [p for p in set(protected[k]) | set(current[k]) if protected[k].get(p) != current[k].get(p)] for k in protected}
        if any(changed.values()):
            raise ValueError('protected originals changed: ' + str(changed))
    commands = read(result / 'gpu_commands.json')
    if any(c['exit_code'] for c in commands):
        process_status = 'COMPLETED_WITH_RETAINED_FAILED_ATTEMPTS'
    else:
        process_status = 'ALL_PROCESSES_SUCCEEDED'
    totals = {'episodes': len(records), 'unique_previously_observed_tasks': len(tasks),
        'logical_conditions': len(summaries), 'GPU_workers': len(commands),
        'generation_requests': sum(r['model_calls'] for r in records),
        'worker_wall_seconds': sum(c['wall_seconds'] for c in commands),
        'peak_allocated_bytes': max(r['peak']['allocated'] for r in receipts),
        'peak_reserved_bytes': max(r['peak']['reserved'] for r in receipts)}
    audit = {'status': 'PASS', 'checked': ['complete rows/no duplicates', 'raw names/args unchanged', 'tool execution and final config replay',
        'original success contract', 'pairing and initial inputs', 'token/request denominators', 'protected files'],
        'process_status': process_status, 'totals': totals, 'preservation': {k: len(v) for k, v in current.items()},
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'logits_or_weight_full_recompute': False,
        'cases001_013_preserved': None if skip_external_preservation else True,
        'external_original_preservation': 'NOT_RUN_EXTERNAL_ORIGINALS' if skip_external_preservation else 'PASS'}
    target = Path(output) if output else result
    if output:
        target.mkdir(parents=True, exist_ok=False)
    write(target / 'summary.json', {'totals': totals, 'conditions': summaries, 'paired': pairs})
    write(target / 'verification.json', audit)
    d16 = next(r for r in summaries if r['method'] == 'D16' and r['condition'] == 'BOTH')
    write(target / 'decision.json', {'EXPERIMENT': 'COMPLETED', 'ENUM_TOOL_NAME_CONTRACT': 'PASS',
        'D16_EIGHT_DEV_TASKS': 'ALL_PASSED' if d16['successes'] == 8 else 'RESIDUAL_FAILURES',
        'ORIGINAL_CASE013_R2': 'PRESERVED', 'FORMAL_SERVING_BENEFIT': 'NOT_ESTABLISHED',
        'generalization': 'not assessed; eight selected existing DEV tasks', 'quantizer_change': False})
    with (target / 'summary.csv').open('w', newline='') as f:
        keys = [k for k in summaries[0] if k != 'failed_task_ids']; writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows({k: r[k] for k in keys} for r in summaries)
    print(json.dumps({'status': 'PASS', 'totals': totals, 'conditions': summaries}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--result', type=Path, required=True)
    p.add_argument('--inputs', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--skip-external-preservation', action='store_true', help='bundle-only replay; does not certify inaccessible originals')
    args = p.parse_args()
    run(args.result, args.inputs, args.output, args.skip_external_preservation)
