"""Independent stdlib readback of saved diagnostic/intervention records.

No model loading, tensors, probability distributions or additional inference.
The existing diova-compare scorer/validator preserves the action contract.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from contracts import read, rows, sha, token_hash, worker_dir, METHODS, TASK_IDS
from diova_compare.case013 import canonical, digest, score_agent, Reader
from diova_compare.core import validate, compare


def audit(result, case=None):
    result = Path(result)
    owned = result / 'owned_inputs.json'
    if not owned.exists():
        if case is None:
            raise ValueError('first readback requires --case; public bundle then keeps owned inputs')
        reader = Reader(case)
        tasks = reader.json('records/inputs/agent_tasks.json')
        tokens = reader.json('records/inputs/agent_token_manifest.json')
        historical = {}
        for method in METHODS:
            historical[method] = {}
            for record in reader.rows('records/raw/4b/' + method + '/agent_dev.jsonl'):
                if record['mode'] == 'FREE' and record['id'] in TASK_IDS:
                    historical[method][record['id']] = next(e for e in record['events'] if 'text' in e)
        owned.write_text(json.dumps({'tasks': tasks, 'tokens': tokens, 'historical_first': historical,
                                    'source_identity': reader.used}, indent=2) + '\n')
    owned_data = read(owned)
    tasks = {r['id']: r for r in owned_data['tasks']}
    input_hashes = {r['id']: r['initial_token_sha256'] for r in owned_data['tokens']}
    registration = read(result / 'preregistration.json')
    runtime = read(result / 'runtime_manifest.json')
    inputs = read(result / 'input_manifest.json')
    if sha(result / 'input_manifest.json') != registration['input_manifest_sha256']:
        raise ValueError('registered input bytes changed')
    natural = {r['task_id']: r for r in inputs['natural']}
    for row in natural.values():
        if token_hash(row['input_ids']) != row['input_hash'] or row['input_hash'] != input_hashes[row['task_id']]:
            raise ValueError('initial token bytes mismatch')
    commands = read(result / 'gpu_commands.json')
    if len(commands) != 9 or any(r['exit_code'] != 0 for r in commands):
        raise ValueError('missing/failed GPU worker commands')
    wall = math.fsum(r['worker_wall_seconds'] for r in commands)
    if wall > 7200:
        raise ValueError('registered worker budget exceeded')
    findings = {'status': 'PASS', 'scope': 'Saved input bytes, extracted/action correctness, paired counts and logprob reductions; not regeneration of logits/weights',
                'audit_code_sha256': sha(Path(__file__)), 'new_GPU_runs_in_audit': 0,
                'P1': {}, 'controlled': [], 'common': [], 'P2': {},
                'GPU_workers': len(commands), 'GPU_worker_wall_seconds': wall,
                'GPU_budget_seconds': 7200, 'generation_requests': 24,
                'head_parity_checks': [], 'artifact_readbacks': []}
    first_by_method = {}
    for method in METHODS:
        first, common = worker_dir(result, 'first', method), worker_dir(result, 'common', method)
        responses = rows(first / 'first_responses.jsonl')
        if len(responses) != 4 or {r['task_id'] for r in responses} != set(TASK_IDS):
            raise ValueError('natural response inventory')
        first_by_method[method] = {r['task_id']: r for r in responses}
        reproduction = []
        for row in responses:
            if row['input_token_sha256'] != natural[row['task_id']]['input_hash']:
                raise ValueError('natural response prompt mismatch')
            old = owned_data['historical_first'][method][row['task_id']]
            reproduction.append({'task_id': row['task_id'], 'output_token_ids_equal_original': row['output_token_ids'] == old['output_token_ids'],
                                 'tool_names': [c.get('name') for c in row['parsed_tool_calls']]})
        recovery = rows(first / 'error_recovery.jsonl')
        if len(recovery) != 2 or {r['task_id'] for r in recovery} != {'AGENT-6', 'AGENT-7'}:
            raise ValueError('recovery inventory')
        probes = {r['task_id']: r for r in inputs['recovery']}
        for row in recovery:
            if row['input_token_sha256'] != probes[row['task_id']]['input_hash']:
                raise ValueError('recovery input mismatch')
        findings['P1'][method] = {'reproduction': reproduction,
            'recovery_registered_only': sum(r['registered_names_only'] for r in recovery),
            'recovery_repeats_wrong': sum(r['repeats_original_wrong_name'] for r in recovery), 'recovery_n': 2}
        scores = rows(first / 'controlled_name_scores.jsonl')
        if len(scores) != 4:
            raise ValueError('controlled-name inventory')
        for row in scores:
            prefix = row['prefix_ids']
            if token_hash(prefix) != row['prefix_hash']:
                raise ValueError('controlled prefix hash')
            matching = next(p for p in inputs['controlled'] if p['task_id'] == row['task_id'] and p['correct_name'] == row['correct_name'])
            if prefix != matching['prefix_ids'] or row['candidates'] != matching['candidates']:
                raise ValueError('controlled tokenization mismatch')
            a, b = row['candidates']; lp = row['next_token']['logprob']
            margin = lp[str(a['first_different_token'])] - lp[str(b['first_different_token'])]
            if margin != row['first_token_correct_minus_wrong_margin']:
                raise ValueError('stored margin mismatch')
            for candidate in row['candidates']:
                seq = row['sequence_scores'][candidate['name']]
                values = seq['token_logprobs']
                if len(values) != len(candidate['continuation_ids']) or not all(math.isfinite(v) for v in values) \
                        or abs(math.fsum(values) - seq['total_logprob']) > 1e-10 \
                        or abs(math.fsum(values) / len(values) - seq['mean_logprob']) > 1e-10:
                    raise ValueError('name score reduction mismatch')
            correct = row['sequence_scores'][row['correct_name']]
            wrong = row['sequence_scores'][row['wrong_name']]
            findings['controlled'].append({'method': method, 'task_id': row['task_id'], 'tool': row['correct_name'],
                'first_token_correct_minus_wrong_margin_nats': margin,
                'total_name_plus_quote_logprob_correct': correct['total_logprob'],
                'total_name_plus_quote_logprob_wrong': wrong['total_logprob'],
                'total_correct_minus_wrong_nats': correct['total_logprob'] - wrong['total_logprob'],
                'correct_length': correct['length'], 'wrong_length': wrong['length'],
                'quote_boundary_penalty_correct_nats': correct['token_logprobs'][-1],
                'quote_boundary_penalty_wrong_nats': wrong['token_logprobs'][-1]})
        probes = [p for p in read(result / 'common_prefix_input_manifest.json') if p['status'] == 'DIVERGENCE']
        saved_common = rows(common / 'common_prefix_scores.jsonl')
        if len(saved_common) != len(probes):
            raise ValueError('common-prefix score inventory')
        for row in saved_common:
            probe = next(p for p in probes if all(p[k] == row[k] for k in ('task_id', 'baseline', 'candidate')))
            if token_hash(probe['prefix_ids']) != row['prefix_hash']:
                raise ValueError('common prefix hash')
            findings['common'].append({k: row[k] for k in
                ('method', 'task_id', 'baseline', 'candidate', 'common_response_token_count', 'baseline_token_minus_candidate_margin')})
        for phase, directory in [('first', first), ('common', common)]:
            parity = read(directory / 'head_parity.json')
            if parity['status'] != 'PASS' or not parity['finite'] or parity['max_abs_difference'] > .25 or parity['argmax_mismatches'] != 0:
                raise ValueError('head parity failed')
            findings['head_parity_checks'].append({'phase': phase, 'method': method, **parity})
            readback = read(directory / 'artifact_readback.json')
            if readback['new_solver_calls'] != 0 or readback['artifact_writes'] != 0:
                raise ValueError('forbidden artifact changes')
            artifact_method = 'RTN' if method == 'RTN' else 'D16'
            expected = runtime['artifacts'][artifact_method]['per_projection_identity'] if method != 'BF16' else []
            actual = [{k: x[k] for k in ('layer', 'shape', 'bytes', 'sha256')} for x in readback['artifacts']]
            if actual != expected:
                raise ValueError('artifact identity readback mismatch')
            findings['artifact_readbacks'].append({'phase': phase, 'method': method, 'files': len(actual), 'status': 'PASS'})
    p2 = worker_dir(result, 'p2', 'D16')
    decision = read(p2 / 'p2_decision.json')
    if decision['P2'] != 'COMPLETED' or not read(p2 / 'I0_gate.json')['passed']:
        raise ValueError('P2 gate/complete status mismatch')
    selection = read(result / 'p2_selection.json')
    error_rows = {key: read(worker_dir(result, 'first', 'D16') / ('projection_errors_' + key + '.json')) for key in ('AGENT-6', 'AGENT-7')}
    eligible = {n: math.fsum(error_rows[key][n]['difference_sse'] / error_rows[key][n]['baseline_energy']
                            for key in ('AGENT-6', 'AGENT-7')) / 2 for n in error_rows['AGENT-6']
                if all(error_rows[key][n]['baseline_energy'] > 0 for key in ('AGENT-6', 'AGENT-7'))}
    target = sorted(eligible, key=lambda n: (-eligible[n], n))[0]
    if target != selection['target'] or abs(eligible[target] - selection['mean_relative_output_error']) > 1e-12:
        raise ValueError('target selection criterion mismatch')
    entries = {x['layer']: x for x in runtime['artifacts']['D16']['per_projection_identity']}
    t, c = entries[target], entries[selection['control']]
    if t['shape'] != c['shape'] or t['bytes'] != c['bytes'] or target.split('.')[-1] != c['layer'].split('.')[-1]:
        raise ValueError('matched control type/shape/payload mismatch')
    import random
    possible = sorted(n for n, r in entries.items() if n != target and n.split('.')[-1] == target.split('.')[-1]
                      and n.split('.')[2] != target.split('.')[2] and r['shape'] == t['shape'] and r['bytes'] == t['bytes'])
    if random.Random(610132).choice(possible) != selection['control']:
        raise ValueError('matched control fixed RNG mismatch')
    records = rows(p2 / 'p2_interventions.jsonl')
    if len(records) != 24 or len({(r['condition'], r['id']) for r in records}) != 24:
        raise ValueError('P2 missing/duplicate records')
    by_condition = {}; exports = {}
    findings['peak_allocated_bytes'] = max(read(worker_dir(result, r['stage'], r['method']) / 'receipt.json')['peak_allocated_bytes'] for r in commands)
    findings['peak_reserved_bytes'] = max(read(worker_dir(result, r['stage'], r['method']) / 'receipt.json')['peak_reserved_bytes'] for r in commands)
    p2_exit = next(r['exit_code'] for r in commands if r['stage'] == 'p2')
    for condition in ('I0', 'I1', 'I2'):
        selected = [r for r in records if r['condition'] == condition]
        if {r['id'] for r in selected} != set(tasks):
            raise ValueError('P2 task IDs mismatch')
        by_condition[condition] = {r['id']: r for r in selected}
        module = read(p2 / (condition + '-module_identity.json'))
        expected_change = [] if condition == 'I0' else [selection['target'] if condition == 'I1' else selection['control']]
        if module['changed_modules'] != expected_change or not module['all_other_artifacts_identical']:
            raise ValueError('module intervention scope mismatch')
        identity = digest({'D16': runtime['artifacts']['D16']['manifest_sha256'], 'reference_intervention': module,
                           'registered_source': registration['followup_code_hashes']})
        exports[condition] = []
        for record in selected:
            parsed, correct, first, execution = score_agent(record, tasks[record['id']], record['final_config'])
            if first['input_token_sha256'] != input_hashes[record['id']] \
                    or record['final_config'] != read(p2 / condition / (record['id'] + '-FREE-config.json')):
                raise ValueError('P2 first input/final file mismatch')
            findings['generation_requests'] += record['model_calls']
            gold = canonical(tasks[record['id']]['gold'])
            exports[condition].append(dict(schema_version=1, sample_id=record['id'], task='case013.followup.P2.original8.FREE',
                task_version='diagnostic-dev-v1:' + sha(result / 'preregistration.json'),
                prompt_hash=digest({'first_token_hash': first['input_token_sha256'], 'generation': registration['generation'],
                                    'tool_schema': inputs['tool_schema'], 'scorer': 'first-json-and-fixture-actions-v1'}),
                gold=[gold], gold_definition='first-json-and-fixture-actions-v1',
                model_id=runtime['model']['id'] + '@' + runtime['model']['revision'],
                artifact_id='dequantized-D16-reference:' + condition + ':' + identity,
                output_type='extracted_answer', answer=canonical(parsed) if parsed is not None else '[PARSE_FAILURE]',
                correct=correct, evidence_kind='measurement', process_exit_code=p2_exit))
        validate(exports[condition])
        findings['P2'][condition] = {'n': 8, 'successes': sum(r['success'] for r in selected),
            'successful_ids': [r['id'] for r in selected if r['success']],
            'failed_ids': [r['id'] for r in selected if not r['success']],
            'model_requests': sum(r['model_calls'] for r in selected),
            'wall_seconds_all_episodes': math.fsum(r['wall_seconds'] for r in selected), 'module': module}
    for condition in ('I1', 'I2'):
        report = compare(exports['I0'], exports[condition], replicates=2000, seed=909220)
        counts = next(iter(report['tasks'].values()))['counts']
        manual = {'both_correct': 0, 'correct_to_wrong': 0, 'wrong_to_correct': 0, 'both_wrong': 0}
        for key in tasks:
            a, b = by_condition['I0'][key]['success'], by_condition[condition][key]['success']
            name = 'both_correct' if a and b else 'correct_to_wrong' if a else 'wrong_to_correct' if b else 'both_wrong'
            manual[name] += 1
        if any(counts[k] != v for k, v in manual.items()):
            raise ValueError('CPU comparison engine/independent paired counts mismatch')
        findings['P2'][condition]['paired_vs_I0'] = manual
    export_dir = result / 'p2_compare_inputs'
    if not export_dir.exists():
        export_dir.mkdir()
        for condition, records in exports.items():
            (export_dir / (condition + '.jsonl')).write_text(''.join(canonical(r) + '\n' for r in records))
    else:
        for condition, records in exports.items():
            if rows(export_dir / (condition + '.jsonl')) != records:
                raise ValueError('stale diagnostic comparison export')
    findings['P2_selection'] = selection
    findings['projection_SSE_scope'] = 'Recorded last-position reductions; ordering recomputed, excluded activation arrays not reconstructed by stdlib auditor'
    return findings


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--case', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise SystemExit('choose fresh audit output')
    answer = audit(a.result, a.case)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(answer, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: answer[k] for k in ('status', 'GPU_workers', 'GPU_worker_wall_seconds', 'generation_requests')}))
