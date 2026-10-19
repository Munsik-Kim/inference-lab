"""CPU-only raw replay and descriptive tables; no inference or stored-score trust."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import tempfile

from state import P2, append, digest, inputs, read, replay_cell, write
from json_recovery_p2.upstream import parse_slots, extract


def focal_metrics(row, scenario):
    index = 1 if row['arm'] in ('N', 'E') else 0
    if len(row['responses']) <= index:
        return {'focal_observed': False, 'parse_success': None, 'arguments': []}
    response = row['responses'][index]
    slots = parse_slots(response['raw_text'])
    parsed = [s for s in slots if s['parsed'] is not None]
    parse_success = bool(parsed) if slots else extract(response['text']) is not None
    events = [e for e in row['tool_events'] if e['generated_in_turn'] == response['generated_in_turn']]
    target = scenario['target_order_id']; original = scenario['orders'][target]
    arguments = []
    for event in events:
        semantic = event['semantic_tool_id']; args = event['raw_arguments']
        key_type_ok = event['schema_result'] in ('VALID', 'UNKNOWN_ORDER')
        value = None
        if semantic is not None and key_type_ok:
            if semantic == 'amount.multiply':
                value = sorted([args['a'], args['b']]) == sorted([original['quantity'], original['unit_price_minor']])
            elif semantic == 'order.write':
                value = args['order_id'] == target and args['total_minor'] == original['quantity'] * original['unit_price_minor']
            else:
                value = args['order_id'] == target
        arguments.append({'event_id': event['event_id'], 'name_valid': semantic is not None,
                          'key_type_valid': key_type_ok, 'value_correct': value,
                          'schema_result': event['schema_result'], 'execution_status': event['execution_status']})
    return {'focal_observed': True, 'parse_success': parse_success, 'marker_slots': len(slots),
            'accepted_slots': len(parsed), 'malformed_slots': len(slots) - len(parsed),
            'arguments': arguments, 'EOS': response['EOS'], 'cap_reached': response['reached_token_cap'],
            'output_tokens': response['output_tokens'], 'request_id': response['request_id']}


def pair(left, right):
    counts = Counter()
    for key in sorted(set(left) & set(right)):
        a = left[key]['VERIFIED_WORKFLOW_SUCCESS']; b = right[key]['VERIFIED_WORKFLOW_SUCCESS']
        if a is None or b is None:
            counts['unobserved'] += 1
        else:
            counts['both_success' if a and b else 'lost' if a else 'recovered' if b else 'both_fail'] += 1
    return {name: counts[name] for name in ['both_success', 'lost', 'recovered', 'both_fail', 'unobserved']}


def audit(root, output):
    root = Path(root); output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    points, snapshots, scenarios, old = inputs()
    records = [read(f) for f in sorted((root / 'cells').glob('*.json'))]
    keys = [(r['method'], r['checkpoint_id'], r['arm']) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate measured cells')
    requests = [json.loads(x) for x in (root / 'requests.jsonl').read_text().splitlines() if x]
    by_request = {r['request_id']: r for r in requests}
    if len(by_request) != len(requests):
        raise ValueError('duplicated real GPU requests')
    for row in records:
        for response in row['responses']:
            saved = by_request[response['request_id']]
            # Shared C0 appears in N/E cell histories but was generated only once.
            if {k: v for k, v in saved.items() if k not in ['checkpoint_id', 'arm']} != response:
                raise ValueError('cell response differs from unique request journal')
    with tempfile.TemporaryDirectory(prefix='case015-p2-raw-audit-') as temp:
        for i, row in enumerate(records):
            replay_cell(row, Path(temp) / str(i))
    reproduction = []
    for row in records:
        if row['arm'] != 'C0' or not row['responses']:
            continue
        response = row['responses'][0]; original = old[(row['checkpoint_id'], row['method'])]
        reproduction.append({'method': row['method'], 'checkpoint_id': row['checkpoint_id'],
            'input_hash_match': response['input_token_sha256'] == original['input_token_sha256'],
            'output_ids_match': response['output_token_ids'] == original['output_token_ids'],
            'historical_raw_sha256': original['raw_response_sha256'], 'new_raw_sha256': response['raw_utf8_sha256'],
            'request_id': response['request_id']})
    if (root / 'reproduction.json').exists():
        expected = read(root / 'reproduction.json')['rows']
        if sorted(expected, key=lambda r: (r['method'], r['checkpoint_id'])) != sorted(reproduction, key=lambda r: (r['method'], r['checkpoint_id'])):
            raise ValueError('C0 reproduction summary differs from raw')
    metrics = []
    for row in records:
        metrics.append({'method': row['method'], 'api_version': row['api_version'], 'checkpoint_id': row['checkpoint_id'], 'arm': row['arm'],
            **focal_metrics(row, scenarios[row['scenario_id']]), 'workflow_success': row['VERIFIED_WORKFLOW_SUCCESS'],
            'state_only_success': row['STATE_ONLY_SUCCESS'], 'final_correct': row['FINAL_ANSWER_CORRECT'],
            'diagnostic_labels': row['diagnostic_labels'], 'terminal_reason': row['terminal_reason'],
            'observation_violations': row['observation_violations'], 'duplicate_writes': row['auxiliary']['duplicate_successful_write_events']})
    table = []
    for method in ('BF16', 'RTN', 'D16'):
        for api in ('API_V1', 'API_V2'):
            for arm in ('C0', 'G', 'N', 'E'):
                cells = [r for r in metrics if r['method'] == method and r['api_version'] == api and r['arm'] == arm]
                args = [a for r in cells for a in r['arguments']]
                observed = [r for r in cells if r['workflow_success'] is not None]
                table.append({'method': method, 'api_version': api, 'arm': arm, 'measured_cells': len(cells),
                    'workflow_denominator': len(observed), 'workflow_successes': sum(r['workflow_success'] for r in observed),
                    'focal_parse_denominator': sum(r['focal_observed'] for r in cells),
                    'focal_parse_successes': sum(r['parse_success'] is True for r in cells),
                    'focal_argument_event_denominator': len(args), 'name_valid': sum(a['name_valid'] for a in args),
                    'keys_types_valid': sum(a['key_type_valid'] for a in args),
                    'value_evaluable': sum(a['value_correct'] is not None for a in args),
                    'value_correct': sum(a['value_correct'] is True for a in args),
                    'unobserved': len(cells) - len(observed)})
    paired = []
    for method in ('BF16', 'RTN', 'D16'):
        subset = [r for r in records if r['method'] == method]
        for a, b in [('C0', 'G'), ('N', 'E')]:
            paired.append({'method': method, 'left': a, 'right': b,
                          **pair({r['checkpoint_id']: r for r in subset if r['arm'] == a}, {r['checkpoint_id']: r for r in subset if r['arm'] == b})})
    planned = read(root / 'planned_cells.json')
    not_applicable = [read(f) for f in sorted((root / 'not_applicable').glob('*.json'))]
    measured_ids = {r['method'] + '__' + r['checkpoint_id'] + '__' + r['arm'] for r in records}
    na_ids = {r['cell_id'] for r in not_applicable}
    if measured_ids & na_ids or len(na_ids) != len(not_applicable):
        raise ValueError('duplicate/not-applicable measured cell conflict')
    missing = [p for p in planned if p['cell_id'] not in measured_ids | na_ids]
    for item in table:
        planned_group = [p for p in planned if p['method'] == item['method'] and p['arm'] == item['arm'] and p['checkpoint_id'].startswith(item['api_version'] + '__')]
        item['planned_checkpoints'] = len(planned_group)
        item['not_applicable_cells'] = sum(p['cell_id'] in na_ids for p in planned_group)
        item['missing_cells'] = sum(p['cell_id'] not in measured_ids | na_ids for p in planned_group)
    journal = [json.loads(x) for x in (root / 'attempt_journal.jsonl').read_text().splitlines() if x]
    started = [r for r in journal if r['event'] == 'REQUEST_STARTED']
    failures = [r for r in journal if r['event'] == 'REQUEST_ERROR']
    if len(started) != len(requests) + len(failures):
        raise ValueError('request attempt/completion/error ledger mismatch')
    costs = {'actual_new_request_attempts': len(started), 'completed_generated_responses': len(requests),
        'first_diagnostic_responses': sum(r['request_role'] == 'FOCAL' for r in requests),
        'continuation_responses': sum(r['request_role'] == 'CONTINUATION' for r in requests),
        'smoke_model_requests': 0, 'failed_request_attempts': len(failures),
        'generated_tokens_observed': sum(r['output_tokens'] for r in requests),
        'generated_tokens_in_failed_requests': 'UNKNOWN' if failures else 0,
        'generation_seconds_sum': sum(r['seconds'] for r in requests),
        'peak_allocated_bytes': max((r['peak_allocated_bytes'] for r in requests), default=None),
        'peak_reserved_bytes': max((r['peak_reserved_bytes'] for r in requests), default=None),
        'total_worker_wall_seconds': read(root / 'receipt.json')['total_worker_wall_seconds'],
        'not_serving_speed': True}
    summary = {'status': 'COMPLETED' if not missing and not failures else 'PARTIAL',
        'C0_reproduction': {'matched': sum(r['input_hash_match'] and r['output_ids_match'] for r in reproduction), 'observed': len(reproduction), 'planned': 24},
        'measured_cells': len(records), 'not_applicable_cells': len(not_applicable), 'missing_cells': missing,
        'independent_selected_fixtures': 4, 'API_checkpoints': 8, 'table': table, 'paired': paired, 'costs': costs,
        'F_actual': sum(r['arm'] == 'C0' and len(r['responses']) == 1 and r['terminal_reason'] == 'PARSE_ERROR'
                        and r['budget']['model_calls'] < 8 and r['budget']['tool_attempts'] < 6
                        and r['budget']['elapsed_seconds'] < 900 and focal_metrics(r, scenarios[r['scenario_id']]).get('malformed_slots', 0) > 0
                        and focal_metrics(r, scenarios[r['scenario_id']]).get('accepted_slots', 0) == 0 for r in records),
        'scope': 'selected post-A1 DEV checkpoints; raw/fixture CPU replay, not independent GPU replication'}
    for name, obj in [('summary.json', summary), ('focal_metrics.json', metrics), ('reproduction.json', reproduction)]:
        write(output / name, obj)
    write(output / 'verification.json', {'status': 'PASS', 'raw_cells_replayed': len(records), 'unique_requests_checked': len(requests),
          'shared_C0_not_double_counted': True, 'independent_endpoint': True, 'reproduction_computed_from_raw': True,
          'new_model_forwards_in_audit': 0, 'full_logits_verification': 'NOT_RUN; no full logits stored'})
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--results', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); value = audit(args.results, args.output)
    print(json.dumps({k: v for k, v in value.items() if k not in ['table', 'paired', 'missing_cells']}))
