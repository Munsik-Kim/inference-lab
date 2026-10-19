"""Descriptive error transitions from raw rows, without altering the scorer."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

P2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P2 / 'execution'))
from state import inputs, read, write
from json_recovery_p2.upstream import parse_slots, correct_event


def diagnose(root, output):
    root = Path(root); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    _, _, scenarios, _ = inputs()
    records = [read(f) for f in sorted((root / 'cells').glob('*.json'))]
    details = []
    for row in records:
        scenario = scenarios[row['scenario_id']]
        focal_index = 1 if row['arm'] in ('N', 'E') else 0
        focal = row['responses'][focal_index] if len(row['responses']) > focal_index else None
        focal_events = [e for e in row['tool_events'] if focal and e['generated_in_turn'] == focal['generated_in_turn']]
        current_events = [e for e in row['tool_events'] if e['generated_in_turn'] >= 2]
        correct_writes = [e for e in current_events if e['semantic_tool_id'] == 'order.write' and correct_event(scenario, e)]
        invalid_ids = [e['event_id'] for e in current_events if isinstance(e['raw_arguments'], dict)
                       and 'order_id' in e['raw_arguments'] and e['raw_arguments']['order_id'] != scenario['target_order_id']]
        last_slots = parse_slots(row['responses'][-1]['raw_text']) if row['responses'] else []
        if row['VERIFIED_WORKFLOW_SUCCESS'] is True:
            last_stage = 'COMPLETED_WORKFLOW'
        elif last_slots and not any(s['parsed'] is not None for s in last_slots):
            last_stage = 'JSON_SERIALIZATION'
        elif invalid_ids:
            last_stage = 'ARGUMENT_ID'
        elif row['observation_violations']:
            last_stage = 'OBSERVATION_ORDER'
        elif any(e['schema_result'] not in ('VALID',) for e in current_events):
            last_stage = 'ARGUMENT_SCHEMA_OR_VALUE'
        else:
            last_stage = 'VALIDATION_OR_COMPLETION'
        whole_slots = []
        for response in row['responses']:
            for slot in parse_slots(response['raw_text']):
                segment = response['raw_text'].split('<tool_call>')[slot['call_index'] + 1].split('</tool_call>')[0]
                segment = segment.removesuffix('<|im_end|>').removesuffix('<|endoftext|>').strip()
                try:
                    strict = isinstance(json.loads(segment), dict)
                except ValueError:
                    strict = False
                whole_slots.append({'request_id': response['request_id'], 'slot': slot['call_index'],
                    'whole_slot_JSON_dict': strict, 'original_first_dict_accepted': slot['parsed'] is not None})
        details.append({'method': row['method'], 'api_version': row['api_version'], 'checkpoint_id': row['checkpoint_id'], 'arm': row['arm'],
            'workflow_success': row['VERIFIED_WORKFLOW_SUCCESS'], 'state_only_success': row['STATE_ONLY_SUCCESS'],
            'focal_actions': [e['semantic_tool_id'] for e in focal_events], 'focal_executed_events': len(focal_events),
            'correct_actual_write_events': len(correct_writes), 'invalid_ID_events_including_rejected': invalid_ids,
            'raw_scorer_ARGUMENT_VALUE_WRONG': row['ARGUMENT_VALUE_WRONG'],
            'terminal_failure_stage': last_stage, 'model_calls_charged': row['budget']['model_calls'],
            'tool_attempts_charged': row['budget']['tool_attempts'], 'state_version': row['final_state_version'],
            'accepted_slots_with_trailing_material': sum(s['original_first_dict_accepted'] and not s['whole_slot_JSON_dict'] for s in whole_slots),
            'slot_diagnostics': whole_slots})
    groups = []
    for method in ('BF16', 'RTN', 'D16'):
        for arm in ('C0', 'G', 'N', 'E'):
            rows = [r for r in details if r['method'] == method and r['arm'] == arm]
            groups.append({'method': method, 'arm': arm, 'observed_cells': len(rows),
                'focal_actions': dict(Counter(a for r in rows for a in r['focal_actions'])),
                'workflow_success': sum(r['workflow_success'] is True for r in rows),
                'correct_write_cells': sum(r['correct_actual_write_events'] > 0 for r in rows),
                'state_only_success': sum(r['state_only_success'] for r in rows),
                'terminal_failure_stages': dict(Counter(r['terminal_failure_stage'] for r in rows)),
                'invalid_ID_cells_including_rejected': sum(bool(r['invalid_ID_events_including_rejected']) for r in rows),
                'accepted_slots_with_trailing_material': sum(r['accepted_slots_with_trailing_material'] for r in rows)})
    write(output / 'error_transitions.json', {'groups': groups, 'cells': details,
        'scope': 'registered descriptive error-stage diagnostics; scorer and original verdicts unchanged; whole-slot JSON and original first-dict acceptance intentionally distinct'})
    columns = ['method', 'api_version', 'checkpoint_id', 'arm', 'workflow_success', 'state_only_success', 'focal_actions', 'correct_actual_write_events', 'terminal_failure_stage', 'model_calls_charged', 'tool_attempts_charged', 'state_version']
    with (output / 'error_transitions.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for row in details:
            writer.writerow({k: json.dumps(row[k]) if isinstance(row[k], list) else row[k] for k in columns})
    return groups


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--results', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); print(json.dumps(diagnose(args.results, args.output)))
