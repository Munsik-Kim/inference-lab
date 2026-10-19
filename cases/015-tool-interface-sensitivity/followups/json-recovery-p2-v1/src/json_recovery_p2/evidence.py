"""Rebuild original requests and selected pre-write checkpoints from raw records."""
import copy
import hashlib
import json
import re
import tempfile
from pathlib import Path
from .io import read, rows, sha, write
from .upstream import A1, CASE, ROOT, Fixture, Registry, canonical, correct_event, digest, history, independent_endpoint, parse_slots, run_episode

METHODS = ('BF16', 'RTN', 'D16')
SELECTED = ('A1-00', 'A1-06', 'A1-08', 'A1-14')
STOP_IDS = (151645, 151643)


def ordered_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def observation(events, available):
    return digest([{'semantic_tool_id': e['semantic_tool_id'], 'arguments': e['raw_arguments'], 'result': e['tool_result']}
                   for e in events if e['event_id'] in available])


def check_response(response, messages, tools, state, version, events, available):
    expected = {
        'raw_utf8_sha256': hashlib.sha256(response['raw_text'].encode()).hexdigest(),
        'messages_tools_hash': digest({'messages': messages, 'tools': tools}),
        'pre_generation_history_hash': digest(messages),
        'pre_generation_state_digest': digest(state),
        'pre_generation_state_version': version,
        'canonical_observation_hash': observation(events, available),
        'available_result_event_ids': list(available),
        'output_tokens': len(response['output_token_ids']),
    }
    for key, value in expected.items():
        if response[key] != value:
            raise ValueError('recorded response mismatch: ' + key)
    if not isinstance(response['input_token_sha256'], str) or len(response['input_token_sha256']) != 64:
        raise ValueError('missing recorded input-token hash')
    return expected


def replay_prefix(row, scenario, path):
    registry = Registry(row['api_version'])
    fixture = Fixture(scenario, registry, path)
    messages = copy.deepcopy(scenario['messages'])
    available = []
    attempted = 0
    for turn, response in enumerate(row['responses'][:2]):
        check_response(response, messages, registry.tools(), fixture.snapshot(), fixture.version, fixture.events, available)
        calls, results, new_ids = [], [], []
        for slot in parse_slots(response['raw_text']):
            attempted += 1
            if slot['parsed'] is None:
                raise ValueError('malformed frozen prefix')
            obj = slot['parsed']
            event = fixture.execute(obj.get('name'), obj.get('arguments'), turn, slot['call_index'], tuple(available))
            if not correct_event(scenario, event):
                raise ValueError('incorrect frozen prefix action')
            calls.append(obj); results.append(event['tool_result']); new_ids.append(event['event_id'])
        messages = history(messages, calls, results)
        available.extend(new_ids)
    if attempted != 2 or [e['semantic_tool_id'] for e in fixture.events] != ['order.read', 'amount.multiply']:
        raise ValueError('checkpoint requires two actual read/multiply events')
    original = [e for e in row['tool_events'] if e['generated_in_turn'] < 2]
    if fixture.events != original:
        raise ValueError('frozen prefix event/state mismatch')
    current = row['responses'][2]
    check_response(current, messages, registry.tools(), fixture.snapshot(), fixture.version, fixture.events, available)
    snapshot = {'messages': messages, 'tools': registry.tools()}
    point = {
        'messages_tools_sha256': digest(snapshot),
        'ordered_model_payload_sha256': ordered_digest(snapshot),
        'history_sha256': digest(messages),
        'common_observation_sha256': observation(fixture.events, available),
        'available_result_event_ids': available,
        'state_sha256': digest(fixture.snapshot()),
        'state_version': fixture.version,
        'recorded_input_token_sha256': current['input_token_sha256'],
        'recorded_input_tokens': current['input_tokens'],
        'model_calls_used': 2, 'tool_attempts_used': attempted,
    }
    return point, snapshot, fixture


def equivalent(points):
    if set(points) != set(METHODS):
        raise ValueError('three methods are required at each checkpoint')
    reference = points['BF16']
    if any(p != reference for p in points.values()):
        raise ValueError('checkpoint input/observations/state/counters differ across methods')
    return reference


def slot_diagnostic(row, response, slot):
    segment = response['raw_text'].split('<tool_call>')[slot['call_index'] + 1]
    payload = segment.split('</tool_call>')[0].removesuffix('<|im_end|>').removesuffix('<|endoftext|>').strip()
    try:
        document = json.loads(payload)
        syntax = isinstance(document, dict)
    except ValueError:
        syntax = False
    parsed = slot['parsed']
    missing_key = '"total_minor"' not in payload
    form = ('AMOUNT_KEY_MISSING_AND_QUOTED_DIGITS' if missing_key and re.search(r':\s*"-?\d+"', payload)
            else 'EXTRA_QUOTE_BEFORE_TOTAL_MINOR' if re.search(r',"\s*,\s*"total_minor"', payload)
            else 'OTHER_MALFORMED')
    return {
        'episode_id': row['episode_id'], 'scenario_id': row['scenario_id'], 'api_version': row['api_version'], 'method': row['method'],
        'turn': response['generated_in_turn'], 'slot_index': slot['call_index'],
        'raw_response_sha256': response['raw_utf8_sha256'],
        'syntactic_whole_document_json': syntax,
        'original_first_dict_accepted': parsed is not None,
        'closing_marker_present': '</tool_call>' in segment,
        'failure_form': form if parsed is None else None,
        'name_evaluation': 'NOT_EVALUABLE_UNPARSED' if parsed is None else 'PARSED',
        'argument_keys_types_values': 'NOT_EVALUABLE_UNPARSED' if parsed is None else 'SEE_ACTUAL_TOOL_EVENT',
        'number_text_is_not_validated_argument': True,
        'output_tokens': response['output_tokens'], 'last_token_id': response['output_token_ids'][-1],
        'EOS': response['output_token_ids'][-1] in STOP_IDS, 'reached_192_token_cap': response['output_tokens'] >= 192,
        'model_calls_used_at_response': response['generated_in_turn'] + 1,
        'episode_model_calls': row['model_calls'], 'episode_tool_attempts': row['attempted_tool_calls'],
        'episode_executed_events': len(row['tool_events']),
        'final_state_version': row['final_state_version'],
        'workflow_success_original': row['VERIFIED_WORKFLOW_SUCCESS'],
        'raw_terminal_reason': row['terminal_reason'],
        'interpretation': 'UNPARSED_MARKER_SLOT; not evidence of feedback recovery',
    }


def derive(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = rows(A1 / 'results/a1_r1/episodes.jsonl')
    scenarios = {s['scenario_id']: s for s in read(A1 / 'data/scenarios.json')}
    by_id = {r['episode_id']: r for r in records}
    if len(records) != 96 or len(by_id) != 96:
        raise ValueError('missing/duplicate A1 episodes')
    generation = read(A1 / 'specs/protocol.json')['generation']
    flat = rows(A1 / 'results/a1_r1/tool_events.jsonl')
    expected_flat = [{'episode_id': r['episode_id'], **e} for r in records for e in r['tool_events']]
    if flat != expected_flat:
        raise ValueError('flat tool-event records differ from raw episode events')
    captured = {}; malformed = []; requests = tools = attempts = 0
    with tempfile.TemporaryDirectory(prefix='case015-p2-replay-') as tmp:
        for row in records:
            cell = A1 / 'results/a1_r1/workers' / row['method'] / 'attempt01/cells' / (row['episode_id'] + '.json')
            if read(cell) != row:
                raise ValueError('worker/aggregate row mismatch')
            pending = iter(row['responses']); snapshots = []
            def generate(messages, declared_tools, deadline):
                response = copy.deepcopy(next(pending))
                snapshots.append(copy.deepcopy({'messages': messages, 'tools': declared_tools}))
                if digest(snapshots[-1]) != response['messages_tools_hash']:
                    raise ValueError('actual reconstructed messages/tools mismatch')
                return response
            identity = {k: row[k] for k in ['method', 'model_revision', 'artifact_hash', 'execution_source_sha256', 'generation_config_hash']}
            replay = run_episode(scenarios[row['scenario_id']], row['api_version'], generate, Path(tmp) / row['episode_id'], generation, identity, row['episode_id'])
            for key in ['responses', 'tool_events', 'final_answer', 'final_fixture', 'final_state_version', 'attempted_tool_calls', 'VERIFIED_WORKFLOW_SUCCESS', 'terminal_reason']:
                if replay[key] != row[key]:
                    raise ValueError('A1 replay mismatch: ' + row['episode_id'] + ' ' + key)
            if independent_endpoint(scenarios[row['scenario_id']], replay) != row['VERIFIED_WORKFLOW_SUCCESS']:
                raise ValueError('independent workflow endpoint mismatch')
            captured[row['episode_id']] = snapshots
            for response in row['responses']:
                if response['raw_utf8_sha256'] != hashlib.sha256(response['raw_text'].encode()).hexdigest():
                    raise ValueError('raw UTF8 bytes changed')
                if response['output_tokens'] != len(response['output_token_ids']):
                    raise ValueError('output token count mismatch')
                for slot in parse_slots(response['raw_text']):
                    if slot['parsed'] is None:
                        malformed.append(slot_diagnostic(row, response, slot))
            requests += len(row['responses']); tools += len(row['tool_events']); attempts += row['attempted_tool_calls']
        checkpoints = []; model_snapshots = []; continuations = []
        for scenario_id in SELECTED:
            for api in ['API_V1', 'API_V2']:
                checkpoint_id = api + '__' + scenario_id + '__before-generation-2'
                points = {}; prefix_payload = None; prefix_events = None
                wall_debits = []
                for method in METHODS:
                    row = by_id[method + '__' + api + '__' + scenario_id]
                    point, payload, fixture = replay_prefix(row, scenarios[scenario_id], Path(tmp) / ('prefix-' + row['episode_id']))
                    if payload != captured[row['episode_id']][2]:
                        raise ValueError('prefix reconstruction differs from full raw replay')
                    points[method] = point; prefix_payload = payload; prefix_events = fixture.events
                    wall_debits.append(row['wall_seconds'])
                    continuations.append({'checkpoint_id': checkpoint_id, 'method': method,
                        'evidence_kind': 'historical_A1_audit_only_not_a_new_C0',
                        'source_episode_id': row['episode_id'], 'raw_text': row['responses'][2]['raw_text'],
                        'raw_response_sha256': row['responses'][2]['raw_utf8_sha256'],
                        'output_token_ids': row['responses'][2]['output_token_ids'],
                        'input_token_sha256': row['responses'][2]['input_token_sha256']})
                reference = equivalent(points)
                snapshot = {'checkpoint_id': checkpoint_id, **prefix_payload,
                    'messages_tools_sha256': reference['messages_tools_sha256'],
                    'ordered_model_payload_sha256': reference['ordered_model_payload_sha256']}
                model_snapshots.append(snapshot)
                checkpoints.append({'checkpoint_id': checkpoint_id, 'scenario_id': scenario_id, 'api_version': api,
                    'evidence_kind': 'CPU_RECONSTRUCTED_HISTORICAL_CHECKPOINT', **reference,
                    'per_method_identity': points, 'prefix_events': prefix_events,
                    'initial_fixture_for_audit_only': scenarios[scenario_id]['orders'],
                    'prefix_elapsed_seconds_debit': max(wall_debits),
                    'prefix_wall_policy': 'conservative common debit=max original full episode wall across methods; exact prefix wall unavailable, includes later work, never reset to fresh900s',
                    'model_input_snapshot_sha256': digest(snapshot)})
    failures = [x for x in malformed if not x['workflow_success_original']]
    mixed = [x for x in malformed if x['workflow_success_original']]
    if len(failures) != 8 or len(mixed) != 2 or requests != 463 or tools != 368 or attempts != 378:
        raise ValueError('registered historical evidence counts differ')
    forms = {name: sum(x['failure_form'] == name for x in failures)
             for name in ['AMOUNT_KEY_MISSING_AND_QUOTED_DIGITS', 'EXTRA_QUOTE_BEFORE_TOTAL_MINOR', 'OTHER_MALFORMED']}
    if forms != {'AMOUNT_KEY_MISSING_AND_QUOTED_DIGITS': 5, 'EXTRA_QUOTE_BEFORE_TOTAL_MINOR': 3, 'OTHER_MALFORMED': 0}:
        raise ValueError('historical JSON classification differs')
    for item in failures:
        if not (item['method'] == 'D16' and item['turn'] == 2 and 43 <= item['output_tokens'] <= 48 and item['last_token_id'] == 151645
                and item['episode_model_calls'] == 3 and item['episode_tool_attempts'] == 3 and item['episode_executed_events'] == 2
                and item['final_state_version'] == 0 and not item['reached_192_token_cap']):
            raise ValueError('registered failure facts mismatch')
    summary = {'status': 'PASS', 'evidence_kind': 'CPU_HISTORICAL_RECORD_REPLAY', 'base_commit': '51385286063eb55e791f13a0b97f1e341ce6a879',
        'episodes_replayed': 96, 'generation_requests_reconstructed': requests, 'attempted_tool_calls': attempts, 'actual_tool_events': tools,
        'malformed_marker_slots': len(malformed), 'malformed_only_failure_episodes': len(failures), 'mixed_success_marker_slots': len(mixed),
        'failure_forms': forms, 'checkpoint_count': len(checkpoints), 'selected_fixtures': 4, 'matched_method_inputs': 24,
        'output_EOS_151645_failures': sum(x['last_token_id'] == 151645 for x in failures),
        'raw_BUDGET_EXHAUSTED_is_not_actual_budget_exhaustion': True,
        'mixed_responses_are_not_error_feedback_recovery': True,
        'input_token_hash_scope': 'recorded input hashes compared across methods; NOT retokenized by this command',
        'new_model_responses': 0, 'recovery_results': 'NOT_RUN',
        'original_parser': 'first valid dict anywhere in slot; closing wrapper/whole-document validity are separate diagnostics'}
    for name, value in [('cpu_diagnostic_summary.json', summary), ('malformed_slots.json', malformed),
                        ('checkpoints.json', checkpoints), ('model_request_snapshots.json', model_snapshots),
                        ('historical_continuations.json', continuations)]:
        write(output / name, value)
    return summary

