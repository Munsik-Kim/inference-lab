"""CPU-testable strict episode state machine; model adapter injected explicitly."""
import copy
import time
from .common import HARNESS, digest, history, parse_calls, extract
from .fixture import Fixture
from .registry import Registry
from .scoring import score


class ObservationError(RuntimeError):
    def __init__(self, label, message):
        super().__init__(message)
        self.label = label


def run_episode(task, interface, generate, root, generation, identity, episode_id):
    start = time.monotonic()
    fixture = Fixture(task, Registry(interface['mapping']), root)
    messages = copy.deepcopy(interface['messages']); responses = []; errors = []
    accepted_count = 0; terminal = 'BUDGET_EXHAUSTED'; final = None; observed = 'OBSERVED'
    for turn in range(generation['max_model_calls']):
        if time.monotonic() - start >= generation['deadline_seconds']:
            terminal = 'TIMEOUT'; errors.append({'label': terminal, 'turn': turn}); break
        try:
            response = generate(messages, interface['tools'], start + generation['deadline_seconds'])
        except ObservationError as e:
            terminal = e.label
            errors.append({'label': e.label, 'turn': turn, 'message': str(e)})
            if e.label in ('OOM', 'INFRA_ERROR', 'WORKER_BUDGET'):
                observed = 'UNOBSERVED_INFRA'
            break
        response['model_turn'] = turn
        responses.append(response)
        calls, failed = parse_calls(response['raw_text'])
        errors.extend({'label': 'PARSE_ERROR', 'turn': turn, 'call_index': e['index']} for e in failed)
        if calls:
            executed = []; results = []
            for index, call in enumerate(calls):
                if accepted_count >= generation['max_tool_calls']:
                    terminal = 'BUDGET_EXHAUSTED'
                    errors.append({'label': terminal, 'turn': turn, 'reason': 'attempted_tool_call_limit'})
                    break
                accepted_count += 1
                event = fixture.execute(call.get('name'), call.get('arguments'), turn, index)
                errors.extend({'label': label, 'turn': turn, 'event_id': event['event_id']} for label in event['error_labels'])
                executed.append(call); results.append(event['tool_result'])
            if not executed:
                break
            messages = history(messages, executed, results)
            continue
        final = extract(response['text'])
        if final is None:
            terminal = 'PARSE_ERROR'; errors.append({'label': terminal, 'turn': turn})
        else:
            terminal = 'FINAL_ANSWER'
        break
    verdict = score(task, final, fixture.state, fixture.events)
    if observed != 'OBSERVED':
        verdict['functional_success'] = None
        verdict['legacy_success'] = None
    if terminal in ('BUDGET_EXHAUSTED', 'TIMEOUT', 'CONTEXT_LIMIT') and observed == 'OBSERVED':
        verdict['functional_success'] = verdict['legacy_success'] = False
        if not any(e['label'] == terminal for e in errors):
            errors.append({'label': terminal, 'turn': len(responses)})
    labels = set(verdict['final_failure_labels']) if not verdict['functional_success'] else set()
    if not verdict['functional_success'] and terminal in ('BUDGET_EXHAUSTED','TIMEOUT','CONTEXT_LIMIT','OOM','INFRA_ERROR','WORKER_BUDGET','PARSE_ERROR'):
        labels.add(terminal)
    if not verdict['functional_success'] and not errors and verdict['final_failure_labels']:
        errors += [{'label': label, 'turn': len(responses)-1} for label in verdict['final_failure_labels']]
    recovery_opportunities = [i for i, e in enumerate(fixture.events) if e['error_labels']]
    recoveries = []
    for i in recovery_opportunities:
        later = fixture.events[i+1:]
        recoveries.append({'event_id': fixture.events[i]['event_id'], 'subsequent_action_observed': bool(later),
                           'registered_valid_later_action': any(e['semantic_tool_id'] and e['execution_status'] == 'SUCCESS' and not e['error_labels'] for e in later)})
    row = {'episode_id': episode_id, 'task_id': task['id'], 'family_id': task['family'],
           'historical_role': 'previously_observed_DEV', 'evidence_kind': 'measurement',
           'interface_id': interface['interface_id'], 'canonical_task_hash': digest(task),
           'rendered_prompt_hash': interface['rendered_prompt_hash'],
           'input_token_hash': responses[0].get('input_token_sha256') if responses else None,
           'harness_version': HARNESS, **identity, **verdict,
           'observation_status': observed, 'first_error_type': errors[0]['label'] if errors else None,
           'first_error_turn': errors[0]['turn'] if errors else None, 'final_failure_labels': sorted(labels),
           'errors': errors, 'episode_error_labels': sorted({e['label'] for e in errors}),
           'model_calls': len(responses), 'tool_attempts': accepted_count,
           'generated_tokens': sum(r['output_tokens'] for r in responses),
           'input_tokens_total': sum(r['input_tokens'] for r in responses),
           'wall_seconds': time.monotonic()-start,
           'peak_allocated_bytes': max((r.get('peak_allocated_bytes', 0) for r in responses), default=0),
           'peak_reserved_bytes': max((r.get('peak_reserved_bytes', 0) for r in responses), default=0),
           'terminal_reason': terminal, 'final_answer': final, 'final_fixture': fixture.state,
           'responses': responses, 'tool_events': fixture.events, 'recovery': recoveries,
           'enum_enabled': False, 'enhanced_feedback_enabled': False, 'aliases_enabled': False}
    return row
