"""Model-visible allowlist and paired retries, distinct from audit checkpoints."""
import copy
import hashlib
import json
from .upstream import digest, parse_slots

FEEDBACK = {
    'N': 'No tool was executed. Retry the previous response.',
    'E': 'PARSE_ERROR: Tool-call JSON could not be parsed. No tool was executed. Retry the previous response.',
}


def visible(snapshot):
    request = {k: copy.deepcopy(snapshot[k]) for k in ('messages', 'tools')}
    if not isinstance(request['messages'], list) or not isinstance(request['tools'], list):
        raise ValueError('messages/tools must be lists')
    if digest(request) != snapshot['messages_tools_sha256']:
        raise ValueError('model-visible snapshot changed')
    ordered = hashlib.sha256(json.dumps(request, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if ordered != snapshot['ordered_model_payload_sha256']:
        raise ValueError('model-visible key serialization order changed')
    return request


def request_identity(request, *, condition, observation_hash, state_hash, available_ids):
    return {
        'condition': condition,
        'messages_tools_sha256': digest(request),
        'intervention_sha256': digest({'condition': condition, 'feedback': FEEDBACK.get(condition), 'grammar': condition == 'G'}),
        'common_observation_sha256': observation_hash,
        'state_sha256': state_hash,
        'available_result_event_ids': list(available_ids),
        'input_token_sha256': None,
        'input_token_hash_status': 'NOT_RETOKENIZED_REQUEST_BUILDER',
    }


def malformed_only(raw, remaining_tool_attempts):
    slots = parse_slots(raw)
    # The original harness executes at most the remaining attempted-call budget.
    attempted = slots[:remaining_tool_attempts]
    return bool(attempted) and not any(s['parsed'] is not None for s in attempted)


def initial(snapshot, condition):
    if condition not in ('C0', 'G'):
        raise ValueError('initial condition must be C0 or G')
    return visible(snapshot)


def retry(snapshot, condition, fresh_c0, *, model_calls_used, tool_attempts_used, retries_used=0, allow_synthetic_contract_test=False):
    if condition not in FEEDBACK:
        raise ValueError('retry condition must be N or E')
    accepted_kinds = {'measurement_p2_new_c0'}
    if allow_synthetic_contract_test:
        accepted_kinds.add('synthetic_contract_test')
    if fresh_c0.get('evidence_kind') not in accepted_kinds:
        raise ValueError('historical/scripted outputs are not a new C0 measurement')
    if fresh_c0.get('parent_request_sha256') != snapshot['messages_tools_sha256']:
        raise ValueError('C0 input differs from checkpoint')
    if fresh_c0.get('checkpoint_id') != snapshot['checkpoint_id'] or fresh_c0.get('method') not in ('BF16', 'RTN', 'D16'):
        raise ValueError('C0 checkpoint/method identity mismatch')
    if retries_used or not (3 <= model_calls_used < 8) or not (3 <= tool_attempts_used < 6):
        raise ValueError('retry count or cumulative budget exhausted')
    slots = parse_slots(fresh_c0['raw_text'])
    if not slots or any(s['parsed'] is not None for s in slots):
        raise ValueError('retry requires malformed-only C0; mixed responses continue normally')
    request = visible(snapshot)
    request['messages'].extend([
        {'role': 'assistant', 'content': fresh_c0['raw_text']},
        {'role': 'user', 'content': FEEDBACK[condition]},
    ])
    return request

