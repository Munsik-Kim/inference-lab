"""Versioned final state, required actions and legacy answer contracts."""
import math
from .common import extract, digest

SCORER = 'functional-v1-required-original-actions;legacy-case013-v1'
REQUIRED = {'document': ['document.read'], 'combine': ['record.read', 'math.multiply'],
            'updates': ['updates.read'], 'fixture_patch': ['fixture.patch', 'fixture.validate']}


def argument_value_correct(semantic, args, task):
    if semantic == 'math.multiply':
        a, b = task['fixture']['record']['quantity'], task['fixture']['record']['unit_price']
        return sorted([args['a'], args['b']]) == sorted([a, b])
    if semantic == 'fixture.patch':
        return args == task['fixture']['expected_config']
    return True


def allowed_actions(task, state, events):
    family = task['family']
    if family == 'combine':
        return ['record.read', 'math.multiply'] if any(e.get('semantic_tool_id') == 'record.read' and e['execution_status'] == 'SUCCESS' for e in events) else ['record.read']
    if family == 'fixture_patch':
        return ['fixture.patch', 'fixture.validate'] if state == task['fixture']['expected_config'] else ['fixture.patch']
    return REQUIRED[family]


def score(task, final, state, events):
    gold_ok = isinstance(final, dict) and all(final.get(k) == v for k, v in task['gold'].items())
    good = [e for e in events if e.get('execution_status') == 'SUCCESS']
    valid = [e for e in good if e.get('semantic_tool_id') == 'fixture.validate' and e.get('tool_result', {}).get('valid') is True]
    legacy = bool(gold_ok)
    if task['family'] == 'fixture_patch':
        legacy = bool(legacy and state == task['fixture']['expected_config'] and valid)
    required = REQUIRED[task['family']]
    satisfied = []
    last_index = -1
    for semantic in required:
        matches = [(i, e) for i, e in enumerate(events) if i > last_index and e.get('semantic_tool_id') == semantic
                   and e.get('execution_status') == 'SUCCESS' and e.get('argument_value_correct') is True]
        if semantic == 'fixture.validate':
            matches = [(i, e) for i, e in matches if e['tool_result'].get('valid') is True and e['state_after_digest'] == digest(state)]
        if matches:
            last_index = matches[0][0]; satisfied.append(semantic)
    state_ok = state == task['fixture']['expected_config'] if task['family'] == 'fixture_patch' else state == task['fixture']['config']
    missing = [s for s in required if s not in satisfied]
    functional = bool(gold_ok and state_ok and not missing)
    labels = []
    if missing:
        labels.append('MISSING_ACTION')
    if isinstance(final, dict) and final.get('valid') is True and not functional:
        labels.append('UNSUPPORTED_SUCCESS_CLAIM')
    return {'functional_success': functional, 'legacy_success': legacy, 'final_answer_correct': bool(gold_ok),
            'state_correct': state_ok, 'missing_actions': missing, 'required_actions': required,
            'final_failure_labels': labels, 'scorer_version': SCORER}
