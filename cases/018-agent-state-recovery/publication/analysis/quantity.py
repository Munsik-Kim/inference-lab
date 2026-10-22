"""Posthoc observation diagnostic v1. No model/runtime or original scorer imports.

Definitions are fixed in DEFINITION.json before the first full aggregation.
Only model-visible successful read_records results delivered before the anchor
are observations. Recorded state transitions establish expiry, never visibility.
"""
import json


def strict_map(rows, required):
    if not isinstance(rows, list):
        return None, 'not_list'
    out = {}
    for r in rows:
        if (type(r) is not dict or type(r.get('record_id')) is not str
                or type(r.get('quantity')) is not int):
            return None, 'invalid_record_type'
        if r['record_id'] in out:
            return None, 'duplicate_record'
        out[r['record_id']] = r['quantity']
    if set(out) != set(required):
        return out, 'key_coverage'
    return out, None


def diagnose(task, row):
    required = list(task['public_request']['record_ids'])
    scope = task['public_request']['scope_id']
    if len(required) != len(set(required)) or not required:
        raise ValueError('invalid public required scope')
    requests, events = row['requests'], row['events']
    finals = [(i, r) for i, r in enumerate(requests) if r.get('decision') == 'FINAL_ACCEPTED']
    if len(finals) > 1:
        raise ValueError('multiple accepted finals')
    has = bool(finals)
    index, anchor = finals[0] if has else (len(requests)-1, requests[-1] if requests else None)
    available = anchor['available_result_event_ids'] if anchor else []
    if len(available) != len(set(available)):
        raise ValueError('duplicate available result IDs')
    ids = {e['event_id']: i for i, e in enumerate(events)}
    if len(ids) != len(events) or set(available)-set(ids):
        raise ValueError('unknown or duplicate event IDs')
    for eid in available:
        if events[ids[eid]]['generated_in_turn'] >= index:
            raise ValueError('future result supplied at anchor')
    terminal = row['terminal_snapshot']
    if terminal['scope_id'] != scope or set(terminal['records']) != set(required):
        raise ValueError('terminal public scope mismatch')
    terminal_map = {k: terminal['records'][k]['quantity'] for k in required}
    last_change = dict.fromkeys(required, -1)
    all_changes, metadata = [], []
    # Includes the last action on a no-final trace: it cannot be retroactively seen.
    for i, e in enumerate(events):
        before, after = e['state_before'], e['state_after']
        changed = [k for k in required if before['records'][k]['quantity'] != after['records'][k]['quantity']]
        for k in changed:
            last_change[k] = i
        if before != after:
            all_changes.append(i)
            if not changed:
                metadata.append(e['event_id'])
    last_any = max(all_changes, default=-1)
    reads, covered, after_all = [], set(), set()
    first_map, first_reason, first_eid = None, 'no_observed_successful_read', None
    rejected_reads = []
    for i, e in enumerate(events):
        if e['tool'] != 'read_records' or e['event_id'] not in available:
            continue
        if (not e.get('callable_executed') or e['status'] != 'SUCCESS'
                or 'error' in e['result']):
            rejected_reads.append({'event_id': e['event_id'], 'reason': 'unsuccessful_read'})
            continue
        if e['state_before']['scope_id'] != scope or e['state_after']['scope_id'] != scope:
            rejected_reads.append({'event_id': e['event_id'], 'reason': 'wrong_scope'})
            continue
        values = e['result'].get('records')
        m, reason = strict_map(values, required)
        # Partial reads remain usable per record; malformed/duplicate values do not.
        if reason not in (None, 'key_coverage') or m is None or set(m)-set(required):
            rejected_reads.append({'event_id': e['event_id'], 'reason': reason or 'wrong_keys'})
            continue
        if set(m) != set(e['arguments'].get('record_ids', [])):
            raise ValueError('read payload differs from executed request scope')
        if any(v != e['state_before']['records'][k]['quantity'] for k, v in m.items()):
            raise ValueError('public read differs from execution state')
        if first_eid is None:
            first_map, first_reason, first_eid = m, reason, e['event_id']
        valid = [k for k in m if last_change[k] <= i and m[k] == terminal_map[k]]
        covered.update(valid)
        if i > last_any:
            after_all.update(m)
        reads.append({'event_id': e['event_id'], 'event_index': i, 'keys': list(m),
                      'quantities': m, 'still_current_keys': valid,
                      'after_all_mutations': i > last_any,
                      'later_metadata_only_events': [events[j]['event_id'] for j in all_changes if j > i and events[j]['event_id'] in metadata],
                      'jsonpath': f'$.events[{i}].result.records'})
    action = anchor['parse']['action'] if has else None
    if has and action != row['final_action']:
        raise ValueError('accepted final action mapping differs')
    fm, fm_error = strict_map(action['arguments'].get('quantities'), required) if has else (None, 'no_accepted_final')
    match = (fm_error is None and fm == terminal_map) if has else None
    first_match = (fm_error is None and fm == first_map) if has and first_eid is not None and first_reason is None else None
    return {
        'has_accepted_final': int(has), 'anchor_generation': index if anchor else None,
        'anchor_kind': 'ACCEPTED_FINAL_PRE_GENERATION' if has else 'LAST_GENERATION_DIAGNOSTIC',
        'anchor_available_result_event_ids': available,
        'required_record_keys': required, 'scope_id': scope,
        'last_quantity_change_event': {k: events[v]['event_id'] if v >= 0 else None for k, v in last_change.items()},
        'last_state_mutation_event': events[last_any]['event_id'] if last_any >= 0 else None,
        'metadata_only_mutation_events': metadata,
        'observed_read_events': reads, 'rejected_read_events': rejected_reads,
        'current_read_coverage_keys': sorted(covered), 'current_read_coverage': len(covered),
        'required_count': len(required), 'current_quantity_read_observed': int(set(required) == covered),
        'read_after_all_state_mutations_observed': int(set(required) <= after_all),
        'terminal_quantities': terminal_map, 'final_quantities': fm,
        'final_quantity_error': fm_error, 'final_quantity_matches_terminal': match,
        'first_observed_read_event': first_eid, 'first_read_quantities': first_map,
        'first_read_comparison_reason': first_reason,
        'final_matches_first_read': first_match,
        'evidence_paths': {'anchor': f'$.requests[{index}]' if anchor else None,
                           'terminal': '$.terminal_snapshot.records',
                           'final': '$.final_action.arguments.quantities' if has else None}}
