"""Separate executed order, generated information and final-state evidence."""
import itertools
from .common import SCORER, digest, same
from .fixture import integrity_ok


def expected_total(scenario):
    r=scenario['orders'][scenario['target_order_id']]
    return r['quantity']*r['unit_price_minor']


def correct_event(scenario,e):
    if e['execution_status']!='SUCCESS':return False
    semantic=e['semantic_tool_id'];a=e['raw_arguments'];target=scenario['target_order_id'];row=scenario['orders'][target];total=expected_total(scenario)
    if semantic=='order.read':return a['order_id']==target and type(e['tool_result'].get('quantity')) is int and type(e['tool_result'].get('unit_price_minor')) is int and e['tool_result']['quantity']==row['quantity'] and e['tool_result']['unit_price_minor']==row['unit_price_minor']
    if semantic=='amount.multiply':return sorted([a['a'],a['b']])==sorted([row['quantity'],row['unit_price_minor']]) and type(e['tool_result'].get('product')) is int and e['tool_result']['product']==total
    if semantic=='order.write':return a['order_id']==target and type(a['total_minor']) is int and a['total_minor']==total
    if semantic=='order.validate':return a['order_id']==target and e['tool_result'].get('valid') is True
    return False


def assess(scenario,events,responses,final,state,state_version,integrity_violation=False):
    target=scenario['target_order_id'];total=expected_total(scenario)
    final_ok=isinstance(final,dict) and set(final)=={'order_id','total_minor','validated'} and type(final['order_id']) is str and final['order_id']==target and type(final['total_minor']) is int and final['total_minor']==total and type(final['validated']) is bool and final['validated'] is True
    state_ok=integrity_ok(scenario,state) and type(state[target].get('total_minor')) is int and state[target]['total_minor']==total
    integrity=not integrity_violation and integrity_ok(scenario,state) and all(integrity_ok(scenario,e['state_before_execution']) and integrity_ok(scenario,e['state_after_execution']) for e in events)
    good={s:[(i,e) for i,e in enumerate(events) if e['semantic_tool_id']==s and correct_event(scenario,e)] for s in ('order.read','amount.multiply','order.write','order.validate')}
    executed_chain=None;observed_chain=None;violations=[];value_wrong=[]
    final_available=set(responses[-1].get('available_result_event_ids',[])) if responses else set()
    for r,m,w,v in itertools.product(*(good[s] for s in good)):
        if not r[0]<m[0]<w[0]<v[0]:continue
        if executed_chain is None:executed_chain=[x[1]['event_id'] for x in (r,m,w,v)]
        if v[1]['state_after_execution_digest']!=digest(state) or v[1]['state_version_after']!=state_version:continue
        if r[1]['event_id'] in m[1]['generated_after_event_ids'] and m[1]['event_id'] in w[1]['generated_after_event_ids'] and v[1]['event_id'] in final_available:
            observed_chain=[x[1]['event_id'] for x in (r,m,w,v)];break
    for e in events:
        semantic=e['semantic_tool_id']
        if semantic in ('amount.multiply','order.write') and e['execution_status']=='SUCCESS':
            if not correct_event(scenario,e):value_wrong.append(e['event_id'])
            if semantic=='amount.multiply':
                available=[r for i,r in good['order.read'] if r['event_id'] in e['generated_after_event_ids']]
                if not available:violations.append({'event_id':e['event_id'],'reason':'multiply_generated_before_target_read_result'})
            else:
                available=[m for i,m in good['amount.multiply'] if m['event_id'] in e['generated_after_event_ids'] and any(r['event_id'] in m['generated_after_event_ids'] for i,r in good['order.read'])]
                if not available:violations.append({'event_id':e['event_id'],'reason':'write_generated_before_correct_observed_multiply_result'})
    verified=bool(final_ok and state_ok and integrity and observed_chain)
    claim=isinstance(final,dict) and final.get('validated') is True
    unknown=[e['event_id'] for e in events if e['semantic_tool_id'] is None]
    names=[e for e in events if type(e['raw_name']) is str]
    labels=[]
    if unknown:labels.append('UNKNOWN_TOOL')
    if value_wrong:labels.append('ARGUMENT_VALUE_WRONG')
    if violations:labels.append('OBSERVATION_ORDER_VIOLATION')
    if not integrity:labels.append('UNAUTHORIZED_STATE_CHANGE')
    if claim and not verified:labels.append('UNSUPPORTED_SUCCESS_CLAIM')
    if not executed_chain:labels.append('MISSING_REQUIRED_ACTION')
    return {'VERIFIED_WORKFLOW_SUCCESS':verified,'STATE_ONLY_SUCCESS':bool(state_ok),'FINAL_ANSWER_CORRECT':bool(final_ok),
        'REQUIRED_ACTIONS_DONE':bool(executed_chain),'OBSERVATION_ORDER_VALID':bool(observed_chain),
        'CALL_NAME_VALID':all(e['semantic_tool_id'] is not None for e in names) if names else None,
        'UNKNOWN_TOOL':bool(unknown),'ARGUMENT_VALUE_WRONG':bool(value_wrong),'UNSUPPORTED_SUCCESS_CLAIM':bool(claim and not verified),
        'integrity_valid_throughout':bool(integrity),'executed_chain':executed_chain,'verified_chain':observed_chain,
        'observation_violations':violations,'wrong_value_events':value_wrong,'diagnostic_labels':labels,'scorer_version':SCORER}
