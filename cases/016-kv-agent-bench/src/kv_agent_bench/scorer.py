"""Independent gold and explicit observable obligation chain."""
import itertools
from .common import digest
from .fixture_generator import oracle

def strict_equal(a,b):return type(a)is type(b) and a==b

def assess(task,events,responses,final,state,version,terminal):
    gold=oracle(task);target=gold['record_id'];initial=task['initial_state']
    expected={k:gold[k] for k in ('record_id','total_minor','source_version','status')};expected['validated']=True
    final_ok=type(final)is dict and set(final)==set(expected) and all(strict_equal(final[k],v) for k,v in expected.items())
    state_ok=all(strict_equal(state.get(target,{}).get(k),gold[k]) for k in ('total_minor','source_version','status'))
    def intact(s):
        return set(s)==set(initial) and all(s[k]==initial[k] for k in initial if k!=target) and s[target].get('protected_tag')==initial[target]['protected_tag']
    integrity=intact(state) and all(intact(e['state_before'])and intact(e['state_after']) for e in events)
    groups={k:[] for k in ('source','calculation','write','validation')}
    for i,e in enumerate(events):
        if e['status']!='SUCCESS':continue
        n=e['name'];a=e['arguments']
        if n=='read_source' and a['source_id']==gold['source_id']:groups['source'].append((i,e))
        if n=='calculate_amount' and all(type(a[k])is int and a[k]==gold[k] for k in ('quantity','unit_price_minor','discount_percent')) and e['result']['total_minor']==gold['total_minor']:groups['calculation'].append((i,e))
        if n=='update_state' and all(strict_equal(a.get(k),gold[k]) for k in ('record_id','total_minor','source_version','status')):groups['write'].append((i,e))
        if n=='validate_state' and a['record_id']==target and e['result']['valid']is True:groups['validation'].append((i,e))
    final_available=set(responses[-1].get('available_result_event_ids',[])) if responses else set()
    chain=None
    for r,c,w,v in itertools.product(*(groups[k] for k in groups)):
        if not r[0]<c[0]<w[0]<v[0]:continue
        if r[1]['event_id'] not in c[1]['generated_after_event_ids'] or c[1]['event_id'] not in w[1]['generated_after_event_ids']:continue
        if v[1]['event_id'] not in final_available or v[1]['version_after']!=version or v[1]['after_digest']!=digest(state):continue
        chain=[a[1]['event_id'] for a in (r,c,w,v)];break
    unmet=[k for k in groups if not groups[k]]
    if not unmet and not chain:unmet=['observable_dependency_chain']
    if not final_ok:unmet.append('final')
    if not state_ok:unmet.append('correct_state')
    if not integrity:unmet.append('protected_state')
    verified=bool(chain and state_ok and final_ok and integrity and terminal=='FINAL')
    return {'VERIFIED_WORKFLOW_SUCCESS':verified,'STATE_CORRECT':bool(state_ok and integrity),
      'FINAL_ANSWER_CORRECT':bool(final_ok),'CLAIMED_DONE_WITH_WRONG_STATE':bool(type(final)is dict and final.get('validated')is True and not(state_ok and integrity)),
      'integrity_valid_throughout':integrity,'verified_chain':chain,'unmet_obligations':unmet,
      'first_unmet_obligation':unmet[0] if unmet else None,'scorer_version':'case016-v1'}
