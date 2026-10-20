"""CPU replay from raw SSE and pristine fixtures, without trusting saved scores."""
import copy
import json
import pathlib
from .common import digest,load,save
from .tools import Fixture
from .vllm_client import StreamAssembly
from .scorer import assess

def independent_endpoint(task,events,final,state,version,terminal,responses):
    v=task['visible'];eligible=[s for s in v['sources'] if s['arrival_time']<=v['cutoff']]
    if task['family']=='A':s=sorted(eligible,key=lambda x:(x['unit_price_minor'],x['source_id']))[0]
    elif task['family']=='B':s=sorted(eligible,key=lambda x:x['version'])[-1]
    elif task['family']=='C':s=sorted(eligible,key=lambda x:(x['event_time'],x['version']))[-1]
    else:s=sorted(eligible,key=lambda x:(x['version'],int(x['kind']=='cancel')))[-1]
    q=0 if s['kind']=='cancel' else v['quantity'];price=s['unit_price_minor'];pct=v['discount_percent']
    exact=(q*price*(100-pct)+50)//100;status='cancelled' if s['kind']=='cancel' else 'active';target=v['record_id']
    expected={'record_id':target,'total_minor':exact,'source_version':s['version'],'status':status,'validated':True}
    final_ok=type(final)is dict and set(final)==set(expected) and all(type(final[k])is type(a) and final[k]==a for k,a in expected.items())
    state_ok=all(type(state[target].get(k))is type(a) and state[target].get(k)==a for k,a in expected.items() if k not in ('record_id','validated'))
    initial=task['initial_state'];intact=True
    for st in [state]+[e[k] for e in events for k in ('state_before','state_after')]:
        intact &= set(st)==set(initial) and all(st[k]==initial[k] for k in initial if k!=target) and st[target]['protected_tag']==initial[target]['protected_tag']
    reads=set();calculations=set();writes=set();validations=set()
    for e in events:
        if e['status']!='SUCCESS':continue
        a=e['arguments'];n=e['name'];seen=set(e['generated_after_event_ids'])
        if n=='read_source' and a=={'source_id':s['source_id']}:reads.add(e['event_id'])
        elif n=='calculate_amount' and a=={'quantity':q,'unit_price_minor':price,'discount_percent':pct} and seen&reads:calculations.add(e['event_id'])
        elif n=='update_state' and a=={k:a for k,a in expected.items() if k!='validated'} and seen&calculations:writes.add(e['event_id'])
        elif n=='validate_state' and a=={'record_id':target} and e['result']['valid']is True and writes and e['version_after']==version and e['state_after']==state:validations.add(e['event_id'])
    seen=set(responses[-1]['available_result_event_ids']) if responses else set()
    return bool(terminal=='FINAL' and final_ok and state_ok and intact and validations&seen)

def replay(task,episode,root):
    fx=Fixture(task,root,episode['menu']);available=[];responses=[];final=None;attempts=0;unaccepted=0
    for r in episode['responses']:
        if r['available_result_event_ids']!=available:raise ValueError('observation event ID mismatch')
        if r['state_digest']!=digest(fx.state) or r['state_version']!=fx.version:raise ValueError('pre-generation state mismatch')
        a=StreamAssembly()
        for line in r['raw_sse_lines']:
            if line.startswith('data: ') and line[6:]!='[DONE]':a.add(json.loads(line[6:]),0)
        built=a.result()
        if built['tool_calls']!=r['tool_calls'] or built['output_token_ids']!=r['output_token_ids']:raise ValueError('SSE reconstruction mismatch')
        if built['content']!=r['content']:raise ValueError('content mismatch')
        responses.append(r);frozen=tuple(available);new=[]
        marker_count=(r['raw_text']or'').count('<tool_call>')
        slots=[]
        for c in built['tool_calls']:
            try:args=json.loads(c['function']['arguments'])
            except ValueError:args=None
            slots.append((c['function']['name'],args))
        slots += [(None,None)]*max(0,marker_count-len(slots))
        if episode.get('slot_order_policy')=='raw_hermes_slots_v2':
            from .observations import ordered_slots
            ordered=ordered_slots(r['raw_text'],built['tool_calls'])
            slots=[(s['parsed']['name'],s['parsed']['arguments']) if s['parsed']is not None else (None,None) for s in ordered]
        for i,(name,args) in enumerate(slots):
            if attempts>=6:break
            attempts+=1
            if type(args)is not dict:unaccepted+=1;continue
            e=fx.execute(name,args,r['turn'],i,frozen);new.append(e['event_id'])
        available.extend(new)
        if not slots:
            try:final=json.loads(built['content'].strip())
            except ValueError:final=None
    if fx.events!=episode['events'] or fx.state!=episode['final_state']:raise ValueError('tool event/state replay mismatch')
    if attempts!=episode['tool_attempts']:raise ValueError('attempt denominator mismatch')
    verdict=assess(task,fx.events,responses,final,fx.state,fx.version,episode['terminal_reason'])
    independent=independent_endpoint(task,fx.events,final,fx.state,fx.version,episode['terminal_reason'],responses)
    saved=episode['VERIFIED_WORKFLOW_SUCCESS']
    if episode['availability']=='OBSERVED' and (verdict['VERIFIED_WORKFLOW_SUCCESS']!=saved or independent!=saved):raise ValueError('endpoint disagreement')
    return {'episode_id':episode['episode_id'],'replayed_events':len(fx.events),'unaccepted_slots':unaccepted,'verified':independent,
      'raw_sse_reassembled':True,'gpu_inference_reproduced':False,'token_text_decode_reproduced':False}

def audit_directory(tasks_path,result_root,output):
    tasks={x['task_id']:x for x in load(tasks_path)};result_root=pathlib.Path(result_root);out=pathlib.Path(output)
    if out.exists():raise ValueError('fresh audit output required')
    out.mkdir(parents=True);rows=[];seen=set()
    for p in sorted((result_root/'episodes').glob('*.json')):
        e=load(p)
        if e['episode_id']in seen:raise ValueError('duplicate episode')
        seen.add(e['episode_id']);rows.append(replay(tasks[e['task_id']],e,out/'fixtures'/e['episode_id']))
    receipt={'status':'PASS','episodes':len(rows),'events':sum(x['replayed_events'] for x in rows),'rows':rows,
      'scope':'Raw SSE reassembly, fixture replay, independent scalar endpoint; no GPU/logit reconstruction.'}
    save(out/'audit.json',receipt);return receipt
