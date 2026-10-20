#!/usr/bin/env python3
"""Independent CPU reconstruction of raw request, termination and score tables.

No ML imports, model files or network. Never trusts stored success booleans.
"""
import argparse
import copy
import hashlib
import json
import pathlib
import struct
import sys
import tempfile

CASE=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CASE/'src'))
from kv_agent_bench.common import load,save,digest
from kv_agent_bench.tools import Fixture
from kv_agent_bench.scorer import assess
from kv_agent_bench.audit import independent_endpoint
from kv_agent_bench.observations import ordered_slots

def reconstruct(r):
    calls={};content='';ids=[];prompt=None;usage=None;reason=None
    for line in r['raw_sse_lines']:
        if not line.startswith('data: ') or line[6:]=='[DONE]':continue
        obj=json.loads(line[6:])
        if obj.get('usage')is not None:usage=obj['usage']
        if obj.get('prompt_token_ids')is not None:prompt=obj['prompt_token_ids']
        for choice in obj.get('choices',[]):
            ids.extend(choice.get('token_ids')or[]);delta=choice.get('delta')or{};content+=delta.get('content')or''
            if choice.get('finish_reason')is not None:reason=choice['finish_reason']
            for part in delta.get('tool_calls')or[]:
                c=calls.setdefault(part['index'],{'index':part['index'],'id':None,'type':'function','function':{'name':'','arguments':''}})
                if part.get('id'):c['id']=part['id']
                f=part.get('function')or{}
                for k in ('name','arguments'):c['function'][k]+=f.get(k)or''
    result={'tool_calls':[calls[i]for i in sorted(calls)],'content':content,'output_token_ids':ids,'prompt_token_ids':prompt,'usage':usage,'server_finish_reason':reason}
    for k,v in result.items():
        if r.get(k)!=v:raise AssertionError('independent raw SSE assembly mismatch: '+k)
    if usage['completion_tokens']!=len(ids) or usage['prompt_tokens']!=len(prompt):raise AssertionError('usage/token denominator mismatch')
    if hashlib.sha256((r['raw_text']or'').encode()).hexdigest()!=r['raw_utf8_sha256']:raise AssertionError('raw UTF8 hash mismatch')
    return result

def replay_one(task,e,root):
    fx=Fixture(task,root,e['menu']);seen=[];attempts=0;tokens=0;responses=[];terminal='UNKNOWN';final=None;per_request=[]
    for r in e['responses']:
        if terminal!='UNKNOWN':raise AssertionError('response after terminal')
        if r['available_result_event_ids']!=seen or r['state_digest']!=digest(fx.state) or r['state_version']!=fx.version:raise AssertionError('observation/state mismatch')
        built=reconstruct(r);actual_prompt=len(built['prompt_token_ids']);remaining=8192-tokens
        expected_cap=min(e['cap'],remaining,28672-actual_prompt)
        if r['request_cap']!=expected_cap:raise AssertionError('request cap mismatch')
        if len(built['output_token_ids'])>expected_cap:raise AssertionError('engine token-cap violation')
        tokens+=len(built['output_token_ids']);responses.append(r)
        if e.get('slot_order_policy')=='raw_hermes_slots_v2':slots=ordered_slots(r['raw_text'],built['tool_calls'])
        else:
            slots=[]
            for call in built['tool_calls']:
                try:a=json.loads(call['function']['arguments']);parsed={'name':call['function']['name'],'arguments':a}if type(a)is dict else None
                except ValueError:parsed=None
                slots.append({'parsed':parsed})
            slots += [{'parsed':None}]*max(0,r['raw_text'].count('<tool_call>')-len(slots))
        new=[];accepted=0;frozen=tuple(seen)
        if slots:
            for i,s in enumerate(slots):
                if attempts==6:terminal='TOOL_CALL_LIMIT';break
                attempts+=1
                if s['parsed']is None:continue
                obj=s['parsed'];event=fx.execute(obj['name'],obj['arguments'],r['turn'],i,frozen);new.append(event['event_id']);accepted+=1
            if accepted:seen.extend(new)
            elif terminal=='UNKNOWN':terminal='PARSE_STOP'
        else:
            try:obj=json.loads(built['content'].strip());final=obj if type(obj)is dict else None
            except ValueError:final=None
            terminal='FINAL' if final is not None else 'TOKEN_LIMIT' if built['server_finish_reason']=='length' else 'PARSE_STOP'
        per_request.append({'turn':r['turn'],'raw_tokens':len(built['output_token_ids']),'prompt_tokens':actual_prompt,'cap':expected_cap,'server_finish':built['server_finish_reason'],'parsed_slots':sum(s['parsed']is not None for s in slots),'unaccepted_slots':sum(s['parsed']is None for s in slots)})
    if terminal=='UNKNOWN':
        if len(responses)==8:terminal='MODEL_CALL_LIMIT'
        elif e['availability']!='OBSERVED':terminal=e['terminal_reason']
        elif tokens>=8192:terminal='TOKEN_LIMIT'
        else:raise AssertionError('unexplained observed terminal')
    if terminal!=e['terminal_reason']:raise AssertionError('terminal classification differs')
    if fx.events!=e['events'] or fx.state!=e['final_state'] or fx.version!=e['final_state_version']:raise AssertionError('fresh fixture replay mismatch')
    if attempts!=e['tool_attempts'] or tokens!=e['output_tokens'] or len(responses)!=e['model_calls']:raise AssertionError('cost denominator mismatch')
    verdict=assess(task,fx.events,responses,final,fx.state,fx.version,terminal)
    other=independent_endpoint(task,fx.events,final,fx.state,fx.version,terminal,responses)
    if other!=verdict['VERIFIED_WORKFLOW_SUCCESS']:raise AssertionError('independent endpoint differs')
    for k in ('VERIFIED_WORKFLOW_SUCCESS','STATE_CORRECT','FINAL_ANSWER_CORRECT','CLAIMED_DONE_WITH_WRONG_STATE','first_unmet_obligation'):
        if verdict[k]!=e[k]:raise AssertionError('saved endpoint differs: '+k)
    return {'episode_id':e['episode_id'],'task_id':e['task_id'],'kv':e['kv'],'menu':e['menu'],'cap':e['cap'],'family':e['family'],
      'verified':other,'state':verdict['STATE_CORRECT'],'final':verdict['FINAL_ANSWER_CORRECT'],'false_done':verdict['CLAIMED_DONE_WITH_WRONG_STATE'],
      'first_unmet':verdict['first_unmet_obligation'],'terminal':terminal,'model_calls':len(responses),'tool_attempts':attempts,'tool_events':len(fx.events),
      'tokens':tokens,'requests':per_request,'initial_token_hash':hashlib.sha256(b''.join(struct.pack('<I',i)for i in responses[0]['prompt_token_ids'])).hexdigest() if responses else None}

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=pathlib.Path,default=CASE);p.add_argument('--run',default='p1_r1');p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('fresh output directory required')
    args.output.mkdir(parents=True);case=args.case.resolve();tasks={t['task_id']:t for t in load(case/'data/tasks.json')};root=case/'results'/args.run
    planned=load(root/'planned_cells.json');rows=[];seen=set()
    for file in sorted((root/'episodes').glob('*.json')):
        e=load(file)
        if e['episode_id']in seen:raise AssertionError('duplicate episode')
        seen.add(e['episode_id']);row=replay_one(tasks[e['task_id']],e,args.output/'fixtures'/e['episode_id']);rows.append(row)
    if seen!={r['episode_id']for r in planned}:raise AssertionError('missing or unplanned cells')
    groups=[]
    for kv in ('B16','F4'):
        for menu in (8,32):
            for cap in (256,1024):
                r=[x for x in rows if (x['kv'],x['menu'],x['cap'])==(kv,menu,cap)]
                if len(r)!=12:raise AssertionError('wrong cell denominator')
                groups.append({'kv':kv,'menu':menu,'cap':cap,'n':12,'verified':sum(x['verified']for x in r),'state':sum(x['state']for x in r),'final':sum(x['final']for x in r),'false_done':sum(x['false_done']for x in r)})
    for key in {(r['task_id'],r['menu'])for r in rows}:
        paired=[r for r in rows if(r['task_id'],r['menu'])==key]
        if len({r['initial_token_hash']for r in paired})!=1:raise AssertionError('initial KV/cap input identity mismatch')
    lookup={(r['task_id'],r['kv'],r['menu'],r['cap']):r for r in rows};tids=sorted({r['task_id']for r in rows});pairs=[]
    def paired(a,b,label):
        counts=dict.fromkeys(['both_success','loss','recovery','both_fail'],0)
        for ka,kb in zip(a,b):
            x=lookup[ka]['verified'];y=lookup[kb]['verified'];counts['both_success'if x and y else'loss'if x else'recovery'if y else'both_fail']+=1
        pairs.append({'comparison':label,'n':12,**counts})
    for kv in ('B16','F4'):
        for m in (8,32):paired([(t,kv,m,256)for t in tids],[(t,kv,m,1024)for t in tids],f'{kv}/menu{m}: cap256 to1024')
    for m in (8,32):
        for cap in (256,1024):paired([(t,'B16',m,cap)for t in tids],[(t,'F4',m,cap)for t in tids],f'menu{m}/cap{cap}: B16 toF4')
    original=load(root/'summary.json')
    for a in groups:
        b=next(x for x in original['groups']if(x['kv'],x['menu'],x['cap'])==(a['kv'],a['menu'],a['cap']))
        for key in ('verified','state','final','false_done'):
            if a[key]!=b[key]:raise AssertionError('summary/raw mismatch')
    for a,b in zip(pairs,original['paired']):
        if a['comparison']!=b['comparison'] or any(a[k]!=b[k]for k in ('both_success','loss','recovery','both_fail')):raise AssertionError('paired summary mismatch')
    save(args.output/'audit.json',{'status':'PASS','episodes':len(rows),'model_calls':sum(r['model_calls']for r in rows),'tool_attempts':sum(r['tool_attempts']for r in rows),'tool_events':sum(r['tool_events']for r in rows),'tokens':sum(r['tokens']for r in rows),'groups':groups,'paired':pairs,'rows':rows,'same_initial_input_groups':24,'gpu_model_forward_reproduced':False,'token_id_to_text_redecoded':False,'scope':'Independent SSE/token/count/terminal reconstruction, fresh fixture replay, separate scalar endpoint and paired-table calculation.'})
    print(json.dumps({'status':'PASS','episodes':len(rows),'model_calls':sum(r['model_calls']for r in rows),'scope':'CPU raw replay; no GPU inference reproduced'}))

if __name__=='__main__':main()
