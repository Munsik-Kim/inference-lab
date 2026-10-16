"""Reparse raw responses and reconstruct file state and observation receipt."""
import copy
import hashlib
import tempfile
from pathlib import Path
from .common import ROOT,read,rows,require,digest,parse_slots,canonical
from .runner import run_episode
from .scoring import expected_total
from .analysis import summarize


def independent_endpoint(scenario,row):
    target=scenario['target_order_id'];total=expected_total(scenario);original=scenario['orders'];final=row['final_answer'];state=row['final_fixture']
    answer=isinstance(final,dict) and sorted(final)==['order_id','total_minor','validated'] and type(final['order_id']) is str and final['order_id']==target and type(final['total_minor']) is int and final['total_minor']==total and final['validated'] is True
    if not answer:return False
    states=[state]+[x for e in row['tool_events'] for x in [e['state_before_execution'],e['state_after_execution']]]
    for st in states:
        if set(st)!=set(original):return False
        for oid in original:
            left=st[oid].copy();right=original[oid].copy()
            if oid==target:left.pop('total_minor',None);right.pop('total_minor',None)
            if canonical(left)!=canonical(right):return False
    if type(state[target]['total_minor']) is not int or state[target]['total_minor']!=total:return False
    events=row['tool_events'];available=set(row['responses'][-1]['available_result_event_ids']) if row['responses'] else set();r0=original[target]
    for vi,v in enumerate(events):
        if v['semantic_tool_id']!='order.validate' or v['execution_status']!='SUCCESS' or v['tool_result'].get('valid') is not True or v['raw_arguments']['order_id']!=target or v['event_id'] not in available or v['state_version_after']!=row['final_state_version'] or v['state_after_execution_digest']!=digest(state):continue
        for wi,w in enumerate(events[:vi]):
            if w['semantic_tool_id']!='order.write' or w['execution_status']!='SUCCESS' or w['raw_arguments']!={'order_id':target,'total_minor':total}:continue
            for mi,m in enumerate(events[:wi]):
                if m['semantic_tool_id']!='amount.multiply' or m['execution_status']!='SUCCESS' or sorted(m['raw_arguments'].values())!=sorted([r0['quantity'],r0['unit_price_minor']]) or m['event_id'] not in w['generated_after_event_ids']:continue
                for r in events[:mi]:
                    if r['semantic_tool_id']=='order.read' and r['execution_status']=='SUCCESS' and r['raw_arguments'].get('order_id')==target and r['event_id'] in m['generated_after_event_ids'] and r['tool_result'].get('quantity')==r0['quantity'] and r['tool_result'].get('unit_price_minor')==r0['unit_price_minor']:return True
    return False


def audit(root=ROOT):
    root=Path(root);records=rows(root/'results/a1_r1/episodes.jsonl');scenarios={s['scenario_id']:s for s in read(root/'data/scenarios.json')};p=read(root/'specs/protocol.json');count=actions=requests=0
    with tempfile.TemporaryDirectory(prefix='case015-a1-audit-') as tmp:
        for row in records:
            if row['observation_status']!='OBSERVED':continue
            queue=iter(row['responses']);scenario=scenarios[row['scenario_id']]
            def generate(messages,tools,deadline):
                r=copy.deepcopy(next(queue));require(r['messages_tools_hash']==digest({'messages':messages,'tools':tools}),'saved history/registry differs')
                require(hashlib.sha256(r['raw_text'].encode()).hexdigest()==r['raw_utf8_sha256'],'raw response bytes changed')
                require(r['pre_generation_history_hash']==digest(messages),'pre-generation history mismatch')
                return r
            ident={k:row[k] for k in ['method','model_revision','artifact_hash','execution_source_sha256','generation_config_hash']}
            replay=run_episode(scenario,row['api_version'],generate,Path(tmp)/row['episode_id'],p['generation'],ident,row['episode_id'])
            for k in ['tool_events','VERIFIED_WORKFLOW_SUCCESS','STATE_ONLY_SUCCESS','FINAL_ANSWER_CORRECT','REQUIRED_ACTIONS_DONE','OBSERVATION_ORDER_VALID','final_fixture','final_state_version','final_answer','observation_violations','attempted_tool_calls','terminal_reason']:
                require(replay[k]==row[k],'replay mismatch: '+row['episode_id']+' '+k)
            require([r['available_result_event_ids'] for r in replay['responses']]==[r['available_result_event_ids'] for r in row['responses']],'result receipt history mismatch')
            require(independent_endpoint(scenario,replay)==row['VERIFIED_WORKFLOW_SUCCESS'],'independent workflow predicate mismatch')
            count+=1;actions+=len(replay['tool_events']);requests+=len(replay['responses'])
    derived=summarize(records,read(root/'results/a1_r1/planned_cells.json'))
    if (root/'results/a1_r1/summary.json').exists():require(derived==read(root/'results/a1_r1/summary.json'),'raw-derived summary differs')
    return {'status':'PASS','raw_episodes_reexecuted':count,'actual_fixture_calls':actions,'saved_generation_requests':requests,'planned':96,'missing':len(derived['missing']),
        'independent_endpoint_predicate':True,'available_event_ids_checked':True,'state_version_and_integrity_checked':True,'scope':'CPU saved response/observation/file-state replay and scalar audit, not model inference reproduction','fresh_gpu_run_in_source':True}
