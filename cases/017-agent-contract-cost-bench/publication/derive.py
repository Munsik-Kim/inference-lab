"""Read lossless recorded episodes into display records. No model dependencies."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import functools
import hashlib
import importlib.util
import json
import math
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('case017_derive_verify', HERE/'verify.py')
v = importlib.util.module_from_spec(spec); spec.loader.exec_module(v)
CASE, original_bytes, read, sha = v.CASE, v.original_bytes, v.read, v.sha

SELECTED = ('HOLDOUT-A00_K1_C0', 'HOLDOUT-A00_K1_G', 'HOLDOUT-B01_K0_G',
            'HOLDOUT-C00_K1_G', 'HOLDOUT-B03_K1_N', 'HOLDOUT-B03_K1_G', 'HOLDOUT-B03_K1_E')

def digest(v):
    return sha(json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())

def fresh(task, episode):
    """Posthoc current-state validation flag, separate from the original endpoint."""
    goal = task['goal']
    names = {'A': ('validate_order', ('order_id',)), 'B': ('validate_transfer', ('source_id','destination_id','sku')),
             'C': ('validate_document', ('document_id',))}
    name, keys = names[task['family']]
    available = episode['final_available_result_event_ids'] if episode.get('final_action') else episode['requests'][-1]['available_result_event_ids']
    return any(e['tool'] == name and all(e['arguments'].get(k) == goal[k] for k in keys)
               and e['status'] == 'SUCCESS' and e['result'].get('valid') is True
               and e['after_digest'] == digest(episode['final_state']) and e['version_after'] == episode['state_version']
               and e['event_id'] in available for e in episode['events'])

def project(case=CASE):
    case = Path(case)
    # Include file contents in the cache key; a rehashed presentation cannot hide changed raw.
    paths = [case/'publication/provenance/original_mapping.json', case/'publication/data/original-evidence.tar.xz',
             *sorted((case/'data').glob('*.json')), *sorted((case/'src/agent_contract_cost').glob('*.py')),
             case/'results/p1_r1/analysis/summary.json', case/'results/p2_r1/analysis/summary.json']
    key = tuple((str(p), sha(p.read_bytes())) for p in paths)
    return json.loads(_project(str(case), key))

@functools.lru_cache(maxsize=3)
def _project(case_path, _key):
    case = Path(case_path)
    sys.path.insert(0, str(case/'src'))
    from agent_contract_cost.audit import independent_endpoint
    tasks = {t['task_id']: t for split in ('dev','holdout') for t in read(case/f'data/{split}_tasks.json')}
    audited = {(r['stage'],r['task_id'],r['kv'],r['policy']):r for s in ('p1_r1','p2_r1')
               for r in read(case/f'results/{s}/audit/audited_rows.json')}
    rows=[]; selected=[]; sources=[]; p0=[]; cache=[]
    for source,b in original_bytes(case):
        path=source['path']
        if path.startswith('results/p0_r1/') and path.endswith('.json'):
            obj=json.loads(b)
            if 'response_id' in obj: p0.append(obj)
            if path.endswith('_smoke_episode.json'): p0.extend(r['response'] for r in obj['requests'])
            if path.endswith('_cache_after.json'):
                cfg=obj['cache_config'];slots=cfg['num_gpu_blocks']*cfg['block_size'];pool=obj['actual_unique_storage_bytes']
                cache.append({'kv':Path(path).name.split('_')[0], 'pool_bytes':pool,'slots':slots,
                              'bytes_per_token':pool//slots,'block_size':cfg['block_size'],'layers':len({t['name'] for t in obj['cache_tensors']}),
                              'source_file':path,'source_sha256':source['sha256']})
        if '/cells/' not in path or not path.endswith('.json'): continue
        e=json.loads(b);t=tasks[e['task_id']]
        ep=independent_endpoint(t,e['events'],e['final_action'],e['final_state'],e['state_version'],
                                e['final_available_result_event_ids'],e['terminal_reason'])
        old=audited[(e['stage'],e['task_id'],e['kv'],e['policy'])]
        if old['source_sha256'] != digest(e) or not old['audit_pass']:
            raise ValueError('Raw no longer matches original independent audit')
        for k in ('VERIFIED_WORKFLOW_SUCCESS','STATE_CORRECT','FINAL_ANSWER_CORRECT'):
            if ep[k] != e['reported_endpoint'][k] or ep[k] != old['endpoint'][k]:
                raise ValueError('Original endpoint discrepancy')
        requests=e['requests']
        if any(r['response']['usage']['prompt_tokens'] != len(r['input_token_ids']) or
               r['response']['usage']['completion_tokens'] != len(r['response']['output_token_ids']) for r in requests):
            raise ValueError('Usage denominator differs from saved token IDs')
        row={'stage':e['stage'],'task_id':t['task_id'],'family':t['family'],'kv':e['kv'],'policy':e['policy'],
             'verified':int(ep['VERIFIED_WORKFLOW_SUCCESS']),'state_correct':int(ep['STATE_CORRECT']),
             'final_correct':int(ep['FINAL_ANSWER_CORRECT']),'fresh_validation':int(fresh(t,e)),
             'requests':len(requests),'input_tokens':sum(len(r['input_token_ids']) for r in requests),
             'output_tokens':sum(len(r['response']['output_token_ids']) for r in requests),
             'wall_seconds':e['wall_seconds'],'terminal_reason':e['terminal_reason'],
             'first_unmet_obligation':e['reported_endpoint']['first_unmet_obligation'],
             'source_file':path,'source_sha256':source['sha256']}
        if row['requests'] != e['budget']['model_request_attempts']:
            raise ValueError('Attempt count differs')
        rows.append(row)
        sources.append({'path':path,'sha256':source['sha256'],'bytes':source['bytes']})
        record_key=Path(path).stem
        if record_key in SELECTED:
            selected.append({'episode_id':record_key,'recorded_episode_id':e['episode_id'],'task_id':e['task_id'],'family':e['family'],
                'kv':e['kv'],'policy':e['policy'],'request':t['prompt'],'initial_state':t['initial_state'],
                'goal':t['goal'],'final_state':e['final_state'],'state_version':e['state_version'],
                'scores':row,'final_action':e['final_action'],
                'turns':[{'generation_turn':r['generation_turn'],'raw_text':r['response']['raw_text'],
                    'raw_utf8_sha256':r['response']['raw_utf8_sha256'],'input_token_sha256':r['input_token_sha256'],
                    'available_result_event_ids':r['available_result_event_ids'],'observation_sha256':r['observation_sha256'],
                    'state_digest':r['state_digest'],'state_version':r['state_version'],'parse':r['parse'],
                    'feedback':r.get('feedback'),'event':r.get('event')} for r in requests]})
    rows.sort(key=lambda r:(r['stage'],r['task_id'],r['kv'],r['policy']))
    if len(rows)!=256 or len({(r['stage'],r['task_id'],r['kv'],r['policy']) for r in rows})!=256:
        raise ValueError('Incomplete or duplicate episode matrix')
    groups=[];comparisons=[]
    for stage in ('P1','P2'):
        original=read(case/f'results/{stage.lower()}_r1/analysis/summary.json')
        for kv in ('K0','K1'):
            for policy in ('C0','G','N','E'):
                rr=[r for r in rows if (r['stage'],r['kv'],r['policy'])==(stage,kv,policy)]
                g={'stage':stage,'kv':kv,'policy':policy,'n':len(rr)}
                for k in ('verified','state_correct','final_correct','fresh_validation','requests','input_tokens','output_tokens'):
                    g[k]=sum(r[k] for r in rr)
                g['wall_seconds']=math.fsum(r['wall_seconds'] for r in rr)
                for k in ('input_tokens','output_tokens','wall_seconds'):
                    g[k+'_per_verified']=g[k]/g['verified'] if g['verified'] else None
                g['family_verified']={f:sum(r['verified'] for r in rr if r['family']==f) for f in ('A','B','C')}
                old=next(r for r in original['groups'] if (r['kv'],r['policy'])==(kv,policy))
                if any(g[a]!=old[b] for a,b in [('verified','verified_success'),('state_correct','state_correct'),('final_correct','final_correct'),('requests','model_request_attempts'),('output_tokens','generated_output_tokens')]):
                    raise ValueError('Raw grouping differs from retained original analysis')
                groups.append(g)
        lookup={(r['task_id'],r['kv'],r['policy']):r for r in rows if r['stage']==stage}
        ids=sorted({r['task_id'] for r in rows if r['stage']==stage})
        for old in original['comparisons']:
            counts=dict.fromkeys(('both_success','loss','recovery','both_fail'),0)
            for task in ids:
                a=lookup[(task,old['candidate_kv'],old['candidate_policy'])]['verified']
                b=lookup[(task,old['baseline_kv'],old['baseline_policy'])]['verified']
                counts['both_success' if a and b else 'loss' if b else 'recovery' if a else 'both_fail']+=1
            if any(counts[k]!=old['counts'].get(k,0) for k in counts): raise ValueError('Paired counts differ')
            comparisons.append({'stage':stage,'comparison_id':old['comparison_id'],'n':len(ids),**counts,
                'difference_pp':100*(counts['recovery']-counts['loss'])/len(ids),
                'ci_95_pp':[100*x for x in old['ci_95_fraction']],
                'ci_source':f'results/{stage.lower()}_r1/analysis/summary.json',
                'ci_type':old['ci_type'],'bootstrap':original['bootstrap']})
    p0_tokens=sum(len(r['output_token_ids']) for r in p0)
    if len(p0)!=22 or p0_tokens!=678: raise ValueError('P0 request census differs')
    worker=[json.loads(s) for s in (case/'provenance/worker_journal.jsonl').read_text().splitlines()]
    workers={r['stage']:r['worker_wall_seconds'] for r in worker if r['event']=='END'}
    freezes=[read(case/f'provenance/execution_freeze_{s}.json') for s in ('p1','p2')]
    common=set(r.get('root_path',r.get('path')) for r in freezes[0]['files']) & set(r.get('root_path',r.get('path')) for r in freezes[1]['files'])
    fmaps=[{r.get('root_path',r.get('path')):r for r in f['files']} for f in freezes]
    if any(fmaps[0][k]!=fmaps[1][k] for k in common): raise ValueError('Common freeze differs')
    return json.dumps({'schema':1,'model':'Qwen/Qwen3-4B-Instruct-2507',
       'model_revision':read(case/'provenance/model_identity.json')['revision'],
       'execution_source_sha256':freezes[1]['execution_source_sha256'],
       'evidence_kind':'STORED_MODEL_MEASUREMENT; CPU endpoint recalculation; Fresh posthoc',
       'rows':rows,'groups':groups,'comparisons':comparisons,'selected_episodes':sorted(selected,key=lambda r:SELECTED.index(r['episode_id'])),
       'p0':{'requests':len(p0),'output_tokens':p0_tokens,'input_tokens':sum(len(r['prompt_token_ids']) for r in p0)},
       'totals':{'requests':len(p0)+sum(r['requests'] for r in rows),'output_tokens':p0_tokens+sum(r['output_tokens'] for r in rows),
                 'worker_wall_seconds':math.fsum(workers.values()),'worker_by_phase':workers},
       'capacity':sorted(cache,key=lambda r:r['kv']),
       'gate':{'required_dev_verified':6,'observed_dev_verified':5,'denominator':8,'success_threshold_in_local_gate':False,
               'separate_user_waiver_record':'NOT_FOUND','p2_interpretation':'EXPLORATORY_WITH_DISCLOSED_PROTOCOL_DEVIATION',
               'common_frozen_files':len(common),'common_frozen_mismatches':0,'holdout_seal':read(case/'provenance/holdout_seal.json')},
       'original_sources':sources,'scope':{'new_publication_model_requests':0,'P3':'NOT_RUN',
           'retokenization':'NOT_RUN; stored IDs/hash/server usage checked','GPU_inference_reproduction':'NOT_RUN',
           'fresh_validation':'POSTHOC_DERIVED; not original registered primary','endpoint_replay':'All 256 full raw records retained losslessly'}},ensure_ascii=False,sort_keys=True,allow_nan=False)

def export(case, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    data=project(case)
    (output/'project.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    for name,rows in [('episode_ledger',data['rows']),('policy_results',data['groups']),('paired_comparisons',data['comparisons'])]:
        with (output/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    return data

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=export(CASE,a.output);print(json.dumps({'episodes':len(d['rows']),'new_model_requests':0}))
