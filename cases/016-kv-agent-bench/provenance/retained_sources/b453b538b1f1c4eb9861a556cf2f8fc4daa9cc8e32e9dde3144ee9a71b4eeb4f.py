"""Sequential, preplanned P0/P1 workers. Budgets include every owned server load."""
import asyncio
import hashlib
import json
import pathlib
import random
import time
import traceback
import urllib.request
from .common import load,save,sha_file,digest,utc
from .server_manager import Server
from .episode_runner import run_episode
from .tokenization import get_tokenizer
from .metrics import summarize

def inspect(base):
    b=json.dumps({'method':'inspect_case016_cache','timeout':30}).encode()
    request=urllib.request.Request(base+'/collective_rpc',data=b,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=40)as r:return json.load(r)['results'][0]

def metrics(base):
    try:
        with urllib.request.urlopen(base+'/metrics',timeout=10)as r:return r.read().decode()
    except Exception as e:return 'METRICS_UNOBSERVED '+type(e).__name__

def freeze(case,output,runtime):
    paths=list((case/'src').rglob('*.py'))+list((case/'specs').glob('*.json'))+[case/'data/tasks.json',case/'data/token_manifest.json']
    rows=[{'path':str(p.relative_to(case)),'bytes':p.stat().st_size,'sha256':sha_file(p)}for p in sorted(paths)]
    receipt={'created_utc':utc(),'files':rows,'source_sha256':digest(rows),'runtime_policy_hash':digest({k:v for k,v in runtime.items() if not k.endswith('_root') and k not in ('python','snapshot')}),'git_commit':runtime['source_commit'],'new_git_commit_permitted':False}
    save(output,receipt);return receipt

def support_check(receipt,kv):
    layers=receipt['layers'];tensors=receipt['cache_tensors']
    if len(layers)!=36:raise ValueError('must inspect all 36 attention layers')
    dtype={'B16':'torch.bfloat16','F8':'torch.float8_e4m3fn','F4':'torch.uint8'}[kv]
    # vLLM stores FP8 bytes in uint8 buffers; backend metadata identifies FP8 interpretation.
    if kv=='F8':allowed={'torch.uint8','torch.float8_e4m3fn'}
    else:allowed={dtype}
    if not tensors or any(t['dtype']not in allowed for t in tensors):raise ValueError('actual cache tensor dtype unsupported/missing')
    if kv=='F4' and not all(x['is_kvcache_nvfp4'] and x['nvfp4_fa2'] for x in layers):raise ValueError('SM120 NVFP4 writer/reader path not confirmed')
    if kv=='F8' and not all(x['cache_dtype']=='fp8_e4m3' for x in layers):raise ValueError('FP8 backend interpretation mismatch')
    return True

def run(case,runtime_path,stage):
    case=pathlib.Path(case).resolve();runtime=load(runtime_path);public=case/'results'/runtime[f'{stage}_run_id'];public.mkdir(parents=True,exist_ok=True)
    private=pathlib.Path(runtime['private_root']);budget_file=private/f'{stage}_budget.json'
    protocol=load(case/'specs/protocol.json');now=time.monotonic()
    if budget_file.exists():budget=load(budget_file)
    else:
        budget={'start_monotonic':now,'started_utc':utc(),'limit_seconds':protocol['budgets_seconds'][stage+'_worker']};save(budget_file,budget)
    def remaining():return budget['limit_seconds']-(time.monotonic()-budget['start_monotonic'])
    source=freeze(case,public/'code_freeze.json',runtime);tok=get_tokenizer(runtime['snapshot']);tasks=load(case/'data/tasks.json')
    cfg={k:protocol[k] for k in ('max_model_calls','max_tool_attempts','episode_token_budget','request_seconds','episode_seconds')}
    eos=load(pathlib.Path(runtime['snapshot'])/'generation_config.json')['eos_token_id'];cfg['eos_token_ids']=eos if type(eos)is list else [eos]
    summaries=[];planned=[];kvs=runtime.get('p0_kvs',['B16','F8','F4']) if stage=='p0' else runtime['p1_kvs']
    if stage=='p0':
        for old_root in runtime.get('p0_reuse_results',[]):
            old=load(case/'results'/old_root/'decision.json')
            summaries.extend([{**s,'reused_from':old_root} for s in old['statuses'] if s['status']=='COMPLETED' and s['kv']not in kvs])
    if stage=='p1':
        if not load(case/'results'/runtime['p0_run_id']/'decision.json').get('p1_eligible'):raise ValueError('P0 support gate is not open')
        cfg['max_model_len']=runtime['p1_max_model_len']
        for kv in kvs:
            rows=[{'episode_id':f'P1_{kv}_{t["task_id"]}_m{m}_c{c}','task_id':t['task_id'],'kv':kv,'menu':m,'cap':c}for t in tasks if t['split']=='DEV' for m in (8,32)for c in (256,1024)]
            random.Random(6101902).shuffle(rows);planned.extend(rows)
        save(public/'planned_cells.json',planned)
        save(public/'configuration_frozen.json',{'created_utc':utc(),'runtime':{k:v for k,v in runtime.items()if k not in('python','snapshot','private_root','prepared_root')},'source_hash':source['source_sha256'],'fresh_dev_requests_before_freeze':0})
    for kv in kvs:
        if remaining()<=0:summaries.append({'kv':kv,'status':'NOT_RUN_BUDGET'});continue
        index=len(list((private/'runs').glob(f'{stage}_{kv}_*')));attempt=private/'runs'/f'{stage}_{kv}_{index:02d}'
        max_len=8192 if stage=='p0' else runtime['p1_max_model_len'];cfg['max_model_len']=max_len
        server=Server(runtime['python'],runtime['snapshot'],protocol['kv'][kv],max_len,runtime['pool_bytes'],attempt,case/'src')
        start=time.monotonic();status='UNKNOWN';completed=0
        try:
            server.start(ready_seconds=min(900,max(1,int(remaining()))));pre=inspect(server.base);save(public/f'{kv}_cache_before.json',pre);support_check(pre,kv)
            (public/f'{kv}_metrics_before.txt').write_text(metrics(server.base))
            rows=([{'episode_id':f'P0_{kv}_{t["task_id"]}','task_id':t['task_id'],'kv':kv,'menu':8,'cap':256}for t in tasks if t['split']=='SMOKE'] if stage=='p0' else [r for r in planned if r['kv']==kv])
            for row in rows:
                target=public/'episodes'/(row['episode_id']+'.json')
                if target.exists():raise ValueError('completed cell exists: resume requires an explicit identity audit, refusing overwrite')
                if remaining()<1:status='PARTIAL_BUDGET';break
                task=next(t for t in tasks if t['task_id']==row['task_id']);length=0 if stage=='p0' else 8192
                prepared=load(pathlib.Path(runtime['prepared_root'])/f'{row["task_id"]}_m{row["menu"]}_l{length}.json')
                ep=asyncio.run(run_episode(task,prepared,tok,server.base,kv,row['cap'],private/'fixtures'/row['episode_id'],cfg,row['episode_id']))
                ep['execution_source_sha256']=source['source_sha256'];save(target,ep);completed+=1
                print(json.dumps({'episode':row['episode_id'],'verified':ep['VERIFIED_WORKFLOW_SUCCESS'],'terminal':ep['terminal_reason'],'calls':ep['model_calls'],'tokens':ep['output_tokens'],'wall':ep['wall_seconds']}),flush=True)
                if ep['availability']!='OBSERVED':status='RAW_OR_INFRA_BLOCKED';break
            else:status='COMPLETED'
            post=inspect(server.base);save(public/f'{kv}_cache_after.json',post);support_check(post,kv)
            if kv=='F8' and [(x['k_scale'],x['v_scale'])for x in pre['layers']]!=[(x['k_scale'],x['v_scale'])for x in post['layers']]:status='F8_SCALE_POLICY_UNRESOLVED'
            (public/f'{kv}_metrics_after.txt').write_text(metrics(server.base))
        except Exception as e:
            status='NOT_RUN_SUPPORT' if completed==0 else 'PARTIAL_IMPLEMENTATION_OR_INFRA'
            save(attempt/'failure_private.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'completed':completed,'time_utc':utc()})
            print(json.dumps({'kv':kv,'status':status,'error_type':type(e).__name__,'error':str(e)}),flush=True)
        finally:
            try:server.stop()
            except Exception as e:status='UNRESOLVED_SERVER_DRAIN';print(str(e),flush=True)
        info={'kv':kv,'stage':stage,'attempt':index,'status':status,'completed_episodes':completed,'attempt_wall_seconds':time.monotonic()-start,'elapsed_worker_seconds':time.monotonic()-budget['start_monotonic'],'source_sha256':source['source_sha256']}
        save(public/f'{kv}_attempt{index:02d}.json',info);summaries.append(info)
        if status=='UNRESOLVED_SERVER_DRAIN':break
    if stage=='p0':
        good={x['kv'] for x in summaries if x['status']=='COMPLETED' and x['completed_episodes']==8};candidate='F4' if 'F4'in good else 'F8' if 'F8'in good else None
        save(public/'decision.json',{'stage':'P0','statuses':summaries,'supported':sorted(good),'p1_eligible':'B16'in good and candidate is not None,'proposed_p1_candidate':candidate,'worker_wall_seconds':time.monotonic()-budget['start_monotonic'],'quality_not_a_support_gate':True,'kernel_write_read_gate':'record separate receipt','raw_cost_and_cancel_gates':'record separate receipt'})
    else:
        summary=summarize(public,planned);summary['worker_wall_seconds']=time.monotonic()-budget['start_monotonic'];summary['attempts']=summaries;save(public/'summary.json',summary);save(public/'decision.json',{'status':summary['status'],'baseline_usable':summary['baseline_usable'],'p2':'NOT_RUN','p3':'NOT_RUN'})

