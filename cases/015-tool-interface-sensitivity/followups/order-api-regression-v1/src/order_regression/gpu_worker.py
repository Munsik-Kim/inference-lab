"""Read-only pinned P1 model adapter; bounded sequential A1 execution."""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from .common import P1,atomic,read,require,sha,digest,utc
from .prepare import validate
from .runner import run_episode


def worker(root,private,method,budget):
    root=Path(root);private=Path(private);start=time.monotonic()
    out=private/'run/workers'/method/'attempt01';out.mkdir(parents=True,exist_ok=False)
    completed=smokes=0;adapter=None
    try:
        validate(root);runtime=read(private/'runtime.json');protocol=read(root/'specs/protocol.json')
        require(sha(private/'runtime.json')==protocol['private_runtime_sha256'],'runtime changed')
        from interface_sensitivity.model_adapter import Adapter,verify_saved_artifacts
        checked=verify_saved_artifacts(runtime,P1)
        atomic(out/'preflight.json',{'saved_artifacts':checked,'new_fitting':0,'new_artifact_writes':0},True)
        adapter=Adapter(P1,runtime,method,out,start+budget)
        tasks={s['scenario_id']:s for s in read(root/'data/scenarios.json')}
        methods=read(P1/'specs/methods.json');gen=protocol['generation']
        identity={'method':method,'model_id':methods['model_id'],'model_revision':methods['revision'],
            'artifact_hash':methods['methods'][method]['artifact_id'],'generation_config_hash':digest(gen),
            'execution_source_sha256':read(root/'provenance/source_freeze.json')['execution_source_sha256']}
        if method=='BF16':
            for api in protocol['apis']:
                row=run_episode(tasks['A1-SMOKE'],api,adapter.generate,out/'smoke_fixtures'/api,gen,
                    {**identity,'evidence_kind':'smoke'},'SMOKE__'+api)
                atomic(out/'smoke'/(api+'.json'),row,True);smokes+=1
                require(row['observation_status']=='OBSERVED','smoke infrastructure failure')
        plan=[r for r in read(root/'results/a1_r1/planned_cells.json') if r['method']==method]
        random.Random(protocol['execution_order_seed']).shuffle(plan)
        atomic(out/'execution_order.json',[r['episode_id'] for r in plan],True)
        for cell in plan:
            if time.monotonic()>=start+budget-1:break
            row=run_episode(tasks[cell['scenario_id']],cell['api_version'],adapter.generate,
                out/'fixtures'/cell['episode_id'],gen,identity,cell['episode_id'])
            if row['input_token_hash'] is not None:require(row['input_token_hash']==cell['input_token_sha256'],'first input token hash changed')
            atomic(out/'cells'/(cell['episode_id']+'.json'),row,True);completed+=1
            print(json.dumps({'completed':completed,'method':method,'scenario':cell['scenario_id'],'api':cell['api_version'],
                'verified':row['VERIFIED_WORKFLOW_SUCCESS'],'state_only':row['STATE_ONLY_SUCCESS'],
                'observation_violations':len(row['observation_violations']),'calls':row['model_calls'],'wall':row['wall_seconds']}),flush=True)
            if completed==12:
                elapsed=time.monotonic()-start
                atomic(out/'first12_cost_projection.json',{'elapsed_worker_seconds':elapsed,'completed_primary':12,
                    'linear_method_wall_estimate':adapter.load_seconds+(elapsed-adapter.load_seconds)*32/12,
                    'estimate_only':True,'matrix_unchanged':True},True)
            if row['observation_status']!='OBSERVED':break
        value={'status':'COMPLETE' if completed==32 else 'PARTIAL','method':method,'completed_primary_episodes':completed,
            'separate_smoke_episodes':smokes,'worker_wall_seconds':time.monotonic()-start,'load_decode_seconds':adapter.load_seconds,
            'peak_allocated_bytes':max([0]+[read(p)['peak_allocated_bytes'] for p in (out/'cells').glob('*.json')]),
            'peak_reserved_bytes':max([0]+[read(p)['peak_reserved_bytes'] for p in (out/'cells').glob('*.json')]),
            'new_fitting':0,'new_artifact_writes':0,'enum':False,'enhanced_feedback':False,'aliases':False}
        atomic(out/'receipt.json',value,True);return value
    except BaseException as e:
        atomic(out/'failure.json',{'exception':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),
            'worker_wall_seconds':time.monotonic()-start,'completed_primary_episodes':completed,'smoke_episodes':smokes},True)
        raise


def launch(root,private):
    root=Path(root);private=Path(private);validate(root);protocol=read(root/'specs/protocol.json')
    run=private/'run';run.mkdir(exist_ok=True);require(not (run/'gpu_commands.json').exists(),'preserve earlier attempts; explicit identity-checked resume only')
    start=time.monotonic();runtime=read(private/'runtime.json');model=Path(runtime['model']['path'])
    model_identity=read(P1/'provenance/model_identity.json');checked=[]
    for name,e in model_identity['files'].items():
        require((model/name).is_file() and (model/name).stat().st_size==e['bytes'] and sha(model/name)==e['sha256'],'local model file mismatch: '+name)
        checked.append({'file':name,**e})
    atomic(run/'model_readback.json',{'status':'LOCAL_IDENTITY_VERIFIED','files':checked,'full_local_weight_shards_hashed':True,
        'D_origin_required':False,'official_digest_comparison':'NOT_PERFORMED','wall_seconds':time.monotonic()-start},True)
    commands=[]
    for method in protocol['model_load_order']:
        remaining=protocol['worker_wall_budget_seconds']-(time.monotonic()-start)
        if remaining<2:break
        command=[sys.executable,'-B','-m','order_regression.cli','--root',str(root),'worker','--private',str(private),'--method',method,'--budget',str(remaining)]
        env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(root/'src'))
        log=run/(method+'-attempt01.log');t=time.monotonic();started=utc()
        with log.open('x') as f:
            try:code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=root,timeout=remaining).returncode
            except subprocess.TimeoutExpired:code=-9
        receipt={'method':method,'attempt':1,'command':command,'started_utc':started,'exit_code':code,'wall_seconds':time.monotonic()-t,
            'remaining_at_launch_seconds':remaining,'log':log.name}
        commands.append(receipt);atomic(run/'gpu_commands.json',commands)
        print(json.dumps(receipt),flush=True)
        if code!=0:break
    result={'worker_commands':len(commands),'total_worker_wall_seconds':time.monotonic()-start,
        'budget_seconds':protocol['worker_wall_budget_seconds'],'all_processes_exit_zero':len(commands)==3 and all(c['exit_code']==0 for c in commands),
        'includes':'full local model hash, model load, artifact hash/decode, smoke, primary, metric and I/O; not GPU busy or serving latency'}
    atomic(run/'launch_receipt.json',result,True);return result
