"""Bounded sequential launch, separate smoke, immutable per-episode writes."""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from .common import METHODS, INTERFACES, atomic, digest, inputs, read, require, sha, utc
from .prepare import validate
from .transform import make_interface
from .runner import run_episode


def worker(case,private,method,budget):
    start=time.monotonic();case=Path(case);private=Path(private)
    out=private/'run/workers'/method/'attempt01';out.mkdir(parents=True,exist_ok=False)
    completed=0;smoke_count=0
    try:
        validate(case);runtime=read(private/'runtime.json')
        require(sha(private/'runtime.json')==read(case/'provenance/private_runtime_identity.json')['sha256'],'private runtime configuration changed')
        model_dir=Path(runtime['model']['path'])
        for name,entry in read(case/'provenance/model_identity.json')['files'].items():
            require((model_dir/name).stat().st_size==entry['bytes'],'local model file bytes changed')
            if not name.endswith('.safetensors'):
                require(sha(model_dir/name)==entry['sha256'],'local model config/tokenizer identity changed')
        from .model_adapter import verify_saved_artifacts,Adapter
        checked=verify_saved_artifacts(runtime,case)
        atomic(out/'preflight.json',{'artifacts':checked,'new_files_created_in_artifact_root':0})
        adapter=Adapter(case,runtime,method,out,start+budget)
        protocol=read(case/'specs/protocol.json');gen=protocol['generation']
        tasks,messages,tools=inputs(case);tasks={t['id']:t for t in tasks}
        p=read(case/'results/p1_r1/planned_cells.json');plan=[r for r in p if r['method']==method]
        first=[r for r in plan if r['interface_id']==INTERFACES[0]]
        rest=[r for r in plan if r['interface_id']!=INTERFACES[0]]
        random.Random(protocol['execution_order_seed']).shuffle(rest)
        order=first+rest
        atomic(out/'execution_order.json',[r['episode_id'] for r in order])
        methods=read(case/'specs/methods.json');ident={'method':method,'model_id':methods['model_id'],'model_revision':methods['revision'],
            'artifact_hash':methods['methods'][method]['artifact_id'],'generation_config_hash':digest(gen),
            'execution_source_sha256':read(case/'provenance/source_freeze.json')['execution_source_sha256']}
        if method=='BF16':
            for tid in protocol['smoke']['tasks']:
                task=tasks[tid];interface=make_interface(task,messages[tid],tools,INTERFACES[0])
                r=run_episode(task,interface,adapter.generate,out/'smoke_fixtures'/tid,gen,ident,'SMOKE__'+tid)
                r['evidence_kind']='smoke';atomic(out/'smoke'/(tid+'.json'),r,True);smoke_count+=1
                require(r['observation_status']=='OBSERVED','smoke infrastructure failed')
        for cell in order:
            if time.monotonic()>=start+budget-1:
                break
            task=tasks[cell['task_id']];interface=make_interface(task,messages[task['id']],tools,cell['interface_id'])
            row=run_episode(task,interface,adapter.generate,out/'fixtures'/cell['episode_id'],gen,ident,cell['episode_id'])
            if row['input_token_hash'] is not None:
                require(row['input_token_hash']==cell['input_token_sha256'],'registered first input hash mismatch')
            atomic(out/'cells'/(cell['episode_id']+'.json'),row,True);completed+=1
            print(json.dumps({'completed':completed,'method':method,'task':task['id'],'interface':cell['interface_id'],
                'functional':row['functional_success'],'legacy':row['legacy_success'],'first_error':row['first_error_type'],
                'calls':row['model_calls'],'wall':row['wall_seconds']}),flush=True)
            if completed==12:
                elapsed=time.monotonic()-start
                atomic(out/'first12_cost_projection.json',{'elapsed_worker_seconds':elapsed,'primary_completed':12,
                    'linear_total_method_seconds_including_load':adapter.load_seconds+(elapsed-adapter.load_seconds)*32/12,
                    'estimate_only':True,'matrix_not_changed':True})
            if row['observation_status']!='OBSERVED':
                break
        receipt={'status':'COMPLETE' if completed==32 else 'PARTIAL','method':method,'completed_primary_episodes':completed,
            'separate_smoke_episodes':smoke_count,'worker_wall_seconds':time.monotonic()-start,
            'peak_allocated_bytes':adapter.torch.cuda.max_memory_allocated(),'peak_reserved_bytes':adapter.torch.cuda.max_memory_reserved(),
            'new_fitting':0,'new_artifact_writes':0,'enum':False,'enhanced_feedback':False,'aliases':False}
        atomic(out/'receipt.json',receipt,True)
        return receipt
    except BaseException as e:
        atomic(out/'failure.json',{'exception':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),
            'wall_seconds':time.monotonic()-start,'completed_primary_episodes':completed,'separate_smoke_episodes':smoke_count})
        raise


def launch(case,private):
    case=Path(case);private=Path(private);validate(case)
    protocol=read(case/'specs/protocol.json');result=private/'run';result.mkdir(exist_ok=True)
    commands=read(result/'gpu_commands.json') if (result/'gpu_commands.json').exists() else []
    for method in protocol['model_load_order']:
        if any(r['method']==method for r in commands):
            raise ValueError('preserve previous attempts; explicit identity-checked resume required')
        remaining=protocol['worker_wall_budget_seconds']-sum(r['wall_seconds'] for r in commands)
        if remaining<=2:
            break
        log=result/(method+'-attempt01.log')
        command=[sys.executable,'-B','-m','interface_sensitivity.cli','--case',str(case),'worker','--private',str(private),'--method',method,'--budget',str(remaining)]
        env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(case/'src'))
        start=time.monotonic();started=utc()
        with log.open('x') as f:
            try:
                r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=case,timeout=remaining)
                code=r.returncode
            except subprocess.TimeoutExpired:
                code=-9
        receipt={'method':method,'attempt':1,'command':command,'started_utc':started,'exit_code':code,
            'wall_seconds':time.monotonic()-start,'remaining_at_launch':remaining,'log':log.name}
        commands.append(receipt);atomic(result/'gpu_commands.json',commands)
        print(json.dumps(receipt),flush=True)
        if code!=0:
            break
    return {'worker_commands':len(commands),'cumulative_wall_seconds':sum(x['wall_seconds'] for x in commands),
            'all_processes_exit_zero':all(x['exit_code']==0 for x in commands)}
