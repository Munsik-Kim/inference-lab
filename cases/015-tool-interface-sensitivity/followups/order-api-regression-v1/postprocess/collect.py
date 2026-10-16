"""Collect immutable private cells; explicit path sanitation only for metadata/logs."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from order_regression.common import P1,read,sha,atomic,canonical,require,utc
from order_regression.analysis import summarize


def collect(private):
    private=Path(private);run=private/'run';result=ROOT/'results/a1_r1';launch=read(run/'launch_receipt.json')
    require(launch['all_processes_exit_zero'],'worker process failure; partial collection needs explicit missing ledger')
    observed=read(private/'environment_observed.json');runtime=read(private/'runtime.json')
    replacements={str(private):'${A1_PRIVATE}',str(ROOT):'${A1}',str(P1):'${CASE015}',
        runtime['model']['path']:'${MODEL_SNAPSHOT}',
        str(Path(observed['python_executable']).parent.parent):'${GPU_ENV}',
        observed['torch_import'].split('/lib/')[0]:'${GPU_ENV_IMPORT}',
        runtime['original_result_root']:'${ORIGINAL_RESULT_ROOT}',runtime['research_source']:'${ORIGINAL_RESEARCH_SOURCE}'}
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,str):
            for old,new in sorted(replacements.items(),key=lambda x:-len(x[0])):v=v.replace(old,new)
        return v
    rows=[];receipts=[];transforms=[]
    for method in ['BF16','D16','RTN']:
        source=run/'workers'/method/'attempt01'
        for p in sorted((source/'cells').glob('*.json')):
            row=read(p);require('/home/' not in canonical(row) and '/mnt/' not in canonical(row),'raw response private-path needs separate preservation/sanitization amendment')
            dest=result/'workers'/method/'attempt01/cells'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);require(not dest.exists(),'preserve completed cell')
            shutil.copyfile(p,dest);require(sha(p)==sha(dest),'cell bytes changed');rows.append(row)
        for p in sorted((source/'smoke').glob('*.json')):
            dest=result/'smoke'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);require(not dest.exists(),'preserve smoke');shutil.copyfile(p,dest)
        for name in ['preflight.json','artifact_readback.json','environment.json','execution_order.json','first12_cost_projection.json','receipt.json']:
            p=source/name;dest=result/'workers'/method/'attempt01'/name;value=clean(read(p));atomic(dest,value,True)
            transforms.append({'source_label':f'A1_PRIVATE/run/workers/{method}/attempt01/{name}','source_sha256':sha(p),'public_relative_path':str(dest.relative_to(ROOT)),'public_sha256':sha(dest),'transformation':'path labels in metadata only; raw cells byte-identical'})
        receipts.append(read(source/'receipt.json'))
        log=run/(method+'-attempt01.log');dest=result/'attempts'/log.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(clean(log.read_text()))
        transforms.append({'source_label':'A1_PRIVATE/run/'+log.name,'source_sha256':sha(log),'public_relative_path':str(dest.relative_to(ROOT)),'public_sha256':sha(dest),'transformation':'private path labels in log'})
    rows.sort(key=lambda r:r['episode_id']);summary=summarize(rows,read(result/'planned_cells.json'));require(summary['status']=='COMPLETED_A1','not all96 observed')
    for name,values in [('episodes.jsonl',rows),('tool_events.jsonl',[{'episode_id':r['episode_id'],**e} for r in rows for e in r['tool_events']])]:
        dest=result/name;require(not dest.exists(),'preserve aggregate');partial=dest.with_suffix(dest.suffix+'.partial');partial.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in values));partial.replace(dest)
    atomic(result/'summary.json',summary,True)
    commands=clean(read(run/'gpu_commands.json'));atomic(result/'command_receipts.json',commands,True)
    atomic(result/'model_readback.json',read(run/'model_readback.json'),True)
    runtime={'fresh_gpu_run':True,'registered_primary_episodes':96,'completed_primary_episodes':96,'separate_smoke_episodes':2,
        'primary_model_requests':sum(r['model_calls'] for r in rows),'primary_tool_attempts':sum(r['attempted_tool_calls'] for r in rows),
        'primary_generated_tokens':sum(r['generated_tokens'] for r in rows),'primary_input_tokens_total':sum(r['input_tokens_total'] for r in rows),
        'total_worker_wall_seconds':launch['total_worker_wall_seconds'],'budget_seconds':7200,'workers':receipts,
        'max_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in rows),'max_peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in rows),
        'device_memory_samples':'per response device_free_total_after_request, separate from Torch allocator and Windows/background use',
        'wall_boundary':launch['includes'],'failure_attempts':0,'retries':0,'new_model_downloads':0,'new_quantization':0,'new_training':0,'new_P1_forwards':0,
        'public_command_paths':'meaningful labels; exact command logs retained privately, hashes in transformation manifest'}
    smoke=[read(p) for p in sorted((result/'smoke').glob('*.json'))]
    runtime['smoke_model_requests']=sum(r['model_calls'] for r in smoke);runtime['smoke_tool_attempts']=sum(r['attempted_tool_calls'] for r in smoke);runtime['smoke_generated_tokens']=sum(r['generated_tokens'] for r in smoke)
    atomic(result/'runtime_manifest.json',runtime,True);atomic(ROOT/'provenance/public_sanitization.json',{'created_utc':utc(),'raw_responses_changed':False,'transforms':transforms},True)
    return {'status':summary['status'],'success_table':summary['success_table'],'runtime':runtime}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--private',type=Path,required=True);a=p.parse_args();print(json.dumps(collect(a.private),ensure_ascii=False))
