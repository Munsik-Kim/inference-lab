"""Assemble small public records and final tables without changing raw files."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from common import CASE,read,write,file_row,utc

def loadcsv(p):return list(csv.DictReader(Path(p).open()))
def savecsv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows or [{'status':'NOT_RUN'}])

def main():
    p=argparse.ArgumentParser();p.add_argument('--private',type=Path,required=True);p.add_argument('--dev',type=Path,required=True);p.add_argument('--eval',type=Path,required=True);p.add_argument('--repeat',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);a=p.parse_args()
    summary=CASE/'results/summary';summary.mkdir(parents=True,exist_ok=False);recorded=CASE/'results/recorded';recorded.mkdir(exist_ok=False)
    selection=read(a.selection);manifest=read(CASE/'inputs/manifest.json');ledger=[json.loads(x) for x in (CASE/'worker_ledger.jsonl').read_text().splitlines()]
    for filename in ['document_metrics.csv','position_metrics.csv','layer_metrics.csv','vector_decomposition.csv']:
        savecsv(summary/filename,loadcsv(a.dev/filename)+loadcsv(a.eval/filename))
    for name in ['primary.csv']:shutil.copyfile(a.eval/name,summary/name)
    for name in ['same_path_repeat.csv','primary.csv']:
        shutil.copyfile(a.repeat/name,summary/('repeat_'+name))
    rawmapping=[];localrows=[];primitive=[];localvec=[];timingraw=[];memory=[];counts={'main':0,'repeat':0,'local_block':0,'timed_full':0,'timing_cold':0,'timing_warmup':0,'timing_workers':0,'optional_replacement':0,'intervention_DEV':0,'intervention_EVAL':0,'preflight_full':0,'historical_isolated_block':0,'intervention_auxiliary_block':0}
    for entry in ledger:
        src=Path(entry['raw_directory']);dest=recorded/entry['label'];dest.mkdir()
        for f in src.iterdir():
            if f.suffix=='.json' or f.name in ['stdout.txt','stderr.txt']:
                shutil.copyfile(f,dest/f.name);rawmapping.append({'source_path':str(f),'public_path':str((dest/f.name).relative_to(CASE)),**file_row(f),'action':'EXACT_COPY'})
            elif f.suffix=='.npy':rawmapping.append({'source_path':str(f),'public_path':None,**file_row(f),'action':'LOCAL_ONLY_FULL_HISTORICAL_ARRAY' if 'preflight-v1' in entry['label'] else 'LOCAL_ONLY_INTERVENTION_ARRAY','reason':'Full historical replay or intervention logits/current-P-input arrays retained locally; public subset is separately labeled'})
        if (src/'preflight.json').exists():
            pre=read(src/'preflight.json')['forward_counts'];counts['preflight_full']+=sum(pre[k] for k in ['historical_full','self_repeat','no_op','block_hook']);counts['historical_isolated_block']+=pre['historical_block']
        if (src/'capture_validation.json').exists():
            pre=read(src/'capture_validation.json');counts['preflight_full']+=pre['full_forwards']+pre['profile_full_forwards']
        if (src/'intervention_preflight.json').exists():
            pre=read(src/'intervention_preflight.json');counts['preflight_full']+=pre['full_forwards'];counts['intervention_auxiliary_block']+=pre['auxiliary_block_forwards']
        if (src/'intervention_receipt.json').exists():
            r=read(src/'intervention_receipt.json');actual=len(list(src.glob('*_logits.npy')))
            if actual!=len(r['rows']):raise ValueError('INTERVENTION_FILE_DENOMINATOR')
            counts['optional_replacement']+=actual;counts['intervention_'+r['split']]+=actual;counts['intervention_auxiliary_block']+=r['auxiliary_block_forwards']
        if (src/'forward_receipt.json').exists():
            r=read(src/'forward_receipt.json');mode=r['mode'];base=Path(r['raw']);metadata=list(base.glob('*.json'));counts['main' if mode=='main' else 'repeat']+=len(metadata)
            mdest=dest/'forward_metadata';mdest.mkdir()
            for f in metadata:
                row=read(f);logit=Path(row['logits']['path'])
                if file_row(logit)!={k:row['logits'][k] for k in ['bytes','sha256']}:raise ValueError('RAW_IDENTITY')
                shutil.copyfile(f,mdest/f.name)
                rawmapping.append({'source_path':str(logit),'public_path':None,**file_row(logit),'action':'LOCAL_ONLY_FULL_SELECTED_LOGITS','reason':'All16 selected positions/full vocabulary retained locally; public audit subset disclosed'})
            inv=r['tensor_inventory'];unique=sum(x['storage_bytes'] for x in inv if x['unique_storage']);memory.append({'worker':entry['label'],'design':'diagnostic_single_path_model','condition':r['condition'],'split':r['split'],'unique_module_storage_bytes':unique,**r['memory']})
        if (src/'local_metrics.json').exists():
            r=read(src/'local_metrics.json');actual_local=len(list(Path(r['selected_raw']).glob('*.npz')))
            if actual_local!=r['block_replay_count']:raise ValueError('LOCAL_REPLAY_FILE_DENOMINATOR')
            counts['local_block']+=actual_local
            for x in r['rows']:
                if 'metrics' in x:localrows.append({k:v for k,v in x.items() if k not in ['metrics','input_identity']}|x['metrics'])
                elif 'decomposition' in x:localvec.append({k:v for k,v in x.items() if k!='decomposition'}|x['decomposition'])
            for x in r['primitive']:primitive.append({k:v for k,v in x.items() if k!='metrics'}|x['metrics'])
        if (src/'timing.json').exists():
            r=read(src/'timing.json');counts['timing_workers']+=1;counts['timed_full']+=len(r['samples']);counts['timing_cold']+=len(r['cold_call_seconds']);counts['timing_warmup']+=sum(x['calls'] for x in r['warmup'])
            timingraw.extend(r['samples'])
            for rounding in ['R0','R1']:
                samples=[x for x in r['samples'] if x['rounding']==rounding];host=np.asarray([x['host_ns']/1e6 for x in samples]);event=np.asarray([x['event_ms'] for x in samples])
                row={'path':r['path'],'rounding':rounding,'block':r['block'],'samples':len(samples),'host_median_ms':float(np.median(host)),'host_mean_ms':float(host.mean()),'host_p95_ms':float(np.quantile(host,.95,method='linear')),'event_median_ms':float(np.median(event)),'event_mean_ms':float(event.mean()),'event_p95_ms':float(np.quantile(event,.95,method='linear')),'setup_load_seconds':r['setup_load_seconds'],'cold_call_seconds':r['cold_call_seconds'][rounding],'parent_worker_wall_seconds':entry['worker_wall_seconds']}
                if 'timing' not in locals():timing=[]
                timing.append(row)
            memory.append({'worker':entry['label'],'design':'hook_free_cost_model','condition':r['path'],'split':'DEV1',**r['timed_memory']})
    savecsv(summary/'local_replay_metrics.csv',localrows);savecsv(summary/'local_rounding_metrics.csv',primitive);savecsv(summary/'local_vector_decomposition.csv',localvec);savecsv(summary/'timing.csv',locals().get('timing',[]));savecsv(summary/'timing_raw.csv',timingraw);savecsv(summary/'memory.csv',memory)
    for split in ['dev','eval']:
        folder=a.private/('analysis_intervention_'+split+'_v1')
        if folder.exists():
            for name in ['intervention_metrics.csv','intervention_positions.csv']:shutil.copyfile(folder/name,summary/(split+'_'+name))
            shutil.copyfile(folder/'summary.json',summary/(split+'_intervention_summary.json'))
    stat=read(a.eval/'summary.json')['primary512'];tau=selection['tau_abs'];delta=stat['mean_delta']
    if abs(delta)<=tau:effect='NUMERICALLY_UNRESOLVED'
    elif stat['ci_low']>0:effect='REDUCED'
    elif stat['ci_high']<0:effect='INCREASED'
    else:effect='INCONCLUSIVE'
    primary=loadcsv(a.eval/'primary.csv');p512=[r for r in primary if int(r['length'])==512];e0=np.mean([float(r['E0']) for r in p512]);e1=np.mean([float(r['E1']) for r in p512])
    stagewall={stage:math.fsum(x['worker_wall_seconds'] for x in ledger if x['stage']==stage) for stage in ['P0','P1','P2','P3']};total=math.fsum(stagewall.values())
    budget={'global_cap_seconds':7200,'worker_count':len(ledger),'worker_wall_seconds':total,'remaining_seconds':7200-total,'stage_wall_seconds':stagewall,'stage_caps_seconds':{'P0':1200,'P1':2400,'P2':2400,'P3':1200},'worker_cpu_user_seconds':math.fsum(x['cpu_user_seconds'] for x in ledger),'worker_cpu_system_seconds':math.fsum(x['cpu_system_seconds'] for x in ledger),'meaning':'Parent spawn-through-exit wall, not GPU busy time; CPU-only analysis is separate'}
    if total>7200 or any(stagewall[k]>budget['stage_caps_seconds'][k] for k in stagewall):raise ValueError('BUDGET_VIOLATION')
    complete=counts['main']==384 and counts['repeat']==54 and counts['local_block']==384 and counts['timed_full']==180 and all(x['exit_code']==0 for x in ledger)
    if selection['intervention']['status']=='SELECTED':complete=complete and counts['intervention_DEV']==32 and counts['intervention_EVAL']==24
    mechanisms=[]
    if any(r['reference']=='D0' and r['candidate']=='D1' and float(r['SSE'])>0 for r in localrows):mechanisms.append('FORMAT_CHANGE_SUPPORTED')
    if any(r['reference']=='D1' and r['candidate']=='P' and r['boundary']=='self_attn.q_proj' and float(r['SSE'])>0 for r in localrows):mechanisms.append('LOCAL_EXECUTION_DIFFERENCE_SUPPORTED')
    if any(r['boundary']=='block_output' and float(r['SSE'])>0 for r in primitive):mechanisms.append('PROPAGATION_SENSITIVITY_SUPPORTED')
    if not mechanisms:mechanisms.append('UNRESOLVED')
    decision={'execution_status':'COMPLETE' if complete else 'PARTIAL','rounding_effect':effect,'rounding_boundary_status':selection['rounding_boundary_status'],'primary512':stat,'mean_E0':float(e0),'mean_E1':float(e1),'relative_delta':None if e0<=tau else float(delta/e0),'tau_abs':tau,'r_repeat':selection['r_repeat'],'documents_delta_positive':sum(float(r['delta'])>0 for r in p512),'documents_delta_negative':sum(float(r['delta'])<0 for r in p512),'counts':counts,'planned':read(CASE/'protocol.json')['counts'],'selected_layers':selection['layers'],'intervention':selection['intervention'],'mechanisms':mechanisms,'alignment':'NOT_ESTABLISHED: D0/D1/P already share the same R0 attention implementation','historical_gate':'Unchanged0.02; D0-P full32 remains outside','publication_status':'LOCAL_ONLY','gpu_source_bundle_sha256':selection['gpu_source_bundle_sha256'],'selection_freeze':file_row(a.selection),'budget':budget}
    write(summary/'budget.json',budget);write(summary/'decision.json',decision);write(CASE/'provenance/public_local_mapping.json',{'entries':rawmapping,'local_full_logits_preserved':True,'model_and_artifact_copied':False})
    print(json.dumps(decision))

if __name__=='__main__':main()
