"""Portable CPU verification of the small local publication candidate."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from common import CASE,read,file_row,verify_files,write

def records(path):return list(csv.DictReader(Path(path).open()))
def require(ok,message):
    if not ok:raise ValueError(message)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--no-public-manifest',action='store_true');p.add_argument('--figures',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.process_time()
    verify_files(read(CASE/'provenance/freeze_v1.json'));verify_files(read(CASE/'provenance/freeze_capture_validation.json'))
    select=read(CASE/'provenance/selection_freeze.json');verify_files(select['analysis_freeze'])
    provisional=read(CASE/'provenance/dev_layers_freeze.json')
    verify_files(provisional['analysis_freeze'],CASE/'provenance/analysis_dev_layers_snapshot')
    require(not provisional['dev_complete'] and select['dev_complete'],'PROVISIONAL_VS_FINAL_EVAL_GATE')
    if not a.no_public_manifest:
        public=read(CASE/'PUBLIC_MANIFEST.json')
        for name,row in public['files'].items():require(file_row(CASE/name)==row,'PUBLIC_HASH:'+name)
        actual={str(x.relative_to(CASE)) for x in CASE.rglob('*') if x.is_file() and x.name!='PUBLIC_MANIFEST.json'}
        require(actual==set(public['files']),'PUBLIC_COVERAGE')
    preservation=read(CASE/'provenance/preservation_before.json')
    for row in preservation['referenced_files']:require(file_row(CASE/row['copied_to'])=={k:row[k] for k in ['bytes','sha256']},'PRESERVED_CASE013_COPY')
    summary=CASE/'results/summary';decision=read(summary/'decision.json');manifest=read(CASE/'inputs/manifest.json')
    metadata=list((CASE/'results/recorded').glob('p1-full-*/forward_metadata/*.json'))+list((CASE/'results/recorded').glob('p2-full-*/forward_metadata/*.json'))
    actual=set()
    for path in metadata:
        row=read(path);key=(row['split'],row['document_id'],row['length'],row['condition']);require(key not in actual,'DUPLICATE_FORWARD');actual.add(key)
        require(row['full_logits_shape']==[1,row['length'],151936] and row['lm_head_input_shapes']==[[1,row['length'],1024]],'LM_HEAD_GEOMETRY')
        require(row['input_sha256']==file_row(CASE/'inputs'/(row['document_id']+'.npy'))['sha256'],'FORWARD_INPUT')
    expected={(r['split'],r['document_id'],n,c) for r in manifest['documents'] for n in [128,512] for c in manifest['condition_order']}
    require(actual==expected and len(actual)==384,'MAIN_DENOMINATOR')
    position=records(summary/'position_metrics.csv');doc=records(summary/'document_metrics.csv')
    keyed={}
    for r in position:keyed.setdefault((r['split'],r['document_id'],r['length'],r['reference'],r['candidate']),[]).append(r)
    for row in doc:
        sub=keyed[(row['split'],row['document_id'],row['length'],row['reference'],row['candidate'])]
        require(len(sub)==16,'POSITION_DENOMINATOR');mean=math.fsum(float(r['KL']) for r in sub)/16
        require(math.isclose(mean,float(row['KL']),rel_tol=1e-12,abs_tol=1e-14),'DOCUMENT_KL_MEAN')
    primary=records(summary/'primary.csv');d=[]
    for row in primary:
        if int(row['length'])!=512:continue
        values={r['reference']:float(r['KL']) for r in doc if r['split']=='EVAL' and r['document_id']==row['document_id'] and int(r['length'])==512 and (r['reference'],r['candidate']) in [('D1_R0','P_R0'),('D1_R1','P_R1')]}
        require(math.isclose(values['D1_R0']-values['D1_R1'],float(row['delta']),rel_tol=1e-12,abs_tol=1e-14),'PRIMARY_PAIR');d.append(float(row['delta']))
    require(len(d)==24,'EVAL_DOCUMENT_COUNT')
    # Independent reconstruction of the fixed bootstrap from the recorded deltas.
    rng=np.random.default_rng(61027023);arr=np.asarray(d);means=arr[rng.integers(0,24,size=(10000,24))].mean(1);ci=np.quantile(means,[.025,.975],method='linear')
    stat=decision['primary512']
    require(np.allclose([arr.mean(),*ci],[stat['mean_delta'],stat['ci_low'],stat['ci_high']],rtol=1e-12,atol=1e-14),'BOOTSTRAP')
    tau=select['tau_abs'];effect='NUMERICALLY_UNRESOLVED' if abs(arr.mean())<=tau else 'REDUCED' if ci[0]>0 else 'INCREASED' if ci[1]<0 else 'INCONCLUSIVE'
    require(effect==decision['rounding_effect'],'EFFECT_RULE')
    if select['intervention']['status']=='SELECTED':
        verify_files(read(CASE/'provenance/freeze_intervention_v1.json'))
        dev=records(summary/'dev_intervention_metrics.csv');candidates=[]
        for layer in sorted(select['layers']):
            for boundary in ['post_RoPE_Q','block_output']:
                values=[float(r['KL_change']) for r in dev if int(r['layer'])==layer and r['boundary']==boundary];require(len(values)==8,'INTERVENTION_DEV_COUNT');candidates.append((abs(math.fsum(values)/8),layer,boundary))
        winner=sorted(candidates,key=lambda r:(-r[0],r[1],0 if r[2]=='post_RoPE_Q' else 1))[0]
        require(select['intervention']['candidate']=={'layer':winner[1],'boundary':winner[2]},'INTERVENTION_SELECTION_RULE')
        ev=records(summary/'eval_intervention_metrics.csv');require(len(ev)==24,'INTERVENTION_EVAL_COUNT')
        pos=records(summary/'eval_intervention_positions.csv')
        for row in ev:
            sub=[r for r in pos if r['document_id']==row['document_id']];require(len(sub)==16,'INTERVENTION_POSITION_COUNT')
            require(math.isclose(math.fsum(float(r['KL_change']) for r in sub)/16,float(row['KL_change']),rel_tol=1e-12,abs_tol=1e-14),'INTERVENTION_DOCUMENT_MEAN')
    ledger=[json.loads(x) for x in (CASE/'worker_ledger.jsonl').read_text().splitlines()];wall=math.fsum(x['worker_wall_seconds'] for x in ledger)
    require(math.isclose(wall,decision['budget']['worker_wall_seconds'],abs_tol=1e-9),'WORKER_DEBIT');require(wall<=7200,'GLOBAL_GPU_BUDGET')
    for stage,cap in [('P0',1200),('P1',2400),('P2',2400),('P3',1200)]:require(math.fsum(x['worker_wall_seconds'] for x in ledger if x['stage']==stage)<=cap,'STAGE_GPU_BUDGET')
    timing=records(summary/'timing.csv');raw=records(summary/'timing_raw.csv');require(len(raw)==180 and len(timing)==18,'TIMING_DENOMINATOR')
    for row in timing:
        sub=[r for r in raw if all(r[k]==row[k] for k in ['path','rounding','block'])];require(len(sub)==10,'TIMING_SAMPLES')
        h=np.asarray([float(r['host_ns'])/1e6 for r in sub]);e=np.asarray([float(r['event_ms']) for r in sub])
        for name,value in [('host_median_ms',np.median(h)),('host_p95_ms',np.quantile(h,.95,method='linear')),('event_mean_ms',e.mean())]:require(math.isclose(value,float(row[name]),rel_tol=1e-12,abs_tol=1e-12),'TIMING_STATISTIC')
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['MPLCONFIGDIR']=str(a.output/'mpl_cache');env['OPENBLAS_NUM_THREADS']='2'
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(CASE/'tests'),'-v'],env=env,capture_output=True,text=True);(a.output/'tests.log').write_text(test.stdout+test.stderr);require(test.returncode==0,'CPU_UNIT_TESTS')
    audit=subprocess.run([sys.executable,str(CASE/'analysis/audit.py'),'--output',str(a.output/'public_logits_audit')],env=env,capture_output=True,text=True);(a.output/'public_audit.log').write_text(audit.stdout+audit.stderr);require(audit.returncode==0,'PUBLIC_SUBSET_KL')
    if a.figures:
        run=subprocess.run([sys.executable,str(CASE/'analysis/figures.py'),'--summary',str(summary),'--output',str(a.output/'figures')],env=env,capture_output=True,text=True);(a.output/'figures.log').write_text(run.stdout+run.stderr);require(run.returncode==0,'FIGURES_REGENERATION')
    result={'verification':'VERIFIED','scope':'Public identities,384 metadata records, scalar per-position to document/bootstrap, worker budget and timing; independent full-vocabulary KL only for disclosed DEV4 x2position subset','main_records':len(actual),'document_statistics':len(doc),'per_position_statistics':len(position),'workers':len(ledger),'worker_wall_seconds':wall,'rounding_effect':effect,'public_manifest':'SKIPPED_EXPLICITLY' if a.no_public_manifest else 'VERIFIED','figures_regenerated':a.figures,'torch_imported':'torch' in sys.modules,'cuda_initialization':'NOT_CALLED','cpu_process_seconds_parent':time.process_time()-start}
    write(a.output/'verification.json',result);print(json.dumps(result))

if __name__=='__main__':main()
