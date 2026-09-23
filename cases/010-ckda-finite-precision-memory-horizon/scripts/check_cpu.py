"""Restore historical paths and run model-free audits in an external output tree."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from verify_unified import CASE, verify, verify_snapshots
from restore_workspace import restore

def run(case, output):
    case=Path(case).resolve();output=Path(output).absolute()
    if output.exists() or output.resolve().is_relative_to(case):raise ValueError('use a new external output directory')
    verify(case);output.mkdir(parents=True)
    restore(case,output/'workspace')
    v1=output/'workspace/cases/010-ckda-finite-precision-memory-horizon'
    v2=output/'workspace/research/case010-failure-aware-v2'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    receipts=[]
    def command(name,args,cwd):
        result=subprocess.run([sys.executable,'-B',*map(str,args)],cwd=cwd,env=env,capture_output=True,text=True)
        (output/f'{name}.log').write_text(result.stdout+result.stderr)
        receipts.append({'name':name,'exit_code':result.returncode,'log':f'{name}.log'})
        if result.returncode:raise RuntimeError(f'{name} failed; see external log')
    command('unified-tests',['-m','unittest','discover','-s',case/'tests','-v'],case)
    command('v1-publication',['scripts/verify_publication.py','--root',v1],v1)
    primary=['scripts/audit_records.py','--toy-original',v1/'results/toy-run','--toy-repair',v1/'results/toy-repair-v1b','--output',output/'v1-scalars.json']
    supplement=['scripts/audit_budget_supplement.py','--output',output/'v1-supplement.json']
    for seed in range(3):
        primary+=['--learned',v1/f'results/learned-seed{seed}']
        supplement+=['--supplement',v1/f'results/budget-supplement-seed{seed}']
    command('v1-scalars',primary,v1)
    command('v1-supplement',supplement,v1)
    command('v2-scalars',['analysis/audit_v2.py','--results','results/fresh','--output',output/'v2-scalars.json'],v2)
    command('v2-aggregate',['analysis/aggregate_v2.py','--results','results/fresh','--output',output/'v2-regenerated'],v2)
    compared = []
    for name in ('fresh_table.csv','paired_table.csv','bytes_table.csv','combined.json','source_hashes.json'):
        if (output/'v2-regenerated'/name).read_bytes() != (v2/'results/fresh-summary'/name).read_bytes():
            raise ValueError(f'regenerated original aggregate differs: {name}')
        compared.append(name)
    command('v2-codec-tests',['-m','unittest','discover','-s','tests','-p','test_online_v2.py','-v'],v2)
    command('synthetic-restart',[case/'scripts/restart_demo.py','--case',case,'--output',output/'synthetic-restart.json'],output)
    command('v1-publication-after',['scripts/verify_publication.py','--root',v1],v1)
    verify_snapshots(case)
    result={'status':'PASS','commands':receipts,'scope':'Model-free CPU scalar audits, integration contracts, synthetic codec/restart tests.',
            'v2_regenerated_byte_identical':compared,
            'trained_checkpoint_runs':0,'new_GPU_runs':0,'not_run':['trained model replay','upstream-dependent native parity','fresh TEST inference','timing remeasurement'],
            'historical_tests_not_reexecuted':'Full v1 and v2 suites require separate environment/upstream checks; this wrapper runs the explicitly named synthetic subset.'}
    (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,default=CASE);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.case,a.output)))
if __name__=='__main__':main()
