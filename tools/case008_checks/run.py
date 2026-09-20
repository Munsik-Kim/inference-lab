"""Explicit Case008 evidence or tiny-model CPU checks; never writes frozen files."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
CASE='cases/008-build-reconstruct-reload'

def inventory(root):
    paths=[p for folder in (CASE,'tools/modelpack') for p in (root/folder).rglob('*') if p.is_file()]
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def run(mode, output):
    output=output.resolve()
    if output.exists() or output.is_relative_to(ROOT):raise ValueError('Use a new external output directory')
    output.mkdir(parents=True);before=inventory(ROOT);results=[]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1'}
    def call(name,args,cwd=ROOT):
        r=subprocess.run([sys.executable,'-B',*args],cwd=cwd,env=env,text=True,capture_output=True,timeout=300)
        (output/(name+'.log')).write_text(r.stdout+r.stderr);results.append({'check':name,'exit_code':r.returncode})
        print(name+': '+('PASS' if r.returncode==0 else 'FAIL'),flush=True)
        if r.returncode:raise ValueError(name+' failed')
    try:
        call('publication',[CASE+'/publication/verify_publication.py','--root',str(ROOT),'--output',str(output/'publication.json')])
        if mode=='evidence':
            call('publication-tests',['-m','unittest','discover','-s',CASE+'/publication/tests','-v'])
            call('analysis',['-m','tools.modelpack.analysis','--case',CASE,'--output',str(output/'analysis')])
            call('scalars',['-m','tools.modelpack.audit','--case',CASE,'--summary',str(output/'analysis/summary.json'),'--output',str(output/'scalars.json')])
            call('timing',[CASE+'/scripts/check_timing.py','--case',CASE,'--summary',str(output/'analysis/summary.json'),'--output',str(output/'timing.json')])
            for name in ('additional_tables','local_tables'):
                call(name,[CASE+'/scripts/'+name+'.py','--case',CASE,'--output',str(output/(name+'.json'))])
            call('posthoc',[CASE+'/publication/posthoc/recalculate.py','--case',CASE,'--output',str(output/'posthoc.json')])
            pairs=[('analysis/summary.json','results/derived/summary.json'),('additional_tables.json','results/derived/additional_tables.json'),('local_tables.json','results/derived/local_tables.json'),('posthoc.json','publication/posthoc/metrics.json')]
            for actual,expected in pairs:
                if (output/actual).read_bytes()!=(ROOT/CASE/expected).read_bytes():raise ValueError('Regeneration mismatch: '+actual)
            results.append({'check':'four-JSON-byte-identity','exit_code':0})
        else:
            call('restore',[CASE+'/publication/restore_original.py','--root',str(ROOT),'--output',str(output/'historical')])
            call('historical-inventory',['-m','tools.modelpack.package','verify'],output/'historical')
            call('historical-cpu-tests',['-m','unittest','discover','-s',CASE+'/tests','-v'],output/'historical')
            call('tiny-demo-tests',['-m','unittest','discover','-s','tests/modelpack_demo','-v'])
        if before!=inventory(ROOT):raise ValueError('Protected source inventory changed')
    except Exception as exc:
        report={'status':'FAIL','mode':mode,'checks':results,'error':str(exc),'source_unchanged':before==inventory(ROOT)}
        (output/'checks.json').write_text(json.dumps(report,indent=2)+'\n');raise
    report={'status':'PASS','mode':mode,'checks':results,'source_unchanged':True,'new_gpu_runs':0,
            'scope':'Retained scalar consistency' if mode=='evidence' else 'Historical tests and random tiny CPU serialization fixtures; no trained-model quality evaluation'}
    (output/'checks.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('evidence','modelpack'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.mode,a.output)
