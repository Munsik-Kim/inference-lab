"""Clone-safe CPU replay; never imports a model or relaxes the original verifier."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from restore_p1 import restore,CASE


def run(case,output,tests=False):
    case=Path(case);output=Path(output).resolve()
    if output.exists() or output.is_relative_to(case.resolve()):raise ValueError('choose a new output outside Case015')
    output.mkdir(parents=True);original=output/'p1_original_view';preservation=restore(case,original)
    a1=case/'followups/order-api-regression-v1';commands=[];receipts=[]
    if tests:commands.append(('p1-tests',original,['-m','unittest','discover','-s','tests','-v'],original/'src'))
    commands.append(('p1-closed-audit',original,['scripts/audit_public.py','--output',str(output/'p1-audit.json')],original/'src'))
    if tests:commands.append(('a1-tests',a1,['-m','unittest','discover','-s','tests','-v'],a1/'src'))
    if tests:commands.append(('publication-tests',case,['-m','unittest','discover','-s','publication/tests','-v'],a1/'src'))
    commands.extend([
        ('a1-validate',a1,['-m','order_regression.cli','validate'],a1/'src'),
        ('a1-oracle',a1,['-m','order_regression.cli','oracle-check','--output',str(output/'a1-oracle.json')],a1/'src'),
        ('a1-audit',a1,['-m','order_regression.cli','audit','--output',str(output/'a1-audit.json')],a1/'src'),
        ('a1-analysis',a1,['-m','order_regression.cli','analyze','--output',str(output/'a1-summary.json')],a1/'src'),
        ('a1-review-render',a1,['postprocess/render_review.py','--output',str(output/'a1-explorer')],a1/'src'),
        ('unified-manifest',case,['publication/verify.py','--output',str(output/'public-identity.json')],a1/'src'),
    ])
    for name,cwd,args,source in commands:
        env=os.environ.copy();env.update(PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_VISIBLE_DEVICES='')
        command=[sys.executable,'-B',*args];r=subprocess.run(command,cwd=cwd,env=env,capture_output=True,text=True)
        (output/(name+'.log')).write_text(r.stdout+r.stderr)
        receipts.append({'check':name,'exit_code':r.returncode})
        print(name+': '+('PASS' if r.returncode==0 else 'FAIL'),flush=True)
        if r.returncode:break
    ok=len(receipts)==len(commands) and all(x['exit_code']==0 for x in receipts)
    if ok:
        spec=importlib.util.spec_from_file_location('case015_public_derive',case/'publication/scripts/derive.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        data=mod.derive(case);stored=json.loads((case/'publication/summary/project.json').read_text())
        if data!=stored:raise ValueError('raw-derived display mismatch')
        if json.loads((output/'a1-summary.json').read_text())!=json.loads((a1/'results/a1_r1/summary.json').read_text()):raise ValueError('A1 summary mismatch')
        (output/'project.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    value={'status':'PASS' if ok else 'FAIL','checks':receipts,'original':preservation,'P1_primary_episodes':96,'A1_primary_episodes':96,
        'test_scope':'27 frozen P1 regression tests;18 A1 and6 publication methods (5 existing+1 public-v2 preservation), counted separately; reruns are not additional unique tests' if tests else 'record audits only; use --tests for CPU regression suites',
        'scope':'raw response reparse, actual fixture replay, generation result receipt, state version, independent endpoint, paired tables and display identities',
        'new_model_forwards':0,'not_reproduced':['GPU model inference','full logits','private quantized payload decode']}
    (output/'verification.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--tests',action='store_true');a=p.parse_args()
    result=run(CASE,a.output,a.tests);raise SystemExit(0 if result['status']=='PASS' else 1)
