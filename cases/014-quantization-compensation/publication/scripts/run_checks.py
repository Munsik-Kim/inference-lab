"""Model-free publication checks in fresh scratch: historical views and raw audits."""
import argparse,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
from restore_stage import restore
from derive import derive

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--output',type=Path,required=True);p.add_argument('--tests',action='store_true');a=p.parse_args();case=a.case.resolve();out=a.output.resolve()
    if out.exists() or out.is_relative_to(case):raise ValueError('new external output directory required')
    out.mkdir(parents=True);start=time.monotonic();views={};steps=[];env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for stage in ['P0','P1','P1b','P2']:views[stage]=restore(case,stage,out/('view-'+stage))
    def run(name,argv,cwd):
        tick=time.monotonic();r=subprocess.run([sys.executable,'-B',*argv],cwd=cwd,env=env,text=True,capture_output=True);text=r.stdout+r.stderr;(out/(name+'.log')).write_text(text)
        count=re.search(r'Ran (\d+) tests?',text);steps.append({'name':name,'command':['${CPU_PYTHON}','-B',*[str(v).replace(str(out),'${SCRATCH}').replace(str(case),'${CASE}') for v in argv]],'exit_code':r.returncode,'wall_seconds':time.monotonic()-tick,'tests_executed':int(count[1]) if count else None,'log_sha256':hashlib.sha256(text.encode()).hexdigest()})
        print(name+': '+('PASS' if r.returncode==0 else 'FAIL'),flush=True)
        if r.returncode:raise RuntimeError(name+' failed: '+text[-2000:])
    status='FAIL'
    try:
        run('P1-raw-and120-CIs',['analysis/p1_r2/audit_public.py','--case',str(out/'view-P1'),'--bootstrap','--output',str(out/'p1-audit.json')],out/'view-P1')
        run('P1b-raw-and462-CIs',['followups/p1b_selection/source/audit_saved.py','--run','results/p1b_r2','--output',str(out/'p1b-audit.json')],out/'view-P1b')
        run('P2-raw-and20-CI-fields',['followups/p2_core/source/audit.py','--run','results/p2_r1','--output',str(out/'p2-audit.json')],out/'view-P2')
        regenerated=derive(case);published=json.loads((case/'publication/summary/project.json').read_text())
        if regenerated!=published:raise ValueError('display JSON differs from raw-derived projection')
        (out/'regenerated-display.json').write_text(json.dumps(regenerated,ensure_ascii=False,indent=2))
        if a.tests:
            for name,path in [('P0-existing15','tests'),('P1b-existing10','followups/p1b_selection/tests'),('P2-existing9','followups/p2_core/tests'),('publication-new','publication/tests')]:run(name,['-m','unittest','discover','-s',path,'-v'],case)
        status='PASS'
    finally:
        receipt={'status':status,'stage_views':views,'steps':steps,'record_counts':{'P1':2880,'P1b':3360,'P2':320},'raw_derived_display_matches':status=='PASS','P0_MC_rerun':False,'new_model_forwards':0,'K_refits_or_requantizations':0,'wall_seconds':time.monotonic()-start,
           'scope':'Historical closed views, stored exact/dither scalars, original paired bootstrap seeds, full raw identities/units/aliases and public display. No private payload decode or GPU reproduction; historical GPU receipts stay separate.'}
        (out/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print(json.dumps({'status':status,'steps':len(steps),'wall_seconds':time.monotonic()-start}))
if __name__=='__main__':main()
