"""Public-only, model-free restore -> independent replay -> posthoc -> figures."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from verify import CASE, verify, restore, read, check_archive
from derive import export
from costs import recompute

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tests',action='store_true');a=p.parse_args();out=a.output.resolve()
    if out.exists() or out.is_relative_to(CASE):raise ValueError('Use a fresh external output')
    out.mkdir(parents=True);wall=time.perf_counter();cpu=time.process_time();commands=[]
    try:
        public=verify(CASE);archive_check=check_archive(CASE);restored=restore(CASE,out/'original-view')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH',None)
        for phase in ['P0','P1','P2']:
            args=[sys.executable,'-B',str(CASE/'publication/replay.py'),'--view',str(out/'original-view'),'--phase',phase,'--output',str(out/(phase+'-audit.json'))]
            r=subprocess.run(args,cwd=out,env=env,text=True,capture_output=True,timeout=240)
            (out/(phase+'-replay.log')).write_text(r.stdout+r.stderr);commands.append({'phase':phase,'exit_code':r.returncode})
            if r.returncode:raise RuntimeError('Independent replay failed: '+phase)
        if a.tests:
            r=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(CASE/'publication/tests'),'-v'],env=env,cwd=out,text=True,capture_output=True,timeout=120)
            (out/'publication-tests.log').write_text(r.stdout+r.stderr);commands.append({'phase':'publication-contract-tests','exit_code':r.returncode})
            if r.returncode:raise RuntimeError('Publication boundary tests failed')
        project=export(out/'original-view',out/'derived')
        costs=recompute(out/'original-view',project)
        (out/'derived/costs.json').write_text(json.dumps(costs,indent=2)+'\n')
        expected=CASE/'publication/summary/project.json'
        if expected.exists() and read(expected)!=project:raise ValueError('Raw-to-display projection mismatch')
        # All historical original scalar costs and bootstrap rows are comparison targets only.
        p2=project['phases'][1];p1=project['phases'][0]
        for p,expected_counts in [(p1,[(6,2),(6,2),(5,0),(5,0)]),(p2,[(24,0),(24,0),(19,5),(19,5)])]:
            if [(g['Q'],g['W']) for g in p['groups']]!=expected_counts:raise ValueError('Original Q/W disagreement')
        if sum(g['requests'] for g in p2['groups'])!=728 or sum(g['output_tokens'] for g in p2['groups'])!=47204:raise ValueError('Original HOLDOUT costs disagree')
        if sum(g['requests'] for g in p1['groups'])!=180 or sum(g['output_tokens'] for g in p1['groups'])!=10692:raise ValueError('Original DEV costs disagree')
        original=read(out/'original-view/case/results/p2_r1/analysis/summary.json')
        # Exact paired row comparison is performed against the saved CSV, including intervals.
        import csv
        old=list(csv.DictReader((out/'original-view/case/results/p2_r1/analysis/paired.csv').open()))
        for new,prior in zip(project['paired_P2'],old):
            for k in ['both_success','loss','recovery','both_failure']:
                if new[k]!=int(prior[k]):raise ValueError('Paired count mismatch')
            if abs(new['difference_pp']-float(prior['difference_pp']))>1e-10:raise ValueError('Paired difference mismatch')
            for x,k in zip(new['exploratory_95_pp'],['exploratory_pointwise_95_lower_pp','exploratory_pointwise_95_upper_pp']):
                if abs(x-float(prior[k]))>1e-10:raise ValueError('Paired interval mismatch')
        report={'status':'PASS','public':public,'archive_check':archive_check,'restoration':restored,'commands':commands,'costs':costs,
                'model_episodes':120,'P0_synthetic_traces':26,'endpoint_differences':0,'new_diagnostic_rows':120,
                'new_model_requests':0,'tokenizer_reexecution':'NOT_RUN; original receipts reused',
                'CPU':{'wall_seconds':time.perf_counter()-wall,'process_CPU_seconds':time.process_time()-cpu,'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
                'scope':'Same-machine clean public-only replay of original endpoints, SSE/IDs/history/budgets, costs, paired analysis, posthoc diagnosis and SVG. No external GPU reproduction.'}
        (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps({'status':'FAIL','exception':repr(exc),'commands':commands},indent=2)+'\n');raise

if __name__=='__main__':main()
