"""CPU-only checks over saved records, restored originals, and synthetic fixtures."""
from pathlib import Path
import argparse,json,os,re,subprocess,sys,traceback
from integrity import CASE,read,sha,verify_manifest
from verify_publication import verify
from restore_original import restore

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--require-full-repository',action='store_true')
    parser.add_argument('--synthetic',action='store_true',help='Also run the original Torch CPU synthetic suite')
    parser.add_argument('--synthetic-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if out.exists() or out.is_relative_to(CASE.parents[1]):
        parser.error('Choose a NEW directory outside the repository')
    out.mkdir(parents=True);checks=[];tests=[]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',
             MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    def write(name,value):
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def run(name,arguments,cwd=CASE):
        result=subprocess.run([sys.executable,'-B',*map(str,arguments)],cwd=cwd,env=env,
                              text=True,capture_output=True)
        log=result.stdout+result.stderr;(out/(name+'.log')).write_text(log)
        checks.append({'check':name,'exit_code':result.returncode,'log_sha256':sha(out/(name+'.log'))})
        if result.returncode:raise RuntimeError(name+' failed; inspect '+name+'.log')
        match=re.search(r'Ran (\d+) tests?',log)
        if match:
            skip=re.search(r'skipped=(\d+)',log)
            tests.append({'suite':name,'tests':int(match.group(1)),'skipped':int(skip.group(1)) if skip else 0})
        print(name+': PASS',flush=True)
    status='FAIL'
    try:
        identity=verify(args.require_full_repository);write('identity.json',identity)
        write('restoration.json',restore(out/'restored'))
        restored=out/'restored/cases'/CASE.name
        if not args.synthetic_only:
            run('historical-identity',['analysis/verify_case.py','--output',out/'historical-identity.json'],restored)
            run('historical-scalars',['analysis/aggregate.py','--input-only-fit','results/input_only_fit.json',
                                     '--output',out/'recomputed','--no-figures'],restored)
            a=read(CASE/'results/derived/aggregate.json');b=read(out/'recomputed/aggregate.json')
            if set(a)!=set(b) or any(a[k]!=b[k] for k in a if k!='figures'):
                raise ValueError('Historical scalar recomputation differs from retained results')
            write('scalar_comparison.json',{'status':'PASS','matched_fields':[k for k in a if k!='figures'],
                  'figures':'original bytes checked separately; no rendering in scalar audit'})
            run('posthoc',['publication/posthoc/analyze.py','--output',out/'posthoc-recomputed'])
            for name in ('summary.json','items.json'):
                if read(CASE/'publication/posthoc/data'/name)!=read(out/'posthoc-recomputed'/name):
                    raise ValueError('Post-hoc recomputation differs: '+name)
            write('posthoc_comparison.json',{'status':'PASS','files':['summary.json','items.json'],
                  'scope':'POST_HOC_SAME_RECORDS; no fitting or model forward'})
            run('publication-tests',['-m','unittest','discover','-s','publication/tests','-v'])
        if args.synthetic or args.synthetic_only:
            run('original-synthetic-tests',['-m','unittest','discover','-s','tests','-v'],restored)
        verify_manifest()
        status='PASS'
    except Exception:
        (out/'failure.log').write_text(traceback.format_exc())
        print(traceback.format_exc(),file=sys.stderr)
    finally:
        write('checks.json',{'status':status,'checks':checks,'test_suites':tests,
             'counts_are_this_run_only':True,'historical_real_reload_receipt_checks':12,
             'historical_saved_boundary_receipt_checks':30,'real_checkpoint_replays_this_run':0,
             'new_research_fitting':0,'synthetic_solver_contracts_executed':bool(args.synthetic or args.synthetic_only),
             'new_test_inference':0,'gpu_runs':0,
             'scope':'saved-record CPU recomputation and optional synthetic contracts; no learned model execution'})
    return 0 if status=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
