"""Case019 quick slice or complete recorded replay; no full ODE sweep/network."""
from pathlib import Path
import argparse,json,os,sys,time,resource,subprocess,shutil

PUB=Path(__file__).resolve().parent
sys.path.insert(0,str(PUB))
from verify import verify,restore,verify_restored,read,sha,safe,canonical

RUNS={'p0':'p0_20261023-095100','p1':'p1_20261023-104046','p2':'p2_20261023-135644'}

def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def install_guard(paths):
    """Python open-event boundary, not an OS sandbox or automatic child rule."""
    denied=[Path(p).resolve() for p in paths]
    def hook(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if any(p.is_relative_to(root) for root in denied):raise PermissionError('CASE019_FORBIDDEN_ORIGINAL_PATH')
    sys.addaudithook(hook)

def phase_source(case,phase,dest):
    dest.mkdir(parents=True,exist_ok=False)
    f=read(case/f'publication/identity/{phase}/execution_freeze.json')
    for r in f['files']:
        p=case/safe(r['path'])
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('PHASE_SOURCE_BYTES')
        q=dest/r['path'];q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    if phase!='p0':
        for relative in ['results/p0_20261023-095100/execution_freeze.json','results/p0_20261023-095100/assets/C_target.npz']:
            q=dest/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(case/'publication/quick'/relative,q)
    return dest

def quick(case,out):
    import numpy as np
    from portable import compare_tree
    # Exact source/data manifests plus an honestly limited raw slice.
    sl=read(case/'publication/quick/SLICE.json')
    for r in sl['files']:
        p=case/'publication/quick'/r['path']
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError('QUICK_SLICE_BYTES')
    checks=[]
    for phase,key in [('p0',sl['P0_scalar_id']),('p1',sl['P1_keys'][0]),*[('p2',k) for k in sl['P2_keys']]]:
        run=case/'publication/quick/results'/RUNS[phase]
        if phase=='p0':
            meta=next(r for r in read(case/'publication/identity/p0/input_manifest.json')['trajectories'] if r['id']==key)
            with np.load(run/'assets'/meta['target_asset'],allow_pickle=False) as a:C,sigma=a['C'],a['sigma']
        else:
            with np.load(run/'assets/target_g2.npz',allow_pickle=False) as a:C,sigma=a['C'],a['sigma']
        with np.load(run/'raw'/(key+'.npz'),allow_pickle=False) as a:
            ss=np.sum(a['z_singular_sorted']**2,axis=1);L=.5*np.sum(C*C)+.5*ss-a['target_mode_signed']@sigma
            err=float(np.max(abs(L-a['loss'])))
            if err>1e-8 or np.max(abs(np.sqrt(ss)-a['z_frobenius']))>1e-8:raise ValueError('QUICK_RAW_ALGEBRA')
            checks.append({'phase':phase,'key':key,'grid_points':len(a['times']),'max_loss_error':err})
    # Recompute J/G/A from saved starting and fixed-time losses, not stored flags.
    summary=read(case/'publication/summary/p2/summary.json');fixed=summary['fixed_time_paired'];rows=[]
    for row in fixed:
        baseline=next(x for x in fixed if x['parent_key']==row['parent_key'] and x['arm']=='CONTINUE' and x['tau']==row['tau'])
        arm0=next(x for x in fixed if x['key']==row['key'] and x['tau']==0)
        base0=next(x for x in fixed if x['key']==baseline['key'] and x['tau']==0)
        J=base0['loss']-arm0['loss'];G=(arm0['loss']-row['loss'])-(base0['loss']-baseline['loss']);A=baseline['loss']-row['loss']
        for k,v in [('J',J),('G',G),('A',A)]:compare_tree(v,row[k])
        rows.append((J,G,A))
    return {'status':'PASS','scope':'SLICE:4 stored numerical trajectories plus source identities, all small summary tables and J/G/A arithmetic',
            'raw_slice':checks,'paired_fixed_time_rows':len(rows),'full_recorded_check':'NOT_RUN_IN_QUICK','full_ODE_sweep':0}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['quick','full'],default='quick');p.add_argument('--evidence',type=Path)
    p.add_argument('--tests',action='store_true');p.add_argument('--deny-path',type=Path,action='append',default=[])
    a=p.parse_args();case=PUB.parent;out=a.output.resolve()
    if out.exists() or out.is_relative_to(case):p.error('Choose a never-used output outside the candidate')
    if a.mode=='full' and a.evidence is None:p.error('--mode full requires explicit local --evidence; no automatic download')
    out.mkdir(parents=True);start=time.monotonic();cpu=time.process_time();denied=[str(x.resolve()) for x in a.deny_path]
    install_guard(denied)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',
             PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(out/'mpl-cache'),CASE019_DENY_PATHS=json.dumps(denied))
    env.pop('PYTHONPATH',None)
    # This command itself is an owned child process too. Configure its cache
    # and BLAS before the first numerical-library import, not only its children.
    for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS',
              'PYTHONDONTWRITEBYTECODE','PYTHONNOUSERSITE','MPLBACKEND','MPLCONFIGDIR']:
        os.environ[k]=env[k]
    os.environ.pop('PYTHONPATH',None)
    commands=[]
    try:
        identity=verify(case);q=quick(case,out);write(out/'quick.json',q)
        roots={phase:phase_source(case,phase,out/('source-'+phase)) for phase in RUNS}
        def child(phase,target,tests=False,run=None):
            cmd=[sys.executable,'-I','-B',str(PUB/'phase_check.py'),'--phase',phase,'--source',str(roots[phase]),'--original',str(case),'--output',str(target)]
            if tests:cmd.append('--tests')
            if run:cmd+=['--run',str(run)]
            before=time.monotonic()
            with (out/(phase+('-tests' if tests else '-full')+'.log')).open('w') as log:r=subprocess.run(cmd,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT)
            commands.append({'phase':phase,'kind':'unit' if tests else 'recorded','command':cmd,'returncode':r.returncode,'wall_seconds':time.monotonic()-before})
            if r.returncode:raise ValueError('PHASE_FAILED: '+phase+(' unit' if tests else ' recorded'))
            return read(target/'verification.json')
        units=[]
        if a.tests:
            for phase in RUNS:units.append(child(phase,out/('tests-'+phase),tests=True))
            cmd=[sys.executable,'-I','-B','-m','unittest','discover','-s',str(PUB/'tests'),'-v']
            with (out/'publication-tests.log').open('w') as log:r=subprocess.run(cmd,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT)
            commands.append({'kind':'publication_unit','command':cmd,'returncode':r.returncode})
            if r.returncode:raise ValueError('PUBLICATION_TEST_FAILURE')
        phases=[];restored=None
        if a.mode=='full':
            if a.evidence.is_dir():restored=verify_restored(case,a.evidence);restored_root=a.evidence.resolve()
            else:restored=restore(case,a.evidence,out/'restored');restored_root=out/'restored'
            write(out/'restoration.json',restored)
            for phase,name in RUNS.items():phases.append(child(phase,out/('recorded-'+phase),run=restored_root/'case/results'/name))
        import numpy,scipy,matplotlib
        result={'status':'PASS','mode':a.mode,'public_identity':identity,'quick':q,'restoration':restored,'recorded_phases':phases,
                'scientific_units':[{'phase':x['phase'],'tests':x['tests'],'unique_test_ids':x['unique_test_ids']} for x in units],
                'full_ODE_sweeps':0,'new_full_reference_integrations':0,'model_requests':0,'network_downloads':0,
                'wall_seconds':time.monotonic()-start,'process_cpu_seconds_parent':time.process_time()-cpu,
                'peak_rss_kib_parent':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'children_cpu_seconds':resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
                'children_peak_rss_kib':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                'python':sys.version,'isolated':bool(sys.flags.isolated),'executable':sys.executable,
                'libraries':{n:m.__version__ for n,m in [('numpy',numpy),('scipy',scipy),('matplotlib',matplotlib)]},
                'library_files':{n:m.__file__ for n,m in [('numpy',numpy),('scipy',scipy),('matplotlib',matplotlib)]},
                'sys_path':sys.path,'commands':commands,'original_paths_denied':denied,
                'guard_scope':'Python open hooks installed explicitly in parent/phase children, not OS sandbox; actual imports reported',
                'remote_publication':'NOT_RUN'}
        write(out/'verification.json',result);print(a.mode,'PASS',flush=True)
    except BaseException as exc:
        write(out/'failure.json',{'status':'FAIL','error':type(exc).__name__,'message':str(exc),'commands':commands,'wall_seconds':time.monotonic()-start});raise
if __name__=='__main__':main()
