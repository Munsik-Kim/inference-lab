"""Isolated one-phase consumer of verified frozen scientific bytes.

No full trajectory/reference integration. Small ODE fixtures run only in the
separate --tests process. New output contains derived files and receipts only.
"""
from pathlib import Path
import argparse,sys,json,time,resource,unittest,os

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['p0','p1','p2'],required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--original',type=Path,required=True)
    p.add_argument('--run',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--tests',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();cpu=time.process_time()
    denied=[Path(x).resolve() for x in json.loads(os.environ.get('CASE019_DENY_PATHS','[]'))]
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            q=Path(os.fsdecode(args[0])).resolve()
            if any(q.is_relative_to(d) for d in denied):raise PermissionError('CASE019_FORBIDDEN_ORIGINAL_PATH')
    sys.addaudithook(guard)
    sys.path.insert(0,str(a.source/'src'))
    import numpy as np
    from silent_alignment.io import read_json,write_json,check_files,canonical,digest
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from portable import compare_tree,compare_csv
    freeze=read_json(a.original/f'publication/identity/{a.phase}/execution_freeze.json')
    check_files(a.source,freeze['files'])
    import silent_alignment
    modules={'silent_alignment':silent_alignment.__file__}
    if a.tests:
        folder={'p0':'tests','p1':'p1_tests','p2':'p2_tests'}[a.phase]
        suite=unittest.TestLoader().discover(str(a.source/folder),pattern='test*.py')
        with (a.output/'tests.log').open('w') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
        names=[]
        def ids(s):
            for t in s:
                if isinstance(t,unittest.TestSuite):yield from ids(t)
                elif t is not None:yield a.phase+':'+t.id()
        # Discover fresh only for IDs, not another execution/ODE call.
        names=list(ids(unittest.TestLoader().discover(str(a.source/folder),pattern='test*.py')))
        receipt={'status':'PASS' if result.wasSuccessful() else 'FAIL','phase':a.phase,'unique_test_ids':names,'tests':result.testsRun,
                 'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),
                 'unit_fixture_ODEs':'small original unit fixtures only; not full primary/reference runs'}
        if not result.wasSuccessful():write_json(a.output/'failure.json',receipt);raise SystemExit(2)
    else:
        run=a.run;rm=read_json(run/'run_manifest.json');check_files(run,rm['files'])
        expected=rm.get('manifest_rows_sha256',rm.get('bundle_sha256'))
        if digest(canonical(rm['files']))!=expected:raise ValueError('RUN_MANIFEST')
        frozen=read_json(run/'execution_freeze.json');check_files(run/'freeze',frozen['files'])
        if frozen['bundle_sha256']!=freeze['bundle_sha256']:raise ValueError('WRONG_PHASE_SOURCE')
        config=read_json(run/'config.json');tol=config['tolerances'];plan=read_json(run/'input_manifest.json')
        if a.phase=='p0':
            from silent_alignment.engine import load_assets
            from silent_alignment import math as metrics
            rows=plan['trajectories'];keyname='id'
        elif a.phase=='p1':
            from silent_alignment.p1.contract import load_inputs as load_assets,planned,assert_source
            from silent_alignment.p1.metrics import measure
            from silent_alignment.p1.verify import validate_saved_arrays
            assert_source(a.source,frozen)
            if [r['key'] for r in planned(config)]!=[r['key'] for r in plan['planned_trajectories']]:raise ValueError('P1_KEY_ORDER')
            rows=plan['planned_trajectories'];keyname='key'
        else:
            from silent_alignment.p2.contract import load_branch as load_assets,planned,assert_source
            from silent_alignment.p1.metrics import measure
            assert_source(a.source,frozen);rows=plan['planned_branches'];keyname='key'
            if [r['key'] for r in planned(config)[1]]!=[r['key'] for r in rows]:raise ValueError('P2_KEYS_ORDER')
        if len({r[keyname] for r in rows})!=len(rows):raise ValueError('DUPLICATE_TRAJECTORY')
        grids=cks=0;max_loss=max_norm=max_checkpoint=0.
        for row in rows:
            key=row[keyname]
            if a.phase=='p0':C,left,right,sigma,R=load_assets(run,row)
            elif a.phase=='p1':C,left,right,sigma,R=load_assets(run,row)
            else:C,left,right,sigma,U0,V0=load_assets(run,row)
            with np.load(run/'raw'/(key+'.npz'),allow_pickle=False) as z:raw={k:z[k] for k in z.files}
            t=raw['times'];ss=np.sum(raw['z_singular_sorted']**2,axis=1)
            loss=.5*np.sum(C*C)+.5*ss-raw['target_mode_signed']@sigma
            le=float(np.max(abs(loss-raw['loss'])));ne=float(np.max(abs(ss-raw['z_frobenius']**2)))
            if le>tol['raw_algebra_absolute'] or ne>tol['raw_algebra_absolute']:raise ValueError('RAW_ALGEBRA')
            max_loss=max(max_loss,le);max_norm=max(max_norm,ne);grids+=len(t)
            if a.phase=='p0':
                gram=np.einsum('tij,tji->t',raw['gram_u'],raw['gram_v'])
                if np.max(abs(gram-ss))>tol['raw_algebra_absolute'] or raw['left_rank'][0]!=0 or not np.isnan(raw['left_distance'][0]):raise ValueError('P0_GRAM_ZERO_DIRECTION')
                drift=np.linalg.norm(raw['gram_v']-raw['gram_u']-2*row['rho']*np.eye(row['m']),axis=(1,2))
                if np.max(abs(drift-raw['imbalance_absolute']))>tol['raw_algebra_absolute']:raise ValueError('P0_IMBALANCE')
            elif a.phase=='p1':validate_saved_arrays(raw,row,C,sigma,tol)
            else:
                if np.max(abs((raw['loss']-row['loss_min'])/row['loss_denominator']-raw['learnable_residual']))>1e-10:raise ValueError('P2_COMMON_DENOMINATOR')
            with np.load(run/'checkpoints'/(key+'.npz'),allow_pickle=False) as z:
                for j,tt in enumerate(z['times']):
                    ix=np.flatnonzero(t==tt)
                    if len(ix)!=1:raise ValueError('CHECKPOINT_TIME')
                    ix=int(ix[0]);u,v=z['U'][j],z['V'][j];scale=np.sqrt(2*row['rho'])
                    if a.phase in ('p0','p1'):
                        aname,bname=('primary_solver_scaled_A','primary_solver_scaled_B') if a.phase=='p0' else ('scaled_A','scaled_B')
                        aa,bb=z[aname][j],z[bname][j]
                        if not np.array_equal(u,scale*aa) or not np.array_equal(v,scale*bb):raise ValueError('PHYSICAL_SCALED_CHECKPOINT')
                    else:aa,bb=u/scale,v/scale
                    if a.phase=='p0':q=metrics.metrics(C,left,sigma,right,aa,bb,row['rho'],tol)
                    else:q=measure(C,left,right,sigma,aa,bb,row['rho'],tol)
                    for name,value in q.items():
                        if name not in raw:continue
                        if a.phase=='p2' and name=='z_singular_sorted':value=value[:32]
                        v1,v2=np.asarray(value),np.asarray(raw[name][ix])
                        if not np.array_equal(np.isnan(v1),np.isnan(v2)):raise ValueError('CHECKPOINT_NA')
                        ok=np.isfinite(v1);err=float(np.max(abs(v1[ok]-v2[ok]))) if ok.any() else 0.
                        max_checkpoint=max(max_checkpoint,err)
                        bound=1e-9 if a.phase=='p2' else tol['raw_algebra_absolute']
                        if err>bound:raise ValueError('CHECKPOINT_METRIC '+name)
                    if a.phase=='p2':
                        drift=np.linalg.norm(v.T@v-u.T@u-(V0.T@V0-U0.T@U0))
                        if abs(drift-raw['observed_imbalance_drift_absolute'][ix])>1e-12:raise ValueError('P2_OBSERVED_DRIFT')
                    cks+=1
        derived=a.output/'derived';derived.mkdir()
        if a.phase=='p0':
            from silent_alignment.analysis import regenerate
            from silent_alignment.plots import figures
            summary=regenerate(run,derived);figures(run,derived)
        elif a.phase=='p1':
            from silent_alignment.p1.analysis import regenerate
            from silent_alignment.p1.plots import figures
            summary=regenerate(run,derived);figures(run,derived)
        else:
            from silent_alignment.p2.analysis import summarize
            from silent_alignment.p2.plots import render
            # summarize owns creation of its new derived directory.
            derived.rmdir();summary=summarize(run,derived,config);render(a.run.parent.parent,run,derived/'figures',config,summary)
        compare_tree(summary,read_json(run/'derived/summary.json'))
        csv_checks={p.name:compare_csv(p,run/'derived'/p.name) for p in derived.glob('*.csv')}
        figures=[]
        import xml.etree.ElementTree as ET
        for p in sorted((derived/'figures').iterdir()):
            old=run/'derived/figures'/p.name
            if p.suffix=='.svg':ET.parse(p)
            figures.append({'file':p.name,'byte_identical':p.read_bytes()==old.read_bytes()})
        expected_counts={'p0':(17,17034,136),'p1':(81,137862,891),'p2':(128,214656,1920)}[a.phase]
        if (len(rows),grids,cks)!=expected_counts:raise ValueError('PHASE_DENOMINATOR')
        receipt={'status':'PASS','phase':a.phase,'primary_paths':len(rows),'grid_checks':grids,'physical_checkpoints':cks,
                 'max_loss_algebra_error':max_loss,'max_norm_squared_error':max_norm,'max_checkpoint_metric_error':max_checkpoint,
                 'summary_numeric_and_discrete':'PASS','csv':csv_checks,'figures':figures,'full_ODE_runs':0,'full_reference_integrations':0}
        from recorded import check
        receipt['recorded_extras']=check(a.phase,a.run.parent.parent,run,a.output)
        if a.phase=='p2':
            from make_projection import project
            projection=project(a.run.parent.parent)
            compare_tree(projection,read_json(a.original/'publication/summary/site_projection.json'))
            write_json(a.output/'site_projection.json',projection)
            receipt['display_projection_recalculated']='PASS: actual stored grids and R8 pointwise medians'
    import scipy,matplotlib
    receipt.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,
                   peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   isolated=bool(sys.flags.isolated),source_root=str(a.source),source_bundle=freeze['bundle_sha256'],
                   python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
                   sys_path=sys.path,module_files={k:v.__file__ for k,v in sys.modules.items() if (k.startswith('silent_alignment') or k in ['numpy','scipy','matplotlib']) and getattr(v,'__file__',None)})
    for name,path in receipt['module_files'].items():
        if name.startswith('silent_alignment') and not Path(path).resolve().is_relative_to(a.source.resolve()):raise ValueError('CROSS_PHASE_IMPORT')
    write_json(a.output/'verification.json',receipt);print(a.phase,receipt['status'],flush=True)
if __name__=='__main__':main()
