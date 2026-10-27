"""Supplemental DEV-only measurement noninterference, separately frozen.

The primary harness is unchanged. This adds the actual CPU-copy hook and
rounding-mode restoration checks before any EVAL execution.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
from common import CASE,read,write,append,verify_files,file_row,utc

def child(a):
    import gc
    import numpy as np
    import torch
    import runtime as rt
    from worker import encoded,full_metrics
    rt.flags();verify_files(read(CASE/'provenance/freeze_v1.json'))
    loc=read(Path(a.private)/'locations.json');manifest=read(CASE/'inputs/manifest.json');doc=manifest['document_order']['DEV'][0];rows=[]
    with torch.inference_mode():
        model=rt.load_model(loc,'D0');args=rt.inputs(doc,512)
        for path in ['D0','D1','P']:
            if path!='D0':rt.replace(model,loc['artifact_dir'],path)
            native=encoded(model(**args).logits)
            with rt.rounding(model,'R0'):noop=encoded(model(**args).logits)
            values=[];geometry=[]
            handles=[layer.register_forward_hook(lambda m,x,y:values.append(encoded(y))) for layer in model.model.layers]
            handles.append(model.lm_head.register_forward_pre_hook(lambda m,x:geometry.append(list(x[0].shape))))
            with rt.rounding(model,'R0'):dump=encoded(model(**args).logits)
            for h in handles:h.remove()
            if len(values)!=28 or geometry!=[[1,512,1024]]:raise RuntimeError('ACTUAL_HOOK_GEOMETRY')
            m0=full_metrics(native,noop);m1=full_metrics(native,dump)
            del noop,dump,values;gc.collect()
            with rt.rounding(model,'R1'):r1=encoded(model(**args).logits)
            values=[];handles=[layer.register_forward_hook(lambda m,x,y:values.append(encoded(y))) for layer in model.model.layers]
            with rt.rounding(model,'R1'):dump=encoded(model(**args).logits)
            for h in handles:h.remove()
            m2=full_metrics(r1,dump);del r1,dump,values;gc.collect()
            restored=encoded(model(**args).logits);m3=full_metrics(native,restored);del native,restored
            if any(not x['bit_identical'] for x in [m0,m1,m2,m3]):raise RuntimeError('CAPTURE_OR_MODE_RESTORE_CHANGED_FULL_OUTPUT')
            rows.append({'path':path,'native_vs_R0_noop':m0,'R0_actual_CPU_copy_hook':m1,'R1_actual_CPU_copy_hook':m2,'R0_after_R1_restoration':m3,'full_positions':512,'full_vocabulary':151936,'full_forwards':6})
        from torch.profiler import profile,ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            with rt.rounding(model,'R0'):out=model(**rt.inputs(doc,128)).logits;rt.sync()
        prof.export_chrome_trace(str(a.output/'full_model_trace.json'))
        trace=json.loads((a.output/'full_model_trace.json').read_text());events=trace['traceEvents']
        names=sorted({x.get('name','') for x in events if x.get('cat')=='kernel'});cpuops=sorted({x.get('name','') for x in events if x.get('cat')=='cpu_op'})
        if not any('marlin' in n.lower() for n in names) or not any('scaled_dot_product' in n for n in cpuops):raise RuntimeError('FULL_MODEL_BACKEND_UNOBSERVED')
        result={'status':'VERIFIED','source_bundle_sha256':read(a.freeze)['bundle_sha256'],'primary_source_bundle_sha256':read(CASE/'provenance/freeze_v1.json')['bundle_sha256'],'document':doc,'rows':rows,'full_forwards':18,'profile_full_forwards':1,'actual_SDPA_ops':[n for n in cpuops if 'scaled_dot_product' in n],'cuda_kernel_names':names,'memory':rt.memory(),'scope':'Full512 logits for actual CPU-copy block hook and scoped mode restoration; profile separate DEV128 forward'}
        write(a.output/'capture_validation.json',result);write(a.output/'exit.json',{'status':'COMPLETE','utc':utc()});print(json.dumps({'full_capture_checks':12,'full_forwards':18,'profile_forwards':1,'bit_identical':True}))

def parent(a):
    from supervise import telemetry,child_env
    private=Path(a.private);task=read(private/'task.json');rows=[json.loads(x) for x in (CASE/'worker_ledger.jsonl').read_text().splitlines()]
    remaining=min(7200-sum(x['worker_wall_seconds'] for x in rows),1200-sum(x['worker_wall_seconds'] for x in rows if x['stage']=='P0'))
    if a.timeout>remaining:raise ValueError('BUDGET_EXHAUSTED')
    label='p0-actual-capture-validation';out=private/'raw'/label;out.mkdir(exist_ok=False)
    lock=(private/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=telemetry()
    if any(x['returncode'] for x in before) or before[1]['stdout'] or float(before[0]['stdout'].split(',')[5])>0:raise ValueError('GPU_BUSY')
    cmd=[a.python,str(Path(__file__).resolve()),'--child','--private',str(private),'--freeze',str(Path(a.freeze).resolve()),'--output',str(out)]
    write(out/'launch.json',{'command':cmd,'timeout':a.timeout,'remaining':remaining,'utc':utc(),'telemetry_before':before,'freeze':file_row(a.freeze)})
    append(CASE/'attempts.jsonl',{'event':'WORKER_START','label':label,'stage':'P0','utc':utc()})
    u0=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.perf_counter();timeout=False
    with (out/'stdout.txt').open('w') as so,(out/'stderr.txt').open('w') as se:
        proc=subprocess.Popen(cmd,stdout=so,stderr=se,env=child_env(private),start_new_session=True)
        try:code=proc.wait(timeout=a.timeout-2)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(proc.pid,signal.SIGTERM)
            try:code=proc.wait(timeout=1)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
    wall=time.perf_counter()-start;u1=resource.getrusage(resource.RUSAGE_CHILDREN)
    row={'label':label,'stage':'P0','worker_wall_seconds':wall,'cpu_user_seconds':u1.ru_utime-u0.ru_utime,'cpu_system_seconds':u1.ru_stime-u0.ru_stime,'parent_children_max_rss_kib':u1.ru_maxrss,'exit_code':code,'timeout':timeout,'pid':proc.pid,'source_bundle_sha256':read(a.freeze)['bundle_sha256'],'raw_directory':str(out),'utc':utc()}
    append(CASE/'worker_ledger.jsonl',row);write(out/'parent_receipt.json',{**row,'telemetry_after':telemetry()});append(CASE/'attempts.jsonl',{'event':'WORKER_END',**row});print(json.dumps(row));sys.exit(0 if code==0 else 1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--freeze',required=True);p.add_argument('--python');p.add_argument('--output',type=Path);p.add_argument('--timeout',type=float,default=300);p.add_argument('--child',action='store_true');a=p.parse_args();verify_files(read(a.freeze))
    if a.child:
        try:child(a)
        except BaseException as e:write(a.output/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
    else:parent(a)

if __name__=='__main__':main()
