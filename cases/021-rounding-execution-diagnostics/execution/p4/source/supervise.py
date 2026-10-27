"""One explicit gate-success branch controls worker start. CPU-only parent."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
from common import read,write,append,utc,verify

def gate_then_start(gate, journal_start, spawn):
    result=gate()
    if result is not True: raise RuntimeError('GATE_REJECTED')
    journal_start()
    return spawn()

def query():
    commands=[['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],
              ['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,pstate,temperature.gpu,power.draw,clocks.sm','--format=csv,noheader,nounits']]
    rows=[]
    for cmd in commands:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
        rows.append({'command':cmd,'returncode':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip()})
    return rows

def eligible(rows):
    if any(r['returncode'] for r in rows) or rows[0]['stdout']:return False
    fields=rows[1]['stdout'].split(',')
    return len(fields)>=6 and float(fields[5])==0 and float(fields[4])>=6144

def start_ticks(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--kind',choices=['CPU','GPU'],required=True)
    p.add_argument('--label',required=True);p.add_argument('--timeout',type=float,required=True);p.add_argument('--freeze',type=Path);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    root=a.root;loc=read(root/'references/locations.json');ledger=root/'records/worker_ledger.jsonl'
    lock=(root/'records/worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    if any(r['label']==a.label for r in old):raise ValueError('LABEL_ALREADY_USED')
    cap=loc['new_gpu_cap_seconds'] if a.kind=='GPU' else loc['cpu_worker_wall_cap_seconds']
    spent=math.fsum(r['parent_wall_seconds'] for r in old if r['kind']==a.kind)
    if not 0<a.timeout<=cap-spent:raise ValueError('BUDGET_EXHAUSTED')
    output=root/'records'/a.label
    if output.exists():raise FileExistsError('OUTPUT_ALREADY_EXISTS')
    used=sum(f.stat().st_size for f in root.rglob('*') if f.is_file() and not f.is_symlink())
    if used>=loc['disk_cap_bytes']:raise ValueError('DISK_CAP')
    cmd=[x for x in a.command if x!='--']
    if not cmd:raise ValueError('COMMAND_MISSING')
    env=os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_LOGGING_LEVEL='WARNING')
    for key in ['HF_HOME','HF_HUB_CACHE','TORCH_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','CUDA_CACHE_PATH','VLLM_CACHE_ROOT','MPLCONFIGDIR']:
        path=root/'cache'/key;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
    if a.kind=='CPU':env['CUDA_VISIBLE_DEVICES']=''
    begun=time.perf_counter();usage0=resource.getrusage(resource.RUSAGE_CHILDREN);parent0=time.process_time()
    state={'label':a.label,'kind':a.kind,'utc_start':utc(),'command':cmd,'timeout_seconds':a.timeout,'used_bytes_before':used,'spawned':False};proc=None;streams=[]
    try:
        if a.freeze:verify(root,read(a.freeze))
        if a.kind=='GPU':
            test=read(root/'records/cpu-fixtures/result.json')
            if not test.get('gate_regression_passed'):raise RuntimeError('GATE_REGRESSION_REQUIRED')
            if not a.freeze:raise ValueError('GPU_REQUIRES_FREEZE')
        def gate():
            if a.kind=='CPU':return True
            state['telemetry_before']=query()
            if not eligible(state['telemetry_before']):raise RuntimeError('GPU_BUSY_OR_INSUFFICIENT_HEADROOM')
            return True
        def journal():
            output.mkdir(exist_ok=False);write(output/'launch.json',state)
            append(root/'records/journal.jsonl',{'event':'WORKER_START','label':a.label,'kind':a.kind,'utc':utc()})
        def spawn():
            streams.extend([(output/'stdout.txt').open('w'),(output/'stderr.txt').open('w')])
            return subprocess.Popen(cmd,env=env,stdout=streams[0],stderr=streams[1],start_new_session=True)
        proc=gate_then_start(gate,journal,spawn);state.update(spawned=True,pid=proc.pid,start_ticks=start_ticks(proc.pid))
        try:code=proc.wait(timeout=max(.01,a.timeout-(time.perf_counter()-begun)-2))
        except subprocess.TimeoutExpired:
            state['timeout']=True
            if start_ticks(proc.pid)==state['start_ticks']:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    if start_ticks(proc.pid)==state['start_ticks']:os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()
            code=124
        state['exit_code']=code
    except Exception as e:state.update(exit_code=2,error=type(e).__name__+': '+str(e))
    finally:
        for s in streams:s.close()
        state['parent_wall_seconds']=time.perf_counter()-begun
        usage1=resource.getrusage(resource.RUSAGE_CHILDREN)
        state.update(cpu_user_seconds=usage1.ru_utime-usage0.ru_utime,cpu_system_seconds=usage1.ru_stime-usage0.ru_stime,parent_cpu_seconds=time.process_time()-parent0,peak_child_rss_kib=usage1.ru_maxrss,utc_end=utc())
        if a.kind=='GPU':
            try:state['telemetry_after']=query()
            except Exception as e:state['telemetry_after_error']=str(e)
        append(ledger,state);write(root/'records'/(a.label+'_parent.json'),state)
        append(root/'records/journal.jsonl',{'event':'WORKER_END' if state['spawned'] else 'BLOCKED_BEFORE_START','label':a.label,'exit_code':state['exit_code'],'utc':utc()})
    print(json.dumps({k:state[k] for k in ['label','kind','spawned','exit_code','parent_wall_seconds']}))
    sys.exit(state['exit_code'] if 0<=state['exit_code']<=125 else 1)

if __name__=='__main__':main()
