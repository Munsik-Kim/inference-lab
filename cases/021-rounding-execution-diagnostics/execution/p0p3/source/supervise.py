"""Serial parent wall ledger. This parent imports no GPU libraries."""
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
from common import CASE,read,write,append,utc,verify_files,file_row

def telemetry():
    commands=[['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,pstate,temperature.gpu,power.draw,clocks.sm','--format=csv,noheader,nounits'],['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits']]
    out=[]
    for cmd in commands:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
        out.append({'command':cmd,'returncode':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip()})
    return out

def child_env(private):
    env=os.environ.copy();cache=Path(private)/'cache'
    for name in ('HF_HOME','HF_HUB_CACHE','TORCH_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','CUDA_CACHE_PATH','VLLM_CACHE_ROOT','MPLCONFIGDIR'):
        p=cache/name;p.mkdir(parents=True,exist_ok=True);env[name]=str(p)
    env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VLLM_LOGGING_LEVEL='WARNING')
    return env

def main():
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--python',required=True);p.add_argument('--stage',choices=['P0','P1','P2','P3'],required=True);p.add_argument('--timeout',type=float,required=True);p.add_argument('--label',required=True);p.add_argument('--freeze',required=True);p.add_argument('worker_args',nargs=argparse.REMAINDER);a=p.parse_args()
    private=Path(a.private);task=read(private/'task.json');freeze=read(a.freeze);verify_files(freeze)
    lock=(private/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=CASE/'worker_ledger.jsonl';rows=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    if any(x['label']==a.label for x in rows):raise ValueError('WORKER_LABEL_ALREADY_USED')
    spent=sum(x['worker_wall_seconds'] for x in rows);stage=sum(x['worker_wall_seconds'] for x in rows if x['stage']==a.stage)
    remaining=min(task['global_gpu_wall_cap_seconds']-spent,task['stage_caps_seconds'][a.stage]-stage)
    if a.timeout>remaining or a.timeout<=0:raise ValueError('BUDGET_EXHAUSTED: timeout exceeds remaining cap')
    used=sum(f.stat().st_size for root in (private,Path(task['worktree'])) for f in root.rglob('*') if f.is_file() and not f.is_symlink())
    if used>=task['disk_cap_bytes']:raise ValueError('DISK_CAP_EXHAUSTED')
    before=telemetry()
    if any(x['returncode'] for x in before) or before[1]['stdout']:raise ValueError('GPU_BUSY_OR_UNOBSERVED: '+str(before))
    fields=before[0]['stdout'].split(',')
    if len(fields)<6 or float(fields[5])>0 or float(fields[4])<6144:raise ValueError('GPU_BUSY_OR_INSUFFICIENT_HEADROOM')
    out=private/'raw'/a.label;out.mkdir(exist_ok=False)
    cmd=[a.python,str(CASE/'source/worker.py'),'--private',str(private),'--output',str(out),'--freeze',str(a.freeze)]+[x for x in a.worker_args if x!='--']
    write(out/'launch.json',{'utc':utc(),'command':cmd,'freeze':file_row(a.freeze),'stage':a.stage,'timeout_seconds':a.timeout,'remaining_seconds':remaining,'owned_bytes_before':used,'telemetry_before':before})
    append(CASE/'attempts.jsonl',{'event':'WORKER_START','label':a.label,'stage':a.stage,'utc':utc(),'freeze':freeze['bundle_sha256']})
    wall=time.perf_counter();usage0=resource.getrusage(resource.RUSAGE_CHILDREN)
    timed_out=False
    with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
        process=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=child_env(private),start_new_session=True)
        try:code=process.wait(timeout=max(.01,a.timeout-2))
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(process.pid,signal.SIGTERM)
            try:code=process.wait(timeout=1)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
    elapsed=time.perf_counter()-wall;usage1=resource.getrusage(resource.RUSAGE_CHILDREN)
    row={'label':a.label,'stage':a.stage,'worker_wall_seconds':elapsed,'cpu_user_seconds':usage1.ru_utime-usage0.ru_utime,'cpu_system_seconds':usage1.ru_stime-usage0.ru_stime,'parent_children_max_rss_kib':usage1.ru_maxrss,'exit_code':code,'timeout':timed_out,'pid':process.pid,'source_bundle_sha256':freeze['bundle_sha256'],'raw_directory':str(out),'utc':utc()}
    append(ledger,row);write(out/'parent_receipt.json',{**row,'telemetry_after':telemetry()})
    append(CASE/'attempts.jsonl',{'event':'WORKER_END',**row})
    print(json.dumps(row));sys.exit(code if 0<=code<=125 else 1)

if __name__=='__main__':main()
