"""Bounded read-only NVML wait; cannot spawn a compute worker."""
import argparse,json,time,resource,fcntl,sys
from pathlib import Path
from datetime import datetime,timezone
from common import read,write,append,utc
from supervise_v3 import query,eligible
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;lock=(r/'records/worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=read(r/'records/gpu-C-l0-b0_parent.json');elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(prior['utc_start'])).total_seconds();limit=max(0,min(180,300-elapsed));start=time.perf_counter();cpu=time.process_time();u0=resource.getrusage(resource.RUSAGE_CHILDREN);samples=[];ready=False;begin=utc()
    while True:
        q=query();samples.append({'utc':utc(),'query':q});ready=eligible(q)
        if ready or time.perf_counter()-start>=limit:break
        time.sleep(min(10,max(0,limit-(time.perf_counter()-start))))
    u1=resource.getrusage(resource.RUSAGE_CHILDREN);record={'label':'gpu-resource-wait','kind':'GPU','utc_start':begin,'utc_end':utc(),'spawned':False,'compute_worker_spawned':False,'exit_code':0 if ready else 2,'parent_wall_seconds':time.perf_counter()-start,'cpu_user_seconds':u1.ru_utime-u0.ru_utime,'cpu_system_seconds':u1.ru_stime-u0.ru_stime,'parent_cpu_seconds':time.process_time()-cpu,'peak_child_rss_kib':u1.ru_maxrss,'samples':samples,'budget_scope':'Read-only GPU eligibility wait charged to new GPU parent budget; no CUDA context','ready':ready,'seconds_since_first_busy_before_wait':elapsed,'max_dedicated_wait_seconds':limit}
    append(r/'records/worker_ledger.jsonl',record);write(r/'records/gpu-resource-wait_parent.json',record);print(json.dumps({'ready':ready,'wall':record['parent_wall_seconds'],'samples':len(samples)}));sys.exit(record['exit_code'])
if __name__=='__main__':main()
