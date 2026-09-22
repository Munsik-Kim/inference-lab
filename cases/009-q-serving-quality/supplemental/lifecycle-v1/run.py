"""Fresh-process bounded lifecycle ledger. Never reruns historical full tasks."""
import argparse,hashlib,json,math,os,signal,subprocess,time
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def valid_result(value,path,count):
    if not isinstance(value,list) or len(value)!=count:return False
    if path=='generate_until':return all(isinstance(x,str) for x in value)
    if path=='loglikelihood':
        return all(isinstance(x,list) and len(x)==2 and type(x[1]) is bool and type(x[0]) in (int,float) and math.isfinite(x[0]) for x in value)
    return all(type(x) in (int,float) and math.isfinite(x) for x in value)
def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ')[1][0]!='Z'
    except FileNotFoundError:return False

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--private-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).parent
    protocol=json.loads((root/'protocol.json').read_text());cfg=json.loads(a.private_config.read_text())
    assert sha(root/'child.py')==protocol['child_sha256'] and sha(root/'run.py')==protocol['runner_sha256']
    env=os.environ.copy();env.update(cfg['environment']);start=time.monotonic();rows=[]
    for cell in protocol['attempt_order']:
        cid='-'.join(str(cell[k]) for k in ['config','arm','path','repeat']);folder=a.output/cid
        if time.monotonic()-start>=protocol['walltime_seconds']:
            rows.append({**cell,'status':'NOT_RUN_BUDGET'});continue
        cmd=[cfg['python'],'-B',str(root/'child.py'),'--model',cfg['models'][cell['arm']],'--tokenizer',cfg['tokenizer'],'--protocol',str(root/'protocol.json'),'--config',cell['config'],'--path',cell['path'],'--output',str(folder)]
        t=time.monotonic();log=a.output/(cid+'.log');timed_out=False
        with log.open('wb') as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:rc=proc.wait(timeout=min(protocol['attempt_timeout_seconds'],protocol['walltime_seconds']-(time.monotonic()-start)))
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
                try:rc=proc.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
        stages=[json.loads(x) for x in (folder/'stages.jsonl').read_text().splitlines()] if (folder/'stages.jsonl').exists() else []
        returned=next((x for x in stages if x['stage']=='shutdown_return'),{})
        workers=returned.get('workers',[]);workers_clean=bool(workers) and all(not x['alive'] and x['exitcode']==0 and not alive(x['pid']) for x in workers)
        remaining=[x['pid'] for s in stages if s['stage']=='workers' for x in s['workers'] if alive(x['pid'])]
        cleanup=[]
        for pid in remaining:
            # Only explicitly recorded children in the process group we created.
            if os.getpgid(pid)==proc.pid:
                os.kill(pid,signal.SIGTERM);cleanup.append(pid)
        finite=False;result_sha=None
        if (folder/'result.json').exists():
            try:
                value=json.loads((folder/'result.json').read_text(),parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
                finite=valid_result(value,cell['path'],len(protocol['inputs'][cell['path']]));result_sha=sha(folder/'result.json')
            except (ValueError,TypeError):pass
        ok=rc==0 and finite and stages and stages[-1]['stage']=='complete' and workers_clean and not cleanup and not timed_out
        row={**cell,'elapsed_seconds':time.monotonic()-t,'returncode':rc,'signal':-rc if rc<0 else None,'timeout':timed_out,'status':'PASS' if ok else 'FAIL','result_valid':finite,'result_sha256':result_sha,'worker_clean':workers_clean,'worker_exitcodes':[x['exitcode'] for x in workers],'cleanup_owned_workers':len(cleanup),'stages':[s['stage'] for s in stages],'input_hash':protocol['input_hashes'][cell['path']],'environment_identity':protocol['environment_identity'],'log_sha256':sha(log)}
        rows.append(row);(a.output/'ledger.json').write_text(json.dumps(rows,indent=2)+'\n');print(cid,rc,row['status'],flush=True)
    (a.output/'ledger.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
