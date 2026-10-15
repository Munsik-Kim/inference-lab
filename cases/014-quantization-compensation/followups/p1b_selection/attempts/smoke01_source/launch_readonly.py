"""Launch existing GPU Python with an auditable cumulative worker receipt."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from p1b_common import save, utc, sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--private',required=True);a=p.parse_args()
    case=Path(a.case).resolve();private=Path(a.private);identity=json.loads((private/'identity.private.json').read_text())
    out=case/'results/p1b_r1';attempt=private/'attempts/worker-01.json'
    if attempt.exists():raise FileExistsError('preserve first attempt; use explicitly registered resume if required')
    cmd=[identity['python'],str(Path(__file__).with_name('run_readonly.py')),'--case',str(case),'--private',str(private),
         '--old-private',identity['old_private'],'--model',identity['model_path'],'--output',str(out)]
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    begin=utc();t=time.monotonic()
    with (private/'attempts/worker-01.log').open('x') as log:
        proc=subprocess.run(cmd,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT)
    elapsed=time.monotonic()-t
    record={'started_utc':begin,'ended_utc':utc(),'command_private':cmd,'exit_code':proc.returncode,'worker_wall_seconds':elapsed,
            'worker_budget_seconds':5400,'counting':'subprocess lifetime includes interpreter startup/load/read/CPU/GPU/metrics/I/O/failure; not GPU busy time',
            'log_sha256':sha(private/'attempts/worker-01.log')}
    save(attempt,record)
    save(out/'worker_receipt.json',{k:v for k,v in record.items() if k!='command_private'}|{'command':'${REFERENCE_PYTHON} followups/p1b_selection/source/run_readonly.py --case ${CASE} --private ${P1B_PRIVATE} --old-private ${P1_PRIVATE} --model ${MODEL_SNAPSHOT} --output ${CASE}/results/p1b_r1'})
    print(json.dumps({k:v for k,v in record.items() if k!='command_private'},indent=2));raise SystemExit(proc.returncode)

if __name__=='__main__':main()
