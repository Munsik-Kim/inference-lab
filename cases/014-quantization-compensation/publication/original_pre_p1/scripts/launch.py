"""Record actual command, exit and conservative worker wall. No retries or cleanup."""
import argparse,json,subprocess,time,sys,os
from pathlib import Path
from common import save
p=argparse.ArgumentParser();p.add_argument('--python',required=True);p.add_argument('--model',required=True);p.add_argument('--private',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
if any(out.iterdir()):raise SystemExit('output must be a fresh empty P1 attempt directory')
cmd=[a.python,str(Path(__file__).with_name('run_p1.py')),'--model',a.model,'--private',a.private,'--output',a.output]
start=time.time();env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
log=Path(a.private)/'attempts'/('p1-'+out.name+'.log')
with log.open('x') as f:
 try:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=5450);code=r.returncode
 except subprocess.TimeoutExpired:code=124
receipt={'argv':cmd,'exit_code':code,'wall_seconds':time.time()-start,'log':str(log),'GPU_busy_time':'NOT_MEASURED'}
save(Path(a.private)/'attempts'/('p1-'+out.name+'.json'),receipt)
save(out/'worker_receipt.json',{'argv':['PYTHON','scripts/run_p1.py','--model','MODEL_SNAPSHOT','--private','PRIVATE_RUN_ROOT','--output','FRESH_P1_OUTPUT'],'exit_code':code,'wall_seconds':receipt['wall_seconds'],'GPU_busy_time':'NOT_MEASURED'})
print(receipt);sys.exit(code)
