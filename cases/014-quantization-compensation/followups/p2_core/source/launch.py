"""One GPU model at a time; conservative combined build/evaluation/reload budget."""
import argparse,json,os,subprocess,time
from pathlib import Path
from p2_common import save,sha,utc

def main():
    ap=argparse.ArgumentParser()
    for k in ['case','private','output']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--previous-receipt',action='append',default=[]);a=ap.parse_args();case=Path(a.case).resolve();pr=Path(a.private);out=Path(a.output).resolve();src=Path(__file__).parent
    identity=json.loads((pr/'identity.private.json').read_text());python=identity['snapshot']['python'];prior=sum(json.loads(Path(p).read_text())['total_worker_wall_seconds'] for p in a.previous_receipt)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    budget=10800.;start=time.monotonic();began=utc();steps=[]
    for phase,filename in [('main','run_p2.py'),('reload','reload_worker.py')]:
        remaining=budget-prior-(time.monotonic()-start)
        if remaining<=0:break
        cmd=[python,str(src/filename),'--case',str(case),'--private',str(pr),'--output',str(out)]
        if phase=='main':cmd+=['--prior-worker-seconds',str(prior)]
        else:cmd+=['--remaining-seconds',str(remaining)]
        tick=time.monotonic();log=pr/'attempts'/f'{phase}-01.log';timeout=False
        with log.open('x') as f:
            try:proc=subprocess.run(cmd,cwd=case,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining);exitcode=proc.returncode
            except subprocess.TimeoutExpired:exitcode=124;timeout=True
        seconds=time.monotonic()-tick
        step={'phase':phase,'exit_code':exitcode,'wall_seconds':seconds,'timeout':timeout,'log_sha256':sha(log)};steps.append(step)
        save(pr/'attempts'/f'{phase}-01.receipt.private.json',dict(step,actual_command=cmd,ended_utc=utc()))
        if exitcode:break
    total=time.monotonic()-start;status='PASS' if len(steps)==2 and all(s['exit_code']==0 for s in steps) else 'PARTIAL_OR_FAILED'
    result={'status':status,'started_utc':began,'ended_utc':utc(),'steps':steps,'total_worker_wall_seconds':total,'previous_worker_seconds':prior,'cumulative_worker_wall_seconds':total+prior,'budget_seconds':budget,
         'accounting':'subprocess lifetimes include imports/load/CPU fit/packing/native captures/metrics/I/O/failure; not GPU busy time',
         'commands':['${REFERENCE_PYTHON} followups/p2_core/source/run_p2.py --case ${CASE} --private ${P2_PRIVATE} --output ${OUTPUT} --prior-worker-seconds ${PRIOR}',
                     '${REFERENCE_PYTHON} followups/p2_core/source/reload_worker.py --case ${CASE} --private ${P2_PRIVATE} --output ${OUTPUT} --remaining-seconds ${REMAINING}']}
    save(out/'worker_receipt.json',result);print(json.dumps(result,indent=2));raise SystemExit(0 if status=='PASS' else 1)
if __name__=='__main__':main()
