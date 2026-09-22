"""Bounded real harness API calls; results are private, parent judges process exit."""
import argparse, faulthandler, gc, hashlib, json, math, os, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['model','tokenizer','protocol','config','path','output']:p.add_argument('--'+n,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=False,parents=True)
    faulthandler.enable()
    def mark(stage,**kw):
        with (out/'stages.jsonl').open('a') as f:f.write(json.dumps({'stage':stage,'time':time.time(),**kw},allow_nan=False)+'\n')
    mark('launch',pid=os.getpid())
    from lm_eval.models.vllm_causallms import VLLM
    from lm_eval.api.instance import Instance
    protocol=json.loads(Path(a.protocol).read_text())
    model=VLLM(pretrained=a.model,tokenizer=a.tokenizer,**protocol['runtime'])
    mark('init')
    process_handles=list(model.model.llm_engine.engine_core.resources.engine_manager.processes)
    mark('workers',workers=[{'pid':x.pid,'name':x.name} for x in process_handles])
    items=protocol['inputs'][a.path]
    requests=[]
    for i,args in enumerate(items):
        parts=[v for v in args if isinstance(v,str)]
        lengths=[len(model.tok_encode(v)) for v in parts]
        if sum(lengths)>protocol['input_token_cap']:raise ValueError('input budget exceeded')
        requests.append(Instance(request_type=a.path,doc={},arguments=tuple(args),idx=i,metadata=('synthetic_lifecycle',i,1)))
    mark('forward',requests=len(requests))
    result=getattr(model,a.path)(requests,disable_tqdm=True)
    if len(result)!=len(items):raise ValueError('result count')
    if a.path=='generate_until':
        if not all(isinstance(v,str) and len(model.tok_encode(v))<=32 for v in result):raise ValueError('generation shape/budget')
    else:
        nums=[v[0] for v in result] if a.path=='loglikelihood' else result
        if not all(math.isfinite(v) for v in nums):raise ValueError('nonfinite likelihood')
    (out/'result.json').write_text(json.dumps(result,allow_nan=False)+'\n')
    mark('result_write')
    mark('shutdown_start')
    model.model.llm_engine.engine_core.shutdown(timeout=30)
    mark('shutdown_return',workers=[{'pid':x.pid,'alive':x.is_alive(),'exitcode':x.exitcode} for x in process_handles])
    if a.config=='D1':
        del requests,model
        mark('explicit_collect',unreachable=gc.collect())
    mark('complete')
if __name__=='__main__':main()
