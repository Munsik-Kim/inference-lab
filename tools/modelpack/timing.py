"""Uninstrumented fixed-work timing; output JSON stays outside the repository."""
from pathlib import Path
import gc
import time
import numpy as np
from .common import CASE, new_external, read, write, require

ARMS=['R-B','I25','I25-R','P25','P25-R','S50','S50-R']


def timing_rows(track):
    ids=read(CASE/'inputs'/track/'manifest.json')['timing_ids']
    lookup={r['id']:r for r in read(CASE/'inputs'/track/'heldout.json')}
    require(len(ids)==12 and len(set(ids))==12,'Timing manifest incomplete')
    return [lookup[i] for i in ids]


def q_measure(llm, output, round_id):
    from vllm import SamplingParams
    rows=timing_rows('Q');records=[]
    for row in rows:
        for count in [1,8]:
            params=SamplingParams(temperature=0,max_tokens=count,ignore_eos=True)
            def call():
                result=llm.generate([{'prompt_token_ids':row['token_ids']}],params,use_tqdm=False)[0]
                require(len(result.outputs[0].token_ids)==count,'Timing generated length mismatch')
            for _ in range(5):call()
            for block in range(5):
                start=time.perf_counter()
                for _ in range(3):call()
                records.append({'id':row['id'],'task':row['task'],'length':row['length'],'round':round_id,
                    'block':block,'calls':3,'boundary':'one_token_request' if count==1 else 'fixed_8_token_request',
                    'new_tokens':count,'wall_seconds_per_call':(time.perf_counter()-start)/3})
        print('Q_TIMING',round_id,row['id'],flush=True)
    write(output/'timing.json',records)
    # generate blocks until worker completion; this includes vLLM scheduling/readback.
    write(output/'timing_scope.json',{'wall':'blocking vLLM local request, pretokenized prompt; no server/network',
         'one_token':'prefill + sampler + scheduling + result readback; not isolated transformer prefill',
         'kv_cache':'new sequence for every call; prefix cache disabled','cuda_events':'NOT_MEASURED',
         'process_pairing':'one model per separate engine process, paired round/input/block cohorts',
         'diagnostic_hooks':False,'profiler':False})


def r_round(artifacts: Path, output: Path, round_id: int):
    import torch
    from .artifact import load_dense
    from .runtime import gpu_guard,forward
    gpu_guard();out=new_external(output);rows=timing_rows('R');records=[];memory=[]
    arms=ARMS[round_id-1:]+ARMS[:round_id-1]
    for arm in arms:
        model=load_dense(artifacts/arm,'cuda');torch.cuda.reset_peak_memory_stats()
        for row in rows:
            ids=torch.tensor([row['token_ids']],device='cuda')
            def call(count):
                result=forward(model,ids,cache=count>1)
                if count>1:
                    for _ in range(count-1):
                        token=result.logits[:,-1].argmax(-1,keepdim=True)
                        result=forward(model,token,cache=True,past=result.past_key_values)
                return result.logits[:,-1].argmax(-1)
            for count in [1,8]:
                for _ in range(5):call(count)
                torch.cuda.synchronize()
                for block in range(5):
                    start=time.perf_counter()
                    for _ in range(3):call(count)
                    torch.cuda.synchronize()
                    records.append({'arm':arm,'id':row['id'],'task':row['task'],'length':row['length'],
                        'round':round_id,'block':block,'calls':3,'new_tokens':count,
                        'boundary':'prefill_and_argmax' if count==1 else 'fixed_8_token_request',
                        'wall_seconds_per_call':(time.perf_counter()-start)/3})
        memory.append({'arm':arm,'max_allocated':torch.cuda.max_memory_allocated(),
                         'max_reserved':torch.cuda.max_memory_reserved(),'scope':'one complete model, fixed workload, allocator'})
        del model;gc.collect();torch.cuda.empty_cache();print('R_TIMING',round_id,arm,flush=True)
    write(out/'timing.json',records);write(out/'memory.json',memory)
    write(out/'status.json',{'status':'PASS','round':round_id,'arms':arms,'rows':len(records),
           'profiler':False,'diagnostic_hooks':False,'cuda_events':'NOT_MEASURED','whole_device_peak':'NOT_MEASURED'})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--round',type=int,choices=[1,2,3],required=True)
    a=p.parse_args();r_round(a.artifacts,a.output,a.round)
