"""Existing vLLM environment only. Prompt scoring from full native vocabulary."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
import numpy as np
from .common import CASE, digest, new_external, read, require, write
from .numerics import score


def runtime_state(worker):
    import torch
    from tools.modelpack.artifact import tensor_hash
    model=worker.model_runner.model
    for n,p in model.named_parameters():
        if not bool(torch.isfinite(p).all()):raise ValueError('Nonfinite runtime parameter')
        if 'weight_scale' in n and not bool((p>0).all()):raise ValueError('Invalid runtime scale')
    tensors={n:{'shape':list(p.shape),'dtype':str(p.dtype),'bytes':p.numel()*p.element_size(),
                'hash':tensor_hash(p)} for n,p in model.named_parameters()}
    routes={}
    for n,m in model.named_modules():
        q=getattr(m,'quant_method',None)
        if q is not None:
            scheme=getattr(m,'scheme',None)
            routes[n]={'quant_method':type(q).__name__,'scheme':type(scheme).__name__ if scheme else None,
                       'kernel':str(type(getattr(scheme,'kernel',None)).__name__) if scheme else None}
    return {'parameters':tensors,'routes':routes,'allocated':torch.cuda.memory_allocated(),
            'reserved':torch.cuda.memory_reserved(),'max_allocated':torch.cuda.max_memory_allocated(),
            'max_reserved':torch.cuda.max_memory_reserved(),'diagnostics':getattr(worker,'_c008_validity',None)}


def diagnostics_on(worker):
    import torch
    worker._c008_validity={'outputs_checked':0,'invalid':0,'boundaries':{}}
    worker._c008_hooks=[]
    def checker(name):
        worker._c008_validity['boundaries'][name]=0
        def check(m,args,out):
            tensors=out if isinstance(out,tuple) else (out,)
            for t in tensors:
                if isinstance(t,torch.Tensor):
                    worker._c008_validity['outputs_checked']+=1
                    worker._c008_validity['boundaries'][name]+=1
                    if not bool(torch.isfinite(t).all()):
                        worker._c008_validity['invalid']+=1
                        raise ValueError('BLOCKED_NUMERICAL_VALIDITY full vLLM output')
        return check
    model=worker.model_runner.model
    for i,m in enumerate(model.model.layers):worker._c008_hooks.append(m.register_forward_hook(checker('layer_'+str(i))))
    worker._c008_hooks.append(model.model.norm.register_forward_hook(checker('final_norm')))
    return {'layers':len(model.model.layers),'installed':True}


def profile_on(worker):
    import torch
    worker._c008_prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
    worker._c008_prof.start()
    return True


def profile_off(worker):
    import torch
    torch.cuda.synchronize();worker._c008_prof.stop()
    events=worker._c008_prof.events()
    names=sorted({e.name for e in events if e.device_type==torch.autograd.DeviceType.CUDA})
    ops=sorted({e.name for e in events if any(k in e.name.lower() for k in ['marlin','gemm','quant','linear'])})
    return {'cuda_kernel_names':names,'related_operations':ops,'scope':'one untimed diagnostic request'}


class AuditWorkerExtension:
    """Named local methods avoid callable/pickle RPC serialization."""
    def c008_runtime_state(self):return runtime_state(self)
    def c008_diagnostics_on(self):return diagnostics_on(self)
    def c008_profile_on(self):return profile_on(self)
    def c008_profile_off(self):return profile_off(self)


def run(checkpoint: Path, output: Path, split: str, fixture: bool = False, timing_round: int = 0):
    if split=='heldout':
        from .freeze import verify_eval
        verify_eval('Q')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['VLLM_NO_USAGE_STATS']='1';os.environ['DO_NOT_TRACK']='1'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    # Same explicit WSL-compatible runner as Case002; V2 requires unavailable UVA.
    os.environ['VLLM_USE_V2_MODEL_RUNNER']='0'
    os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
    from vllm import LLM,SamplingParams
    out=new_external(output)
    rows=read(CASE/'inputs/Q'/(split+'.json'))
    started=time.perf_counter()
    llm=LLM(model=str(checkpoint),tokenizer=str(checkpoint),dtype='bfloat16',trust_remote_code=False,
             enforce_eager=True,max_model_len=1056,max_num_seqs=1,enable_prefix_caching=False,
             enable_chunked_prefill=False,async_scheduling=False,max_num_batched_tokens=1056,
             attention_config={'backend':'FLASH_ATTN'},generation_config='vllm',kv_cache_dtype='bfloat16',
             kv_cache_memory_bytes=1024**3,gpu_memory_utilization=.78,max_logprobs=-1,
             logprobs_mode='raw_logits',seed=808190,disable_log_stats=True,worker_extension_cls='tools.modelpack.q_runtime.AuditWorkerExtension')
    write(out/'start.json',{'process_load_seconds':time.perf_counter()-started,'scope':'process startup with OS cache uncontrolled',
                         'kv_cache_bytes':1024**3,'fixture':fixture,'split':split})
    before=llm.collective_rpc('c008_runtime_state');write(out/'runtime_before.json',before)
    cfg=read(checkpoint/'config.json')
    if cfg.get('quantization_config'):
        routes=[v for v in before[0]['routes'].values() if v['quant_method']=='CompressedTensorsLinearMethod']
        require(len(routes)==cfg['num_hidden_layers']*4 and all(v['kernel']=='MarlinLinearKernel' for v in routes),'Unexpected W4 runtime/fallback')
    if timing_round:
        from .timing import q_measure
        q_measure(llm,out,timing_round)
        write(out/'runtime_after.json',llm.collective_rpc('c008_runtime_state'))
        write(out/'status.json',{'status':'PASS','round':timing_round,'kind':'timing'})
        return
    write(out/'diagnostic_install.json',llm.collective_rpc('c008_diagnostics_on'))
    params=SamplingParams(temperature=0,max_tokens=1,logprobs=-1)
    (out/'vectors').mkdir();records=[]
    for row in rows:
        response=llm.generate([{'prompt_token_ids':row['token_ids']}],params,use_tqdm=False)[0]
        values=response.outputs[0].logprobs[0]
        vocab=read(checkpoint/'config.json')['vocab_size']
        require(len(values)>=vocab and all(i in values for i in range(vocab)),'Incomplete full logits')
        z=np.array([values[i].logprob for i in range(vocab)],dtype=np.float64)
        require(np.isfinite(z).all(),'Invalid full native logits')
        np.save(out/'vectors'/(row['id']+'.npy'),z,allow_pickle=False)
        record={'id':row['id'],'task':row['task'],'split':split,'token_hash':row['token_hash'],
                'score':score(z,row['label_ids'],row['gold']),'evidence_kind':'unit_test_fixture' if fixture else 'gpu_measurement',
                'readout':'vLLM raw_logits; CPU FP64 statistics','full_vocab_checked':vocab}
        write(out/'cells'/(row['id']+'.json'),record);records.append(record)
    # Actual prefill + decode from prompt only; not primary answer scoring.
    decode=llm.generate([{'prompt_token_ids':rows[0]['token_ids']}],SamplingParams(temperature=0,max_tokens=4,ignore_eos=True),use_tqdm=False)[0]
    write(out/'decode.json',{'id':rows[0]['id'],'tokens':list(decode.outputs[0].token_ids),'fixed_new_tokens':4})
    llm.collective_rpc('c008_profile_on')
    llm.generate([{'prompt_token_ids':rows[0]['token_ids']}],SamplingParams(temperature=0,max_tokens=1),use_tqdm=False)
    write(out/'profiler.json',llm.collective_rpc('c008_profile_off'))
    after=llm.collective_rpc('c008_runtime_state');write(out/'runtime_after.json',after)
    require(after[0]['parameters']==before[0]['parameters'],'Runtime weights changed/dense fallback')
    require(after[0]['diagnostics']['outputs_checked']>0 and after[0]['diagnostics']['invalid']==0 and all(n>0 for n in after[0]['diagnostics']['boundaries'].values()),'Absent/invalid full-output evidence')
    write(out/'records.json',records)
    write(out/'status.json',{'status':'PASS','split':split,'fixture':fixture,'count':len(records),
                            'runtime_weights_unchanged':True,'full_outputs_finite':True,'records_hash':digest(records)})
    print('Q_RUNTIME_PASS',split,len(records))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['smoke','development','heldout'],required=True)
    p.add_argument('--fixture',action='store_true');p.add_argument('--timing-round',type=int,default=0)
    a=p.parse_args();run(a.checkpoint,a.output,a.split,a.fixture,a.timing_round)
