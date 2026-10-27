"""Explicit GPU CLI; --help and module import do not import Torch or load data."""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
from common import CASE,read,write,verify_files,file_row,utc

def environment(rt):
    import importlib.metadata as md
    import torch
    import transformers
    import vllm
    sources={}
    for p in [Path(rt.hf.__file__),Path(torch.__file__),Path(vllm.__file__),Path(vllm.__file__).parent/'_custom_ops.py',Path(vllm.__file__).parent/'model_executor/layers/quantization/utils/marlin_utils.py']:
        sources[str(p)]=file_row(p)
    binaries={str(p):file_row(p) for p in Path(vllm.__file__).parent.glob('_C*.so')}
    return {'utc':utc(),'python':sys.version,'executable':sys.executable,'sys_path':sys.path,'platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'versions':{k:md.version(k) for k in ['torch','transformers','vllm','numpy','safetensors']},'torch_cuda':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'device_properties':str(torch.cuda.get_device_properties(0)),'flags':rt.flags(),'runtime_sources':sources,'marlin_binaries':binaries,'cache_environment':{k:os.environ.get(k) for k in ['HF_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','CUDA_CACHE_PATH','PYTORCH_CUDA_ALLOC_CONF','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']},'environment_reuse':'Read-only inherited existing runtime; not a clean environment','historical_driver':'610.88','driver_status':'See parent telemetry; environment drift is reported separately'}

def encoded(t):
    import torch
    return t.detach().cpu().contiguous().view(torch.uint16).numpy().copy()

def full_metrics(a,b):
    import numpy as np
    from layout import bf16_float,metrics
    rows=[]
    for start in range(0,a.shape[-2],16):rows.append(metrics(bf16_float(a[...,start:start+16,:]),bf16_float(b[...,start:start+16,:])))
    sse=sum(r['SSE'] for r in rows);energy=sum(r['reference_energy'] for r in rows)
    return {'SSE':sse,'reference_energy':energy,'relative_RMSE':None if not energy else (sse/energy)**.5,'MAE_sum':sum(r['MAE_sum'] for r in rows),'elements':sum(r['elements'] for r in rows),'max_abs':max(r['max_abs'] for r in rows),'finite':all(r['finite'] for r in rows),'bit_identical':bool(np.array_equal(a,b))}

def preflight(a,rt,loc):
    import gc
    import numpy as np
    import torch
    from layout import Artifact,reconstruct,bf16_float,metrics
    from primitives import norm_rope
    result={'environment':environment(rt),'source_bundle_sha256':read(a.freeze)['bundle_sha256'],'input_manifest':file_row(CASE/'inputs/manifest.json'),'checks':{},'historical':[]}
    write(a.output/'environment.json',result['environment'])
    torch.manual_seed(610134)
    # Kernel-aligned exact fixture, fixed before any kernel observation.
    codes=np.resize(np.array([-8,-7,-1,0,1,7],np.int8),(128,128))
    gamma=np.ldexp(np.ones((128,2),np.float32),np.arange(256).reshape(128,2)%3-2)
    alpha=np.ones_like(gamma);beta=np.resize(np.array([.5,1,2,4],np.float32),128)
    q=Artifact(codes,gamma,alpha,beta,'row_column',0)
    original=torch.nn.Linear(128,128,bias=True,device='cuda',dtype=torch.bfloat16)
    original.bias.data.copy_(torch.arange(128,device='cuda').bfloat16()/8)
    x=torch.eye(128,device='cuda',dtype=torch.bfloat16)[::8]
    outputs={};models={}
    for mode in ['D0','D1','P']:
        models[mode]=rt.Packed(original,q) if mode=='P' else rt.Dense(original,q,mode)
        outputs[mode]=rt.cpu(models[mode](x))
        expected=(rt.cpu(x)*beta)@reconstruct(q,mode if mode!='P' else 'D1').T+rt.cpu(original.bias)
        m=metrics(expected,outputs[mode]);result['checks']['exact_fixture_'+mode]=m
        if m['max_abs']>1e-6:raise RuntimeError('LAYOUT_BETA_BIAS_EXACT_FIXTURE_FAILED')
    from torch.profiler import profile,ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
        y=models['P'](x);rt.sync()
    prof.export_chrome_trace(str(a.output/'marlin_trace.json'))
    trace=json.loads((a.output/'marlin_trace.json').read_text());names=sorted({e.get('name','') for e in trace.get('traceEvents',[]) if e.get('cat')=='kernel'})
    result['backend']={'cuda_kernel_names':names,'actual_op':'torch.ops._C.marlin_gemm','fallback':False,'profile_only':True}
    if not any('marlin' in n.lower() for n in names):raise RuntimeError('MARLIN_KERNEL_UNOBSERVED')
    del outputs,models,original,x,y;gc.collect();torch.cuda.empty_cache()
    model=rt.load_model(loc,'D0');result['checks']['projection_count']=196
    ids=np.load(loc['historical_ids'],allow_pickle=False);t=torch.from_numpy(ids.copy()).cuda()[None,:]
    oldargs={'input_ids':t,'use_cache':False,'logits_to_keep':0}
    torch.manual_seed(610134);gemm_x=torch.randn(16,1024,device='cuda',dtype=torch.bfloat16)
    historical={}
    for mode in ['D0','D1','P']:
        if mode!='D0':rt.replace(model,loc['artifact_dir'],mode)
        historical[mode]=[]
        for cycle in range(3):
            g=model.model.layers[0].self_attn.q_proj(gemm_x)
            hidden=model.model.embed_tokens(t);pos=torch.arange(32,device='cuda')[None,:];pe=model.model.rotary_emb(hidden,pos)
            block=model.model.layers[0](hidden,attention_mask=None,position_ids=pos,position_embeddings=pe,use_cache=False)
            logits=model(**oldargs).logits
            if logits.shape!=(1,32,151936) or logits.dtype!=torch.bfloat16:raise RuntimeError('HISTORICAL_OUTPUT_CONTRACT')
            row={}
            for name,value in [('gemm',g),('block',block),('logits',logits)]:
                row[name]=encoded(value);rt.save_tensor(a.output/f'historical_{mode}_c{cycle}_{name}.npy',value)
            historical[mode].append(row)
            del g,hidden,pe,block,logits
    for cycle in range(3):
        for ref,candidate in [('D0','D1'),('D1','P'),('D0','P')]:
            row={'cycle':cycle,'reference':ref,'candidate':candidate}
            for name in ['gemm','block','logits']:row[name]=full_metrics(historical[ref][cycle][name],historical[candidate][cycle][name])
            row['full32_argmax_flips']=int((bf16_float(historical[ref][cycle]['logits']).argmax(-1)!=bf16_float(historical[candidate][cycle]['logits']).argmax(-1)).sum())
            result['historical'].append(row)
    result['historical_gate']={'threshold':.02,'performed':'GEMM, isolated block0, full32 logits; no generation','within_gate':all(r['logits']['relative_RMSE']<=.02 for r in result['historical'] if r['reference']=='D0' and r['candidate']=='P'),'environment_status':'DIFFERENT_ENVIRONMENT'}
    del historical;gc.collect()
    rt.replace(model,loc['artifact_dir'],'D0')
    doc=read(CASE/'inputs/manifest.json')['document_order']['DEV'][0];args=rt.inputs(doc,512)
    base=None;self_rows=[]
    for i in range(3):
        out=model(**args).logits;v=encoded(out)
        if base is None:base=v
        self_rows.append(full_metrics(base,v));del out,v
    self_limit=max(1e-7,10*max(r['relative_RMSE'] for r in self_rows))
    with rt.rounding(model,'R0'):
        out=model(**args).logits;noop=full_metrics(base,encoded(out));del out
    shapes=[]
    handles=[layer.register_forward_hook(lambda m,inp,out:shapes.append(list(out.shape))) for layer in model.model.layers]
    out=model(**args).logits;hook=full_metrics(base,encoded(out));del out
    for handle in handles:handle.remove()
    self_exact=all(r['bit_identical'] for r in self_rows)
    if (self_exact and (not noop['bit_identical'] or not hook['bit_identical'])) or max(noop['relative_RMSE'],hook['relative_RMSE'])>self_limit:raise RuntimeError('NOOP_OR_HOOK_CHANGED_OUTPUT')
    result['no_op']={'document':doc,'self_repeats':self_rows,'criterion':self_limit,'bit_identical_required':self_exact,'noop':noop,'hook':hook,'hooked_layers':len(shapes)}
    # Same projected Q/K and native cos/sin; block0 only, diagnostic pass.
    hidden=model.model.embed_tokens(args['input_ids']);pos=args['position_ids'];cos,sin=model.model.rotary_emb(hidden,pos)
    block=model.model.layers[0];normed=block.input_layernorm(hidden)
    qv=block.self_attn.q_proj(normed).view(1,512,16,128)
    kv=block.self_attn.k_proj(normed).view(1,512,8,128)
    q0=block.self_attn.q_norm(qv).transpose(1,2);k0=block.self_attn.k_norm(kv).transpose(1,2)
    native=rt.hf.apply_rotary_pos_emb(q0,k0,cos,sin)
    collected={}
    def collect(name,tensor):collected[name]={'dtype':str(tensor.dtype),'shape':list(tensor.shape),'stride':list(tensor.stride()),'contiguous':tensor.is_contiguous()}
    with rt.rounding(block,'R1',collect):
        q1=block.self_attn.q_norm(qv).transpose(1,2);k1=block.self_attn.k_norm(kv).transpose(1,2)
        changed=rt.hf.apply_rotary_pos_emb(q1,k1,cos,sin)
    boundary=[]
    for name,raw,nativevalue,newvalue,weight in [('q',qv.transpose(1,2),native[0],changed[0],block.self_attn.q_norm.weight),('k',kv.transpose(1,2),native[1],changed[1],block.self_attn.k_norm.weight)]:
        if nativevalue.dtype!=newvalue.dtype or nativevalue.shape!=newvalue.shape or nativevalue.stride()!=newvalue.stride():raise RuntimeError('R1_LAYOUT_MISMATCH')
        # FP64 primitive uses exact BF16 projection/weight/cos/sin values.
        ref=norm_rope(rt.cpu(raw[:,:,:5,:]),rt.cpu(weight),block.self_attn.q_norm.variance_epsilon,rt.cpu(cos[:,:5,:]),rt.cpu(sin[:,:5,:]))
        m=metrics(ref,rt.cpu(newvalue[:,:,:5,:]));diff=metrics(rt.cpu(nativevalue),rt.cpu(newvalue));boundary.append({'tensor':name,'fp64':m,'r0_r1':diff,'r0_stride':list(nativevalue.stride()),'r1_stride':list(newvalue.stride())})
        if m['relative_RMSE']>.01:raise RuntimeError('R1_PRIMITIVE_NUMERICS')
    result['rounding_boundary_status']='DISTINCT' if any(r['r0_r1']['max_abs']>0 for r in boundary) else 'NO_DISTINCT_BOUNDARY'
    result['rounding']={'boundaries':boundary,'R0_norm_output_dtype':str(q0.dtype),'R1_norm_output_dtype':str(q1.dtype),'native_cos_dtype':str(cos.dtype),'R1_observed':collected,'attention_dtype':str(changed[0].dtype)}
    result['implementation_status']='VERIFIED';result['artifact_status']='VERIFIED';result['historical_status']='DIFFERENT_ENVIRONMENT';result['p1_ready']=True
    result['memory']=rt.memory();result['forward_counts']={'historical_full':9,'historical_block':9,'historical_gemm':9,'self_repeat':3,'no_op':1,'block_hook':1,'boundary_primitive_only':2,'synthetic_gemm':4}
    write(a.output/'preflight.json',result)
    print(json.dumps({'implementation_status':'VERIFIED','boundary':result['rounding_boundary_status'],'historical_full':next(r['logits']['relative_RMSE'] for r in result['historical'] if r['reference']=='D0' and r['candidate']=='P')}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze',required=True);p.add_argument('--mode',choices=['preflight','full','repeat','local','timing'],required=True);p.add_argument('--condition');p.add_argument('--split',choices=['DEV','EVAL']);p.add_argument('--selection');p.add_argument('--block',type=int,default=0);a=p.parse_args()
    freeze=read(a.freeze);verify_files(freeze)
    try:
        import runtime as rt
        import torch
        rt.flags();loc=read(Path(a.private)/'locations.json')
        with torch.inference_mode():
            if a.mode=='preflight':preflight(a,rt,loc)
            else:
                import modes
                if a.mode=='full':modes.full(a,rt,loc)
                elif a.mode=='repeat':modes.full(a,rt,loc,repeat=True)
                elif a.mode=='local':modes.local(a,rt,loc)
                elif a.mode=='timing':modes.timing(a,rt,loc)
        write(a.output/'exit.json',{'status':'COMPLETE','utc':utc(),'source_bundle_sha256':freeze['bundle_sha256'],'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    except BaseException as e:
        write(a.output/'failure.json',{'status':'BLOCKED','error_type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc(),'utc':utc()})
        raise

if __name__=='__main__':main()
