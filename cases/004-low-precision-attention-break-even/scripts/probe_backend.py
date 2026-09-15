"""Explicit GPU Gate 0. Profiler and correctness checks are outside timing."""
import argparse,json,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import CASE,write_json,utc


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=CASE/'provenance/backend_probe.json');a=p.parse_args();assert not a.output.exists()
    import torch,numpy as np
    from src.backends import Adapter,tensor_hash,assert_unmodified,quantized_kernel_call
    from src.reference import attention_fp64
    from src.metrics import unit_error
    from src.runtime import profile_call,environment
    from scripts.benchmark import safe_error
    result=dict(stage='development_gate0',started_at_utc=utc(),checks=[])
    if not torch.cuda.is_available():result['status']='GPU_NOT_AVAILABLE';write_json(a.output,result);return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    result['environment']=environment();adapters={}
    for name in ['default','flash','sage']:
        try:adapters[name]=Adapter(name)
        except Exception:result['checks'].append(dict(backend=name,status='BACKEND_BLOCKED',error=safe_error()))
    with torch.inference_mode():
        for dim in [64,128]:
            g=torch.Generator().manual_seed(400007+dim)
            host=[torch.randn(1,h,32,dim,generator=g).to(torch.bfloat16) for h in [4,2,2]]
            q,k,v=[t.cuda() for t in host];before=[tensor_hash(t) for t in [q,k,v]]
            for causal in [True,False]:
                ref,_=attention_fp64(*[t.float().numpy() for t in host],scale=dim**-.5,causal=causal)
                for name,adapter in adapters.items():
                    rec=dict(backend=name,dim=dim,causal=causal,hq=4,hkv=2,length=32,scale=dim**-.5,source='separate development Gaussian fixture')
                    try:
                        start=time.perf_counter();out=adapter(q,k,v,causal=causal,scale=dim**-.5);torch.cuda.synchronize();rec['first_call_seconds']=time.perf_counter()-start
                        rec['output_dtype']=str(out.dtype);rec['output_shape']=list(out.shape);rec['oracle']=unit_error(out.float().cpu().numpy(),ref)
                        assert out.dtype==torch.bfloat16 and out.shape==q.shape
                        assert rec['oracle']['invalid_rows']==0 and rec['oracle']['relative_output_error']<(.01 if name!='sage' else .05),'Coarse semantic smoke tolerance exceeded'
                        rec['profile']=profile_call(lambda:adapter(q,k,v,causal=causal,scale=dim**-.5))
                        assert rec['profile']['low_precision' if name=='sage' else 'fused_bf16']
                        assert_unmodified(before,[tensor_hash(t) for t in [q,k,v]])
                        if causal:
                            # Future permutation keeps global per-channel ranges fixed.
                            perm=v.clone();perm[:,:,16:]=v[:,:,16:].flip(2)
                            test=adapter(q,k,perm,causal=True,scale=dim**-.5)
                            rec['causal_future_permutation_bitwise_equal']=bool(torch.equal(test[:,:,:16],out[:,:,:16]));assert rec['causal_future_permutation_bitwise_equal']
                            noncausal=adapter(q,k,v,causal=False,scale=dim**-.5)
                            rec['causal_flip_changes_output']=not bool(torch.equal(out,noncausal));assert rec['causal_flip_changes_output']
                            scaled=adapter(q,k,v,causal=True,scale=.3)
                            rec['explicit_scale_changes_output']=not bool(torch.equal(out,scaled));assert rec['explicit_scale_changes_output']
                            distinct=torch.empty_like(v);distinct[:,0]=2.;distinct[:,1]=7.
                            expected=distinct.repeat_interleave(2,dim=1) # oracle only; never an adapter input
                            got=adapter(q,k,distinct,causal=True,scale=dim**-.5)
                            rec['distinct_kv_head_relative_error']=float((got-expected).float().norm()/expected.float().norm());assert rec['distinct_kv_head_relative_error']<.03
                            future=v.clone();future[:,:,16:]*=1000
                            sensitive=adapter(q,k,future,causal=True,scale=dim**-.5)
                            rec['future_range_stress_prefix_change']=unit_error(sensitive[:,:,:16].float().cpu().numpy(),out[:,:,:16].float().cpu().numpy())
                            rec['future_range_stress_interpretation']='Separate stress diagnostic: whole-sequence quantization/smoothing statistics can change earlier approximate outputs. Not a decode or prefix-invariance guarantee.'
                        if name=='sage':
                            inner,details=quantized_kernel_call(q,k,v,causal,dim**-.5);rec['kernel_only_bitwise_equal']=bool(torch.equal(inner(),out));rec['internal_operand_dtypes']=details;assert rec['kernel_only_bitwise_equal']
                        rec['status']='OK'
                    except Exception:rec['status']='FAILED';rec['error']=safe_error()
                    result['checks'].append(rec)
    result['status']='PASS' if len(result['checks'])==12 and all(c['status']=='OK' for c in result['checks']) else 'BACKEND_BLOCKED'
    result['coarse_smoke_relative_tolerance']={'bf16':.01,'sage':.05,'scope':'Geometry/scale/mask smoke only, not the stricter held-out local numerical screen'}
    result['completed_at_utc']=utc();write_json(a.output,result)
    print(json.dumps({'status':result['status'],'checks':[{k:v for k,v in c.items() if k in ['backend','dim','causal','status','error','future_range_stress_prefix_change']} for c in result['checks']]},indent=2))
    if result['status']!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
