"""GPU smoke per configuration; instrumentation is restored before any timing."""
import argparse,json,sys,warnings,traceback,contextlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import CASE,write_json,sha,utc
from src.backends import adapter,effective,IDS,tensor_hash
from src.base_metrics import unit_error
from src.reference import attention_fp64
from src.runtime import profile_call

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',choices=IDS,required=True);a=p.parse_args();cid=a.config;out=CASE/'provenance'/f'probe-{cid}.json';assert not out.exists()
 import torch,numpy as np,sageattention.core as core
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 result=dict(config_id=cid,started_at_utc=utc(),settings=effective(cid),checks=[],warnings=[],status='SOURCE_PRESENT_EXECUTION_UNVERIFIED')
 if torch.cuda.mem_get_info()[0]<3*2**30:result['status']='GPU_BUSY';write_json(out,result);raise SystemExit(3)
 try:
  fn=adapter(cid)
  with torch.inference_mode(),warnings.catch_warnings(record=True) as warned:
   warnings.simplefilter('always')
   g=torch.Generator().manual_seed(400135)
   host=[torch.randn(1,h,32,128,generator=g).bfloat16() for h in [4,2,2]];q,k,v=[x.cuda() for x in host];before=[tensor_hash(x) for x in [q,k,v]]
   outputs={}
   for causal in [True,False]:
    ref,_=attention_fp64(*[x.float().numpy() for x in host],128**-.5,causal)
    o=fn(q,k,v,causal=causal,scale=128**-.5);torch.cuda.synchronize();metric=unit_error(o.float().cpu().numpy(),ref)
    assert o.shape==q.shape and o.dtype==torch.bfloat16 and o.is_contiguous()
    assert torch.isfinite(o).all() and metric['relative_output_error']<(.01 if cid=='B' else .05),'Coarse fixture semantic error exceeded'
    outputs[causal]=o;result['checks'].append({'causal':causal,'oracle':metric,'output_dtype':str(o.dtype),'layout':'contiguous BHND'})
   assert not torch.equal(outputs[True],outputs[False]),'Causal flag ignored'
   distinct=torch.empty_like(v);distinct[:,0]=2;distinct[:,1]=7;expected=distinct.repeat_interleave(2,dim=1)
   got=fn(q,k,distinct,causal=True,scale=128**-.5)
   err=float((got.float()-expected.float()).norm()/expected.float().norm());assert err<.03,'GQA mismatch'
   scaled=fn(q,k,v,causal=True,scale=.3);assert not torch.equal(scaled,outputs[True]),'Explicit scale ignored'
   result.update(gqa_distinct_head_error=err,causal_flip=True,explicit_scale=True,profile=profile_call(lambda:fn(q,k,v,causal=True,scale=128**-.5)))
   assert result['profile']['fused_bf16' if cid=='B' else 'low_precision'],'Actual kernel not observed'
   if cid!='B':
    # Record internal operand dtypes and quantizer arguments from one extra call.
    observed=[];replacements=[]
    def wrap(obj,name):
     original=getattr(obj,name)
     def spy(*args,**kwargs):
      observed.append({'function':name,'kwargs':kwargs,'tensor_operands':[{'index':i,'dtype':str(x.dtype),'shape':list(x.shape)} for i,x in enumerate(args) if isinstance(x,torch.Tensor)]})
      return original(*args,**kwargs)
     replacements.append((obj,name,original));setattr(obj,name,spy)
    for name in ['per_warp_int8_cuda','per_channel_fp8']:wrap(core,name)
    obj=core.sm80_compile if cid=='V4' else core.sm89_compile
    names=[n for n in dir(obj) if n.startswith('qk_int8') and callable(getattr(obj,n))]
    for name in names:wrap(obj,name)
    try:fn(q,k,v,causal=True,scale=128**-.5);torch.cuda.synchronize()
    finally:
     for obj,name,original in reversed(replacements):setattr(obj,name,original)
    result['observed_calls']=observed
   assert before==[tensor_hash(x) for x in [q,k,v]];result['inputs_unchanged']=True
   if cid in ['B','A_PUBLIC']:
    perm=v.clone();perm[:,:,16:]=v[:,:,16:].flip(2)
    stress=v.clone();stress[:,:,16:]*=1000
    result['prefix_stress']={}
    for label,x in [('range_preserving_permutation',perm),('future_range_x1000',stress)]:
     y=fn(q,k,x,causal=True,scale=128**-.5);result['prefix_stress'][label]=unit_error(y[:,:,:16].float().cpu().numpy(),outputs[True][:,:,:16].float().cpu().numpy())
   if cid=='A_PUBLIC':
    result['A_MATCHED']={'status':'ALIAS_OF_OTHER_CONFIG','alias_of':'A_PUBLIC','reason':'Byte-identical Case004 adapter already uses explicit FP8 function; no extra wrapper or timing variant'}
    original=core.sageattn_qk_int8_pv_fp8_cuda;forwarded={}
    def routing_spy(*args,**kwargs):forwarded.update(kwargs);return 'routing-only sentinel'
    core.sageattn_qk_int8_pv_fp8_cuda=routing_spy
    try:core.sageattn(q,k,v,tensor_layout='HND',is_causal=True,pv_accum_dtype='fp32',smooth_v=True)
    finally:core.sageattn_qk_int8_pv_fp8_cuda=original
    result['top_level_routing_only_probe']={'requested_extra':{'pv_accum_dtype':'fp32','smooth_v':True},'forwarded':forwarded,'scope':'Python routing spy only; not a distinct precision execution'}
   result['warnings']=[str(w.message) for w in warned];result['status']='EXECUTABLE_VERIFIED'
 except Exception as e:
  import re
  result['error']=re.sub(r'/home/[^\s\"\']+','<local-path>',traceback.format_exc())
  result['error_type']=type(e).__name__
  result['status']='UNSUPPORTED_ON_THIS_SETUP' if isinstance(e,(RuntimeError,ImportError)) else 'IMPLEMENTATION_ERROR'
 finally:
  result['completed_at_utc']=utc();write_json(out,result)
 print(json.dumps({'config_id':cid,'status':result['status'],'error':result.get('error'),'warnings':result['warnings']}))
 if result['status']!='EXECUTABLE_VERIFIED':raise SystemExit(2)
if __name__=='__main__':main()
