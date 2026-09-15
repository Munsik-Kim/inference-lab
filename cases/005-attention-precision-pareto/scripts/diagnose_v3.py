"""Bounded follow-up to the failed constant-V semantic fixture; no real DEV data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import CASE,write_json
from src.backends import adapter
from src.runtime import profile_call

def main():
 import torch
 from sageattention.quant import per_channel_fp8
 torch.set_num_threads(4);g=torch.Generator().manual_seed(400135)
 q,k,v=[torch.randn(1,h,32,128,generator=g).bfloat16().cuda() for h in [4,2,2]]
 constant=torch.empty_like(v);constant[:,0]=2;constant[:,1]=7
 rows=[]
 with torch.inference_mode():
  for label,x in [('constant',constant),('constant_plus_small_variation',constant+v*.01)]:
   vf,scale,mean=per_channel_fp8(x,tensor_layout='HND',scale_max=448.,smooth_v=True)
   r=dict(fixture=label,v_scale_min=float(scale.min()),v_scale_max=float(scale.max()),v_quantized_nonfinite=int((~torch.isfinite(vf.float())).sum()),v_mean_per_head=mean.mean(-1).cpu().tolist(),outputs={})
   for cid in ['B','V1','V3']:
    o=adapter(cid)(q,k,x,causal=True,scale=128**-.5)
    r['outputs'][cid]={'nonfinite':int((~torch.isfinite(o)).sum()),'first_query_channel0':[[float(z) if torch.isfinite(z) else None for z in o[0,:,0,0]]],'finite_mean_by_head':[float(o[0,h].float().mean()) if torch.isfinite(o[0,h]).all() else None for h in range(4)]}
   rows.append(r)
  profile=profile_call(lambda:adapter('V3')(q,k,v,causal=True,scale=128**-.5))
 out=CASE/'provenance/v3_diagnostic.json';assert not out.exists();write_json(out,{'scope':'Two predetermined semantic fixtures after constant-V failure; no quality/timing selection','records':rows,'profile':profile})
 print(rows)
if __name__=='__main__':main()
