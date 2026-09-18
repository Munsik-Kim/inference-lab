"""Pinned BF16 model. Diagnostic scans are excluded from timing."""
import hashlib,platform,time,inspect,os
import numpy as np
from .core import positions,norm_stats,score,groups
from .surgery import replace_mlp,contributions,sliced,masked
REVISION='c1899de289a04d12100db370d81485cdf75e47ca'

def thash(t):
 import torch
 return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def array(t):return t.detach().float().cpu().numpy().astype(np.float64)

def resource_gate():
 import torch,subprocess,csv,io
 samples=[]
 for _ in range(3):
  free,total=torch.cuda.mem_get_info();samples.append({'free_bytes':free,'total_bytes':total});time.sleep(.3)
 other=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
 external=[line for line in other.splitlines() if int(line.split(',')[0].strip())!=os.getpid()]
 if external:raise RuntimeError('BLOCKED_RESOURCE: existing external GPU compute process')
 if min(x['free_bytes'] for x in samples)<8*2**30:raise RuntimeError('BLOCKED_RESOURCE: less than 8 GiB free')
 return samples
class Runtime:
 def __init__(self,snapshot):
  import torch
  from transformers import AutoModelForCausalLM
  if snapshot.name!=REVISION:raise ValueError('Wrong snapshot revision')
  self.torch=torch;torch.set_num_threads(4);torch.manual_seed(707180)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
  torch.cuda.set_per_process_memory_fraction(13*2**30/torch.cuda.get_device_properties(0).total_memory)
  self.model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
  cfg=self.model.config
  if (cfg.num_hidden_layers,cfg.hidden_size,cfg.intermediate_size,cfg.hidden_act)!=(28,1024,3072,'silu'):raise ValueError('Unexpected architecture')
  self.layer=self.model.model.layers[13];self.original=self.layer.mlp;self.mapping=groups(3072);self.modules={'B':self.original}
 def add(self,selections):
  for name,sel in selections.items():self.modules[name]=sliced(self.original,self.mapping,sel['removed'])
 def forward(self,ids):
  from torch.nn.attention import sdpa_kernel,SDPBackend
  with self.torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
   return self.model(input_ids=ids,use_cache=False,logits_to_keep=1)
 def diagnose(self,row,method='B',expected_hash=None):
  torch=self.torch;seen=[];capture={};hooks=[]
  def finite(_m,args,out):
   t=out[0] if isinstance(out,tuple) else out
   ok=bool(torch.isfinite(t).all());seen.append(ok)
   if not ok:raise ValueError('BLOCKED_IMPLEMENTATION: full block/nonfinite output')
  def before(m,args):capture['x']=args[0].detach().clone();capture['before']=thash(args[0])
  def after(m,args,out):
   capture['after']=thash(args[0]);capture['o']=out.detach().clone()
   if not bool(torch.isfinite(out).all()):raise ValueError('Nonfinite full MLP output')
  for block in self.model.model.layers:hooks.append(block.register_forward_hook(finite))
  hooks.append(self.model.model.norm.register_forward_hook(finite))
  module=self.modules[method];hooks.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
  try:
   ids=torch.tensor([row['token_ids']],device='cuda')
   with replace_mlp(self.layer,module):out=self.forward(ids)
   z=array(out.logits[0,-1]);ps=positions(row['length']);x=capture['x'][0,ps];o=capture['o'][0,ps]
   valid=len(seen)==29 and all(seen) and np.isfinite(z).all() and capture['before']==capture['after'] and (expected_hash is None or expected_hash==capture['before'])
   if not valid:raise ValueError('Missing/mismatched validity evidence')
   record={'id':row['id'],'split':row['split'],'task':row['task'],'method':method,'token_hash':row['token_hash'],'evidence_kind':'gpu_measurement','valid':True,'full_blocks_checked':28,'full_final_norm_checked':True,'full_logits_finite':True,'mlp_input_hash':capture['before'],'score':score(z,row['label_ids'],row['gold'])}
   return record,z,x,o,capture['x'][0]
  finally:
   for h in hooks:h.remove()
 def q(self,x):
  with self.torch.inference_mode():cs,full,h=contributions(self.original,x,self.mapping)
  c=array(cs);q=np.einsum('tih,tjh->ij',c,c)/len(c);a=np.square(c).sum(axis=(0,2))/len(c)
  return q,{'pre_symmetry_max_abs':float(np.max(abs(q-q.T))),'diagonal_max_abs':float(np.max(abs(q.diagonal()-a)))},cs,full,h
 def equivalence(self,x,selections):
  torch=self.torch;stats=[]
  with torch.inference_mode():
   for name,sel in selections.items():
    a=self.modules[name](x);m=masked(self.original,x,self.mapping,sel['removed']);s=norm_stats(array(a),array(m));s['method']=name
    if a.dtype!=torch.bfloat16 or a.shape!=m.shape or s['relative_error'] is None or s['relative_error']>.01:raise ValueError('BLOCKED_IMPLEMENTATION: sliced/masked equivalence')
    stats.append(s)
  return stats
