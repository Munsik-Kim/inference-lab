"""Lazy Torch adapters. Only instantiated objects import Torch."""
from contextlib import contextmanager
import copy
from .core import kept

def sliced(original,mapping,removed):
 import torch
 ids=torch.tensor(kept(mapping,removed),device=original.gate_proj.weight.device)
 result=copy.deepcopy(original)
 def linear(src,rows=None,cols=None):
  w=src.weight.detach();w=w.index_select(0,rows) if rows is not None else w;w=w.index_select(1,cols) if cols is not None else w
  dst=torch.nn.Linear(w.shape[1],w.shape[0],bias=src.bias is not None,device=w.device,dtype=w.dtype)
  with torch.no_grad():
   dst.weight.copy_(w)
   if src.bias is not None:dst.bias.copy_(src.bias if rows is None else src.bias.index_select(0,rows))
  return dst
 result.gate_proj=linear(original.gate_proj,rows=ids);result.up_proj=linear(original.up_proj,rows=ids);result.down_proj=linear(original.down_proj,cols=ids);result.intermediate_size=len(ids)
 # Deep-copied config avoids changing shared model configuration.
 result.config.intermediate_size=len(ids)
 return result.eval()
def masked(original,x,mapping,removed):
 import torch
 h=original.act_fn(original.gate_proj(x))*original.up_proj(x)
 keep=torch.zeros(h.shape[-1],device=h.device,dtype=h.dtype);keep[kept(mapping,removed)]=1
 return original.down_proj(h*keep)
@contextmanager
def replace_mlp(layer,replacement):
 old=layer.mlp;layer.mlp=replacement
 try:yield
 finally:layer.mlp=old

def contributions(original,x,mapping):
 """FP32 arithmetic on represented BF16 inputs/weights, not native BF16 output."""
 import torch
 import torch.nn.functional as F
 def bias(m):return None if m.bias is None else m.bias.detach().float()
 xx=x.float();h=original.act_fn(F.linear(xx,original.gate_proj.weight.detach().float(),bias(original.gate_proj)))*F.linear(xx,original.up_proj.weight.detach().float(),bias(original.up_proj))
 cs=torch.stack([F.linear(h[:,g],original.down_proj.weight.detach().float()[:,g]) for g in mapping],dim=1)
 full=F.linear(h,original.down_proj.weight.detach().float(),bias(original.down_proj))
 return cs,full,h
