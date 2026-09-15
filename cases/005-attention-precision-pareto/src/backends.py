"""Only the four requested precision bundles; no implicit top-level dispatch."""
from .anchor import Adapter as AnchorAdapter, validate_geometry, tensor_hash

COMMON = dict(tensor_layout='HND',qk_quant_gran='per_warp',smooth_k=True,return_lse=False)
OPTIONS = {
 'A_PUBLIC':dict(COMMON,pv_accum_dtype='fp32+fp16',smooth_v=False),
 'V1':dict(COMMON,pv_accum_dtype='fp32',smooth_v=False),
 'V2':dict(COMMON,pv_accum_dtype='fp32+fp32',smooth_v=False),
 'V3':dict(COMMON,pv_accum_dtype='fp32',smooth_v=True),
 'V4':dict(COMMON,pv_accum_dtype='fp32',smooth_v=False)}
IDS=['B','A_PUBLIC','V1','V2','V3','V4']
ALIASES={'A_MATCHED':'A_PUBLIC'}

def effective(config):
 if config in ALIASES:config=ALIASES[config]
 if config=='B':return {'function':'torch.nn.functional.scaled_dot_product_attention','backend':'default fused SDPA, inherited Case004 choice','input_dtype':'BF16','output_dtype':'BF16'}
 return dict(function='sageattn_qk_int8_pv_fp16_cuda' if config=='V4' else 'sageattn_qk_int8_pv_fp8_cuda',requested_kwargs=OPTIONS[config],effective_kwargs=OPTIONS[config],qk_dtype='INT8',p_dtype='FP16' if config=='V4' else 'E4M3',v_dtype='FP16' if config=='V4' else 'E4M3',v_scale_max=None if config=='V4' else 2.25 if config=='A_PUBLIC' else 448.0,output_dtype='BF16',bundle='Accumulation option also changes V quantization scale from anchor; V3 adds V centering; V4 changes PV format and casts V inside call',gqa='Native H16/H8; no full KV expansion; upstream may repeat the sequence-mean vector')

class Variant:
 def __init__(self,config):
  import torch,sageattention
  if config not in ['V1','V2','V3','V4']:raise ValueError(config)
  self.torch=torch;self.options=OPTIONS[config];self.fn=getattr(sageattention,effective(config)['function'])
 def __call__(self,q,k,v,*,causal,scale):
  validate_geometry(q,k,v)
  if q.dtype!=self.torch.bfloat16:raise TypeError('BF16 input required')
  if not q.is_cuda:raise ValueError('GPU adapter requires CUDA')
  q,k,v=q.contiguous(),k.contiguous(),v.contiguous()
  o=self.fn(q,k,v,is_causal=causal,sm_scale=scale,**self.options)
  o=o.to(dtype=self.torch.bfloat16).contiguous()
  if o.shape!=q.shape:raise ValueError('Wrong output shape')
  return o

def adapter(config):
 config=ALIASES.get(config,config)
 if config=='B':return AnchorAdapter('default')
 if config=='A_PUBLIC':return AnchorAdapter('sage')
 return Variant(config)
