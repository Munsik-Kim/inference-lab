"""Read-only official worker extension. No upstream kernel/model modification."""
from kv_agent_bench.profiler_extension import ProfileInspector

class TechnicalInspector(ProfileInspector):
 def technical_inspect(self):
  import torch,hashlib
  r=self.inspect_case016_cache();runner=self.model_runner;model=runner.get_model()if hasattr(runner,'get_model')else runner.model;r['cuda_capability']=list(torch.cuda.get_device_capability());r['runner_class']=type(runner).__module__+'.'+type(runner).__name__;r['compilation_mode']=str(self.vllm_config.compilation_config.cudagraph_mode)
  h=hashlib.sha256()
  for name,p in model.named_parameters():
   flat=p.detach().reshape(-1);sample=torch.cat((flat[:8],flat[-8:]));h.update(name.encode());h.update(sample.view(torch.int16).cpu().numpy().tobytes())
  r['parameter_sample_sha256']=h.hexdigest();r['parameter_signature_scope']='first/last8 values per named parameter, not a full GPU tensor hash'
  views=[]
  from vllm.v1.attention.backends.flashinfer import _nvfp4_kv_views
  for name,m in model.named_modules():
   if getattr(m,'kv_cache_dtype',None)=='nvfp4':
    data,scales,layout=_nvfp4_kv_views(m.kv_cache,8);views.append({'layer':name,'layout':layout,'data_dtype':[str(v.dtype)for v in data],'scale_dtype':[str(v.dtype)for v in scales],'data_shape':[list(v.shape)for v in data],'scale_shape':[list(v.shape)for v in scales],'data_scale_storage_shared':all(v.untyped_storage().data_ptr()==m.kv_cache.untyped_storage().data_ptr()for v in [*data,*scales])})
  r['nvfp4_views']=views;return r
 def technical_finite_start(self):
  import torch
  runner=self.model_runner;model=runner.get_model()if hasattr(runner,'get_model')else runner.model;self._finite_rows={};self._finite_hooks=[]
  for name,m in model.named_modules():
   if hasattr(m,'kv_cache_dtype')and hasattr(m,'impl'):
    self._finite_rows[name]={'calls':0,'finite':True,'shapes':[]}
    def hook(module,args,out,label=name):
     tensors=[out]if isinstance(out,torch.Tensor)else[v for v in out if isinstance(v,torch.Tensor)];row=self._finite_rows[label]
     row['calls']+=1
     for v in tensors:
      row['finite'] &= bool(torch.isfinite(v).all().item())
      shape=list(v.shape)
      if shape not in row['shapes']:row['shapes'].append(shape)
    self._finite_hooks.append(m.register_forward_hook(hook))
  if len(self._finite_hooks)!=36:raise AssertionError('36 actual attention modules required')
  return {'layers':36,'scope':'P0 eager actual model attention module outputs; observer removed before timing'}
 def technical_finite_stop(self):
  for h in self._finite_hooks:h.remove()
  rows=self._finite_rows;self._finite_hooks=[]
  return {'rows':rows,'all36_finite':len(rows)==36 and all(r['finite']and r['calls']>0 for r in rows.values()),'scope':'Actual compressed-KV attention outputs, not a second whole decoder equivalence test. No LM-head full-vocabulary logit claim.'}
 def technical_reset_peaks(self):
  import torch
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();return {'reset':True}
