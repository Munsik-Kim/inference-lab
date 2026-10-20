"""Official worker extension: inspect tensors/config, never alter kernels/model."""
class CacheInspector:
    def inspect_case016_cache(self):
        import torch
        runner=self.model_runner;model=runner.get_model() if hasattr(runner,'get_model') else runner.model
        layers=[];storages={};tensors=[]
        parameters={};parameter_dtypes={}
        for name,p in model.named_parameters():
            storage=p.untyped_storage();key=(str(p.device),storage.data_ptr(),storage.nbytes());parameters[key]=storage.nbytes()
            parameter_dtypes[str(p.dtype)]=parameter_dtypes.get(str(p.dtype),0)+p.numel()
        def inspect_tensor(t,label):
            if isinstance(t,(list,tuple)):
                for i,v in enumerate(t):inspect_tensor(v,f'{label}/{i}')
            elif isinstance(t,dict):
                for k,v in t.items():inspect_tensor(v,f'{label}/{k}')
            elif isinstance(t,torch.Tensor):
                st=t.untyped_storage();key=(str(t.device),st.data_ptr(),st.nbytes());storages[key]=st.nbytes()
                tensors.append({'name':label,'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),'logical_bytes':t.numel()*t.element_size(),'storage_bytes':st.nbytes()})
        for name,m in model.named_modules():
            if hasattr(m,'kv_cache_dtype') and hasattr(m,'impl'):
                cache=getattr(m,'kv_cache',None);inspect_tensor(cache,name)
                impl=m.impl;layers.append({'name':name,'cache_dtype':m.kv_cache_dtype,
                  'k_scale':float(getattr(m,'_k_scale_float',1.0)),'v_scale':float(getattr(m,'_v_scale_float',1.0)),
                  'q_scale':float(getattr(m,'_q_scale_float',1.0)),
                  'impl':type(impl).__module__+'.'+type(impl).__name__,
                  'nvfp4_fa2':getattr(impl,'nvfp4_fa2',None),'nvfp4_trtllm':getattr(impl,'nvfp4_trtllm',None),
                  'is_kvcache_nvfp4':getattr(impl,'is_kvcache_nvfp4',None),'query_dtype':'bfloat16 (model dtype; verify kernel path)',
                  'use_xqa':getattr(impl,'use_xqa',None)})
        if not tensors:inspect_tensor(getattr(runner,'kv_caches',None),'runner.kv_caches')
        conf=self.vllm_config.cache_config
        try:layout=str(conf.get_resolved_kv_cache_layout())
        except Exception:layout=getattr(conf,'kv_cache_layout',None)
        return {'layers':layers,'cache_tensors':tensors,'unique_storage_count':len(storages),'actual_unique_storage_bytes':sum(storages.values()),
          'cache_config':{k:getattr(conf,k,None) for k in ('cache_dtype','block_size','num_gpu_blocks','kv_cache_size_tokens','kv_cache_memory_bytes','enable_prefix_caching','kv_cache_dtype_skip_layers')},
          'resolved_layout':layout,'torch_cuda_version':torch.version.cuda,'device':torch.cuda.get_device_name(),
          'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),
          'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
          'unique_parameter_storage_bytes':sum(parameters.values()),'parameter_elements_by_dtype':parameter_dtypes,
          'model_load_memory_usage':getattr(runner,'model_memory_usage',None),
          'other_allocator_bytes_including_workspace_buffers':torch.cuda.memory_allocated()-sum(parameters.values())-sum(storages.values()),
          'cudagraph_policy':'disabled by common enforce_eager',
          'free_total_bytes':list(torch.cuda.mem_get_info()),'tf32':torch.backends.cuda.matmul.allow_tf32,
          'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
