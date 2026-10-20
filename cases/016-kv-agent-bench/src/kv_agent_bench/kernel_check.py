"""GPU-only NVFP4 page writer/read check, adapted from pinned vLLM upstream test.

The test changes heads to Qwen's 32 query/8 KV (GQA4) and retains head_dim128.
It compares two decoders of the SAME compressed cache separately from quantization loss.
"""
def main():
    import argparse
    import time
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();start=time.monotonic()
    import torch
    import flashinfer
    from vllm.config import VllmConfig,set_current_vllm_config
    from vllm.v1.attention.backends import flashinfer as fi
    from .common import save
    torch.manual_seed(6101903);torch.backends.cuda.matmul.allow_tf32=False
    rows=[]
    for layout in ('NHD','HND'):
        # L=2; page count6; 2*KVheads16; page16; packed64+scale8 bytes.
        shape=(2,6,16,16,72)
        layers=torch.zeros(shape,device='cuda',dtype=torch.uint8)
        if layout=='NHD':layers=layers.transpose(2,3).contiguous().transpose(2,3)
        kv=torch.randn((2,96,8,128),device='cuda',dtype=torch.bfloat16);one=torch.ones((),device='cuda')
        layer=torch.nn.Module();layer._k_scale=layer._v_scale=one
        with set_current_vllm_config(VllmConfig()):
            impl=fi.FlashInferImpl(32,128,128**-0.5,8,None,None,'nvfp4')
        impl.do_kv_cache_update(layer,kv[0],kv[1],layers[1],torch.arange(96,device='cuda'))
        data,scales,actual_layout=fi._nvfp4_kv_views(layers[1],8)
        pages=torch.arange(6,device='cuda',dtype=torch.int32)[None];seq=torch.tensor([96],device='cuda',dtype=torch.int32)
        deq=torch.empty_like(kv)[:,None]
        flashinfer.nvfp4_kv_dequantize_paged(data,scales,pages,seq,one,one,*deq,actual_layout)
        independent=[];lut=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device='cuda')
        for d,s in zip(data,scales):
            if actual_layout=='HND':d=d.transpose(1,2);s=s.transpose(1,2)
            packed=d.reshape(96,8,64).to(torch.int64);scale=s.reshape(96,8,8).float().repeat_interleave(16,dim=-1)
            unpacked=torch.stack((lut[packed&15],lut[packed>>4]),dim=-1).flatten(-2)
            independent.append((unpacked*scale).to(torch.bfloat16))
        independent=torch.stack(independent)
        torch.testing.assert_close(deq[:,0],independent,atol=0.02,rtol=0.02)
        rel=float((deq[:,0].float()-kv.float()).norm()/kv.float().norm())
        if not torch.isfinite(deq).all() or layers[0].any():raise AssertionError('finite/other-layer integrity failure')
        rows.append({'requested_layout':layout,'actual_layout':actual_layout,'status':'PASS','query_heads':32,'kv_heads':8,'head_dim':128,
          'same_compressed_cache_two_decoder_max_abs':float((deq[:,0].float()-independent.float()).abs().max()),
          'bf16_original_vs_quantized_relative_norm_error':rel,'quantization_error_is_not_decoder_disagreement':True,
          'packed_data_dtype':str(data[0].dtype),'block_scale_dtype':str(scales[0].dtype),'writer':'FlashInferImpl.do_kv_cache_update/nvfp4_quantize_append_paged_kv_cache_with_slot_mapping',
          'reader':'flashinfer.nvfp4_kv_dequantize_paged','other_layer_untouched':True})
    torch.cuda.synchronize();save(args.output,{'status':'PASS','rows':rows,'wall_seconds':time.monotonic()-start,'torch_peak_allocated':torch.cuda.max_memory_allocated(),'torch_peak_reserved':torch.cuda.max_memory_reserved(),'model_forwards':0,'small_kernel_operations':True,'same_cache_decoder_tolerance':{'atol':0.02,'rtol':0.02},'upstream_test':'vllm@5df63a7/tests/kernels/attention/test_flashinfer.py::test_flashinfer_nvfp4_kv_cache_update','upstream_test_suite_executed':False})

if __name__=='__main__':main()
