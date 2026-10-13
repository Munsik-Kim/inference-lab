"""Bounded existing public W4A16 Marlin probe; not author's W4A4 implementation."""
import gc,hashlib,json,time,traceback,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from reference_int4 import QuantizedWeight
from paper_int4 import replace_projection
from paper_model_runner import atomic

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'results'/(ROOT/'results/ACTIVE_RUN.txt').read_text().strip()

class PackedLinear(nn.Module):
    def __init__(self,q,bias=None):
        super().__init__()
        from vllm import _custom_ops as ops
        from vllm.scalar_type import scalar_types
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales,marlin_make_workspace_new
        self.ops=ops;self.qtype=scalar_types.uint4b8;self.out_features,self.in_features=q.shape
        # Pure layout conversion, signed code c -> unsigned c+8; no requantization.
        unsigned=(q.codes.to(torch.int32)+8).T.contiguous().cuda()
        packed=torch.zeros((self.in_features//8,self.out_features),dtype=torch.int32,device='cuda')
        for i in range(8):packed|=unsigned[i::8]<< (4*i)
        empty=torch.empty(0,dtype=torch.int32,device='cuda')
        repacked=ops.gptq_marlin_repack(packed,empty,self.in_features,self.out_features,4)
        scale=(q.alpha_inv*q.gamma).T.contiguous().to(device='cuda',dtype=torch.bfloat16)
        scale=marlin_permute_scales(scale,self.in_features,self.out_features,64)
        self.register_buffer('codes',repacked);self.register_buffer('scales',scale)
        self.register_buffer('beta',q.beta_inv.cuda());self.register_buffer('empty',empty)
        self.register_buffer('workspace',marlin_make_workspace_new(torch.device('cuda',0)))
        self.register_buffer('bias',bias.detach().to(device='cuda',dtype=torch.bfloat16) if bias is not None else None)
    def forward(self,x):
        z=(x.float()*self.beta).to(torch.bfloat16).reshape(-1,self.in_features).contiguous()
        y=self.ops.marlin_gemm(z,None,self.codes,None,self.scales,None,None,self.empty,self.empty,self.empty,self.workspace,self.qtype,z.shape[0],self.out_features,self.in_features,True,False,True,False)
        if self.bias is not None:y=y+self.bias
        return y.reshape(*x.shape[:-1],self.out_features)

@torch.inference_mode()
def main(output=None,matched=False):
    start=time.monotonic();result={'label':'PUBLIC_MARLIN_W4A16_SCALE_BF16_PROBE_VARIANT','official_DualQuant_packed':'NOT_VERIFIED','native_INT4_tensor_core_MMA':False,'runtime':'packed INT4 weights; fused unpack into BF16 tensor-core matmul','beta_preserved':True,'scale_change':'alpha_inv*gamma combined in FP32 then BF16 kernel scales; original FP32 artifact preserved','thresholds_before_probe':{'gemm_relative_rmse':0.02,'full_forward_relative_rmse':0.02},'kernel_exploration_limit_seconds':3600,'formal_serving_benefit':'NOT_ESTABLISHED'}
    try:
        torch.set_num_threads(2);torch.manual_seed(610134);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        result['BF16_reduced_precision_reduction']=False
        result['device']=torch.cuda.get_device_name();result['capability']=list(torch.cuda.get_device_capability())
        directory=RUN/'artifacts/06b/D16';q=QuantizedWeight.deserialize((directory/'model.layers.0.self_attn.q_proj.rpi4').read_bytes())
        x=torch.randn(16,q.shape[1],device='cuda',dtype=torch.bfloat16)
        packed=PackedLinear(q)
        expected=F.linear((x.float()*q.beta_inv.cuda()).bfloat16(),q.weight_pre().cuda().bfloat16())
        y=packed(x);torch.cuda.synchronize()
        rmse=float(((y.float()-expected.float()).square().sum()/expected.float().square().sum()).sqrt())
        if not torch.isfinite(y).all() or rmse>0.02:raise ValueError(f'GEMM gate failed {rmse}')
        result['gemm']={'status':'PASS','relative_rmse':rmse,'max_abs_diff':float((y-expected).abs().max()),'gpu_packed_codes_bytes':packed.codes.numel()*packed.codes.element_size(),'gpu_scale_bytes':packed.scales.numel()*packed.scales.element_size(),'beta_bytes':packed.beta.numel()*packed.beta.element_size(),'operator':'torch.ops._C.marlin_gemm'}
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:packed(x);torch.cuda.synchronize()
            result['profiler_kernel_names']=sorted({e.name for e in prof.events() if str(e.device_type).endswith('CUDA')})
        except Exception as e:result['profiler_error']=str(e)
        del packed,x,y,expected,q;gc.collect();torch.cuda.empty_cache()
        identity=json.loads((RUN/'experiment_manifest.json').read_text())['models']['06b']
        model=AutoModelForCausalLM.from_pretrained(identity['path'],local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').eval().requires_grad_(False)
        files=json.loads((directory/'manifest.json').read_text())['files']
        for f in files:
            qq=QuantizedWeight.deserialize((directory/(f['layer']+'.rpi4')).read_bytes());replace_projection(model,f['layer'],qq)
        model.cuda();ids=torch.tensor(np.load(RUN/'inputs/wiki_06b.npy')[0,:32],device='cuda',dtype=torch.long).unsqueeze(0)
        position=torch.arange(32,device='cuda').unsqueeze(0);emb=model.model.embed_tokens(ids);pe=model.model.rotary_emb(emb,position)
        reference_block=model.model.layers[0](emb,position_ids=position,position_embeddings=pe)
        reference_logits=model(ids,use_cache=False).logits.cpu();primary_logits=reference_logits.clone()
        if matched:
            # Diagnose the exact BF16-scale format used by the existing kernel.
            # This does NOT change the primary FP32-scale reference or its gate.
            scale_errors=[]
            for f in files:
                qq=QuantizedWeight.deserialize((directory/(f['layer']+'.rpi4')).read_bytes())
                scale=(qq.gamma*qq.alpha_inv).bfloat16().float()
                mw=(qq.codes.reshape(qq.shape[0],-1,64).float()*scale.unsqueeze(-1)).reshape(qq.shape).bfloat16()
                primary=qq.weight_pre().bfloat16().float();diff=mw.float()-primary
                scale_errors.append({'layer':f['layer'],'relative_weight_rmse':float((diff.square().sum()/primary.square().sum()).sqrt())})
                model.get_submodule(f['layer']).weight.copy_(mw.cuda())
            reference_block=model.model.layers[0](emb,position_ids=position,position_embeddings=pe)
            reference_logits=model(ids,use_cache=False).logits.cpu()
            result['diagnostic_reference']='same signed codes, BF16-rounded gamma*alpha group scales, same beta; no fitting/requantization'
            result['weight_scale_rounding_diagnostics']=scale_errors
            result['matched_dense_vs_primary_logit_relative_rmse']=float(((reference_logits.float()-primary_logits.float()).square().sum()/primary_logits.float().square().sum()).sqrt())
        torch.cuda.reset_peak_memory_stats()
        reference_times=[]
        for repeat in range(3):
            torch.cuda.synchronize();tick=time.monotonic();reference_ids=model.generate(ids,do_sample=False,max_new_tokens=4,logits_to_keep=1,use_cache=True,pad_token_id=model.config.eos_token_id).cpu();torch.cuda.synchronize();reference_times.append(time.monotonic()-tick)
        result['reference_request_probe']={'input_tokens':32,'max_new_tokens':4,'actual_new_tokens':reference_ids.shape[1]-32,'wall_seconds_repeats':reference_times,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'load_and_convert_excluded':True}
        # Release every BF16 projection weight from GPU before packed whole-model probe.
        model.cpu();torch.cuda.empty_cache()
        for f in files:
            path,leaf=f['layer'].rsplit('.',1);parent=model.get_submodule(path);original=getattr(parent,leaf)
            qq=QuantizedWeight.deserialize((directory/(f['layer']+'.rpi4')).read_bytes());setattr(parent,leaf,PackedLinear(qq,original.bias))
        model.cuda();torch.cuda.reset_peak_memory_stats()
        packed_block=model.model.layers[0](emb,position_ids=position,position_embeddings=pe)
        brmse=float(((packed_block.float()-reference_block.float()).square().sum()/reference_block.float().square().sum()).sqrt())
        if not torch.isfinite(packed_block).all() or brmse>0.02:raise ValueError(f'block gate {brmse}')
        result['one_block']={'status':'PASS','relative_rmse':brmse}
        actual=model(ids,use_cache=False).logits.cpu();packed_times=[]
        for repeat in range(3):
            torch.cuda.synchronize();tick=time.monotonic();actual_ids=model.generate(ids,do_sample=False,max_new_tokens=4,logits_to_keep=1,use_cache=True,pad_token_id=model.config.eos_token_id).cpu();torch.cuda.synchronize();packed_times.append(time.monotonic()-tick)
        result['packed_request_probe']={'input_tokens':32,'max_new_tokens':4,'actual_new_tokens':actual_ids.shape[1]-32,'wall_seconds_repeats':packed_times,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'load_and_convert_excluded':True,'inference_dtype':'BF16 activations, packed INT4 weights, Marlin BF16 MMA'}
        frmse=float(((actual.float()-reference_logits.float()).square().sum()/reference_logits.float().square().sum()).sqrt())
        primary_rmse=float(((actual.float()-primary_logits.float()).square().sum()/primary_logits.float().square().sum()).sqrt())
        result['short_full_decode']={'status':'PASS' if frmse<=0.02 else 'FAIL_TOLERANCE','forward_relative_rmse':frmse,'vs_primary_FP32_scale_reference_relative_rmse':primary_rmse,'argmax_flip_fraction':float((actual.argmax(-1)!=reference_logits.argmax(-1)).float().mean()),'greedy_suffix_equal':bool(torch.equal(actual_ids,reference_ids)),'reference_suffix':reference_ids[0,-4:].tolist(),'packed_suffix':actual_ids[0,-4:].tolist(),'projection_count':len(files),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        if not torch.isfinite(actual).all() or frmse>0.02:raise ValueError(f'full forward gate {frmse}')
        result['status']='FORMAT_MATCHED_DIAGNOSTIC_PASS_PRIMARY_EQUIVALENCE_BLOCKED' if matched and primary_rmse>0.02 else 'PACKED_PROBE_VERIFIED_PUBLIC_MARLIN_W4A16_ONLY'
    except Exception as e:
        traceback.print_exc();result.update(status='BLOCKED',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc())
    result['wall_seconds']=time.monotonic()-start;atomic(Path(output) if output else RUN/'packed_probe.json',result);print(json.dumps(result))
    return 1 if result['status']=='BLOCKED' else 0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--matched-scale-diagnostic',action='store_true');a=p.parse_args();raise SystemExit(main(a.output,a.matched_scale_diagnostic))
