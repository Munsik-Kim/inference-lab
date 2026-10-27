"""GPU-only runtime. Imported by worker after CLI parsing and freeze checks.

Reuses Case013 equations and vLLM/Torch operators. No new kernel or quantizer.
"""
import contextlib
import gc
import inspect
import types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen3 import modeling_qwen3 as hf
from common import CASE,read,file_row,write,byte_sha
from layout import load_rpi4,pack_marlin_input,marlin_scale_permute

def flags():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    return {'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'allow_fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,'marlin_use_fp32_reduce':True,'marlin_use_atomic_add':False,'threads':torch.get_num_threads(),'attention_backend':'sdpa','eager':True,'inference_mode':True}

def sync():torch.cuda.synchronize()

def memory():
    sync()
    free,total=torch.cuda.mem_get_info()
    return {'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'device_sample_used':total-free,'device_total':total}

def cpu(t):return t.detach().float().cpu().numpy()

def save_tensor(path,t):
    t=t.detach().cpu().contiguous()
    if t.dtype==torch.bfloat16:
        a=t.view(torch.uint16).numpy();encoding='BF16_BITS_UINT16'
    else:a=t.numpy();encoding=str(t.dtype)
    np.save(path,a,allow_pickle=False)
    return {'path':str(path),'dtype':str(t.dtype),'encoding':encoding,'shape':list(t.shape),'stride':list(t.stride()),**file_row(path)}

def tensor_inventory(module):
    seen=set();rows=[]
    for kind,entries in (('parameter',module.named_parameters(remove_duplicate=False)),('buffer',module.named_buffers(remove_duplicate=False))):
        for name,t in entries:
            storage=t.untyped_storage();key=(str(t.device),storage.data_ptr());unique=key not in seen;seen.add(key)
            rows.append({'name':name,'kind':kind,'dtype':str(t.dtype),'device':str(t.device),'shape':list(t.shape),'numel':t.numel(),'payload_bytes':t.numel()*t.element_size(),'storage_bytes':storage.nbytes(),'storage_id':len(seen) if unique else 'alias','unique_storage':unique})
    return rows

class Dense(nn.Module):
    def __init__(self,layer,q,mode):
        super().__init__();self.in_features=q.shape[1];self.out_features=q.shape[0];dev=next(layer.parameters(),layer.beta_inverse if hasattr(layer,'beta_inverse') else torch.empty(0,device='cuda')).device
        codes=torch.from_numpy(q.codes.copy()).to(dev);gamma=torch.from_numpy(q.gamma).to(dev);alpha=torch.from_numpy(q.alpha_inv).to(dev)
        if mode=='D0':weight=(codes.to(gamma.dtype).reshape(self.out_features,-1,64)*gamma[...,None])*alpha[...,None]
        elif mode=='D1':
            scale=(alpha.float()*gamma.float()).bfloat16()
            weight=codes.float().reshape(self.out_features,-1,64)*scale.float()[...,None]
        else:raise ValueError(mode)
        self.weight=nn.Parameter(weight.reshape(q.shape).bfloat16(),requires_grad=False);self.bias=layer.bias
        self.register_buffer('beta_inverse',torch.from_numpy(q.beta_inv).to(dev,dtype=torch.float32))
    def forward(self,x):
        z=(x.float()*self.beta_inverse).to(self.weight.dtype)
        return F.linear(z,self.weight,self.bias)

class Packed(nn.Module):
    def __init__(self,layer,q):
        super().__init__()
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_permute_scales,marlin_make_workspace_new
        from vllm.scalar_type import scalar_types
        self.ops=ops;self.qtype=scalar_types.uint4b8;self.in_features=q.shape[1];self.out_features=q.shape[0];self.bias=layer.bias
        required={'a','c','b_q_weight','b_bias','b_scales','a_scales','global_scale','b_zeros','g_idx','perm','workspace','b_q_type','size_m','size_n','size_k','use_fp32_reduce'}
        if not required.issubset(inspect.signature(ops.marlin_gemm).parameters):raise RuntimeError('MARLIN_API_MISMATCH')
        empty=torch.empty(0,dtype=torch.int32,device='cuda');self.register_buffer('empty',empty)
        packed=torch.from_numpy(pack_marlin_input(q.codes)).cuda()
        self.register_buffer('codes',ops.gptq_marlin_repack(b_q_weight=packed,perm=empty,size_k=self.in_features,size_n=self.out_features,num_bits=4))
        alpha=torch.from_numpy(q.alpha_inv).cuda().float();gamma=torch.from_numpy(q.gamma).cuda().float()
        plain=(alpha*gamma).T.contiguous().bfloat16()
        scales=marlin_permute_scales(plain,self.in_features,self.out_features,64)
        expected=marlin_scale_permute(plain.view(torch.uint16).cpu().numpy())
        if not np.array_equal(scales.view(torch.uint16).cpu().numpy(),expected):raise RuntimeError('SCALE_PERMUTATION_MISMATCH')
        self.register_buffer('scales',scales);self.register_buffer('beta_inverse',torch.from_numpy(q.beta_inv).cuda().float())
        self.register_buffer('workspace',marlin_make_workspace_new(torch.device('cuda:0')))
    def forward(self,x):
        shape=x.shape[:-1];z=(x.float()*self.beta_inverse).bfloat16().reshape(-1,self.in_features).contiguous()
        y=self.ops.marlin_gemm(a=z,c=None,b_q_weight=self.codes,b_bias=None,b_scales=self.scales,a_scales=None,global_scale=None,b_zeros=self.empty,g_idx=self.empty,perm=self.empty,workspace=self.workspace,b_q_type=self.qtype,size_m=z.shape[0],size_n=self.out_features,size_k=self.in_features,is_k_full=True,use_atomic_add=False,use_fp32_reduce=True,is_zp_float=False)
        if self.bias is not None:y=y+self.bias
        return y.reshape(*shape,self.out_features)

def replace(model,artifact_dir,mode,layer_indices=None):
    inventory=read(CASE/'artifact_manifest.json')['artifacts'];count=0
    for row in inventory:
        name=row['projection'];idx=int(name.split('.')[2])
        if layer_indices is not None and idx not in layer_indices:continue
        rel=name if hasattr(model,'model') else '.'.join(name.split('.')[3:])
        parts=rel.split('.');parent=model
        for key in parts[:-1]:parent=getattr(parent,key)
        layer=getattr(parent,parts[-1]);path=Path(artifact_dir)/(name+'.rpi4')
        if file_row(path)!={k:row[k] for k in ('bytes','sha256')}:raise ValueError('ARTIFACT_CHANGED')
        q=load_rpi4(path);new=Packed(layer,q) if mode=='P' else Dense(layer,q,mode)
        setattr(parent,parts[-1],new);count+=1
    gc.collect();sync()
    return count

def load_model(locations,mode):
    model=AutoModelForCausalLM.from_pretrained(locations['model'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,trust_remote_code=False).eval().cuda()
    if replace(model,locations['artifact_dir'],mode)!=196:raise ValueError('PROJECTION_COUNT')
    return model

def norm_fp32(module,x):
    h=x.float();h=h*torch.rsqrt(h.pow(2).mean(-1,keepdim=True)+module.variance_epsilon)
    return module.weight.float()*h

@contextlib.contextmanager
def rounding(model,mode,trace=None):
    original_rope=hf.apply_rotary_pos_emb;original_norm=[]
    norms=[(name,m) for name,m in model.named_modules() if name.endswith(('q_norm','k_norm'))]
    def wrap_norm(module,original,name):
        def forward(x):
            y=norm_fp32(module,x) if mode=='R1' else original(x)
            if trace is not None:trace(name+'.output',y)
            return y
        return forward
    def rope(q,k,cos,sin,unsqueeze_dim=1):
        if trace is not None:
            for name,t in [('pre_rope_q',q),('pre_rope_k',k),('cos',cos),('sin',sin)]:trace(name,t)
        if mode=='R1':
            c=cos.float().unsqueeze(unsqueeze_dim);s=sin.float().unsqueeze(unsqueeze_dim)
            qo=(q.float()*c+hf.rotate_half(q.float())*s).bfloat16()
            ko=(k.float()*c+hf.rotate_half(k.float())*s).bfloat16()
        else:qo,ko=original_rope(q,k,cos,sin,unsqueeze_dim)
        if trace is not None:trace('post_rope_q',qo);trace('post_rope_k',ko)
        return qo,ko
    try:
        if mode!='NATIVE' or trace is not None:
            for name,m in norms:
                original_norm.append((m,m.forward));m.forward=wrap_norm(m,m.forward,name)
            hf.apply_rotary_pos_emb=rope
        yield
    finally:
        hf.apply_rotary_pos_emb=original_rope
        for m,fn in original_norm:m.forward=fn

def inputs(document,length):
    ids=np.load(CASE/'inputs'/(document+'.npy'),allow_pickle=False)[:length]
    t=torch.from_numpy(ids.copy()).cuda()[None,:]
    return {'input_ids':t,'attention_mask':torch.ones_like(t),'position_ids':torch.arange(length,device='cuda')[None,:],'use_cache':False,'logits_to_keep':0}

def boundary_hooks(block,trace):
    handles=[]
    for name,module in block.named_modules():
        if name.endswith(('q_proj','k_proj','o_proj')) or name=='mlp':
            handles.append(module.register_forward_hook(lambda m,a,y,n=name:trace(n,y)))
        if name.endswith('o_proj'):handles.append(module.register_forward_pre_hook(lambda m,a:trace('attention_output',a[0])))
    handles.append(block.register_forward_hook(lambda m,a,y:trace('block_output',y)))
    return handles

def clear(model=None):
    del model;gc.collect();sync();torch.cuda.empty_cache()
