"""Complete BF16-input/BF16-output adapters. Imports are lazy for CPU tests."""
from contextlib import nullcontext

SAGE_OPTIONS=dict(tensor_layout='HND',qk_quant_gran='per_warp',pv_accum_dtype='fp32+fp16',smooth_k=True,smooth_v=False,return_lse=False)


def validate_geometry(q,k,v):
    if len(q.shape)!=4 or len(k.shape)!=4 or k.shape!=v.shape:raise ValueError('Expected BHND Q and equal K/V shapes')
    if q.shape[0]!=k.shape[0] or q.shape[2:]!=k.shape[2:]:raise ValueError('Only square self-attention with common head dimension')
    if q.shape[1]%k.shape[1]:raise ValueError('Query heads must be divisible by KV heads')
    if q.shape[-1] not in [64,128]:raise ValueError('Only head dimensions 64/128')
    if not (q.dtype==k.dtype==v.dtype):raise TypeError('Input dtypes differ')
    if not (q.device==k.device==v.device):raise ValueError('Input devices differ')


class Adapter:
    def __init__(self,name):
        if name not in ['default','flash','sage']:raise ValueError(name)
        import torch
        self.torch=torch;self.name=name
        if name=='sage':
            from sageattention import sageattn_qk_int8_pv_fp8_cuda
            self.fn=sageattn_qk_int8_pv_fp8_cuda
        else:self.fn=torch.nn.functional.scaled_dot_product_attention

    def __call__(self,q,k,v,*,causal,scale):
        validate_geometry(q,k,v)
        if q.dtype!=self.torch.bfloat16:raise TypeError('BF16 input required')
        if not q.is_cuda:raise ValueError('GPU adapter requires CUDA')
        q,k,v=q.contiguous(),k.contiguous(),v.contiguous()
        if self.name=='sage':o=self.fn(q,k,v,is_causal=causal,sm_scale=scale,**SAGE_OPTIONS)
        else:
            from torch.nn.attention import sdpa_kernel,SDPBackend
            context=sdpa_kernel(SDPBackend.FLASH_ATTENTION) if self.name=='flash' else nullcontext()
            with context:o=self.fn(q,k,v,dropout_p=0.,is_causal=causal,scale=scale,enable_gqa=q.shape[1]!=k.shape[1])
        o=o.to(dtype=self.torch.bfloat16).contiguous()
        if o.shape!=q.shape:raise ValueError('Wrong output shape')
        return o


def assert_unmodified(before,after):
    if before!=after:raise RuntimeError('INPUT_MUTATION')


def tensor_hash(t):
    import hashlib
    return hashlib.sha256(t.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def quantized_kernel_call(q,k,v,causal,scale):
    """Auxiliary only. Isolate the exact pinned upstream internal invocation.

    This imports the authors' functions; no attention/quantization kernel is
    implemented here. Preprocessing and allocation are outside this callable.
    Public-wrapper equality must be checked per input before timing it.
    """
    import torch
    from sageattention.quant import per_warp_int8,per_channel_fp8
    from sageattention import sm89_compile
    validate_geometry(q,k,v)
    q,k,v=q.contiguous(),k.contiguous(),v.contiguous()
    km=k.mean(dim=2,keepdim=True)
    qi,qs,ki,ks=per_warp_int8(q,k,km,tensor_layout='HND',BLKQ=128,WARPQ=32,BLKK=64)
    vf,vs,_=per_channel_fp8(v,tensor_layout='HND',scale_max=2.25,smooth_v=False)
    out=torch.empty_like(q)
    def call():
        sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(qi,ki,vf,out,qs,ks,vs,1,int(causal),2,scale,0)
        return out
    return call,{'q_dtype':str(qi.dtype),'k_dtype':str(ki.dtype),'v_dtype':str(vf.dtype),'output_dtype':str(out.dtype),'scope':'prequantized, preallocated upstream custom-op call; Python/custom-op overhead included'}
