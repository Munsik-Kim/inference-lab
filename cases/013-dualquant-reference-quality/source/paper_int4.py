"""Private-library-free element INT4 reference; dense BF16 runtime, not INT4 GEMM.

PAPER_INT4_REFERENCE_VARIANT: FP32 scales, explicit zero guards, standard final
group RTN. No vendor imports. All rows participate in each column update.
"""
from __future__ import annotations
import torch,struct,sys
from torch import nn
from torch.nn import functional as F
from reference_int4 import QuantizedWeight, EPS, pack_nibbles

PROJECTIONS = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}

def serialize_fast(q):
    """Same checked RPI4 bytes; NumPy bulk copy replaces Python byte iteration."""
    q.validate()
    if sys.byteorder!='little':raise RuntimeError('little endian required')
    header=struct.pack('<4sBBBBIIII',b'RPI4',1,1 if q.dtype==torch.float32 else 2,['rtn','row_only','row_column'].index(q.method),0,*q.shape,64,q.num_iter)
    parts=[pack_nibbles(q.codes).numpy().tobytes(),q.gamma.contiguous().numpy().tobytes()]
    if q.method!='rtn':parts.append(q.alpha_inv.contiguous().numpy().tobytes())
    if q.method=='row_column':parts.append(q.beta_inv.contiguous().numpy().tobytes())
    return header+b''.join(parts)

def fast_ledger(q,data):
    code=q.codes.numel()//2;gamma=q.gamma.numel()*q.gamma.element_size()
    alpha=0 if q.method=='rtn' else q.alpha_inv.numel()*q.alpha_inv.element_size()
    beta=q.beta_inv.numel()*q.beta_inv.element_size() if q.method=='row_column' else 0
    if code+gamma+alpha+beta+24!=len(data):raise ValueError('ledger does not match actual bytes')
    return {'code_bytes':code,'gamma_bytes':gamma,'alpha_bytes':alpha,'beta_bytes':beta,'header_bytes':24,'serialized_bytes':len(data),'scale_dtype':str(q.dtype),'packed_gpu_execution':False}

def checked(x, name):
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f'{name}: nonfinite; no repair')
    return x

@torch.no_grad()
def solve(weight, method='row_column', iterations=15, group_chunk=8):
    """GPU or CPU FP32; chunk independent input groups, never output rows."""
    if weight.ndim != 2 or weight.shape[1] % 64 or method not in {'rtn','row_only','row_column'}:
        raise ValueError('invalid matrix/method/group64')
    if iterations < 0 or group_chunk < 1: raise ValueError('invalid iteration/chunk')
    rows, cols = weight.shape
    all_codes, all_a, all_b, all_g, records = [], [], [], [], []
    for start in range(0, cols, group_chunk*64):
        W = checked(weight[:,start:start+group_chunk*64].float().contiguous().reshape(rows,-1,64),'W')
        zr=(W==0).all(-1); zc=(W==0).all(0)
        a=torch.ones_like(W[:,:,0]); b=torch.ones_like(W[0])
        if method=='row_column': b=torch.where(zc,torch.ones_like(b),W.abs().sum(0).clamp_min(1e-8))
        if method!='rtn': a=torch.where(zr,torch.ones_like(a),(W/b.unsqueeze(0)).abs().amax(-1)/7)
        for it in range(iterations if method!='rtn' else 0):
            M=(a.reciprocal().unsqueeze(-1)*W*b.reciprocal().unsqueeze(0)).round().clamp(-8,7)
            den=(M*b.unsqueeze(0)).square().sum(-1)+EPS
            aa=(W*M*b.unsqueeze(0)).sum(-1)/den
            a=torch.where(zr,torch.ones_like(aa),aa)
            checked(a,'alpha_inv')
            if bool((a<=0).any()): raise ValueError(f'zero/nonpositive alpha at group {start//64} iteration {it+1}')
            fresh=(a.reciprocal().unsqueeze(-1)*W*b.reciprocal().unsqueeze(0)).round().clamp(-8,7)
            changed_a=int((fresh!=M).sum())
            if method=='row_column':
                denb=(fresh*a.unsqueeze(-1)).square().sum(0)+EPS
                bb=(W*fresh*a.unsqueeze(-1)).sum(0)/denb
                b=torch.where(zc,torch.ones_like(bb),bb)
                checked(b,'beta_inv')
                if bool((b<=0).any()): raise ValueError(f'zero/nonpositive beta at group {start//64} iteration {it+1}')
            final=(a.reciprocal().unsqueeze(-1)*W*b.reciprocal().unsqueeze(0)).round().clamp(-8,7)
            err=(W-final*a.unsqueeze(-1)*b.unsqueeze(0)).square()
            records.append({'group_start':start//64,'group_count':W.shape[1],'iteration':it+1,
                'inner_mse':float(err.mean()),'inner_sse':float(err.double().sum()),'elements':W.numel(),
                'alpha_code_changes':changed_a,'beta_code_changes':int((final!=fresh).sum()),
                'alpha_min':float(a.min()),'alpha_max':float(a.max()),'beta_min':float(b.min()),'beta_max':float(b.max()),
                'zero_rows':int(zr.sum()),'zero_columns':int(zc.sum()),'nonfinite_count':0,'denominator_epsilon':EPS})
        scaled=checked(a.reciprocal().unsqueeze(-1)*W*b.reciprocal().unsqueeze(0),'final scaled')
        zero=(scaled==0).all(-1)
        # Declared zero exception. Nonzero tiny scales are not clipped or searched.
        g=torch.where(zero,torch.ones_like(a),scaled.abs().amax(-1)/7)
        checked(g,'gamma')
        if bool((g<=0).any()): raise ValueError('zero final nonzero scale')
        codes=checked(scaled/g.unsqueeze(-1),'final normalized').round().clamp(-8,7).to(torch.int8)
        all_codes.append(codes.cpu());all_a.append(a.cpu());all_b.append(b.cpu());all_g.append(g.cpu())
    result=QuantizedWeight(torch.cat(all_codes,1).reshape(rows,cols),torch.cat(all_g,1),torch.cat(all_a,1),torch.cat(all_b,0).reshape(-1),method,iterations if method!='rtn' else 0)
    result.validate()
    return result, records

def activation_int4(x):
    """W4A4_REFERENCE_VARIANT: dynamic token/group64 FP32 absmax, zeros exact."""
    if x.shape[-1]%64: raise ValueError('activation channels not divisible by64')
    z=x.float().reshape(*x.shape[:-1],-1,64)
    scale=z.abs().amax(-1,keepdim=True)/7
    scale=torch.where(scale==0,torch.ones_like(scale),scale)
    return ((z/scale).round().clamp(-8,7)*scale).reshape(x.shape).to(x.dtype)

class ReferenceLinear(nn.Module):
    def __init__(self, layer, quantized, activation4=False):
        super().__init__()
        self.in_features=layer.in_features;self.out_features=layer.out_features
        self.weight=nn.Parameter(quantized.weight_pre().to(layer.weight.device,dtype=layer.weight.dtype),requires_grad=False)
        self.bias=layer.bias
        self.register_buffer('beta_inverse',quantized.beta_inv.to(layer.weight.device,dtype=torch.float32))
        self.activation4=activation4
    def forward(self,x):
        z=x.float()*self.beta_inverse
        if self.activation4: z=activation_int4(z)
        return F.linear(z.to(self.weight.dtype),self.weight,self.bias)

def replace_projection(model,name,quantized,activation4=False):
    parent_path, leaf=name.rsplit('.',1)
    parent=model.get_submodule(parent_path);layer=getattr(parent,leaf)
    if leaf not in PROJECTIONS or not isinstance(layer,nn.Linear): raise ValueError('unapproved projection')
    setattr(parent,leaf,ReferenceLinear(layer,quantized,activation4))

