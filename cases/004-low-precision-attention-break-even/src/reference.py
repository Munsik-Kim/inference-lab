"""Independent NumPy FP64 oracle and bounded-memory FP32 row reference."""
import numpy as np


def attention_fp64(q,k,v,scale,causal,mask=None):
    q,k,v=[np.asarray(x,dtype=np.float64) for x in [q,k,v]]
    if q.ndim!=4 or k.shape!=v.shape or q.shape[0]!=k.shape[0] or q.shape[-1]!=k.shape[-1]:raise ValueError('Bad geometry')
    if q.shape[1]%k.shape[1]:raise ValueError('Bad GQA')
    if causal and q.shape[2]!=k.shape[2]:raise ValueError('Non-square causal is excluded')
    if not all(np.isfinite(x).all() for x in [q,k,v]):raise ValueError('Nonfinite input')
    out=np.zeros_like(q);valid=np.zeros(q.shape[:-1],bool)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            kh=h//(q.shape[1]//k.shape[1])
            for i in range(q.shape[2]):
                scores=np.array([np.dot(q[b,h,i],x)*scale for x in k[b,kh]])
                keep=np.arange(k.shape[2])<=i if causal else np.ones(k.shape[2],bool)
                if mask is not None:keep &= np.broadcast_to(mask,(*q.shape[:3],k.shape[2]))[b,h,i]
                if not keep.any():continue
                w=np.exp(scores[keep]-max(scores[keep]));w/=w.sum()
                out[b,h,i]=sum((weight*value for weight,value in zip(w,v[b,kh,keep])),start=np.zeros(q.shape[-1]))
                valid[b,h,i]=True
    return out,valid


def sampled_fp32(q,k,v,positions,scale,causal):
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    out=[]
    for h in range(q.shape[1]):
        kh=h//(q.shape[1]//k.shape[1])
        scores=q[:,h,positions].float()@k[:,kh].float().transpose(-1,-2)*scale
        if causal:
            keep=torch.arange(k.shape[-2],device=q.device)[None,:]<=torch.tensor(positions,device=q.device)[:,None]
            scores.masked_fill_(~keep,-torch.inf)
        out.append((torch.softmax(scores,dim=-1)@v[:,kh].float()).cpu())
    return torch.stack(out,dim=1)
