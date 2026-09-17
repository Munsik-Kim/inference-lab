"""Small independent CPU oracle and chunked GPU reference (never injected)."""
from __future__ import annotations
import math
import numpy as np


def positions(length: int) -> list[int]:
    if length < 32:
        return list(range(length))
    return [i*(length-1)//31 for i in range(32)]


def tiny_reference(q, k, v, scale: float, causal: bool = True):
    q, k, v = (np.asarray(x, dtype=np.float64) for x in (q, k, v))
    if q.ndim != 4 or k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError('Expected B,H,N,D tensors')
    b, hq, n, d = q.shape
    hkv = k.shape[1]
    if hq % hkv or q.shape[-2] != k.shape[-2]:
        raise ValueError('Square GQA attention only')
    if not all(np.isfinite(x).all() for x in (q, k, v)):
        raise ValueError('Nonfinite input')
    out = np.empty_like(q)
    for batch in range(b):
        for head in range(hq):
            kh = head // (hq//hkv)
            for row in range(n):
                end = row+1 if causal else n
                scores = np.array([math.fsum(float(q[batch,head,row,j])*float(k[batch,kh,col,j])
                                            for j in range(d))*scale for col in range(end)])
                exp = np.exp(scores-scores.max()); p = exp/exp.sum()
                out[batch,head,row] = p @ v[batch,kh,:end]
    return out


def sampled_fp32(q, k, v, scale: float):
    import torch
    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError('TF32 must be disabled before reference')
    selected = positions(q.shape[-2])
    outputs = []
    for head in range(q.shape[1]):
        kv = head // (q.shape[1]//k.shape[1])
        rows = []
        for row in selected:
            # One query by all valid keys; never a full N x N matrix.
            score = (q[:,head,row:row+1].float() @ k[:,kv,:row+1].float().transpose(-1,-2))*scale
            rows.append(score.softmax(-1) @ v[:,kv,:row+1].float())
        outputs.append(torch.cat(rows,dim=1))
    return torch.stack(outputs,dim=1), selected
