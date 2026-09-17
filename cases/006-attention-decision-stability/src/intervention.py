"""Scoped installed-Qwen SDPA interception. GPU route requires semantic smoke.

This module imports no GPU library until a real operator call. The context manager
restores the registry even if inference raises. It is single-process, not thread-safe.
"""
from __future__ import annotations
from contextlib import AbstractContextManager
import math
from typing import Any, Callable

COMMON = dict(tensor_layout='HND', qk_quant_gran='per_warp', smooth_k=True,
              smooth_v=False, return_lse=False)
SETTINGS = {
    'A_PUBLIC': dict(function='sageattn_qk_int8_pv_fp8_cuda', pv_accum_dtype='fp32+fp16', v_scale_max=2.25),
    'V4': dict(function='sageattn_qk_int8_pv_fp16_cuda', pv_accum_dtype='fp32', v_scale_max=None),
}


def route(arm: str, layer: int, q_len: int, kv_len: int, *, mask_present: bool,
          dropout: float, geometry: tuple[int, int, int, int]) -> str:
    if arm not in ('B', 'A_PUBLIC', 'V4'):
        raise ValueError('Unknown arm')
    if layer != 13 or q_len == 1 or arm == 'B':
        return 'native_BF16'
    if q_len != kv_len or mask_present or dropout != 0 or geometry != (1, 16, 8, 128):
        raise ValueError('Unsupported intervention geometry/mask; no fallback')
    return SETTINGS[arm]['function']


class ScopedAttention(AbstractContextManager):
    def __init__(self, registry: Any, arm: str, observer: Callable | None = None, record_calls: bool = True):
        self.registry, self.arm, self.observer = registry, arm, observer
        self.calls: list[dict] = []
        self.original = None
        self.record_calls = record_calls

    def __enter__(self):
        self.original = self.registry['sdpa']
        self.registry.register('sdpa', self.call)
        return self

    def __exit__(self, *exc):
        self.registry.register('sdpa', self.original)
        return False

    def call(self, module, q, k, v, attention_mask, dropout=0.0, scaling=None, **kwargs):
        geometry = (q.shape[0], q.shape[1], k.shape[1], q.shape[-1])
        layer = module.layer_idx
        selected = route(self.arm, layer, q.shape[-2], k.shape[-2],
                         mask_present=attention_mask is not None, dropout=dropout, geometry=geometry)
        if selected != 'native_BF16' and kwargs.get('sliding_window') is not None:
            raise ValueError('Sliding-window mask unsupported')
        if selected != 'native_BF16' and (not module.is_causal or scaling is None or not math.isfinite(scaling) or scaling <= 0):
            raise ValueError('Explicit causal attention and finite positive scaling required')
        if self.observer:
            self.observer('before', layer, q, k, v, None, selected)
        if selected == 'native_BF16':
            result = self.original(module, q, k, v, attention_mask,
                                   dropout=dropout, scaling=scaling, **kwargs)
        else:
            import torch
            import sageattention
            if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype or not q.is_cuda:
                raise ValueError('CUDA BF16 Q/K/V required')
            if v.shape != k.shape or q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
                raise ValueError('GQA tensor mismatch')
            out = getattr(sageattention, selected)(q.contiguous(), k.contiguous(), v.contiguous(),
                    is_causal=True, sm_scale=scaling, pv_accum_dtype=SETTINGS[self.arm]['pv_accum_dtype'], **COMMON)
            if out.dtype != torch.bfloat16 or out.shape != q.shape:
                raise ValueError('Wrong public adapter output; no silent cast/repair')
            result = (out.transpose(1, 2).contiguous(), None)
        if self.record_calls:
            self.calls.append({'layer': layer, 'q_len': q.shape[-2], 'kv_len': k.shape[-2], 'route': selected})
        if self.observer:
            self.observer('after', layer, q, k, v, result[0], selected)
        return result
