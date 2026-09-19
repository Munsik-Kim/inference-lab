"""Local-only execution guardrails and full-output diagnostics."""
from contextlib import contextmanager
import os
import subprocess
import numpy as np
from .common import require
from .artifact import tensor_hash


def gpu_guard(minimum_free_gib: float = 8) -> dict:
    import torch
    require(torch.cuda.is_available(), 'BLOCKED_GPU_ENVIRONMENT')
    torch.set_num_threads(4)
    require(torch.cuda.get_device_name(0) == 'NVIDIA GeForce RTX 5080', 'Unexpected device')
    torch.manual_seed(808190)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    free,total = torch.cuda.mem_get_info()
    jobs = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    require(all(int(line.strip())==os.getpid() for line in jobs.splitlines() if line.strip()), 'BLOCKED_RESOURCE: other compute process')
    require(free >= minimum_free_gib*2**30, 'BLOCKED_RESOURCE: insufficient free VRAM')
    torch.cuda.set_per_process_memory_fraction(min(13*2**30/total, .85))
    return {'device':torch.cuda.get_device_name(0),'free_bytes':free,'total_bytes':total,
            'process_allocation_cap_bytes':int(min(13*2**30,total*.85))}


@contextmanager
def replace(layer, module):
    original=layer.mlp;layer.mlp=module
    try:yield
    finally:layer.mlp=original


def forward(model, ids, cache=False, past=None):
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return model(input_ids=ids, use_cache=cache, past_key_values=past, logits_to_keep=1)


def diagnostic(model, row: dict, capture_layer: int = 13):
    import torch
    hooks=[];flags=[];capture={}
    def finite(module,args,output):
        t=output[0] if isinstance(output,tuple) else output
        ok=bool(torch.isfinite(t).all());flags.append(ok)
        require(ok,'BLOCKED_NUMERICAL_VALIDITY: full output')
    def before(module,args):capture['input']=args[0].detach().clone();capture['input_hash']=tensor_hash(args[0])
    def after(module,args,output):
        finite(module,args,output);capture['output']=output.detach().clone()
        require(tensor_hash(args[0])==capture['input_hash'],'Input mutated')
    for block in model.model.layers:hooks.append(block.register_forward_hook(finite))
    hooks.append(model.model.norm.register_forward_hook(finite))
    hooks.append(model.model.layers[capture_layer].mlp.register_forward_pre_hook(before))
    hooks.append(model.model.layers[capture_layer].mlp.register_forward_hook(after))
    try:
        ids=torch.tensor([row['token_ids']],device='cuda')
        output=forward(model,ids)
        z=output.logits[0,-1].detach().float().cpu().numpy().astype(np.float64)
        require(len(flags)==len(model.model.layers)+2 and all(flags) and np.isfinite(z).all(), 'Incomplete validity evidence')
        return z,capture,{'full_block_outputs':len(model.model.layers),'full_norm':True,'full_mlp':True,
                          'full_final_vocab':True,'input_immutable':True,'input_hash':capture['input_hash']}
    finally:
        for hook in hooks:hook.remove()
