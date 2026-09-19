"""Fresh-process reload smoke, prompt-only prefill and incremental greedy decode."""
import os
from pathlib import Path
from .common import read, require
from .artifact import load_dense, tensor_hash


def reload_test(artifact: Path, input_path: Path, device: str) -> dict:
    import torch
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    torch.set_num_threads(4)
    if device=='cuda':
        from .runtime import gpu_guard
        gpu_guard()
    model = load_dense(artifact, device)
    inputs = read(input_path)
    rows = []
    flags = []
    def check(module, args, output):
        tensors = output if isinstance(output, tuple) else (output,)
        for t in tensors:
            if isinstance(t, torch.Tensor):
                ok = bool(torch.isfinite(t).all()); flags.append(ok)
                require(ok, 'Nonfinite full reloaded model output')
    hooks = [block.register_forward_hook(check) for block in model.model.layers]
    hooks.append(model.model.norm.register_forward_hook(check))
    with torch.inference_mode():
        for row in inputs:
            ids = torch.tensor([row['token_ids']], device=device)
            output = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            require(bool(torch.isfinite(output.logits).all()), 'Nonfinite full vocabulary')
            hashes = [tensor_hash(output.logits)]
            tokens = []
            for _ in range(4):
                token = output.logits[:, -1].argmax(-1, keepdim=True)
                tokens.append(int(token.item()))
                output = model(input_ids=token, past_key_values=output.past_key_values,
                               use_cache=True, logits_to_keep=1)
                require(bool(torch.isfinite(output.logits).all()), 'Nonfinite decode vocabulary')
                hashes.append(tensor_hash(output.logits))
            rows.append({'id': row['id'], 'tokens': tokens, 'full_logits_hashes': hashes})
    for hook in hooks: hook.remove()
    require(len(flags) >= len(inputs)*5*(len(model.model.layers)+1) and all(flags), 'Absent full reload validity')
    return {'status': 'PASS', 'device': device, 'offline_environment': True,
            'full_outputs_checked': len(flags),
            'no_original_checkpoint_loaded': True, 'rows': rows,
            'scope': 'deterministic 4-token smoke, not task-quality or latency evaluation'}
