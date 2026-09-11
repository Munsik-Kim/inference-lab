"""Capture actual post-QK-norm/post-RoPE inputs at Transformers' SDPA call."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from numerics import dense
from prepare import CASE, sha, write_json


def verify_frozen():
    path = CASE / 'configs/experiment_spec.json'
    spec = json.loads(path.read_text())
    for name, digest in spec['frozen_files'].items():
        assert sha((CASE / name).read_bytes()) == digest, name
    return spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['validate', 'dev', 'eval'], required=True)
    p.add_argument('--model-cache', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if args.stage == 'eval':
        verify_frozen()
    plan = json.loads((CASE / 'configs/development_plan.json').read_text())
    manifest = json.loads((CASE / 'inputs/manifest.json').read_text())
    docs = [d for d in manifest['documents'] if d['split'] == ('dev' if args.stage == 'validate' else args.stage)]
    if args.stage == 'validate':
        docs = docs[:1]
    torch.manual_seed(360011)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    assert torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info()
    if free < 4 * 2**30:
        raise RuntimeError('GPU_BUSY: less than 4 GiB free; no other job was stopped.')
    torch.cuda.set_per_process_memory_fraction(.45)
    snapshot = args.model_cache / ('models--' + plan['model_id'].replace('/', '--')) / 'snapshots' / plan['model_revision']
    model = AutoModel.from_pretrained(snapshot, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True, trust_remote_code=False).eval().cuda()
    assert model.config.num_hidden_layers == 28 and model.config.num_attention_heads == 16
    assert model.config.num_key_value_heads == 8 and model.config.head_dim == 128
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    state, validation, captured = {}, [], []

    def capture(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        output, weights = original(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)
        if module.layer_idx not in plan['layers']:
            return output, weights
        assert attention_mask is None, 'Plain full-prefix SDPA should use native causal mask.'
        assert module.is_causal and module.sliding_window is None and dropout == 0
        assert query.shape[0] == 1 and query.shape[-2] == key.shape[-2]
        heads = torch.tensor(plan['heads'], device=query.device)
        kv_heads = heads // module.num_key_value_groups
        q_all = query[0].index_select(0, heads)
        k = key[0].index_select(0, kv_heads)
        v = value[0].index_select(0, kv_heads)
        n = query.shape[-2]
        if args.stage == 'validate':
            native = output[0].index_select(1, heads).transpose(0, 1)
            reconstructed = F.scaled_dot_product_attention(q_all[None], k[None], v[None], is_causal=True, dropout_p=0., scale=scaling)[0]
            s = q_all.float() @ k.float().transpose(-1, -2) * scaling
            s.masked_fill_(torch.ones(n, n, device=s.device, dtype=torch.bool).triu(1), -torch.inf)
            fp32 = dense(s, v.float())['out']
            rel_native = ((reconstructed.float() - native.float()).norm() / native.float().norm()).item()
            rel_fp32 = ((fp32 - native.float()).norm() / native.float().norm()).item()
            validation.append({'layer': module.layer_idx, 'heads': plan['heads'], 'kv_heads': kv_heads.tolist(),
                'length': n, 'same_backend_relative_error': rel_native, 'fp32_vs_native_bf16_relative_error': rel_fp32,
                'same_backend_pass': rel_native <= plan['trace_validation']['same_backend_relative_tolerance'],
                'fp32_pass': rel_fp32 <= plan['trace_validation']['fp32_vs_native_bf16_relative_tolerance']})
            if module.layer_idx == plan['layers'][0]:
                positions = np.array([0, 7, 63, 127], dtype=np.int64)
                np.savez_compressed(CASE / 'tests/fixtures/qwen_dev_small.npz',
                    q=q_all[0, positions].float().cpu().numpy(), k=k[0].float().cpu().numpy(),
                    v=v[0].float().cpu().numpy(), positions=positions, scaling=np.array(scaling))
        else:
            positions = torch.tensor(plan['query_positions'][str(n)], device=query.device)
            q = q_all.index_select(1, positions)
            meta = {'document_id': state['doc']['document_id'], 'split': args.stage, 'language': state['doc']['language'],
                    'length': n, 'layer': module.layer_idx, 'heads': plan['heads'], 'kv_heads': kv_heads.tolist(),
                    'positions': positions.tolist(), 'scaling': scaling,
                    'model_revision': plan['model_revision'], 'input_sha256': state['doc']['prefix_sha256'][str(n)],
                    'capture_point': 'Actual SDPA query/key after QK RMSNorm and RoPE; value after v_proj; native causal mask.'}
            record = {'q': q.cpu().contiguous(), 'k': k.cpu().contiguous(), 'v': v.cpu().contiguous(), 'meta': meta}
            target = args.work_dir / 'traces' / args.stage / f"{meta['document_id']}-n{n}-l{module.layer_idx}.pt"
            assert not target.exists(), target.name
            torch.save(record, target)
            captured.append({**meta, 'trace_file': target.name, 'trace_sha256': sha(target.read_bytes()), 'bytes': target.stat().st_size})
        return output, weights

    ALL_ATTENTION_FUNCTIONS.register('sdpa', capture)
    started = time.time()
    try:
        with torch.inference_mode():
            for doc in docs:
                state['doc'] = doc
                for n in ([128] if args.stage == 'validate' else plan['lengths']):
                    expected = [args.work_dir / 'traces' / args.stage / f"{doc['document_id']}-n{n}-l{layer}.pt" for layer in plan['layers']]
                    if args.stage != 'validate' and args.resume and all(x.exists() for x in expected):
                        for x in expected:
                            record = torch.load(x, map_location='cpu', weights_only=True)
                            assert record['meta']['input_sha256'] == doc['prefix_sha256'][str(n)]
                            assert record['meta']['model_revision'] == plan['model_revision']
                            captured.append({**record['meta'], 'trace_file': x.name, 'trace_sha256': sha(x.read_bytes()), 'bytes': x.stat().st_size})
                        continue
                    inputs = torch.tensor([doc['token_ids_4096'][:n]], device='cuda')
                    model(input_ids=inputs, use_cache=False, return_dict=True)
                    torch.cuda.synchronize()
                    print(json.dumps({'stage': args.stage, 'document': doc['document_id'], 'length': n, 'elapsed_s': round(time.time()-started, 2)}), flush=True)
    finally:
        ALL_ATTENTION_FUNCTIONS.register('sdpa', original)
    runtime = {'gpu': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
               'free_before_bytes': free, 'total_bytes': total,
               'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'elapsed_s': time.time()-started}
    if args.stage == 'validate':
        write_json(CASE / 'provenance/trace_validation.json', {'runtime': runtime, 'checks': validation})
        assert len(validation) == 3 and all(x['same_backend_pass'] and x['fp32_pass'] for x in validation)
    else:
        write_json(CASE / f'provenance/{args.stage}_traces.json', {'runtime': runtime, 'traces': captured})


if __name__ == '__main__':
    main()
