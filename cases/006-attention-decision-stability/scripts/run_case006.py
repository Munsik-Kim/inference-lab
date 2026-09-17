"""Guarded environment gate and semantic-smoke entry point, not an evaluation run.

No downloads, installation or fallback. A busy GPU stops before importing Torch.
Later study stages require verified smoke and DEV-derived freezes; they are not
implemented by this CPU-only delivery. Exit 20 means a resource/access block.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.storage import load, write_new


def classify_resource(samples: list[dict]) -> str:
    if not samples:
        return 'GPU_ACCESS_BLOCKED'
    if min(s['free_mib'] for s in samples)<13312:
        return 'INSUFFICIENT_FREE_MEMORY'
    # Registry presence is not activity. Require sustained observed device load.
    if sum(s['utilization_percent']>10 for s in samples)>=3:
        return 'GPU_BUSY'
    return 'RESOURCE_GATE_PASSED'


def gpu_gate() -> dict:
    samples=[]
    for i in range(5):
        sample=_gpu_sample()
        if sample.get('status')=='GPU_ACCESS_BLOCKED':
            return sample
        samples.append(sample)
        if i<4:time.sleep(1)
    result=dict(samples[-1])
    result['status']=classify_resource(samples)
    result['samples']=samples
    result['policy']='Five pre-model samples; >=3 with GPU utilization >10% blocks. Registration alone is not proof of activity. Minimum free memory remains 13312 MiB.'
    return result


def _gpu_sample() -> dict:
    try:
        active = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                                capture_output=True,text=True,timeout=15)
        gpu = subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu',
                              '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {'status':'GPU_ACCESS_BLOCKED','active_compute_process_count':None}
    if active.returncode or gpu.returncode:
        return {'status':'GPU_ACCESS_BLOCKED','active_compute_process_count':None,
                'reason':'nvidia-smi failed; do not infer that physical GPU is absent'}
    pids = [x.strip() for x in active.stdout.splitlines() if x.strip().isdigit()]
    fields = gpu.stdout.strip().split(',')
    if len(fields) != 5:
        return {'status':'GPU_ACCESS_BLOCKED','reason':'Unexpected GPU query shape'}
    result = {'gpu':fields[0].strip(),'driver':fields[1].strip(),
              'total_mib':int(fields[2]),'free_mib':int(fields[3]),'utilization_percent':int(fields[4]),
              'active_compute_process_count':len(pids)}
    return result


def semantic_smoke(snapshot: Path, case: Path) -> dict:
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from src.intervention import ScopedAttention
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.cuda.set_per_process_memory_fraction(13*2**30/torch.cuda.get_device_properties(0).total_memory)
    inputs = [x for x in load(case/'inputs/prompts.json') if x['split']=='smoke']
    model = AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
                                                dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    cfg = model.config
    if (cfg.num_hidden_layers,cfg.num_attention_heads,cfg.num_key_value_heads,cfg.head_dim)!=(28,16,8,128):
        raise ValueError('Model geometry mismatch')
    def thash(t):
        return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    rows=[]
    with torch.inference_mode():
        for item in inputs:
            ids = torch.tensor([item['token_ids']],device='cuda')
            native = model(input_ids=ids,use_cache=False,logits_to_keep=1).logits.detach().clone()
            baseline_qkv=None
            for arm in ('B','A_PUBLIC','V4'):
                finite=[]; blocks=[]; before={}; qkv_hash=None; immutable=[]; target_output=None
                def observer(when,layer,q,k,v,out,selected):
                    nonlocal qkv_hash,target_output
                    if when=='before':
                        before[layer]=[thash(x) for x in (q,k,v)]
                        if layer==13 and q.shape[-2]>1:
                            qkv_hash=before[layer]
                    else:
                        immutable.append(before[layer]==[thash(x) for x in (q,k,v)])
                        finite.append(bool(torch.isfinite(out).all()))
                        if not finite[-1]:
                            raise ValueError(f'BLOCKED_NUMERICAL_VALIDITY: {arm} full attention output layer {layer}')
                        if layer==13 and q.shape[-2]>1:
                            target_output=out.detach().cpu()
                hooks=[]
                def block_check(_module,_args,out):
                    blocks.append(bool(torch.isfinite(out).all()))
                    if not blocks[-1]:
                        raise ValueError(f'BLOCKED_NUMERICAL_VALIDITY: {arm} full block output')
                for layer in model.model.layers:
                    hooks.append(layer.register_forward_hook(block_check))
                try:
                    with ScopedAttention(ALL_ATTENTION_FUNCTIONS,arm,observer) as scoped:
                        result=model(input_ids=ids,use_cache=True,logits_to_keep=1)
                        logits=result.logits.detach().clone()
                        if not bool(torch.isfinite(logits).all()):
                            raise ValueError(f'BLOCKED_NUMERICAL_VALIDITY: {arm} full returned logits')
                        if arm=='B' and not torch.equal(native,logits):
                            raise ValueError('BLOCKED_SEMANTICS: native/B mismatch requires repeat-floor investigation')
                        prefix_calls=list(scoped.calls)
                        # Own prompt cache; only one generated token, no gold continuation.
                        token=logits[:,-1].argmax(-1,keepdim=True)
                        decode=model(input_ids=token,past_key_values=result.past_key_values,use_cache=True,logits_to_keep=1)
                        decode_calls=scoped.calls[len(prefix_calls):]
                finally:
                    for hook in hooks:
                        hook.remove()
                if arm=='B':
                    baseline_qkv=qkv_hash
                elif qkv_hash!=baseline_qkv:
                    raise ValueError('BLOCKED_SEMANTICS: layer-13 QKV differ across arms')
                route_ok=(len(prefix_calls)==28 and len(decode_calls)==28
                          and all(c['route']=='native_BF16' for c in decode_calls)
                          and all(c['route']=='native_BF16' for c in prefix_calls if c['layer']!=13))
                rows.append({'item_id':item['item_id'],'arm':arm,'evidence_kind':'gpu_semantic_smoke',
                             'native_B_bitwise':bool(torch.equal(native,logits)) if arm=='B' else None,
                             'same_layer13_qkv':qkv_hash==baseline_qkv,'inputs_unchanged':all(immutable),
                             'all_attention_outputs_finite':all(finite),'all_blocks_finite':all(blocks),
                             'full_logits_finite':bool(torch.isfinite(logits).all() and torch.isfinite(decode.logits).all()),
                             'routing_valid':route_ok,'prefill_calls':prefix_calls,'decode_calls':decode_calls,
                             'target_output_dtype':str(target_output.dtype),'final_logits_dtype':str(logits.dtype),
                             'profiler_kernel_verification':'NOT_RUN','status':'PARTIAL_SEMANTIC_CHECK_NOT_EVALUATION_READY'})
                del result,decode,logits,target_output
    return {'rows':rows,'status':'PARTIAL_SEMANTIC_CHECK_NOT_EVALUATION_READY',
            'remaining':['Full final-norm and boundary capture checks','Standalone/integrated operator correspondence',
                         'Profiler fused/native and low-precision kernel verification','B repeat floor and full/no-last logits check',
                         'DEV scoring and two real freezes before any evaluation']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['smoke'],required=True)
    p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='New external JSON path')
    args=p.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.case.resolve()):
        p.error('Use a new output outside the case; measurements remain immutable')
    gate=gpu_gate()
    record={'stage':'smoke','resource_gate':gate,'model_forward_started':False,'gpu_measurements':0}
    if gate['status']!='RESOURCE_GATE_PASSED':
        write_new(args.output,record);print(json.dumps(record));return 20
    try:
        record['model_forward_started']=True
        record['semantic_smoke']=semantic_smoke(args.snapshot,args.case)
    except Exception as exc:
        # Failure is preserved; no retry, fallback or output repair.
        record['failure']={'type':type(exc).__name__,'message':str(exc)}
        write_new(args.output,record)
        raise
    write_new(args.output,record)
    print(json.dumps({'status':record['semantic_smoke']['status'],'output_written':True}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
