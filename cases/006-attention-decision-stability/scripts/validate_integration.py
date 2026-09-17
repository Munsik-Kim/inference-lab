"""Complete pre-evaluation smoke checks; outputs are diagnostics, not task results."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import load,write_new
from src.intervention import COMMON,SETTINGS,ScopedAttention
from src.model_runtime import Runtime,ARMS,tensor_hash
from scripts.run_case006 import gpu_gate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Never overwrite prior validation')
    gate=gpu_gate();report={'resource_gate':gate,'status':'IN_PROGRESS'}
    if gate['status']!='RESOURCE_GATE_PASSED':write_new(a.output,report);return 20
    import torch,sageattention,sageattention.core as core
    runtime=Runtime(a.snapshot);item=next(x for x in load(a.case/'inputs/prompts.json') if x['split']=='smoke')
    gold=next(x for x in load(a.case/'inputs/gold.json') if x['base_id']==item['base_id'])
    gold_index='ABCD'.index(gold['gold']);native=[]
    with torch.inference_mode():
        ids=runtime.ids(item['token_ids'])
        native=[runtime.model(input_ids=ids,use_cache=False,logits_to_keep=1).logits.detach().cpu() for _ in range(2)]
        b=runtime.forward(item['token_ids'],'B',False).logits.detach().cpu()
        small=ids[:,:64]
        captured=[]
        handle=runtime.model.model.norm.register_forward_hook(lambda _m,_a,out:captured.append(out[:,-1:].detach().clone()))
        full=runtime.model(input_ids=small,use_cache=False,logits_to_keep=0).logits[:,-1].detach().cpu()
        last=runtime.model(input_ids=small,use_cache=False,logits_to_keep=1).logits[:,-1].detach().cpu()
        handle.remove()
        manual=runtime.model.lm_head(captured[-1])[:,-1].detach().cpu()
    report['native_B_validated']=bool(torch.equal(native[0],native[1]) and torch.equal(native[0],b))
    report['native_repeat_bitwise']=bool(torch.equal(native[0],native[1]))
    report['last_logits_matches_full']=bool(torch.equal(full,last))
    report['last_logit_max_abs_difference']=float((full.float()-last.float()).abs().max())
    report['full_and_last_hidden_bitwise']=bool(torch.equal(captured[0],captured[1]))
    report['native_last_matches_same_shape_head_bitwise']=bool(torch.equal(last,manual))
    report['head_shape_diagnostic']='Native full-position and last-position BF16 GEMM shapes may round differently. All arms use the same native logits_to_keep=1, which must match the same-shape LM head on the identical final hidden state.'
    if not report['native_B_validated'] or not report['full_and_last_hidden_bitwise'] or not report['native_last_matches_same_shape_head_bitwise']:
        report['status']='BLOCKED_SEMANTICS';write_new(a.output,report);return 21
    report['profiles']={};report['verified_arms']={}
    for arm in ARMS:
        observed=[];saved=[]
        def spy(obj,name):
            original=getattr(obj,name)
            def wrapped(*args,**kwargs):
                observed.append({'function':name,'kwargs':{k:v for k,v in kwargs.items() if isinstance(v,(str,int,float,bool,type(None)))},
                                 'operands':[{'dtype':str(x.dtype),'shape':list(x.shape)} for x in args if isinstance(x,torch.Tensor)]})
                return original(*args,**kwargs)
            setattr(obj,name,wrapped);saved.append((obj,name,original))
        if arm!='B':
            spy(core,'per_warp_int8_cuda')
            if arm=='A_PUBLIC':
                spy(core,'per_channel_fp8');spy(core.sm89_compile,'qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf')
            else:spy(core.sm80_compile,'qk_int8_sv_f16_accum_f32_attn')
        try:
            warm=runtime.forward(item['token_ids'],arm,False);del warm;torch.cuda.synchronize()
            observed.clear()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                result=runtime.forward(item['token_ids'],arm,False);torch.cuda.synchronize()
            del result
        finally:
            for obj,name,original in reversed(saved):setattr(obj,name,original)
        ops=sorted({e.key for e in prof.key_averages() if 'attention' in e.key or 'attn' in e.key})
        kernels=sorted({e.name for e in prof.events() if str(e.device_type).endswith('CUDA') and ('flash' in e.name or 'qk_int' in e.name)})
        if arm=='B':verified=any('flash_fwd' in k for k in kernels) and not any('math' in k for k in ops)
        elif arm=='A_PUBLIC':verified=any('qk_int_sv_f8' in k for k in kernels) and any(o['function']=='per_channel_fp8' and o['kwargs'].get('scale_max')==2.25 for o in observed)
        else:verified=any('qk_int_sv_f16' in k for k in kernels) and any(o['function']=='qk_int8_sv_f16_accum_f32_attn' and any(t['dtype']=='torch.float16' for t in o['operands']) for o in observed)
        report['verified_arms'][arm]=verified;report['profiles'][arm]={'operators':ops,'kernels':kernels,'observed_calls':observed,'instruction_level_evidence':'Not newly acquired; profiler and source evidence only'}
    if not all(report['verified_arms'].values()):
        report['status']='BLOCKED_RUNTIME_ROUTE';write_new(a.output,report);return 22
    checks=[];baseline=None
    for arm in ARMS:
        record,private=runtime.diagnose(item,gold_index,arm,report,baseline)
        q,k,v=[x.cuda() for x in private['qkv']]
        with torch.inference_mode():
            if arm=='B':
                original=runtime.registry['sdpa'];module=runtime.model.model.layers[13].self_attn
                replay=original(module,q,k,v,None,dropout=0.,scaling=128**-.5)[0]
            else:replay=getattr(sageattention,SETTINGS[arm]['function'])(q,k,v,is_causal=True,sm_scale=128**-.5,pv_accum_dtype=SETTINGS[arm]['pv_accum_dtype'],**COMMON).transpose(1,2).contiguous()
        record['standalone_integrated_bitwise']=bool(torch.equal(replay.cpu(),private['operator_out']))
        checks.append(record)
        if arm=='B':baseline=private
        del q,k,v,replay
    report['diagnostic_checks']=checks
    report['status']='PASS' if all(r['validity_status']['valid'] and r['standalone_integrated_bitwise'] for r in checks) else 'BLOCKED_SEMANTICS'
    report['memory']={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    write_new(a.output,report);print(json.dumps({'status':report['status'],'native_identity':report['native_B_validated'],'arms':report['verified_arms']}))
    return 0 if report['status']=='PASS' else 23


if __name__=='__main__':raise SystemExit(main())
