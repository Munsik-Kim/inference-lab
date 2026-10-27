"""Bounded single-boundary sensitivity diagnostics, separately frozen.

One P model plus one D1 reference block. Reference values are recomputed from
the current P block input; no dense activation from another history is injected.
"""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
from common import CASE,read,write,append,verify_files,file_row,utc

def reference_block(model,idx,rt,loc):
    import torch
    block=rt.hf.Qwen3DecoderLayer(model.config,idx).eval().to(device='cuda',dtype=torch.bfloat16)
    original=model.model.layers[idx]
    norms=['input_layernorm','post_attention_layernorm','self_attn.q_norm','self_attn.k_norm']
    for name in norms:block.get_submodule(name).weight.copy_(original.get_submodule(name).weight)
    if rt.replace(block,loc['artifact_dir'],'D1',layer_indices=[idx])!=7:raise RuntimeError('AUXILIARY_PROJECTION_COUNT')
    if len(list(block.named_parameters()))!=11:raise RuntimeError('AUXILIARY_PARAMETER_COVERAGE')
    return block

@contextlib.contextmanager
def replace_boundary(model,idx,boundary,reference,rt,save_input=None,noop=False):
    original_rope=rt.hf.apply_rotary_pos_emb;target=model.model.layers[idx]
    state={'active':False,'applied':0,'reference_calls':0}
    def pre(module,args,kwargs):
        hidden=args[0] if args else kwargs['hidden_states']
        if save_input is not None:state['input']=rt.save_tensor(save_input,hidden)
        if not noop:
            def capture(q,k,cos,sin,unsqueeze_dim=1):
                qo,ko=original_rope(q,k,cos,sin,unsqueeze_dim)
                if boundary=='post_RoPE_Q':state['replacement']=qo
                return qo,ko
            rt.hf.apply_rotary_pos_emb=capture
            try:out=reference(*args,**kwargs)
            finally:rt.hf.apply_rotary_pos_emb=rope
            if boundary=='block_output':state['replacement']=out
            state['reference_calls']+=1
        state['active']=True
    def rope(q,k,cos,sin,unsqueeze_dim=1):
        qo,ko=original_rope(q,k,cos,sin,unsqueeze_dim)
        if state['active'] and boundary=='post_RoPE_Q':
            replacement=qo.clone() if noop else state['replacement']
            if replacement.shape!=qo.shape or replacement.dtype!=qo.dtype or replacement.stride()!=qo.stride():raise RuntimeError('REPLACEMENT_Q_LAYOUT')
            state['applied']+=1
            return replacement,ko
        return qo,ko
    def post(module,args,out):
        state['active']=False
        if boundary=='block_output':
            replacement=out.clone() if noop else state['replacement']
            if replacement.shape!=out.shape or replacement.dtype!=out.dtype or replacement.stride()!=out.stride():raise RuntimeError('REPLACEMENT_BLOCK_LAYOUT')
            state['applied']+=1
            return replacement
    handles=[target.register_forward_pre_hook(pre,with_kwargs=True),target.register_forward_hook(post)]
    rt.hf.apply_rotary_pos_emb=rope
    try:yield state
    finally:
        rt.hf.apply_rotary_pos_emb=original_rope
        for h in handles:h.remove()

def child(a):
    import gc
    import numpy as np
    import torch
    import runtime as rt
    from worker import encoded,full_metrics
    from modes import to_gpu
    from layout import metrics
    rt.flags();verify_files(read(CASE/'provenance/freeze_v1.json'))
    loc=read(Path(a.private)/'locations.json');selection=read(a.selection);manifest=read(CASE/'inputs/manifest.json');private=Path(a.private)
    if a.job=='eval':
        verify_files(selection['analysis_freeze'])
        if not selection['dev_complete'] or selection['intervention']['status']!='SELECTED':raise ValueError('EVAL_INTERVENTION_NOT_SELECTED')
    model=rt.load_model(loc,'P');rows=[];aux=[]
    with torch.inference_mode():
        if a.job=='preflight':
            doc=manifest['document_order']['DEV'][0];args=rt.inputs(doc,512);base=encoded(model(**args).logits)
            for idx in sorted(selection['layers']):
                ref=reference_block(model,idx,rt,loc);aux.append({'layer':idx,'storage_bytes':sum(x['storage_bytes'] for x in rt.tensor_inventory(ref) if x['unique_storage'])})
                ctx=private/'contexts/DEV'/doc;hidden=to_gpu(np.load(ctx/f'layer{idx}_input.npy',allow_pickle=False));cos=to_gpu(np.load(ctx/'cos.npy',allow_pickle=False));sin=to_gpu(np.load(ctx/'sin.npy',allow_pickle=False))
                value=ref(hidden,attention_mask=None,position_ids=torch.from_numpy(np.load(ctx/'position_ids.npy',allow_pickle=False)).cuda(),position_embeddings=(cos,sin),use_cache=False)
                expected=np.load(private/'local/DEV'/f'{doc}_layer{idx}_D1_R0_selected.npz',allow_pickle=False)['block_output']
                observed=rt.cpu(value[:,manifest['selected_positions']['512'],:]);m=metrics(expected,observed)
                if m['max_abs']!=0:raise RuntimeError('AUXILIARY_D1_BLOCK_MISMATCH')
                for boundary in ['post_RoPE_Q','block_output']:
                    with replace_boundary(model,idx,boundary,ref,rt,noop=True) as state:out=model(**args).logits
                    check=full_metrics(base,encoded(out));del out
                    if not check['bit_identical'] or state['applied']!=1:raise RuntimeError('REPLACEMENT_NOOP_CHANGED_OUTPUT')
                    rows.append({'layer':idx,'boundary':boundary,'noop':check,'auxiliary_D1_same_input':m,'applied':state['applied']})
                del ref,value,hidden,cos,sin;gc.collect()
            write(a.output/'intervention_preflight.json',{'source_bundle_sha256':read(a.freeze)['bundle_sha256'],'input_manifest':file_row(CASE/'inputs/manifest.json'),'rows':rows,'full_forwards':5,'auxiliary_block_forwards':2,'auxiliary_storage':aux,'one_auxiliary_block_at_a_time':True})
        else:
            preflight=read(private/'raw/p2-intervention-preflight/intervention_preflight.json')
            if preflight['source_bundle_sha256']!=read(a.freeze)['bundle_sha256'] or preflight['input_manifest']!=file_row(CASE/'inputs/manifest.json'):raise ValueError('INTERVENTION_PREFLIGHT_MISMATCH')
            split='DEV' if a.job=='dev' else 'EVAL';docs=manifest['document_order'][split]
            candidates=[{'layer':i,'boundary':b} for i in sorted(selection['layers']) for b in ['post_RoPE_Q','block_output']] if a.job=='dev' else [selection['intervention']['candidate']]
            for candidate in candidates:
                idx,boundary=candidate['layer'],candidate['boundary'];ref=reference_block(model,idx,rt,loc)
                aux.append({'layer':idx,'boundary':boundary,'storage_bytes':sum(x['storage_bytes'] for x in rt.tensor_inventory(ref) if x['unique_storage'])})
                for doc in docs:
                    stem=f'{doc}_layer{idx}_{boundary}';args=rt.inputs(doc,512)
                    with replace_boundary(model,idx,boundary,ref,rt,save_input=a.output/(stem+'_P_input.npy')) as state:out=model(**args).logits
                    if state['applied']!=1 or state['reference_calls']!=1 or not torch.isfinite(out).all():raise RuntimeError('INTERVENTION_FORWARD_CONTRACT')
                    saved=rt.save_tensor(a.output/(stem+'_logits.npy'),out[0,manifest['selected_positions']['512'],:]);del out
                    row={'document_id':doc,'split':split,'layer':idx,'boundary':boundary,'full_logits_shape':[1,512,151936],'positions':manifest['selected_positions']['512'],'logits':saved,'same_current_P_input':state['input'],'reference_block_calls':1,'replacement_count':1,'rounding':'R0','path':'P','auxiliary_path':'D1'}
                    write(a.output/(stem+'.json'),row);rows.append(row)
                del ref;gc.collect()
            write(a.output/'intervention_receipt.json',{'source_bundle_sha256':read(a.freeze)['bundle_sha256'],'split':split,'full_forwards':len(rows),'auxiliary_block_forwards':len(rows),'candidates':candidates,'rows':rows,'auxiliary_storage':aux,'one_auxiliary_block_at_a_time':True,'memory':rt.memory()})
    write(a.output/'exit.json',{'status':'COMPLETE','utc':utc()});print(json.dumps({'job':a.job,'rows':len(rows)}))

def parent(a):
    from supervise import telemetry,child_env
    private=Path(a.private);ledger=[json.loads(x) for x in (CASE/'worker_ledger.jsonl').read_text().splitlines()];remaining=min(7200-sum(x['worker_wall_seconds'] for x in ledger),2400-sum(x['worker_wall_seconds'] for x in ledger if x['stage']=='P2'))
    if a.timeout>remaining:raise ValueError('BUDGET_EXHAUSTED')
    lock=(private/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);before=telemetry()
    if any(x['returncode'] for x in before) or before[1]['stdout'] or float(before[0]['stdout'].split(',')[5])>0 or float(before[0]['stdout'].split(',')[4])<6144:raise ValueError('GPU_BUSY_OR_HEADROOM')
    used=sum(f.stat().st_size for root in [private,CASE.parents[1]] for f in root.rglob('*') if f.is_file() and not f.is_symlink())
    if used+512*1024**2>8589934592:raise ValueError('DISK_HEADROOM')
    label='p2-intervention-'+a.job;out=private/'raw'/label;out.mkdir(exist_ok=False)
    cmd=[a.python,str(Path(__file__).resolve()),'--child','--job',a.job,'--private',str(private),'--freeze',str(Path(a.freeze).resolve()),'--selection',str(Path(a.selection).resolve()),'--output',str(out)]
    write(out/'launch.json',{'utc':utc(),'command':cmd,'timeout':a.timeout,'remaining':remaining,'freeze':file_row(a.freeze),'telemetry_before':before,'owned_disk_bytes_before':used});append(CASE/'attempts.jsonl',{'event':'WORKER_START','label':label,'stage':'P2','utc':utc()})
    u0=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.perf_counter();timeout=False
    with (out/'stdout.txt').open('w') as so,(out/'stderr.txt').open('w') as se:
        proc=subprocess.Popen(cmd,stdout=so,stderr=se,env=child_env(private),start_new_session=True)
        try:code=proc.wait(timeout=a.timeout-2)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(proc.pid,signal.SIGTERM)
            try:code=proc.wait(timeout=1)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
    wall=time.perf_counter()-start;u1=resource.getrusage(resource.RUSAGE_CHILDREN);row={'label':label,'stage':'P2','worker_wall_seconds':wall,'cpu_user_seconds':u1.ru_utime-u0.ru_utime,'cpu_system_seconds':u1.ru_stime-u0.ru_stime,'parent_children_max_rss_kib':u1.ru_maxrss,'exit_code':code,'timeout':timeout,'pid':proc.pid,'source_bundle_sha256':read(a.freeze)['bundle_sha256'],'raw_directory':str(out),'utc':utc()};append(CASE/'worker_ledger.jsonl',row);write(out/'parent_receipt.json',{**row,'telemetry_after':telemetry()});append(CASE/'attempts.jsonl',{'event':'WORKER_END',**row});print(json.dumps(row));sys.exit(0 if code==0 else 1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--freeze',required=True);p.add_argument('--selection',required=True);p.add_argument('--python');p.add_argument('--output',type=Path);p.add_argument('--timeout',type=float,default=300);p.add_argument('--job',choices=['preflight','dev','eval'],required=True);p.add_argument('--child',action='store_true');a=p.parse_args();verify_files(read(a.freeze))
    if a.child:
        try:child(a)
        except BaseException as e:write(a.output/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
    else:parent(a)

if __name__=='__main__':main()
