"""P1b read-only artifact evaluator. Never fits K or creates quantized payloads."""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from p1b_common import ALPHAS, BLOCKS, PROJECTIONS, TARGETS, REVISION, POSITIONS, target, save, append, sha, digest, utc, forbidden
from policies import select

def main():
    ap=argparse.ArgumentParser()
    for k in ['case','private','old-private','model','output']:ap.add_argument('--'+k,required=True)
    a=ap.parse_args();start=time.monotonic();case=Path(a.case).resolve();private=Path(a.private);old=Path(a.old_private);out=Path(a.output)
    spec=json.loads((case/'followups/p1b_selection/specs/p1b.json').read_text())
    baseline=json.loads((out/'reused_artifact_manifest.json').read_text());manifest=json.loads((out/'input_manifest.json').read_text())
    assert sha(private/'inputs/documents.private.json')==manifest['private_tokens_sha256']
    docs=json.loads((private/'inputs/documents.private.json').read_text());smoke=next(d for d in docs if d['split']=='REUSED_SMOKE')
    source=Path(__file__).parent
    files={str(p.relative_to(case)):sha(p) for p in sorted(source.glob('*.py'))}
    for name in ['codec.py','decompose.py','common.py','fit.py']:
        files['scripts/'+name]=sha(case/'scripts'/name)
    freeze=digest(json.dumps(files,sort_keys=True).encode())
    save(out/'source_freeze.json',{'recorded_utc':utc(),'execution_source_sha256':freeze,'files':files,'spec_sha256':sha(case/'followups/p1b_selection/specs/p1b.json'),
                                 'input_manifest_sha256':sha(out/'input_manifest.json'),'reused_artifact_manifest_sha256':sha(out/'reused_artifact_manifest.json'),
                                 'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=case,text=True).strip(),
                                 'historical_P1_source_sha256':spec['historical_P1_source_sha256'],'scope':'source tree hash is not a Git commit or model hash'})
    sys.path.insert(0,str(case/'scripts'))
    import codec
    import fit
    import decompose
    guard_calls=[]
    def deny(*args,**kwargs):
        guard_calls.append('FORBIDDEN_FIT_OR_ENCODE');forbidden()
    codec.encode=codec.write_reload=fit.fit_numpy=fit.fit_torch=deny
    assert Path(codec.__file__).resolve()==case/'scripts/codec.py'
    assert Path(decompose.__file__).resolve()==case/'scripts/decompose.py'
    torch.set_num_threads(4);torch.manual_seed(spec['seeds']['execution'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    assert torch.cuda.is_available() and Path(a.model).name==REVISION
    def budget():
        if time.monotonic()-start>spec['worker_budget_seconds']:raise TimeoutError('cumulative P1b worker wall budget')
    def event(stage,**kw):
        append(out/'events.jsonl',dict(stage=stage,utc=utc(),worker_elapsed_seconds=time.monotonic()-start,**kw));print(stage,kw,flush=True)
    def device():
        free,total=torch.cuda.mem_get_info()
        return {'device_free_bytes':free,'device_total_bytes':total,'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
    def guarded_artifact(rel):
        p=old/'artifacts'/rel;entry=next(r for r in baseline['artifacts'] if r['label']=='P1_PRIVATE/artifacts/'+rel)
        assert p.is_file() and p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],rel
        return codec.decode(p.read_bytes())
    event('SPEC_FROZEN',source_sha256=freeze,input_sha256=sha(out/'input_manifest.json'))
    t=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    load_seconds=time.monotonic()-t
    assert model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr()
    modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n.startswith('model.layers.')}
    originals={n:m.weight.detach().cpu().clone() for n,m in modules.items()}
    assert len(originals)==196
    for n in TARGETS:assert list(originals[n].shape)==([1024,2048] if n.endswith('o_proj') else [1024,3072])
    env={'recorded_utc':utc(),'python_version':platform.python_version(),'python_label':'EXISTING_REFERENCE_ENV','torch':torch.__version__,
         'transformers':__import__('transformers').__version__,'numpy':np.__version__,'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(),
         'capability':list(torch.cuda.get_device_capability()),'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
         'flags':spec['runtime'],'imported_reference_files':{'codec':'scripts/codec.py','decompose':'scripts/decompose.py','fit':'scripts/fit.py; functions guarded, not called'},
         'initial_memory':device(),'load_seconds':load_seconds,'model_revision':REVISION,'execution_source_sha256':freeze,'tied_embedding_head':True}
    save(out/'runtime_manifest.json',env)
    traces=private/'traces';counts=Counter();metric_seconds=0.;stage=['SMOKE'];ref={};stage_times={};used_prefix=set()
    def path(kind,d,n,part):return traces/kind/d['document_id_hash']/(n.replace('.','_')+'_'+part+'.npy')
    def put(p,arr):
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():raise FileExistsError(p)
        np.save(p,arr,allow_pickle=False)
    def restore():
        for n,w in originals.items():modules[n].weight.data.copy_(w.to('cuda'))
    def prefix(depth):
        restore();changed=[]
        for n in originals:
            if int(n.split('.')[2])<depth:
                q,_,_=guarded_artifact('prefix/'+n+'.int4')
                modules[n].weight.data.copy_(torch.from_numpy(q).to('cuda',torch.bfloat16));used_prefix.add(n);changed.append(n)
        assert len(changed)==7*depth
        assert all(torch.equal(modules[n].weight.detach().cpu(),w) for n,w in originals.items() if int(n.split('.')[2])>=depth)
        assert model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr()
    @torch.inference_mode()
    def forward(d,capture):
        budget();buf={};hooks=[]
        def hook(n):
            def h(m,inp,y):
                if n in buf:raise ValueError('repeated target execution')
                buf[n]=(inp[0].detach().clone().float().cpu().numpy()[0].T,y.detach().clone().float().cpu().numpy()[0].T)
            return h
        try:
            for n in capture:hooks.append(model.get_submodule(n).register_forward_hook(hook(n)))
            ids=torch.tensor([d['input_ids']],device='cuda',dtype=torch.long);assert ids.shape==(1,256)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
            result=model(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(256,device='cuda').unsqueeze(0),use_cache=False,return_dict=True)
            logits=result.logits[0];assert logits.shape==(256,model.config.vocab_size) and torch.isfinite(logits).all()
            f=logits[:-1].float();loss=torch.nn.functional.cross_entropy(f,ids[0,1:],reduction='sum');arg=f.argmax(-1).cpu().numpy()
            selected=logits[torch.tensor(POSITIONS,device='cuda')].float().cpu().numpy();torch.cuda.synchronize()
            stats={'NLL_sum':float(loss),'valid_token_count':255,'wall_seconds':time.monotonic()-tick,
                   'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),**device()}
            counts[stage[0]]+=1
            append(out/'forward_calls.jsonl',dict(stage=stage[0],document_id_hash=d['document_id_hash'],token_hash=d['token_hash'],split=d['split'],capture=list(capture),**stats))
            assert set(buf)==set(capture)
            return stats,arg,selected,buf
        finally:
            for h in hooks:h.remove()
    def teacher(ds):
        restore()
        for d in ds:
            st,arg,sel,buf=forward(d,TARGETS)
            for n,(x,y) in buf.items():put(path('teacher',d,n,'X'),x);put(path('teacher',d,n,'Y'),y)
            p=traces/'reference'/d['document_id_hash'];put(p/'selected.npy',sel);put(p/'argmax.npy',arg)
            ref[d['document_id_hash']]=st
            append(out/'teacher.jsonl',dict(document_id_hash=d['document_id_hash'],split=d['split'],token_hash=d['token_hash'],**st))
    def compare_ref(d,arg,sel):
        r=traces/'reference'/d['document_id_hash'];rs=np.load(r/'selected.npy')
        lp=torch.log_softmax(torch.from_numpy(sel).cuda(),-1);rlp=torch.log_softmax(torch.from_numpy(rs).cuda(),-1)
        return {'sampled_full_vocab_KL_sum':float((rlp.exp()*(rlp-lp)).sum()),'KL_position_count':16,'flip_count':int((arg!=np.load(r/'argmax.npy')).sum())}
    old_stats={(r['target'],r['alpha']):r for r in json.loads((case/'results/p1_r2/artifact_manifest.json').read_text())['candidates']}
    old_fit={r['target']:r for r in baseline['old_fit']}
    # Reuse old SMOKE1 only. No K fitting, encoding or new-document model outputs.
    teacher([smoke]);restore();st,arg,sel,buf=forward(smoke,TARGETS)
    assert all(np.array_equal(buf[n][0],np.load(path('teacher',smoke,n,'X'))) and np.array_equal(buf[n][1],np.load(path('teacher',smoke,n,'Y'))) for n in TARGETS)
    assert np.array_equal(sel,np.load(traces/'reference'/smoke['document_id_hash']/'selected.npy'))
    old_ref=old/'traces/reference'/smoke['document_id_hash']/'selected.npy'
    historical_teacher_exact=np.array_equal(sel,np.load(old_ref));assert historical_teacher_exact
    n=target(4,'o_proj');prefix(4);st,arg,sel,buf=forward(smoke,[n]);z=buf[n][0]
    historical_prefix_exact=np.array_equal(z,np.load(old/'traces/prefix'/smoke['document_id_hash']/(n.replace('.','_')+'_Z.npy')));assert historical_prefix_exact
    st2,_,_,b2=forward(smoke,[n]);assert np.array_equal(z,b2[n][0]) and np.array_equal(buf[n][1],b2[n][1])
    q,_,_=guarded_artifact(f'candidates/{n}.alpha0.25.int4');q0,_,_=guarded_artifact(f'candidates/{n}.alpha0.int4');k=np.load(old/f'artifacts/{n}_K.npy')
    try:
        modules[n].weight.data.copy_(torch.from_numpy(q).to('cuda',torch.bfloat16));st3,_,_,cb=forward(smoke,[n]);assert np.array_equal(z,cb[n][0])
        x=np.load(path('teacher',smoke,n,'X'));y=np.load(path('teacher',smoke,n,'Y'));w=originals[n].float().numpy()
        check=decompose.measure(w,x,z,.25*k,q,q0,y,cb[n][1],modules[n].weight.detach().float().cpu().numpy())
        assert check['identity_relative_gap']<=baseline['old_smoke_tolerance']['identity_relative']
        np.savez(private/'smoke_arrays.private.npz',W=w[:8,:32],X=x[:32,:8],Z=z[:32,:8],K=k[:8,:32],Q=q[:8,:32],Q0=q0[:8,:32])
    finally:restore()
    save(out/'smoke.json',{'status':'PASS','input':'REUSED_SMOKE','historical_teacher_selected_logits_exact':historical_teacher_exact,
        'historical_prefix_Z_exact':historical_prefix_exact,'teacher_reinstall_exact':True,'same_Z_after_target_change':True,'full_model_head_shape':[256,model.config.vocab_size],
        'finally_restore_verified':all(torch.equal(modules[n].weight.detach().cpu(),w) for n,w in originals.items()),
        'tolerance':baseline['old_smoke_tolerance'],'decomposition':check,'forward_count':5,'memory':st3,'elapsed_seconds':time.monotonic()-start})
    event('SMOKE_PASS')
    def evaluate_phase(split):
        nonlocal metric_seconds
        if split!='SELECT-ID':
            f=out/'selection_frozen.json';assert f.is_file(),'all selectors must freeze before EVAL'
            assert sha(f)==selection_sha,'selection file changed'
        tick=time.monotonic();ds=sorted([d for d in docs if d['split']==split],key=lambda d:d['document_id_hash'])
        stage[0]=split+'_TEACHER';teacher(ds)
        order=list(BLOCKS);np.random.default_rng(spec['seeds']['execution']).shuffle(order)
        for l in order:
            prefix(l);stage[0]=split+'_PREFIX'
            for d in ds:
                st,arg,sel,buf=forward(d,[target(l,p) for p in PROJECTIONS])
                append(out/'prefix_only.jsonl',dict(block=l,split=split,document_id_hash=d['document_id_hash'],token_hash=d['token_hash'],**st,**compare_ref(d,arg,sel)))
                for n,(z,y) in buf.items():put(path('prefix',d,n,'Z'),z)
        pairs=[(l,p,a) for l in BLOCKS for p in PROJECTIONS for a in ALPHAS]
        np.random.default_rng(spec['seeds']['execution']).shuffle(pairs);rows=[];stage[0]=split+'_GRID'
        for l,p,alpha in pairs:
            n=target(l,p);prefix(l);w=originals[n].float().numpy().astype('f8');k=np.load(old/f'artifacts/{n}_K.npy');assert k.dtype==np.float64
            q,c,s=guarded_artifact(f'candidates/{n}.alpha{alpha:g}.int4');q0,c0,s0=guarded_artifact(f'candidates/{n}.alpha0.int4')
            meta=old_stats[n,alpha];assert q.shape==w.shape
            cell=out/'cells'/split/f'b{l}_{p}_a{alpha:g}.jsonl';cell.parent.mkdir(parents=True,exist_ok=True);tmp=cell.with_suffix('.partial')
            if cell.exists() or tmp.exists():raise FileExistsError(cell)
            try:
                modules[n].weight.data.copy_(torch.from_numpy(q).to('cuda',torch.bfloat16));qrun=modules[n].weight.detach().float().cpu().numpy()
                assert np.array_equal(qrun,torch.from_numpy(q).to(torch.bfloat16).float().numpy())
                for d in ds:
                    st,arg,sel,buf=forward(d,[n]);mt=time.monotonic()
                    x=np.load(path('teacher',d,n,'X')).astype('f8');z=np.load(path('prefix',d,n,'Z')).astype('f8');y=np.load(path('teacher',d,n,'Y'))
                    assert np.array_equal(z,buf[n][0]),'target input differs across alpha'
                    m=decompose.measure(w,x,z,alpha*k,q,q0,y,buf[n][1],qrun)
                    assert m['identity_relative_gap']<=baseline['old_smoke_tolerance']['identity_relative']
                    u=k@z;A=w@(x-z);m.update(U_norm2=float(np.sum(u*u)),dot_A_U=float(np.sum(A*u)))
                    row=dict(run_id=out.name,stage='P1B_FIXED_ARTIFACT_DIAGNOSTIC',source_commit=env.get('source_commit',json.loads((out/'source_freeze.json').read_text())['source_commit']),
                        execution_source_sha256=freeze,model_revision=REVISION,tokenizer_revision=REVISION,target=n,block=l,projection=p,alpha=alpha,split=split,
                        document_id_hash=d['document_id_hash'],token_hash=d['token_hash'],offset=d['offset'],lambda_value=old_fit[n]['lambda'],K_sha256=old_fit[n]['K_sha256'],
                        quantizer_variant='P1_SAVED_RTN_RIDGE_COMPENSATION_READ_ONLY',bits=4,group_size=64,scale_dtype='FP32',solver_dtype='P1 CPU FP64; NOT_RUN_P1B',rounding='nearest ties-to-even',
                        input_identical_across_alpha=True,NLL_teacher_sum=ref[d['document_id_hash']]['NLL_sum'],**m,**st,**compare_ref(d,arg,sel),
                        artifact_sha256=meta['sha256'],artifact_bytes=meta['artifact_bytes'],code_change_rate=float(np.mean(c!=c0)),
                        scale_change_mean=float(np.mean(np.abs(s.astype('f8')-s0)/s0)),scale_change_max=float(np.max(np.abs(s.astype('f8')-s0)/s0)),
                        endpoint_rate=meta['endpoint_rate'],minus8_rate=meta['minus8_rate'],clipping_rate=meta['clipping_rate'],pre_range_outside_rate=meta['pre_range_outside_rate'],status='COMPLETE')
                    metric_seconds+=time.monotonic()-mt;append(tmp,row);rows.append(row)
                assert len(rows)>=len(ds)
                tmp.rename(cell)
                event('CELL_COMPLETE',split=split,target=n,alpha=alpha,documents=len(ds))
            finally:modules[n].weight.data.copy_(originals[n].to('cuda'))
        stage_times[split]=time.monotonic()-tick
        return rows
    try:
        select_rows=evaluate_phase('SELECT-ID');tick=time.monotonic()
        choices=select(select_rows,{n:r['selected_alpha'] for n,r in old_fit.items()})
        save(out/'selection_frozen.json',{'frozen_utc':utc(),'selection_source':'SELECT-ID only','choices':choices,'execution_source_sha256':freeze,
             'input_manifest_sha256':sha(out/'input_manifest.json'),'spec_sha256':sha(case/'followups/p1b_selection/specs/p1b.json'),
             'SELECT_cell_hashes':{str(p.relative_to(out)):sha(p) for p in sorted((out/'cells/SELECT-ID').glob('*.jsonl'))},
             'FRESH_EVAL_forward_count_at_freeze':sum(v for k,v in counts.items() if k.startswith('FRESH')),'K_or_quantizer_refit':False})
        selection_sha=sha(out/'selection_frozen.json');stage_times['select_compute']=time.monotonic()-tick
        assert sum(v for k,v in counts.items() if k.startswith('FRESH'))==0
        event('SELECTION_FROZEN',sha256=selection_sha,choices=[{'target':r['target'],'SSE':r['policy_alpha']['SELECT_SSE'],'NLL':r['policy_alpha']['SELECT_NLL']} for r in choices])
        evaluate_phase('FRESH-EVAL-ID');evaluate_phase('FRESH-SHIFT')
    finally:restore()
    assert len(used_prefix)==140 and not guard_calls
    assert sum(counts.values())==spec['expected_full_model_forward_calls']
    assert all(torch.equal(modules[n].weight.detach().cpu(),w) for n,w in originals.items())
    save(out/'completion.json',{'status':'COMPLETED_P1B','completed_candidate_rows':3360,'expected_candidate_rows':3360,'completed_target_alpha_split_files':90,
        'full_model_forward_counts':dict(counts),'total_full_model_forward_calls':sum(counts.values()),'model_restore_verified':True,
        'fit_encode_calls':len(guard_calls),'prefix_files_read':len(used_prefix),'worker_internal_wall_seconds':time.monotonic()-start,
        'stage_seconds_inclusive':dict(stage_times,load=load_seconds),'metric_seconds_within_stage_totals':metric_seconds,'final_memory':device(),
        'selection_frozen_sha256':selection_sha,'EVAL_before_selection':False,'GPU_busy_time':'NOT_MEASURED','P2':'NOT_RUN_SCOPE','P3':'NOT_RUN_SCOPE','remote_publication':False})
    event('COMPLETED_P1B',full_forwards=sum(counts.values()),wall_seconds=time.monotonic()-start)

if __name__=='__main__':main()
