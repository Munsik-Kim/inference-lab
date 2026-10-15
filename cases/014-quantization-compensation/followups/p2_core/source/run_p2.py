"""Native full-forward P2: build own-trajectory INT4 models, freeze, then evaluate."""
import argparse,json,math,platform,subprocess,sys,time,os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from p2_common import REVISION,ALPHA,POSITIONS,groups,all_names,sha,digest,save,append,utc
from p2_fit import factor,solve
from model_state import WeightBank
from independent_codec import read as independent_read
from artifact_io import write_new

def tensor_sha(t):
    c=t.detach().cpu().contiguous()
    return digest(c.view(torch.int16).numpy().tobytes()) if c.dtype==torch.bfloat16 else digest(c.numpy().tobytes())
def configure():
    torch.set_num_threads(4);torch.manual_seed(6101522)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False;torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
def putarray(path,arr):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial')
    if p.exists() or tmp.exists():raise FileExistsError(p)
    with tmp.open('xb') as f:np.save(f,arr,allow_pickle=False)
    tmp.rename(p)

def main():
    ap=argparse.ArgumentParser()
    for k in ['case','private','output']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--prior-worker-seconds',type=float,default=0.)
    a=ap.parse_args();start=time.monotonic();case=Path(a.case).resolve();pr=Path(a.private);out=Path(a.output);src=Path(__file__).parent
    identity=json.loads((pr/'identity.private.json').read_text());spec=json.loads((case/'followups/p2_core/specs/p2.json').read_text());inp=json.loads((out/'input_manifest.json').read_text())
    assert sha(pr/'inputs/documents.private.json')==inp['private_tokens_sha256']
    docs=json.loads((pr/'inputs/documents.private.json').read_text());cal=[d for d in docs if d['split']=='CAL-FIT'];smoke=next(d for d in docs if d['split']=='SMOKE');fresh=[d for d in docs if d['split'] in ['EVAL-ID','SHIFT']]
    assert len(cal)==32 and len(fresh)==80
    sys.path.insert(0,str(case/'scripts'));import codec,decompose
    files={str(p.relative_to(case)):sha(p) for p in sorted(src.glob('*.py'))}
    for n in ['codec.py','decompose.py','common.py']:files['scripts/'+n]=sha(case/'scripts'/n)
    source_hash=digest(json.dumps(files,sort_keys=True).encode());spec_hash=sha(case/'followups/p2_core/specs/p2.json');input_hash=sha(out/'input_manifest.json')
    save(out/'source_freeze.json',{'recorded_utc':utc(),'execution_source_sha256':source_hash,'files':files,'spec_sha256':spec_hash,'input_manifest_sha256':input_hash,
         'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=case,text=True).strip(),'note':'Git commit, execution tree hash, model and artifact identity are separate'})
    configure();assert torch.cuda.is_available()
    def budget():
        if time.monotonic()-start+a.prior_worker_seconds>spec['worker_budget_seconds']:raise TimeoutError('P2 cumulative worker budget exceeded')
    def event(stage,**kw):
        append(out/'events.jsonl',dict(stage=stage,utc=utc(),worker_elapsed_seconds=time.monotonic()-start,**kw));print(stage,kw,flush=True)
    def device():
        free,total=torch.cuda.mem_get_info();return {'device_free_bytes':free,'device_total_bytes':total,'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
    event('SPEC_INPUT_SOURCE_FROZEN',source_sha256=source_hash)
    t=time.monotonic();model=AutoModelForCausalLM.from_pretrained(identity['snapshot']['model_path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();load_time=time.monotonic()-t
    names=all_names(model.config.num_hidden_layers);assert len(names)==196
    actual={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n.startswith('model.layers.')};assert set(actual)==set(names)
    assert all(list(actual[n].weight.shape)==spec['projection_shapes'][n] for n in names)
    assert model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr()
    bank=WeightBank(model,names)
    def protected():return {n:tensor_sha(p) for n,p in model.named_parameters() if not any(n==x+'.weight' or n==x+'.bias' for x in names)}
    protected_start=protected();counts=Counter();stage=['SMOKE'];forward_serial=0
    model_bytes=sum(p.numel()*p.element_size() for p in model.parameters());block_bytes=sum(w.numel()*w.element_size() for w in bank.originals.values())
    save(out/'model_manifest.json',{'model_id':'Qwen/Qwen3-0.6B','revision':REVISION,'snapshot_files':identity['snapshot']['files'],'identity_level':'full local file hashes match original receipts; no new official digest/D-origin requirement',
         'modules':[{'name':n,'shape':list(actual[n].weight.shape),'bias':actual[n].bias is not None} for n in names],'projection_count':196,'protected_parameter_hashes':protected_start,'tied_embedding_lm_head':True,
         'runtime_parameter_bytes_unique':model_bytes,'original_projection_BF16_tensor_bytes':block_bytes,'unchanged_common_parameter_bytes_unique':model_bytes-block_bytes})
    installed_source=Path(__import__('transformers.models.qwen3.modeling_qwen3',fromlist=['__file__']).__file__)
    save(out/'runtime_manifest.json',{'recorded_utc':utc(),'python':platform.python_version(),'python_label':'EXISTING_REFERENCE_ENV','torch':torch.__version__,'transformers':__import__('transformers').__version__,'numpy':np.__version__,'CUDA':torch.version.cuda,
         'GPU':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
         'runtime':spec['runtime'],'initial_memory':device(),'load_seconds':load_time,'native_model_source_sha256':sha(installed_source),'strategy':'native entire CausalLM forward; no blockwise replay or truncated suffix',
         'reference_imports':{'codec':'scripts/codec.py','decompose':'scripts/decompose.py'}})
    save(out/'memory_ledger.json',{'model_BF16_unique_bytes':model_bytes,'CPU_original_projection_copy_bytes':block_bytes,'max_input_dimension':3072,'max_Gram_FP64_bytes':3072**2*8,
         'max_Cholesky_FP64_bytes':3072**2*8,'max_CAL_X_or_Z_FP64_bytes':3072*8192*8,'row_chunk':128,'max_chunk_output_FP64_bytes':128*8192*8,
         'full_head_BF16_bytes':256*model.config.vocab_size*2,'quality_full_head_FP32_temporary_bytes':255*model.config.vocab_size*4,'CAL_X_storage_policy':'CPU/disk native hook captures; not serving offload'})
    @torch.inference_mode()
    def forward(d,capture=(),metrics=False,hash_logits=False):
        nonlocal forward_serial
        budget();buf={};hooks=[];seen=[]
        def hook(n):
            def h(m,args,y):
                if n in buf:raise ValueError('duplicate projection execution')
                buf[n]=args[0].detach().clone().float().cpu().numpy()[0].T;seen.append(n)
            return h
        try:
            for n in capture:hooks.append(actual[n].register_forward_hook(hook(n)))
            ids=torch.tensor([d['input_ids']],device='cuda',dtype=torch.long);assert ids.shape==(1,256)
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
            result=model(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(256,device='cuda').unsqueeze(0),use_cache=False,return_dict=True)
            logits=result.logits[0];assert logits.shape==(256,model.config.vocab_size) and torch.isfinite(logits).all()
            stats={};arg=sel=None
            if metrics:
                f=logits[:-1].float();stats['NLL_sum']=float(torch.nn.functional.cross_entropy(f,ids[0,1:],reduction='sum'));stats['valid_token_count']=255
                arg=f.argmax(-1).cpu().numpy();sel=logits[torch.tensor(POSITIONS,device='cuda')].float().cpu().numpy()
            if hash_logits:stats['full_logits_BF16_sha256']=tensor_sha(logits)
            torch.cuda.synchronize();stats.update(wall_seconds=time.monotonic()-tick,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),**device())
            counts[stage[0]]+=1;forward_serial+=1
            append(out/'forward_calls.jsonl',dict(call_id=forward_serial,stage=stage[0],split=d['split'],document_id_hash=d['document_id_hash'],token_hash=d['token_hash'],capture=list(capture),metrics=metrics,**stats))
            assert seen==list(capture),'native execution order differs from registered group order'
            return stats,arg,sel,buf
        finally:
            for h in hooks:h.remove()
    def check_group_inputs(buf,ns):
        if not all(np.array_equal(buf[ns[0]],buf[n]) for n in ns):raise ValueError('registered shared-input group mismatch')
    # Reused SMOKE only; native full forward is the capture implementation itself.
    smoke0,_,sel0,b0=forward(smoke,names,True,True);bank.restore();smoke1,_,_,b1=forward(smoke,names,True,True)
    assert smoke0['full_logits_BF16_sha256']==smoke1['full_logits_BF16_sha256'] and all(np.array_equal(b0[n],b1[n]) for n in names)
    for block in range(28):
        for g,ns in groups(block):check_group_inputs(b0,ns)
    n=names[0];w=bank.originals[n].float().numpy();q,c,s,stat=write_new(pr/'smoke/first_q.int4',w);iq,ic,is_=independent_read(pr/'smoke/first_q.int4')
    assert np.array_equal(q,iq) and np.array_equal(c,ic) and np.array_equal(s,is_)
    with bank.restored_on_exit():
        bank.install(n,q);_,_,_,qbuf=forward(smoke,names,False)
        assert np.array_equal(qbuf[n],b0[n]);assert not np.array_equal(qbuf[groups(0)[1][1][0]],b0[groups(0)[1][1][0]])
    with bank.restored_on_exit():
        last=names[-1];bank.install(last,codec.decode(codec.encode(bank.originals[last].float().numpy())[0])[0]);_,_,_,futurebuf=forward(smoke,names,False)
        assert np.array_equal(futurebuf[n],b0[n])
    assert bank.unchanged() and protected()==protected_start
    putarray(pr/'smoke/BF16_selected.npy',sel0)
    save(pr/'smoke/BF16_stats.json',smoke0)
    save(out/'smoke.json',{'status':'PASS','input':'REUSED_ORIGINAL_SMOKE1','teacher_reinstall_full_logits_exact':True,'all196_input_captures_repeat_exact':True,'qkv_gateup_common_inputs':'PASS all28blocks',
         'same_block_q_quantization_changes_o_input':True,'future_block_does_not_change_first_q_input':True,'codec_independent_q_scale_weight_exact':True,'restoration_and_protected_parameters':'PASS',
         'blockwise_replay':'NOT_USED; capture is native whole-model full forward','full_head_shape':[256,model.config.vocab_size],'smoke_full_model_calls':4,'tolerance':spec['numerical_tolerance'],
         'initial_memory':smoke0,'worker_elapsed_seconds':time.monotonic()-start})
    del b0,b1,qbuf,futurebuf
    event('SMOKE_PASS')
    trace=pr/'traces';art=pr/'artifacts';meta={};smoke_stats={'BF16':smoke0};fit_counts=Counter();stage_times={};metric_seconds=0.
    def xloc(d,block,g):return trace/'teacher'/f'b{block}_{g}'/(d['document_id_hash']+'.npy')
    def zloc(method,d,block,g):return trace/method/f'b{block}_{g}'/(d['document_id_hash']+'.npy')
    # Capture original teacher CAL inputs once, checking all shared inputs on every document.
    tick=time.monotonic();stage[0]='CAL_TEACHER';bank.restore()
    for d in cal:
        _,_,_,buf=forward(d,names)
        for block in range(28):
            for g,ns in groups(block):check_group_inputs(buf,ns);putarray(xloc(d,block,g),buf[ns[0]])
    stage_times['CAL_teacher_capture']=time.monotonic()-tick;del buf
    def artifact(method,n,w):
        tick=time.monotonic();p=art/method/(n+'.int4');q,c,s,st=write_new(p,w);iq,ic,is_=independent_read(p)
        assert np.array_equal(q,iq) and np.array_equal(c,ic) and np.array_equal(s,is_)
        expected=36+q.shape[0]*((q.shape[1]+63)//64)*4+q.shape[0]*((q.shape[1]+63)//64)*32;assert p.stat().st_size==expected
        row={'method':method,'name':n,'shape':list(q.shape),'path_label':method+'/'+n+'.int4','sha256':sha(p),'installed_BF16_sha256':tensor_sha(torch.from_numpy(q).to(torch.bfloat16)),
             'convert_pack_reload_seconds':time.monotonic()-tick,**st};meta[method,n]=row
        append(out/'artifacts.jsonl',row)
        return q
    def built_smoke(method):
        stage[0]='BUILT_SMOKE_'+method;st,_,sel,_=forward(smoke,(),True,True);smoke_stats[method]=st;putarray(pr/f'smoke/{method}_selected.npy',sel);save(pr/f'smoke/{method}_stats.json',st)
        assert protected()==protected_start
        save(out/(method+'.build_complete.json'),{'method':method,'projection_count':196,'core_bytes':sum(meta[method,n]['artifact_bytes'] for n in names),
             'artifact_hashes':{n:meta[method,n]['sha256'] for n in names},'smoke_full_logits_BF16_sha256':st['full_logits_BF16_sha256'],'protected_parameters':'UNCHANGED'})
    with bank.restored_on_exit():
        tick=time.monotonic();bank.restore()
        for n in names:budget();q=artifact('RTN',n,bank.originals[n].float().numpy());bank.install(n,q)
        stage_times['RTN_build']=time.monotonic()-tick;built_smoke('RTN');event('RTN_BUILD_COMPLETE',core_bytes=sum(meta['RTN',n]['artifact_bytes'] for n in names))
        comp_order=list(np.random.default_rng(6101522).permutation(['COMP_QUARTER','COMP_HALF']))
        save(out/'execution_order.json',{'build':['RTN']+comp_order,'evaluation':['BF16']+list(np.random.default_rng(6101522).permutation(['RTN','COMP_QUARTER','COMP_HALF'])),'seed':6101522,'native_group_order':['qkv','o','gate_up','down']})
        terms=['A_norm2','C_norm2','B_norm2','B0_norm2','deltaB_norm2','dot_A_B0','dot_A_C','dot_B0_C','dot_R0_C','dot_R0minusC_deltaB','SSE_before','SSE_after','reference_SSE','fixedB_SSE','continuous_SSE','T_inherited','T_cancellation','T_cont','T_requant','Delta']
        for method in comp_order:
            tick=time.monotonic();bank.restore();prefix_hash=digest(b'P2_ORIGINAL_UNBUILT');alpha=ALPHA[method]
            for block in range(28):
                before_block=prefix_hash
                for g,ns in groups(block):
                    budget();ct=time.monotonic();stage[0]='CAL_CAPTURE_'+method
                    for d in cal:
                        _,_,_,buf=forward(d,ns);check_group_inputs(buf,ns);putarray(zloc(method,d,block,g),buf[ns[0]])
                    capture_seconds=time.monotonic()-ct;del buf
                    X=np.concatenate([np.load(xloc(d,block,g)).astype('f8') for d in cal],axis=1);Z=np.concatenate([np.load(zloc(method,d,block,g)).astype('f8') for d in cal],axis=1)
                    assert X.shape==Z.shape and X.shape[1]==8192
                    fct=factor(X,Z,cal);fit_counts[method]+=1
                    xhash=digest(json.dumps([sha(xloc(d,block,g)) for d in cal]).encode());zhash=digest(json.dumps([sha(zloc(method,d,block,g)) for d in cal]).encode())
                    if block==0 and g=='qkv':assert np.array_equal(X,Z),'first qkv inherited error must be zero'
                    outputs=[];group_fit=[]
                    for n in ns:
                        w=bank.originals[n].float().numpy().astype('f8');K,diag=solve(w,fct,row_chunk=128)
                        kp=art/method/'K'/(n+'.npy');putarray(kp,K);q=artifact(method,n,w+alpha*K);q0=codec.decode((art/'RTN'/(n+'.int4')).read_bytes())[0]
                        if block==0 and g=='qkv':assert not np.count_nonzero(K) and meta[method,n]['sha256']==meta['RTN',n]['sha256']
                        dt=time.monotonic();chunks=[]
                        for lo in range(0,len(w),128):
                            m=decompose.measure(w[lo:lo+128],X,Z,alpha*K[lo:lo+128],q[lo:lo+128],q0[lo:lo+128]);assert m['identity_relative_gap']<=1e-10;chunks.append(m)
                        total={k:math.fsum(m[k] for m in chunks) for k in terms};scale=max(1,total['SSE_before'],total['SSE_after'],abs(total['T_cont']),abs(total['T_requant']))
                        gap=total['Delta']-total['T_inherited']-total['T_cancellation']-total['T_requant'];assert abs(gap)/scale<=1e-10
                        metric_seconds+=time.monotonic()-dt
                        record={'method':method,'alpha':alpha,'block':block,'group':g,'name':n,'CAL_documents':32,'CAL_tokens':8192,'X_group_sha256':xhash,'Z_group_sha256':zhash,
                                'K_sha256':sha(kp),'lambda':fct['lambda'],'factor_id':f'{method}/b{block}_{g}','factor_diagnostics':fct['diagnostics'],'fit':diag,
                                'decomposition_identity_relative_gap':abs(gap)/scale,'max_chunk_identity_relative_gap':max(m['identity_relative_gap'] for m in chunks),
                                'baseline_interpretation':'local Q(W) on this method current Z, not separate full RTN trajectory',**total}
                        append(out/'CAL_diagnostics.jsonl',record);group_fit.append({'name':n,'K_sha256':record['K_sha256'],'artifact_sha256':meta[method,n]['sha256']});outputs.append((n,q))
                    for n,q in outputs:bank.install(n,q)
                    prefix_hash=digest(json.dumps([meta[method,n]['sha256'] for n in names if (method,n) in meta]).encode())
                    append(out/'build_ledger.jsonl',{'method':method,'block':block,'group':g,'source_sha256':source_hash,'spec_sha256':spec_hash,'input_manifest_sha256':input_hash,
                         'CAL_only':True,'CAL_token_hashes':[d['token_hash'] for d in cal],'X_group_sha256':xhash,'Z_group_sha256':zhash,'prefix_after_sha256':prefix_hash,
                         'group_members':group_fit,'factorizations':1,'shared_factor_projections':len(ns),'capture_seconds':capture_seconds,'factor_diagnostics':fct['diagnostics'],'status':'COMPLETE'})
                    del X,Z,fct,outputs,K,w,q,q0,chunks
                    event('GROUP_COMPLETE',method=method,block=block,group=g,projections=len(ns))
                save(out/'checkpoints'/method/f'block{block:02d}.json',{'method':method,'block':block,'prefix_before_block_sha256':before_block,'prefix_after_block_sha256':prefix_hash,
                     'execution_source_sha256':source_hash,'spec_sha256':spec_hash,'input_manifest_sha256':input_hash,'complete_projection_count':7*(block+1),'resume':'not used; require all identities/previous prefix to match before any future reuse'})
            stage_times[method+'_build']=time.monotonic()-tick;built_smoke(method);event('METHOD_BUILD_COMPLETE',method=method,projections=196)
        assert all(sum(meta[method,n]['artifact_bytes'] for n in names)==spec['core_artifacts_total_bytes_expected_per_method'] for method in ALPHA)
        assert all(meta['RTN',n]['artifact_bytes']==meta['COMP_QUARTER',n]['artifact_bytes']==meta['COMP_HALF',n]['artifact_bytes'] for n in names)
        assert not any(k.startswith('EVAL') or k.startswith('SHIFT') for k in counts)
        frozen={'frozen_utc':utc(),'methods':{method:{'projections':196,'core_bytes':sum(meta[method,n]['artifact_bytes'] for n in names),'artifact_hashes':{n:meta[method,n]['sha256'] for n in names},'alpha':ALPHA[method]} for method in ALPHA},
                'execution_source_sha256':source_hash,'spec_sha256':spec_hash,'input_manifest_sha256':input_hash,'artifact_records_sha256':sha(out/'artifacts.jsonl'),'build_ledger_sha256':sha(out/'build_ledger.jsonl'),
                'fresh_forward_count_at_freeze':0,'fit_factorizations':dict(fit_counts),'projection_K_fits':392,'RTN_K_fits':0,'shared_factors':'within each method/group only; QUARTER/HALF built independently',
                'BF16_SMOKE_full_logits_sha256':smoke0['full_logits_BF16_sha256']}
        save(out/'artifact_frozen.json',frozen);event('ALL_ARTIFACTS_FROZEN',fresh_forwards=0)
        frozen_sha=sha(out/'artifact_frozen.json');eval_order=json.loads((out/'execution_order.json').read_text())['evaluation'];refs={}
        for method in eval_order:
            assert sha(out/'artifact_frozen.json')==frozen_sha;bank.restore();lt=time.monotonic()
            if method!='BF16':
                for n in names:
                    p=art/method/(n+'.int4');assert sha(p)==meta[method,n]['sha256'];q=codec.decode(p.read_bytes())[0];bank.install(n,q)
            reload_time=time.monotonic()-lt;assert protected()==protected_start
            for split in ['EVAL-ID','SHIFT']:
                tick=time.monotonic();stage[0]=split+'_'+method;rows=[]
                for d in [d for d in fresh if d['split']==split]:
                    st,arg,sel,_=forward(d,(),True);refpath=pr/'evaluation_reference'/d['document_id_hash'];mt=time.monotonic()
                    if method=='BF16':
                        putarray(refpath/'selected.npy',sel);putarray(refpath/'argmax.npy',arg);refs[d['document_id_hash']]=st
                        kl=0.;flip=0
                    else:
                        lp=torch.log_softmax(torch.from_numpy(sel).cuda(),-1);rlp=torch.log_softmax(torch.from_numpy(np.load(refpath/'selected.npy')).cuda(),-1)
                        kl=float((rlp.exp()*(rlp-lp)).sum());flip=int((arg!=np.load(refpath/'argmax.npy')).sum())
                    metric_seconds+=time.monotonic()-mt
                    row=dict(run_id=out.name,stage='P2_FULL_MODEL_FRESH_EVALUATION',method=method,alpha=ALPHA.get(method),split=split,document_id_hash=d['document_id_hash'],token_hash=d['token_hash'],
                         model_revision=REVISION,execution_source_sha256=source_hash,spec_sha256=spec_hash,input_manifest_sha256=input_hash,artifact_frozen_sha256=frozen_sha,
                         artifact_projection_count=196 if method!='BF16' else 0,projection_core_bytes=frozen['methods'][method]['core_bytes'] if method!='BF16' else block_bytes,
                         sampled_full_vocab_KL_sum=kl,KL_position_count=16,flip_count=flip,**st,status='COMPLETE')
                    rows.append(row)
                dest=out/'metrics'/f'{method}_{split}.jsonl';tmp=dest.with_suffix('.partial');dest.parent.mkdir(exist_ok=True)
                if dest.exists() or tmp.exists():raise FileExistsError(dest)
                for row in rows:append(tmp,row)
                tmp.rename(dest);stage_times[method+'_'+split]=time.monotonic()-tick;event('EVALUATION_COMPLETE',method=method,split=split,documents=len(rows))
            append(out/'evaluation_reload.jsonl',{'method':method,'reload_seconds':reload_time,'all196_stored_weights_installed':method!='BF16'})
    assert bank.unchanged() and protected()==protected_start
    save(out/'completion.json',{'status':'P2_CORE_EVALUATED','artifact_files':588,'K_matrices':392,'factorizations':dict(fit_counts),'raw_evaluation_rows':320,'expected_raw_rows':320,
         'full_model_forward_counts':dict(counts),'total_full_model_forward_calls':sum(counts.values()),'stage_seconds_inclusive':dict(stage_times,model_load=load_time),
         'metric_seconds_inside_build_evaluation':metric_seconds,'worker_internal_wall_seconds':time.monotonic()-start,'restored_original_model':True,'protected_parameters':'UNCHANGED',
         'fresh_forward_count_at_freeze':0,'artifact_frozen_sha256':frozen_sha,'GPU_busy_time':'NOT_MEASURED','fresh_process_reload':'PENDING_SEPARATE_WORKER','P3':'NOT_RUN_SCOPE','remote_publication':False})
    event('CORE_EVALUATED',rows=320,full_model_forwards=sum(counts.values()),wall=time.monotonic()-start)
if __name__=='__main__':main()
