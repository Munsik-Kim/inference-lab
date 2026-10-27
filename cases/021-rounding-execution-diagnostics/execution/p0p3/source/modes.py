"""Diagnostic forward, same-input local replay and separate cost workers."""
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from common import CASE,read,write,file_row,verify_files
from layout import bf16_float,metrics

def readiness(a):
    gate=read(Path(a.private)/'readiness.json')
    if gate['source_bundle_sha256']!=read(a.freeze)['bundle_sha256'] or not gate['p1_ready']:raise ValueError('PREFLIGHT_SOURCE_MISMATCH')
    if gate['input_manifest']!=file_row(CASE/'inputs/manifest.json'):raise ValueError('PREFLIGHT_INPUT_MISMATCH')
    if a.split=='EVAL':
        if not a.selection:raise ValueError('EVAL_REQUIRES_SELECTION_FREEZE')
        selection=read(a.selection);verify_files(selection['analysis_freeze'])
        if selection['gpu_source_bundle_sha256']!=read(a.freeze)['bundle_sha256'] or selection['input_manifest']!=file_row(CASE/'inputs/manifest.json') or not selection['dev_complete'] or len(selection['layers'])!=2 or 'tau_abs' not in selection or 'intervention' not in selection:raise ValueError('EVAL_SELECTION_GATE_FAILED')
    return gate

def block_bits(t):return t.detach().cpu().contiguous().view(torch.uint16).numpy().copy()

def full(a,rt,loc,repeat=False):
    readiness(a);manifest=read(CASE/'inputs/manifest.json');path,rmode=a.condition.split('_')
    started=time.perf_counter();model=rt.load_model(loc,path);rt.sync();load_seconds=time.perf_counter()-started
    docs=manifest['repeat_documents'] if repeat else manifest['document_order'][a.split]
    lengths=[512] if repeat else [128,512];cycles=range(3) if repeat else range(1)
    raw=Path(a.private)/('repeats' if repeat else 'main')/a.split/a.condition
    raw.mkdir(parents=True,exist_ok=False)
    scratch=Path(a.private)/'scratch'/a.split/a.condition
    if not repeat:scratch.mkdir(parents=True,exist_ok=False)
    rows=[];block_values=[];context={};lm_rows=[]
    def hook(m,inp,y):block_values.append(block_bits(y[0] if isinstance(y,tuple) else y))
    def capture_context(m,args,kwargs):
        context.clear();context['input']=block_bits(args[0] if args else kwargs['hidden_states'])
        for key,value in kwargs.items():
            if isinstance(value,torch.Tensor):context[key]=block_bits(value) if value.dtype==torch.bfloat16 else value.detach().cpu().numpy()
            elif key=='position_embeddings':
                context['cos']=block_bits(value[0]);context['sin']=block_bits(value[1])
            elif value is None or isinstance(value,(bool,int,float,str)):context[key]=value
    handles=[]
    if not repeat:
        handles=[layer.register_forward_hook(hook) for layer in model.model.layers]
        if a.condition=='D0_R0':handles.append(model.model.layers[0].register_forward_pre_hook(capture_context,with_kwargs=True))
    handles.append(model.lm_head.register_forward_pre_hook(lambda m,args:lm_rows.append(list(args[0].shape))))
    with rt.rounding(model,rmode):
        for doc in docs:
            for n in lengths:
                args=rt.inputs(doc,n);positions=manifest['selected_positions'][str(n)]
                for cycle in cycles:
                    block_values.clear();lm_rows.clear();rt.sync();start=time.perf_counter()
                    output=model(**args).logits;rt.sync();diagwall=time.perf_counter()-start
                    if list(output.shape)!=[1,n,151936] or lm_rows!=[[1,n,1024]]:raise RuntimeError('FULL_LM_HEAD_GEOMETRY_CHANGED')
                    selected=output[0,positions,:];name=f'{doc}_n{n}_r{cycle}'
                    saved=rt.save_tensor(raw/(name+'_logits.npy'),selected)
                    if not torch.isfinite(selected).all():raise RuntimeError('NONFINITE_LOGITS')
                    row={'document_id':doc,'length':n,'condition':a.condition,'split':a.split,'repeat':cycle,'positions':positions,'next_token_ids':args['input_ids'][0,torch.tensor(positions,device='cuda')+1].cpu().tolist(),'logits':saved,'full_logits_dtype':str(output.dtype),'full_logits_shape':list(output.shape),'lm_head_input_shapes':lm_rows.copy(),'diagnostic_forward_host_seconds':diagwall,'timing_eligible':False,'input_sha256':file_row(CASE/'inputs'/(doc+'.npy'))['sha256']}
                    if not repeat:
                        if len(block_values)!=28:raise RuntimeError('BLOCK_HOOK_COUNT')
                        blocks=np.concatenate(block_values,axis=0) # [layer, token, hidden]
                        np.save(scratch/(name+'_blocks.npy'),blocks,allow_pickle=False)
                        np.save(raw/(name+'_block_selected.npy'),blocks[:,positions,:],allow_pickle=False)
                        row['block_selected']=file_row(raw/(name+'_block_selected.npy'))
                        if a.condition=='D0_R0' and n==512:
                            ctx=Path(a.private)/'contexts'/a.split/doc;ctx.mkdir(parents=True,exist_ok=False)
                            metadata={}
                            for key,value in context.items():
                                if isinstance(value,np.ndarray):np.save(ctx/(key+'.npy'),value,allow_pickle=False);metadata[key]={'file':key+'.npy','dtype':str(value.dtype),'shape':list(value.shape)}
                                else:metadata[key]=value
                            write(ctx/'metadata.json',metadata)
                    rows.append(row);write(raw/(name+'.json'),row)
                    del output,selected;block_values.clear();gc.collect()
    for h in handles:h.remove()
    write(a.output/'forward_receipt.json',{'mode':'repeat' if repeat else 'main','split':a.split,'condition':a.condition,'count':len(rows),'load_seconds':load_seconds,'raw':str(raw),'source_bundle_sha256':read(a.freeze)['bundle_sha256'],'input_manifest':file_row(CASE/'inputs/manifest.json'),'memory':rt.memory(),'tensor_inventory':rt.tensor_inventory(model)})
    print(json.dumps({'mode':'repeat' if repeat else 'main','condition':a.condition,'forwards':len(rows)}))

def to_gpu(bits):return torch.from_numpy(np.asarray(bits).copy()).view(torch.bfloat16).cuda()

def local(a,rt,loc):
    readiness(a);selection=read(a.selection);layers=selection['layers'];manifest=read(CASE/'inputs/manifest.json')
    docs=manifest['document_order'][a.split];rows=[];primitive=[]
    # One model, no second full model. Only selected block is evaluated.
    model=rt.load_model(loc,'D0');base=Path(a.private)/'local'/a.split;base.mkdir(parents=True,exist_ok=False)
    for idx in layers:
        block=model.model.layers[idx]
        for doc in docs:
            ctx=Path(a.private)/'contexts'/a.split/doc
            hidden=to_gpu(np.load(ctx/f'layer{idx}_input.npy',allow_pickle=False))
            cos=to_gpu(np.load(ctx/'cos.npy',allow_pickle=False));sin=to_gpu(np.load(ctx/'sin.npy',allow_pickle=False))
            meta=read(ctx/'metadata.json');mask=None
            if isinstance(meta.get('attention_mask'),dict):
                arr=np.load(ctx/'attention_mask.npy',allow_pickle=False);mask=to_gpu(arr) if arr.dtype==np.uint16 else torch.from_numpy(arr).cuda()
            kwargs={'attention_mask':mask,'position_ids':torch.arange(512,device='cuda')[None,:],'position_embeddings':(cos,sin),'use_cache':False}
            captures={}
            for mode in ['D0','D1','P']:
                rt.replace(model,loc['artifact_dir'],mode,layer_indices=[idx])
                for rounding in ['R0','R1']:
                    name=mode+'_'+rounding;values={}
                    def trace(key,t):values[key]=rt.cpu(t)
                    hooks=rt.boundary_hooks(block,trace)
                    with rt.rounding(block,rounding,trace):y=block(hidden,**kwargs)
                    for h in hooks:h.remove()
                    if not torch.isfinite(y).all():raise RuntimeError('LOCAL_NONFINITE')
                    captures[name]=values
                    chosen={}
                    positions=manifest['selected_positions']['512']
                    for key,value in values.items():
                        if key in ('cos','sin'):continue
                        # Normalize token axis only for selection, retain original shape in receipt.
                        axis=2 if value.ndim==4 and value.shape[2]==512 else 1
                        chosen[key]=np.take(value,positions,axis=axis)
                    np.savez(base/f'{doc}_layer{idx}_{name}_selected.npz',**chosen)
                    del y
            for r in ['R0','R1']:
                for ref,cand in [('D0','D1'),('D1','P'),('D0','P')]:
                    ca,cb=captures[ref+'_'+r],captures[cand+'_'+r]
                    for boundary in sorted(set(ca)&set(cb)-{'cos','sin'}):
                        rows.append({'document_id':doc,'split':a.split,'layer':idx,'rounding':r,'reference':ref,'candidate':cand,'boundary':boundary,'same_block_input':True,'input_identity':file_row(ctx/f'layer{idx}_input.npy'),'metrics':metrics(ca[boundary],cb[boundary])})
            # Identical projection values through both native and FP32-intermediate functions.
            for mode in ['D0','D1','P']:
                for boundary in ['post_rope_q','post_rope_k','block_output']:
                    primitive.append({'document_id':doc,'layer':idx,'path':mode,'boundary':boundary,'comparison':'R0_R1_same_block_input','metrics':metrics(captures[mode+'_R0'][boundary],captures[mode+'_R1'][boundary])})
            # Preserve FP64 decomposition statistics for identical local tensor coordinates.
            from primitives import vector_decomposition
            for r in ['R0','R1']:
                for boundary in ['self_attn.q_proj','self_attn.k_proj','post_rope_q','post_rope_k','block_output']:
                    rows.append({'document_id':doc,'split':a.split,'layer':idx,'rounding':r,'reference':'D0','candidate':'D1_then_P','boundary':boundary,'same_block_input':True,'decomposition':vector_decomposition(captures['D0_'+r][boundary],captures['D1_'+r][boundary],captures['P_'+r][boundary])})
            del captures,hidden,cos,sin;gc.collect()
    write(a.output/'local_metrics.json',{'rows':rows,'primitive':primitive,'block_replay_count':len(docs)*len(layers)*6,'layers':layers,'documents':docs,'source_bundle_sha256':read(a.freeze)['bundle_sha256'],'memory':rt.memory(),'auxiliary_full_models':0,'auxiliary_blocks':0,'selected_raw':str(base)})
    print(json.dumps({'block_replays':len(docs)*len(layers)*6,'split':a.split}))

def timing(a,rt,loc):
    readiness(a);manifest=read(CASE/'inputs/manifest.json');doc=manifest['document_order']['DEV'][0]
    path=a.condition;initial=rt.memory();start=time.perf_counter();model=rt.load_model(loc,path);rt.sync();load=time.perf_counter()-start
    args=rt.inputs(doc,512);rows=[];warm=[];cold={}
    # Scoped mode transitions change only Q/K norm-RoPE arithmetic. No dump hooks.
    for r in ['R0','R1']:
        with rt.rounding(model,r):
            rt.sync();t=time.perf_counter();out=model(**args).logits;rt.sync();cold[r]=time.perf_counter()-t;del out
            for i in range(3):out=model(**args).logits;del out
        warm.append({'rounding':r,'calls':3})
    rt.sync();torch.cuda.reset_peak_memory_stats();resident=rt.memory()
    for round_id in range(10):
        order=['R0','R1'] if (round_id+a.block)%2==0 else ['R1','R0']
        for r in order:
            with rt.rounding(model,r):
                start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
                rt.sync();start_ns=time.perf_counter_ns();start_event.record();out=model(**args).logits;end_event.record();end_event.synchronize();host_ns=time.perf_counter_ns()-start_ns
                event_ms=start_event.elapsed_time(end_event);del out
            rows.append({'path':path,'rounding':r,'block':a.block,'round':round_id,'order':order,'host_ns':host_ns,'event_ms':event_ms,'full_positions':512})
    write(a.output/'timing.json',{'path':path,'block':a.block,'document':doc,'samples':rows,'warmup':warm,'cold_call_seconds':cold,'setup_load_seconds':load,'initial_memory':initial,'resident_memory':resident,'timed_memory':rt.memory(),'contract':'Same-call dual timer: host includes event record and wait; event is stream elapsed, not summed kernel busy time','timed_forwards':20,'warmup_forwards':6,'cold_forwards':2,'source_bundle_sha256':read(a.freeze)['bundle_sha256']})
    print(json.dumps({'path':path,'block':a.block,'timed':20}))
