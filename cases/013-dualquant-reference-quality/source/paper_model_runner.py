"""Sequential complete-model reference quality, long DEV and local-agent pilot."""
from __future__ import annotations
import argparse,datetime,gc,hashlib,json,math,time,traceback,os,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer,LogitsProcessor
from reference_int4 import QuantizedWeight
from paper_int4 import solve,replace_projection,PROJECTIONS,serialize_fast,fast_ledger

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.partial')
    temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)
def append(p,x):
    with Path(p).open('a') as f:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
def sync():torch.cuda.synchronize()
class FiniteScores(LogitsProcessor):
    def __call__(self,input_ids,scores):
        if not bool(torch.isfinite(scores).all()):raise ValueError('generation logits nonfinite; not converted to valid tokens')
        return scores
def json_result(text):
    """Parser only: no answer correction. Permit code fences/explanatory prefix."""
    dec=json.JSONDecoder()
    for i,c in enumerate(text):
        if c=='{':
            try:
                x,_=dec.raw_decode(text[i:])
                if isinstance(x,dict):return x
            except ValueError:pass
    return None
def score_long(text,task):
    obj=json_result(text)
    return {'parse_success':obj is not None,'exact_success':bool(obj and str(obj.get('answer'))==task['answer'] and sorted(obj.get('evidence',[]))==sorted(task['evidence'])),'parsed':obj}

class Worker:
    def __init__(self,run,model_key,method,output_override=None):
        self.run=run;self.key=model_key;self.method=method
        self.man=json.loads((run/'experiment_manifest.json').read_text());self.identity=self.man['models'][model_key]
        self.out=Path(output_override) if output_override is not None else run/'raw'/model_key/method;self.out.mkdir(parents=True,exist_ok=True)
        if (self.out/'summary.json').exists():raise FileExistsError('completed method exists; use new result directory')
        self.start=time.monotonic();self.convert_seconds=0;self.projection_rows=[]
        torch.set_num_threads(2);torch.manual_seed(610132)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        self.tokenizer=AutoTokenizer.from_pretrained(self.identity['path'],local_files_only=True)
        t=time.monotonic()
        self.model=AutoModelForCausalLM.from_pretrained(self.identity['path'],local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
        self.load_seconds=time.monotonic()-t
        self.base_parameter_bytes=sum(p.numel()*p.element_size() for p in self.model.parameters())
        self.base_shared_embedding=self.model.model.embed_tokens.weight.data_ptr()==self.model.lm_head.weight.data_ptr()
        self.original_small_hashes={n:hashlib.sha256(p.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest() for n,p in self.model.named_parameters() if 'norm' in n}
        self.model.requires_grad_(False)

    def apply_artifacts(self):
        if self.method=='BF16':return
        art_method='RTN' if self.method=='RTN' else 'D16'
        directory=self.run/'artifacts'/self.key/art_method;directory.mkdir(parents=True,exist_ok=True)
        module_names=[n for n,m in self.model.named_modules() if isinstance(m,torch.nn.Linear) and n.split('.')[-1] in PROJECTIONS]
        expected=self.model.config.num_hidden_layers*7
        if len(module_names)!=expected:raise RuntimeError(f'projection coverage {len(module_names)} vs {expected}')
        final_manifest=[];t=time.monotonic()
        for index,name in enumerate(module_names):
            layer=self.model.get_submodule(name);path=directory/(name+'.rpi4')
            stage=time.monotonic()
            if path.exists():
                data=path.read_bytes();q=QuantizedWeight.deserialize(data);action='LOAD_EXISTING'
                if self.method=='D4' and not (directory/'manifest.json').exists():raise ValueError('incomplete shared D16 artifact')
            else:
                if self.method=='D4':raise FileNotFoundError('D4 requires frozen D16 weight artifact')
                gpu_w=layer.weight.detach().to('cuda')
                q,trace=solve(gpu_w,'rtn' if self.method=='RTN' else 'row_column',15)
                del gpu_w;sync();data=serialize_fast(q)
                temp=path.with_suffix('.partial');temp.write_bytes(data);temp.replace(path)
                loaded=QuantizedWeight.deserialize(path.read_bytes())
                if not torch.equal(q.codes,loaded.codes) or not torch.equal(q.weight_pre(),loaded.weight_pre()) or not torch.equal(q.beta_inv,loaded.beta_inv):raise ValueError('artifact roundtrip mismatch')
                for record in trace:append(self.run/'solver_trace.jsonl',dict(record,model=self.key,layer=name,method=art_method))
                action='CONVERT_SAVE_RELOAD';q=loaded
            # Restored representation, never a live exact original cache.
            original=layer.weight.detach().float()
            effective=q.weight_hat()
            err=(original-effective).double().square().sum().item();energy=original.double().square().sum().item()
            row={'layer':name,'shape':list(q.shape),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'action':action,'seconds':time.monotonic()-stage,'weight_error_sse':err,'weight_energy':energy,'normalized_weight_mse':err/energy,'scale_dtype':str(q.dtype),'ledger':fast_ledger(q,data)}
            final_manifest.append(row);self.projection_rows.append(row)
            replace_projection(self.model,name,q,self.method=='D4')
            del q,data,original,effective;gc.collect()
            if index%14==0 or index==len(module_names)-1:print(json.dumps({'stage':'artifacts','model':self.key,'method':self.method,'projection':index+1,'total':len(module_names)}),flush=True)
        self.convert_seconds=time.monotonic()-t
        if self.method!='D4' and not (directory/'manifest.json').exists():atomic(directory/'manifest.json',{'model':self.identity,'method':art_method,'projection_count':len(final_manifest),'files':final_manifest,'all_codes_bit_exact_reloaded':True,'weights_modified_on_disk':False,'total_payload_bytes':sum(x['bytes'] for x in final_manifest)})
        for n,p in self.model.named_parameters():
            if n in self.original_small_hashes and hashlib.sha256(p.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()!=self.original_small_hashes[n]:raise ValueError('norm changed')
        now_shared=self.model.model.embed_tokens.weight.data_ptr()==self.model.lm_head.weight.data_ptr()
        if now_shared!=self.base_shared_embedding:raise ValueError('embedding tie changed')

    @torch.inference_mode()
    def quality(self):
        self.model.to('cuda');torch.cuda.reset_peak_memory_stats();sync()
        toks=np.load(self.run/f'inputs/wiki_{self.key}.npy');positions=self.man['wiki']['kl_positions']
        ref=self.run/'reference_logits'/self.key;ref.mkdir(exist_ok=True)
        # Actual original full512forward first; separate reduced LM-head path audit.
        ids=torch.tensor(toks[0,:512],device='cuda',dtype=torch.long).unsqueeze(0)
        capture={};hooks=[]
        for name in ['model.layers.0.self_attn.q_proj','model.layers.0.self_attn.o_proj','model.layers.0.mlp.gate_proj','model.layers.0.mlp.down_proj']:
            hooks.append(self.model.get_submodule(name).register_forward_hook(lambda m,a,y,n=name:capture.__setitem__(n,y[:,:16].detach().cpu())))
        sync();t=time.monotonic();full=self.model(ids,use_cache=False).logits;sync();secs=time.monotonic()-t
        if not bool(torch.isfinite(full).all()):raise ValueError('512full logits nonfinite')
        hidden=self.model.model(ids,use_cache=False).last_hidden_state
        check=self.model.lm_head(hidden[:,-16:])
        # Same model functions, only selects requested tokens at original boundary.
        maxdiff=float((check-full[:,-16:]).float().abs().max());flips=int((check.argmax(-1)!=full[:,-16:].argmax(-1)).sum())
        for h in hooks:h.remove()
        layer_errors={}
        for n,y in capture.items():
            rp=ref/(n+'.npy')
            if self.method=='BF16':np.save(rp,y.view(torch.uint16).numpy())
            else:
                orig=torch.from_numpy(np.load(rp).copy()).view(torch.bfloat16).float();diff=y.float()-orig
                layer_errors[n]={'relative_mse':float(diff.double().square().sum()/orig.double().square().sum()),'max_abs':float(diff.abs().max())}
        atomic(self.out/'forward512.json',{'finite':True,'seconds':secs,'reduced_lm_head_max_abs_diff':maxdiff,'argmax_mismatch_count':flips,'exact_parity':maxdiff==0,'layer_output_errors':layer_errors,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
        if maxdiff>0.25 or flips>0:raise ValueError('reduced LM head failed frozen parity allowance maxabs0.25 and argmax0')
        del full,hidden,check,ids,capture;gc.collect();torch.cuda.empty_cache()
        records=[]
        for w,ids_np in enumerate(toks):
            sync();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
            ids=torch.tensor(ids_np,device='cuda',dtype=torch.long).unsqueeze(0)
            hidden=self.model.model(ids,use_cache=False).last_hidden_state[0]
            if not bool(torch.isfinite(hidden).all()):raise ValueError(f'window{w}hidden nonfinite')
            losses=[];preds=[];selected=[]
            for start in range(0,2047,64):
                stop=min(start+64,2047);logits=self.model.lm_head(hidden[start:stop]).float()
                if not bool(torch.isfinite(logits).all()):raise ValueError(f'window{w}logits nonfinite')
                loss=F.cross_entropy(logits,ids[0,start+1:stop+1],reduction='none')
                losses.append(loss.cpu().numpy());preds.append(logits.argmax(-1).cpu().numpy())
                for p in positions:
                    if start<=p<stop:selected.append((p,logits[p-start].to(torch.bfloat16).cpu()))
                del logits,loss
            loss_np=np.concatenate(losses);pred_np=np.concatenate(preds)
            selected=sorted(selected,key=lambda x:x[0]);sel=torch.stack([x[1] for x in selected])
            kl_sum=0.;flip_count=0
            if self.method=='BF16':
                np.save(ref/f'window{w:02d}.npy',sel.view(torch.uint16).numpy());np.save(ref/f'argmax{w:02d}.npy',pred_np)
            else:
                baseline=torch.from_numpy(np.load(ref/f'window{w:02d}.npy').copy()).view(torch.bfloat16)
                for j in range(0,32,8):
                    lp=F.log_softmax(baseline[j:j+8].cuda().float(),-1);lq=F.log_softmax(sel[j:j+8].cuda().float(),-1)
                    kl_sum+=float((lp.exp()*(lp-lq)).sum());del lp,lq
                flip_count=int((pred_np!=np.load(ref/f'argmax{w:02d}.npy')).sum())
            sync()
            row={'window':w,'input_sha256':hashlib.sha256(ids_np.tobytes()).hexdigest(),'valid_nll_tokens':2047,'nll_sum':float(loss_np.astype(np.float64).sum()),'kl_sum':kl_sum,'kl_positions':32,'argmax_flips':flip_count,'wall_seconds':time.monotonic()-t,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'finite':True}
            append(self.out/'quality_windows.jsonl',row);records.append(row)
            np.savez(self.out/f'tokens{w:02d}.npz',nll=loss_np,argmax=pred_np)
            del hidden,ids,selected,sel;gc.collect()
            if w%4==0:print(json.dumps({'stage':'quality','model':self.key,'method':self.method,'window':w+1,'total':len(toks)}),flush=True)
        total=sum(x['valid_nll_tokens'] for x in records);nll=sum(x['nll_sum'] for x in records)/total
        return {'windows':len(records),'valid_tokens':total,'nll':nll,'ppl':math.exp(nll),'forward_kl':sum(x['kl_sum'] for x in records)/(len(records)*32),'argmax_flip_fraction':sum(x['argmax_flips'] for x in records)/total,'wall_seconds':sum(x['wall_seconds'] for x in records),'peak_allocated_bytes':max(x['peak_allocated_bytes'] for x in records),'peak_reserved_bytes':max(x['peak_reserved_bytes'] for x in records)}

    @torch.inference_mode()
    def generate(self,ids,max_new):
        ids=torch.tensor(ids,device='cuda',dtype=torch.long).unsqueeze(0)
        if ids.shape[1]+max_new>self.man['generation']['context_limit'] and max_new==192:raise ValueError('agent CONTEXT_LIMIT')
        sync();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
        result=self.model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=max_new,use_cache=True,logits_to_keep=1,pad_token_id=self.tokenizer.eos_token_id,return_dict_in_generate=False,logits_processor=[FiniteScores()])
        sync();new=result[0,ids.shape[1]:].cpu().tolist();seconds=time.monotonic()-t
        raw=self.tokenizer.decode(new,skip_special_tokens=False);text=self.tokenizer.decode(new,skip_special_tokens=True)
        return {'text':text,'raw_text':raw,'output_token_ids':new,'input_tokens':ids.shape[1],'output_tokens':len(new),'input_token_sha256':hashlib.sha256(ids.cpu().numpy().astype('<i4').tobytes()).hexdigest(),'seconds':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'status':'COMPLETE'}

    def short_generation(self):
        for i,prompt in enumerate(json.loads((self.run/'inputs/greedy_prompts.json').read_text())):
            ids=self.tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
            result=self.generate(ids,64);append(self.out/'short_generation.jsonl',dict(result,id=i,prompt=prompt))

    def long_dev(self):
        tasks=json.loads((self.run/'inputs/long_tasks.json').read_text());result=[]
        bytes_token=2*self.model.config.num_hidden_layers*self.model.config.num_key_value_heads*self.model.config.head_dim*2
        for length in (8192,16384,32768):
            # Operational policy, no quality-based seed/method selection.
            torch.cuda.empty_cache();free,total=torch.cuda.mem_get_info()
            workspace=536870912
            if length==32768:
                previous=[r for r in result if r.get('length')==16384 and r.get('status')=='COMPLETE']
                if previous:
                    # Actual16K peak includes method-specific FP32 activation buffers.
                    observed=max(r['peak_allocated_bytes'] for r in previous)
                    static=torch.cuda.memory_allocated()
                    previous_workspace=max(0,observed-static-bytes_token*(16384+96))
                    workspace=max(workspace,int(previous_workspace*2.2))
            need=bytes_token*(length+96)+1073741824+workspace
            if length==32768 and free<need:
                append(self.out/'long_dev.jsonl',{'length':length,'status':'NOT_RUN_MEMORY_POLICY','free_bytes':free,'required_additional_bytes_estimate':need,'kv_theory_bytes_per_token':bytes_token});continue
            ids_all=np.load(self.run/f'inputs/long_{length}.npy');okay=True
            for i,task in enumerate(tasks):
                try:
                    res=self.generate(ids_all[i].tolist(),96);res.update(score_long(res['text'],task))
                    res.update({'id':task['id'],'family':task['family'],'length':length,'gold':task['answer'],'evidence':task['evidence'],'kv_theoretical_bytes':bytes_token*(res['input_tokens']+res['output_tokens'])})
                except torch.cuda.OutOfMemoryError as exc:
                    res={'id':task['id'],'length':length,'status':'OOM','error':str(exc)};okay=False
                    append(self.out/'long_dev.jsonl',res)
                    torch.cuda.empty_cache()
                    return result
                append(self.out/'long_dev.jsonl',res);result.append(res)
                print(json.dumps({'stage':'long_dev','method':self.method,'length':length,'id':task['id'],'success':res['exact_success']}),flush=True)
            if not okay:break
        return result

    def tool(self,name,args,fixture,config_path):
        params={'read_document':[],'read_record':[],'query_table':[],'calculate':['a','b'],'apply_fixture_patch':['enabled','limit'],'validate_fixture_test':[]}
        if name not in params or set(args)!=set(params[name]):raise ValueError('unknown tool/incorrect argument keys')
        if name=='read_document':return fixture['document']
        if name=='read_record':return fixture['record']
        if name=='query_table':return fixture['updates']
        if name=='calculate':
            if any(type(args[k]) not in (int,float) or not math.isfinite(args[k]) for k in ['a','b']):raise ValueError('invalid numerical argument')
            return {'product':args['a']*args['b']}
        if name=='apply_fixture_patch':
            if type(args['enabled'])!=bool or type(args['limit'])!=int:raise ValueError('incorrect patch argument types')
            atomic(config_path,args);return {'applied':True,'config':args}
        return {'valid':json.loads(config_path.read_text())==fixture['expected_config']}

    def agent_dev(self):
        tools=json.loads((self.run/'inputs/tool_schema.json').read_text());tasks=json.loads((self.run/'inputs/agent_tasks.json').read_text());out=[]
        for task in tasks:
            for mode in ['REPLAY','FREE']:
                t=time.monotonic();config=self.out/(task['id']+'-'+mode+'-config.json');atomic(config,task['fixture']['config'])
                initial=json.loads((self.run/'inputs/agent_initial_messages.json').read_text())[task['id']]
                messages=list(initial);calls=0;tool_calls=0;parse_errors=0;tool_errors=0;retries=0;model_seconds=0;tool_seconds=0;total_input=0;total_output=0;events=[];success=False;status='CALL_LIMIT'
                if mode=='REPLAY':
                    # Common gold-defined trajectory, never copied from BF16 output.
                    replay=[(task['replay_tool'],task['replay_args'])]
                    if task['family']=='combine':replay.append(('calculate',{'a':task['fixture']['record']['quantity'],'b':3}))
                    if task['family']=='fixture_patch':replay.append(('validate_fixture_test',{}))
                    for name,args in replay:
                        answer=self.tool(name,args,task['fixture'],config)
                        messages.extend([{'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':name,'arguments':args}}]},{'role':'tool','name':name,'content':json.dumps(answer)}])
                    messages.append({'role':'user','content':'All required tool results are above. Return the requested final JSON only.'})
                max_calls=1 if mode=='REPLAY' else 8
                for turn in range(max_calls):
                    if time.monotonic()-t>900:status='TIMEOUT';break
                    ids=self.tokenizer.apply_chat_template(messages,tools=tools,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
                    if len(ids)+192>24576:status='CONTEXT_LIMIT';break
                    res=self.generate(ids,192);calls+=1;total_input+=len(ids);total_output+=res['output_tokens'];model_seconds+=res['seconds'];events.append(res)
                    raw=res['raw_text'];blocks=[]
                    if '<tool_call>' in raw:
                        for segment in raw.split('<tool_call>')[1:]:
                            chunk=segment.split('</tool_call>')[0];obj=json_result(chunk)
                            if obj is None:parse_errors+=1
                            else:blocks.append(obj)
                    if blocks and mode=='FREE':
                        accepted=[];answers=[]
                        for obj in blocks:
                            if tool_calls>=6:status='TOOL_LIMIT';break
                            tool_calls+=1;name=obj.get('name');args=obj.get('arguments');tool_start=time.monotonic()
                            try:
                                if not isinstance(args,dict):raise ValueError('arguments not an object')
                                answer=self.tool(name,args,task['fixture'],config)
                            except (ValueError,TypeError) as e:tool_errors+=1;answer={'error':str(e)}
                            tool_seconds+=time.monotonic()-tool_start
                            accepted.append({'type':'function','function':{'name':name,'arguments':args}});answers.append((name,answer));events.append({'tool':name,'arguments':args,'result':answer})
                        if not accepted:break
                        messages.append({'role':'assistant','content':'','tool_calls':accepted})
                        messages.extend({'role':'tool','name':name,'content':json.dumps(answer)} for name,answer in answers)
                        continue
                    final=json_result(res['text'])
                    if final is None:parse_errors+=1;status='PARSE_ERROR'
                    else:
                        success=all(final.get(k)==v for k,v in task['gold'].items())
                        if task['family']=='fixture_patch' and mode=='FREE':success=success and json.loads(config.read_text())==task['fixture']['expected_config'] and any(e.get('tool')=='validate_fixture_test' and e.get('result',{}).get('valid') for e in events)
                        status='SUCCESS' if success else 'INCORRECT_FINAL'
                    break
                record={'id':task['id'],'family':task['family'],'mode':mode,'method':self.method,'success':success,'status':status,'model_calls':calls,'tool_calls':tool_calls,'parse_errors':parse_errors,'tool_errors':tool_errors,'retry_count':retries,'total_input_tokens':total_input,'total_output_tokens':total_output,'model_seconds':model_seconds,'tool_seconds':tool_seconds,'wall_seconds':time.monotonic()-t,'gold':task['gold'],'events':events,'cache_policy':'OFF across turns; KV within each request','role':'REFERENCE_PILOT_DEV'}
                append(self.out/'agent_dev.jsonl',record);out.append(record)
                print(json.dumps({'stage':'agent','method':self.method,'id':task['id'],'mode':mode,'status':status}),flush=True)
        return out

    def run_all(self):
        self.apply_artifacts();quality=self.quality();self.short_generation()
        long=[];agents=[]
        if self.key=='4b':long=self.long_dev();agents=self.agent_dev()
        mandatory_long_done=sum(r['length'] in (8192,16384) for r in long)
        summary={'model':self.key,'method':self.method,'quality':quality,'base_parameter_bytes':self.base_parameter_bytes,'load_seconds':self.load_seconds,'artifact_conversion_or_loading_seconds':self.convert_seconds,'projection_count':len(self.projection_rows),'weight_normalized_mse':sum(x['weight_error_sse'] for x in self.projection_rows)/sum(x['weight_energy'] for x in self.projection_rows) if self.projection_rows else 0,'artifact_bytes':sum(x['bytes'] for x in self.projection_rows),'long_dev_completed':len(long),'mandatory_long_dev_completed':mandatory_long_done,'long_dev_successes':sum(x['exact_success'] for x in long),'agent_replay_successes':sum(x['success'] for x in agents if x['mode']=='REPLAY'),'agent_free_successes':sum(x['success'] for x in agents if x['mode']=='FREE'),'wall_seconds':time.monotonic()-self.start,'status':'COMPLETE' if self.key=='06b' or mandatory_long_done==24 else 'PARTIAL_LONG_DEV','actual_runtime':'dequantized BF16/SDPA reference; no native INT4 kernel'}
        atomic(self.out/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--model',choices=['06b','4b'],required=True);p.add_argument('--method',choices=['BF16','RTN','D16','D4'],required=True);args=p.parse_args()
    start=time.monotonic()
    try:Worker(Path(args.run),args.model,args.method).run_all()
    except Exception as exc:
        traceback.print_exc();atomic(Path(args.run)/'raw'/args.model/args.method/'failure.json',{'status':'FAILED','exception':type(exc).__name__,'message':str(exc),'wall_seconds':time.monotonic()-start,'traceback':traceback.format_exc()});sys.exit(1)
