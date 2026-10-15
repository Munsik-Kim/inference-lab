"""Case014 controlled full-model quality intervention. No fitting on DEV/EVAL.
Offline local assets required. Does not download, install, or run P2/P3.
"""
import argparse,contextlib,json,time,os,sys,subprocess,platform
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from common import save,sha,digest,append
from codec import encode,decode,write_reload
from fit import fit_torch
from decompose import measure
ALPHAS=[0,.25,.5,.75,1];BLOCKS=[4,12,20];PROJS=['o_proj','down_proj']
POSITIONS=np.array([0,16,33,50,67,84,101,118,135,152,169,186,203,220,237,254])

def target(l,p):return f'model.layers.{l}.'+('self_attn.' if p=='o_proj' else 'mlp.')+p

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--private',required=True);ap.add_argument('--output',required=True);ap.add_argument('--smoke-only',action='store_true');ap.add_argument('--budget-seconds',type=float,default=5400);args=ap.parse_args()
 case=Path(__file__).resolve().parents[1];private=Path(args.private);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 start=time.monotonic();run_id=out.name;torch.set_num_threads(4);torch.manual_seed(6101515)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
 assert Path(args.model).name=='c1899de289a04d12100db370d81485cdf75e47ca','unregistered snapshot revision'
 manifest=json.loads((private/'inputs/input_manifest.json').read_text());assert sha(private/'inputs/documents.private.json')==manifest['private_token_file_sha256'],'changed input tokens'
 docs=json.loads((private/'inputs/documents.private.json').read_text());cal=[d for d in docs if d['split']=='CAL-FIT'];smoke=next(d for d in docs if d['split']=='SMOKE');main_docs=[d for d in docs if d['split']!='SMOKE']
 code={str(p.relative_to(case)):sha(p) for p in sorted((case/'scripts').glob('*.py'))};freeze=digest(json.dumps(code,sort_keys=True).encode())
 save(out/'code_freeze.json',{'files':code,'code_freeze_sha256':freeze,'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=case).strip()})
 def checkbudget():
  if time.monotonic()-start>args.budget_seconds:raise TimeoutError('conservative P1 worker wall budget exhausted')
 def event(stage,**kw):
  append(out/'events.jsonl',dict(stage=stage,worker_elapsed=time.monotonic()-start,**kw));print(stage,kw,flush=True)
 def device():
  free,total=torch.cuda.mem_get_info();return {'device_free_bytes':free,'device_total_bytes':total,'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved()}
 event('CREATED');assert torch.cuda.is_available(),'CUDA not accessible'
 model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
 assert model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr(),'tie lost'
 modules={n:m for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)}
 originals={n:m.weight.detach().cpu().clone() for n,m in modules.items() if n.startswith('model.layers.')}
 names=[target(l,p) for l in BLOCKS for p in PROJS]
 shape={n:list(modules[n].weight.shape) for n in names}
 env={'python':sys.version,'torch':torch.__version__,'transformers':__import__('transformers').__version__,'numpy':np.__version__,'GPU':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'CUDA':torch.version.cuda,'TF32':False,'BF16_reduced_precision_reduction':False,'use_cache':False,'batch':1,'attention':'sdpa','head':'native full256-position LM head, all arms','initial_memory':device(),'model_config':model.config.to_dict(),'target_shapes':shape,'model_weight_whole_file_hash':'LOCAL_IDENTITY_VERIFIED; full hash in p1_r2 authorization; no independently fetched official digest'}
 save(private/'runtime.private.json',dict(env,python_path=sys.executable,model_path=args.model));save(out/'runtime_manifest.json',env)
 ledger={n:{'W_BF16_bytes':np.prod(s)*2,'W_FP64_temporary_bytes':np.prod(s)*8,'CAL_X_Z_FP64_bytes':2*s[1]*8192*8,'Gram_FP64_bytes':s[1]**2*8,'Cholesky_FP64_bytes':s[1]**2*8,'INT4_core_bytes':36+s[0]*((s[1]+63)//64)*4+s[0]*((s[1]+63)//64)*32} for n,s in shape.items()}
 save(out/'memory_ledger.json',{'per_target':{n:{k:int(v) for k,v in row.items()} for n,row in ledger.items()},'full_logits_BF16_bytes':256*model.config.vocab_size*2,'selected_logits_FP32_bytes':16*model.config.vocab_size*4,'model_parameter_BF16_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'storage_policy':'user-authorized existing WSL snapshot read-only; no D origin prerequisite; no staging copy'})
 traces=private/'traces';art=private/'artifacts';ref={};prefix_art={};artifact_meta={};fits={};prefix_metrics={};forward_counts={};forward_stage=['SMOKE'];prefix_verified=set()
 def location(kind,d,n,part):return traces/kind/d['document_id_hash']/(n.replace('.','_')+'_'+part+'.npy')
 def putarray(p,a):
  p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():raise FileExistsError(p)
  np.save(p,a,allow_pickle=False)
 @torch.inference_mode()
 def forward(d,capture):
  checkbudget();buf={};hooks=[]
  def hook(n):
   def h(m,inp,y):
    if n in buf:raise ValueError('projection called more than once')
    buf[n]=(inp[0].detach().clone().float().cpu().numpy()[0].T,y.detach().clone().float().cpu().numpy()[0].T)
   return h
  try:
   for n in capture:hooks.append(model.get_submodule(n).register_forward_hook(hook(n)))
   ids=torch.tensor([d['input_ids']],device='cuda',dtype=torch.long);mask=torch.ones_like(ids);pos=torch.arange(256,device='cuda').unsqueeze(0)
   torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.monotonic()
   result=model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False,return_dict=True)
   logits=result.logits[0];assert logits.shape==(256,model.config.vocab_size)
   if not torch.isfinite(logits).all():raise ValueError('nonfinite logits')
   f=logits[:-1].float();loss=torch.nn.functional.cross_entropy(f,ids[0,1:],reduction='sum');arg=f.argmax(-1).cpu().numpy()
   sel=logits[torch.tensor(POSITIONS,device='cuda')].float();torch.cuda.synchronize()
   stats={'NLL_sum':float(loss),'valid_token_count':255,'wall_seconds':time.monotonic()-t,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),**device()}
   forward_counts[forward_stage[0]]=forward_counts.get(forward_stage[0],0)+1
   append(out/'forward_calls.jsonl',dict(stage=forward_stage[0],document_id_hash=d['document_id_hash'],split=d['split'],**stats))
   return stats,arg,sel.cpu().numpy(),buf
  finally:
   for h in hooks:h.remove()
 def restore_all():
  for n,w in originals.items():modules[n].weight.data.copy_(w.to('cuda'))
 def setprefix(depth):
  restore_all()
  for n,m in modules.items():
   if n in originals and int(n.split('.')[2])<depth:
    if n not in prefix_art:
     p=art/'prefix'/f'{n}.int4';w,_,_,st=write_reload(p,originals[n].float().numpy());prefix_art[n]=p
    else:w=decode(prefix_art[n].read_bytes())[0]
    m.weight.data.copy_(torch.from_numpy(w).to('cuda',torch.bfloat16))
  if depth not in prefix_verified:
   unchanged=[n for n in originals if int(n.split('.')[2])>=depth]
   assert all(torch.equal(modules[n].weight.detach().cpu(),originals[n]) for n in unchanged)
   assert model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr()
   event('PREFIX_BOUNDARY_VERIFIED',depth=depth,prefix_projection_count=7*depth,unchanged_projection_count=len(unchanged))
   prefix_verified.add(depth)
 def teacher_run(ds):
  restore_all()
  for d in ds:
   stats,arg,sel,buf=forward(d,names)
   for n,(x,y) in buf.items():putarray(location('teacher',d,n,'X'),x);putarray(location('teacher',d,n,'Y'),y)
   p=traces/'reference'/d['document_id_hash'];p.mkdir(parents=True,exist_ok=True);putarray(p/'selected.npy',sel);putarray(p/'argmax.npy',arg)
   ref[d['document_id_hash']]=stats;append(out/'teacher.jsonl',dict(document_id_hash=d['document_id_hash'],token_hash=d['token_hash'],split=d['split'],**stats))
 def capture_prefix(l,ds):
  setprefix(l)
  for d in ds:
   stats,arg,sel,buf=forward(d,[target(l,p) for p in PROJS]);base=ref[d['document_id_hash']]
   lp=torch.log_softmax(torch.from_numpy(sel).cuda(),-1);rlp=torch.log_softmax(torch.from_numpy(np.load(traces/'reference'/d['document_id_hash']/'selected.npy')).cuda(),-1)
   stats.update(sampled_full_vocab_KL_sum=float((rlp.exp()*(rlp-lp)).sum()),KL_position_count=16,flip_count=int((arg!=np.load(traces/'reference'/d['document_id_hash']/'argmax.npy')).sum()))
   append(out/'prefix_only.jsonl',dict(block=l,split=d['split'],document_id_hash=d['document_id_hash'],token_hash=d['token_hash'],**stats))
   for n,(z,y) in buf.items():putarray(location('prefix',d,n,'Z'),z)
 def local_arrays(d,n):
  return np.load(location('teacher',d,n,'X')).astype(np.float64),np.load(location('prefix',d,n,'Z')).astype(np.float64)
 def fit_target(n):
  w=originals[n].float().numpy().astype(np.float64);x=np.concatenate([local_arrays(d,n)[0] for d in cal],axis=1);z=np.concatenate([local_arrays(d,n)[1] for d in cal],axis=1)
  k,lam,diag=fit_torch(w,x,z);fits[n]={'lambda':lam,'K_path':str(art/f'{n}_K.npy'),'diagnostics':diag};putarray(art/f'{n}_K.npy',k)
  # no-prefix is a separate CAL fit using the actual X; reject a hidden inherited correction.
  kn,ln,dn=fit_torch(w,x,x);assert np.count_nonzero(kn)==0
  no_prefix_hashes=[digest(encode(w+a*kn)[0]) for a in ALPHAS];assert len(set(no_prefix_hashes))==1
  save(out/(n+'.fit.json'),{'lambda':lam,'diagnostics':diag,'no_prefix':{'delta_nonzero_count':0,'alpha_artifact_hashes':no_prefix_hashes,'inherited_error_zero':True,'quantization_error_zero':False}})
  sums=[]
  for alpha in ALPHAS:
   p=art/'candidates'/f'{n}.alpha{alpha:g}.int4';q,c,s,st=write_reload(p,w+alpha*k);sse=float(np.sum((w@x-q.astype(np.float64)@z)**2))
   artifact_meta[(n,alpha)]={'path':p,'sha256':sha(p),'stats':st,'CAL_raw_SSE':sse}
   sums.append((sse,alpha))
  assert len({m['stats']['artifact_bytes'] for (nn,a),m in artifact_meta.items() if nn==n})==1
  best=min(sums)[1];event('FIT_TARGET_FROZEN',target=n,selected_alpha=best,lambda_value=lam)
  return {'target':n,'selected_alpha':best,'alpha_CAL_SSE':{str(a):v for v,a in sums},'selected_artifact_sha256':artifact_meta[(n,best)]['sha256'],'K_sha256':sha(art/f'{n}_K.npy'),'lambda':lam}
 # Separate smoke, no CAL/DEV/EVAL reads used to set tolerances.
 teacher_run([smoke]);restore_all();tr,ta,ts,tbuf=forward(smoke,names)
 teacher_reinstall_exact=all(np.array_equal(tbuf[n][0],np.load(location('teacher',smoke,n,'X'))) and np.array_equal(tbuf[n][1],np.load(location('teacher',smoke,n,'Y'))) for n in names) and np.array_equal(ts,np.load(traces/'reference'/smoke['document_id_hash']/'selected.npy')) and np.array_equal(ta,np.load(traces/'reference'/smoke['document_id_hash']/'argmax.npy'))
 assert teacher_reinstall_exact
 setprefix(4);stats,_,_,buf=forward(smoke,[target(4,'o_proj')]);sx,sy=np.load(location('teacher',smoke,target(4,'o_proj'),'X')),np.load(location('teacher',smoke,target(4,'o_proj'),'Y'))
 n=target(4,'o_proj');z,_=buf[n];w=originals[n].float().numpy();k,lam,_=fit_torch(w,sx,z)
 q0=decode(encode(w)[0])[0];q=decode(encode(w+k)[0])[0];m=measure(w,sx,z,k,q,q0)
 # Repeat unmodified path for actual runtime parity.
 st2,_,_,b2=forward(smoke,[n]);repeat_equal=np.array_equal(z,b2[n][0]) and np.array_equal(buf[n][1],b2[n][1]);assert repeat_equal
 saved_q,_,_,_=write_reload(art/'smoke.int4',w+k);np.testing.assert_array_equal(saved_q,q)
 modules[n].weight.data.copy_(torch.from_numpy(saved_q).to('cuda',torch.bfloat16))
 cst,_,_,cb=forward(smoke,[n]);assert np.array_equal(z,cb[n][0])
 smoke_actual=measure(w,sx,z,k,q,q0,sy,cb[n][1],modules[n].weight.detach().float().cpu().numpy());modules[n].weight.data.copy_(originals[n].to('cuda'))
 tolerance={'identity_relative':1e-10,'sign_relative_to_SSE_scale':1e-9,'runtime_repeat_exact':repeat_equal,'teacher_reinstall_exact':teacher_reinstall_exact,'SMOKE_identity_relative_gap':m['identity_relative_gap'],'frozen_before_MAIN':True,'arithmetic_residual':'measured separately; not used as algebra identity gap'}
 assert m['identity_relative_gap']<=tolerance['identity_relative'];save(out/'smoke.json',dict(stats,tolerance=tolerance,model_shapes=shape,actual_candidate=smoke_actual,forward_calls_planned=3865,rough_forward_wall_only_estimate_seconds=3865*stats['wall_seconds'],estimate_excludes_CPU_fit_pack_local_analysis=True));event('SPEC_FROZEN',code_freeze=freeze)
 if args.smoke_only:return
 selections=[]
 forward_stage[0]='CAL_TEACHER';teacher_run(cal)
 fitorder=list(BLOCKS);np.random.default_rng(6101515).shuffle(fitorder)
 for l in fitorder:
  forward_stage[0]='CAL_PREFIX';capture_prefix(l,cal)
  for p in PROJS:selections.append(fit_target(target(l,p)))
 save(out/'fit_frozen.json',{'selected':selections,'code_freeze_sha256':freeze,'input_manifest_sha256':sha(private/'inputs/input_manifest.json'),'selection_data':'CAL-FIT only','frozen_before_non_CAL_forward':True});event('FIT_FROZEN')
 artifact_rows=[]
 for (n,a),meta in artifact_meta.items():artifact_rows.append(dict(target=n,alpha=a,sha256=meta['sha256'],**meta['stats']))
 save(out/'artifact_manifest.json',{'candidates':artifact_rows,'prefix_count':len(prefix_art),'external_private_artifact_not_in_review_zip':True})
 other=[d for d in main_docs if d['split']!='CAL-FIT'];forward_stage[0]='HELDOUT_TEACHER';teacher_run(other)
 for l in fitorder:
  forward_stage[0]='HELDOUT_PREFIX';capture_prefix(l,other)
 event('EVAL_RUNNING');forward_stage[0]='GRID'
 order=[(l,p,a) for l in BLOCKS for p in PROJS for a in ALPHAS];np.random.default_rng(6101515).shuffle(order)
 completed=[]
 for l,p,alpha in order:
  checkbudget();n=target(l,p);setprefix(l);w=originals[n].float().numpy().astype(np.float64);k=np.load(art/f'{n}_K.npy');delta=alpha*k
  meta=artifact_meta[(n,alpha)];q,c,s=decode(meta['path'].read_bytes());q0,c0,s0=decode(artifact_meta[(n,0)]['path'].read_bytes());code_change=float(np.mean(c!=c0));scale_rel=np.abs(s.astype('f8')-s0)/s0
  cell=out/'cells'/f'b{l}_{p}_a{alpha:g}.jsonl';cell.parent.mkdir(exist_ok=True);tmp=cell.with_suffix('.partial')
  if cell.exists() or tmp.exists():raise FileExistsError(cell)
  cellstart=time.monotonic();modules[n].weight.data.copy_(torch.from_numpy(q).to('cuda',torch.bfloat16));qrun=modules[n].weight.detach().float().cpu().numpy()
  try:
   for d in main_docs:
    stats,arg,sel,buf=forward(d,[n]);x,z=local_arrays(d,n);input_match=np.array_equal(z,buf[n][0]);assert input_match,'target intervention changed its own input'
    y=np.load(location('teacher',d,n,'Y'));meas=measure(w,x,z,delta,q,q0,y,buf[n][1],qrun)
    assert meas['identity_relative_gap']<=tolerance['identity_relative'],'decomposition failure'
    rsel=np.load(traces/'reference'/d['document_id_hash']/'selected.npy');lp=torch.log_softmax(torch.from_numpy(sel).cuda(),-1);rlp=torch.log_softmax(torch.from_numpy(rsel).cuda(),-1)
    kl=float((rlp.exp()*(rlp-lp)).sum());flip=int((arg!=np.load(traces/'reference'/d['document_id_hash']/'argmax.npy')).sum());base=ref[d['document_id_hash']]
    row=dict(run_id=run_id,stage='P1',source_commit=env.get('source_commit',subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=case).strip()),code_freeze_sha256=freeze,model_revision='c1899de289a04d12100db370d81485cdf75e47ca',tokenizer_revision='c1899de289a04d12100db370d81485cdf75e47ca',quantizer_variant='RTN_RIDGE_COMPENSATION',bits=4,group_size=64,scale_dtype='FP32',solver_dtype='CPU FP64',rounding='nearest ties-to-even',split=d['split'],document_id_hash=d['document_id_hash'],token_hash=d['token_hash'],offset=d['offset'],calibration_seed=6101515,block=l,projection=p,alpha=alpha,lambda_value=fits[n]['lambda'],input_path='common teacher X and hybrid prefix Z',artifact_sha256=meta['sha256'],input_identical_across_alpha=input_match,**meas,**stats,NLL_teacher_sum=base['NLL_sum'],sampled_full_vocab_KL_sum=kl,KL_position_count=16,flip_count=flip,code_change_rate=code_change,scale_change_mean=float(scale_rel.mean()),scale_change_max=float(scale_rel.max()),**meta['stats'],status='COMPLETE')
    append(tmp,row)
   tmp.rename(cell);completed.append(cell.name);event('CELL_COMPLETE',cell=cell.name,rows=len(main_docs),wall=time.monotonic()-cellstart)
  finally:modules[n].weight.data.copy_(originals[n].to('cuda'))
 # Optional lower priority diagnostics: actual FP32 continuous weight branch, cast boundaries explicit.
 class Continuous(torch.nn.Module):
  def __init__(self,w,b):super().__init__();self.weight=torch.nn.Parameter(torch.from_numpy(w.astype('f4')).cuda(),requires_grad=False);self.bias=None if b is None else torch.nn.Parameter(b.detach().float(),requires_grad=False)
  def forward(self,x):return torch.nn.functional.linear(x.float(),self.weight,self.bias).to(x.dtype)
 forward_stage[0]='CONTINUOUS'
 for l in BLOCKS:
  setprefix(l)
  for p in PROJS:
   if time.monotonic()-start>args.budget_seconds-200:event('CONTINUOUS_NOT_RUN_BUDGET',block=l,projection=p);continue
   n=target(l,p);old=model.get_submodule(n);par_name,attr=n.rsplit('.',1);par=model.get_submodule(par_name);w=originals[n].float().numpy().astype('f8');k=np.load(art/f'{n}_K.npy')
   try:
    setattr(par,attr,Continuous(w+k,old.bias))
    for d in main_docs:
     st,arg,sel,buf=forward(d,[n]);assert np.array_equal(np.load(location('prefix',d,n,'Z')),buf[n][0]);append(out/'continuous.jsonl',dict(block=l,projection=p,alpha=1,split=d['split'],document_id_hash=d['document_id_hash'],runtime='FP32 Wstar and input; output cast BF16; original bias',NLL_teacher_sum=ref[d['document_id_hash']]['NLL_sum'],**st))
   finally:setattr(par,attr,old)
 save(out/'p1_completion.json',{'status':'COMPLETED_FULL' if len(completed)==30 else 'PARTIAL','cells':completed,'rows_per_cell':96,'expected_grid_rows':2880,'teacher_documents':97,'worker_wall_seconds':time.monotonic()-start,'full_generation':0,'forward_counts':forward_counts,'total_full_model_forward_calls':sum(forward_counts.values()),'GPU_busy_time':'NOT_MEASURED','controls':'PREFIX_ONLY shared by targets; no-prefix CAL fit; FP32 continuous alpha1','peak':device(),'code_freeze_sha256':freeze})
 event('REVIEW_DATA_READY')
if __name__=='__main__':main()
