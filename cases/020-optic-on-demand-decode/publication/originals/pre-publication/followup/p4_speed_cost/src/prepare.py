"""P4A immutable historical analysis and fixed P4 plan preparation (CPU only)."""
import csv,datetime,itertools,sys,subprocess,os,platform
from base import *
def csvout(path,rows):
 with path.open('w',newline='') as f:
  fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def legacy():
 run=CASE/'results/20261026-1235';ledger=read(run/'worker_ledger.json');rows=[];index={};memory=[]
 for w in ledger['workers']:
  if w['kind']!='cell':continue
  r=read(run/w['raw']);c=r['cell'];key=(c['shape'],c['M'],c['U'],c['block'],c['method']);index[key]=r
  for metric,values in [('host',r['timing']['host_us']),('event_bracket_per_call',r['timing']['event_bracket_per_call_us'])]:rows.append({**{k:c[k] for k in ['shape','M','U','condition','block','method']},'metric':metric,'unit':'us','K':1 if metric=='host' else r['timing']['event_K'],**stats(values)})
  unique={tuple(x['storage_key']):x['storage_bytes'] for x in r['resident_inventory']};module=sum(unique.values());ids=sum(x['payload_bytes'] for x in r['id_inventory']);baseline=r['baseline']['allocated'];warm=r['resident_after_warmup']['allocated'];memory.append({**{k:c[k] for k in ['shape','M','U','block','method']},'baseline_allocated':baseline,'baseline_reserved':r['baseline']['reserved'],'setup_allocated_peak':r['setup']['allocated_peak'],'cold_allocated_peak':r['cold_peak']['allocated'],'warm_allocated':warm,'module_storage':module,'ID_payload':ids,'warm_minus_module_ID':warm-module-ids,'warm_minus_module_ID_baseline':warm-module-ids-baseline,'after_load_allocated_saved':False,'after_output_release_separate_saved':False})
 paired=[]
 for key in index:
  if key[-1]!='A':continue
  s,M,U,b,_=key;rr={m:index[(s,M,U,b,m)] for m in 'ABC'};st={m:{k:stats(rr[m]['timing'][v]) for k,v in [('host','host_us'),('event','event_bracket_per_call_us')]} for m in 'ABC'}
  for num,den in [('C','B'),('B','A')]:
   q={'shape':s,'M':M,'U':U,'block':b,'comparison':num+'/'+den}
   for met in ['host','event']:
    for stat in ['mean','median','p95','min','IQR']:q[met+'_'+stat+'_ratio']=st[num][met][stat]/max(st[den][met][stat],1e-12)
   paired.append(q)
 cb=[x for x in paired if x['comparison']=='C/B'];hw=[x for x in cb if x['host_median_ratio']<1];ew=[x for x in cb if x['event_mean_ratio']<1];summary={'historical_raw_workers':len(index),'host_samples':sum(r['metric']=='host' for r in rows)*100,'event_brackets':sum(r['metric']=='event_bracket_per_call' for r in rows)*100,'host_median_C_wins':len(hw),'event_mean_C_wins':len(ew),'common_wins':sum(x['host_median_ratio']<1 and x['event_mean_ratio']<1 for x in cb),'direction_disagreements':sum((x['host_median_ratio']<1)!=(x['event_mean_ratio']<1) for x in cb),'host_wins_U_equals_M':sum(x['M']==x['U'] for x in hw),'all3_host_winning_conditions':sum(all(x['host_median_ratio']<1 for x in cb if (x['shape'],x['M'],x['U'])==(s,M,U)) for s,M,U in {(x['shape'],x['M'],x['U']) for x in cb}),'baseline_allocated_unique':sorted({x['baseline_allocated'] for x in memory}),'warm_residual_excluding_baseline_range':[min(x['warm_minus_module_ID_baseline'] for x in memory),max(x['warm_minus_module_ID_baseline'] for x in memory)],'interpretation':'32MiB already existed at post common-CUDA/library baseline before IDs/artifact. Original raw cannot pinpoint pre-library allocation or stack owner. after-load and separate after-output-release allocated snapshots were not saved; setup peak is not an endpoint.','outside_analysis':{'path':'/home/mskim/Codex_main/case020-results-analysis-20261026-1418','found':pathlib.Path('/home/mskim/Codex_main/case020-results-analysis-20261026-1418/VALIDATION.json').exists()},'new_GPU_observations':False}
 csvout(ROOT/'summary/legacy_statistics.csv',rows);csvout(ROOT/'summary/legacy_paired.csv',paired);csvout(ROOT/'summary/legacy_memory_stages.csv',memory);dump(ROOT/'summary/legacy_decision.json',summary);print('P4A',summary)
def prepare():
 original=read(CASE/'provenance/input_manifest.json');inputs={}
 for label,M,U in [('m1',1,1),('unique2048',2048,2048),('repeat2048',2048,512)]:
  item=next(x for x in original if x['shape']=='LARGE' and x['M']==M and x['U']==U);p=CASE/item['path'];ids=np.load(p,allow_pickle=False);validate_ids(ids);assert info(p)['sha256']==item['sha256'];inputs[label]={'path':str(p),'M':M,'U':U,**info(p),'payload_sha256':sha(ids.tobytes())}
 artifacts={}
 for w in [32,128,644]:
  if w!=32:
   meta,a=fixture(w);np.savez(artifact(w),metadata=np.asarray(json.dumps(meta)),**a)
  meta,a=arrays(artifact(w));artifacts[str(w)]={'path':str(artifact(w)),**info(artifact(w)),'metadata':meta,'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'bytes':v.nbytes,'sha256':sha(v.tobytes()),'finite':bool(np.isfinite(v).all()),'min':float(v.min()),'max':float(v.max())} for k,v in a.items()}}
 # Scalar-loop versus chunk/vector FP64 over a small slice of each stored width.
 scalar=[]
 for w in [32,128,644]:
  meta,a=arrays(artifact(w));ids=np.array([0,32767],np.int64);vector=next(ref_chunks(meta,a,ids))[1][:,:3,:16];G=meta['n_groups'];codes=a['token_codes_packed_uint8'].reshape(32768,G);cb=a['codebooks_fp16'].astype(np.float64);W=a['decoder_weight_fp16'].astype(np.float64);b=a['decoder_bias_fp16'].astype(np.float64);loop=np.empty((2,3,16))
  for i,t in enumerate(ids):
   z=[float(cb[g,int(codes[t,g]),d]) for g in range(G) for d in range(4)]
   for l in range(3):
    for h in range(16):loop[i,l,h]=float(b[l,h])+sum(z[j]*float(W[l,h,j]) for j in range(w))
  np.testing.assert_allclose(loop,vector,rtol=1e-10,atol=1e-12);scalar.append({'Z':w,'shape':[2,3,16],'max_abs':float(abs(loop-vector).max()),'rtol':1e-10,'atol':1e-12})
 dump(ROOT/'provenance/cpu_reference_checks.json',scalar)
 jobs=[]
 for width in [32,128,644]:
  cs=list(inputs) if width==32 else ['unique2048','repeat2048'];blocks=6 if width==32 else 3
  for ci,condition in enumerate(cs):
   pass_perms=list(itertools.permutations('HDE'));np.random.default_rng(np.random.SeedSequence([61026024,width,ci])).shuffle(pass_perms)
   if blocks==3:pass_perms=[tuple('HDE'),tuple('DEH'),tuple('EHD')];offset=int(np.random.default_rng(width+ci).integers(3));pass_perms=pass_perms[offset:]+pass_perms[:offset]
   for block in range(blocks):
    names=list('ABCS' if width==32 and condition=='unique2048' else 'ABC');job={'id':f'lat_Z{width}_{condition}_b{block}','kind':'timing','phase':'core' if width==32 else 'width','Z':width,'condition':condition,'block':block,'methods':names,'pass_order':list(pass_perms[block]),'orders':{'initial':balanced_orders(names,20,[61026024,width,ci,block,0])}}
    for pi,pas in enumerate('HDE'):job['orders'][pas+'_warm']=balanced_orders(names,20,[61026024,width,ci,block,pi+1]);job['orders'][pas]=balanced_orders(names,{'H':120,'D':48,'E':24}[pas],[61026024,width,ci,block,pi+4])
    jobs.append(job)
 for phase in ['core','width']:
  these=[x for x in jobs if x['phase']==phase];np.random.default_rng(np.random.SeedSequence([61026025,0 if phase=='core' else 1])).shuffle(these)
  if phase=='core':ordered=these
  else:ordered+=these
 life=[{'id':f'life_{m}_N{N}_b{b}','kind':'lifecycle','phase':'lifecycle','Z':32,'condition':'repeat2048','method':m,'N':N,'block':b} for m in 'ABC' for N in [128,1024,4096] for b in range(3)];np.random.default_rng(61026026).shuffle(life);ordered+=life
 for w in [32,128,644]:ordered.append({'id':f'preflight_Z{w}','kind':'preflight','phase':'preflight','Z':w})
 names=list('ABC');smoke={'id':'smoke_Z644','kind':'smoke','phase':'smoke','Z':644,'condition':'unique2048','block':0,'methods':names,'pass_order':list('HDE'),'orders':{'initial':balanced_orders(names,20,[61026024,999,0])}}
 for pi,pas in enumerate('HDE'):smoke['orders'][pas+'_warm']=balanced_orders(names,20,[61026024,999,pi+1]);smoke['orders'][pas]=balanced_orders(names,6,[61026024,999,pi+4])
 ordered.append(smoke)
 for m in 'ABC':ordered.append({'id':'memory_'+m,'kind':'memory','phase':'memory','Z':32,'condition':'repeat2048','method':m})
 assert len([j for j in ordered if j['kind']=='timing'])==30 and sum(len(j['methods']) for j in ordered if j['kind']=='timing')==96
 dump(ROOT/'provenance/plans.json',ordered);dump(ROOT/'provenance/inputs.json',inputs);dump(ROOT/'provenance/artifacts.json',artifacts)
 budget=read(ROOT/'provenance/baseline_inventory.json');protocol={'run_id':'p4-20261026-1451','designed_after_observing':'Original final P2 and diagnostic raw plus outside analysis already observed. Followup, not retroactive preregistration.','upstream_sha':UPSTREAM,'original_run_id':'20261026-1235','prior_gpu_workers':budget['original_workers'],'prior_gpu_wall_exact':budget['original_gpu_wall_exact'],'P4_wall_cap_seconds':budget['p4_cap_seconds'],'global_gpu_cap_seconds':1800,'additional_allocated_cap_bytes':2*1024**3,'disk_target_bytes':256*1024**2,'disk_hard_cap_bytes':512*1024**2,'CUDA_CACHE_MAXSIZE':128*1024**2,'total_wall_start_KST':'2026-10-26T14:51:53+09:00','stop_GPU_before_KST':'2026-10-26T17:21:53+09:00','time_budget_note':'3h target with30min documentation reserve; no adaptive grid expansion.','runtime':'Read-only /home/mskim/inference-lab/.venv-baseline/bin/python; Python>=3.12,Torch2.13+cu130, no installs','arithmetic':{'TF32':False,'fp16_reduced_precision_reduction':False,'bf16_reduced_precision_reduction':False,'fp16_accumulation':False,'autograd':False,'eager':True,'dtype':'BF16','output':'[M,L,H] raw unscaled all layers','dense_chunk':256},'correctness':{'fp64_rtol':1e-10,'fp64_atol':1e-12,'BF16_relative_L2_max':.01,'zero_max_abs':1e-6,'finite_required':True,'preflight_separate_process':True,'cold_main_not_used_for_correctness':True},'timing':{'initial_warm_each_method':20,'pass_start_warm_each_method':20,'H_rounds':120,'H_K':1,'D_rounds':48,'D_K':1,'D_host_includes_event_records_and_wait':True,'E_brackets':24,'E_K':16,'p95':'linear','trim':False,'pass_order':'core six permutations across6blocks; width cyclic Latin across3blocks','method_order':'all6 or24 permutations, seeded shuffled cycle; full multiples in all main passes'},'fixture_seeds':{'new_width':'SeedSequence([61026020,4,Z]); original distribution/scale retained','ordering':61026024,'worker_order':61026025,'lifecycle_order':61026026},'planned':{'core_workers':18,'width_workers':12,'timing_workers':30,'method_blocks':96,'lifecycle_workers':27,'preflight_workers':3,'smoke_workers':1,'memory_max_workers':3,'profile_workers':0},'lifecycle':{'N':[128,1024,4096],'methods':list('ABC'),'blocks':3,'no_warmup':True,'session':'before artifact read through final owned-tensor cleanup+sync; IDs H2D included','loop':'N real fn+completion synchronize; first call included; output immediately deleted; no events/sample copies','worker_wall':'parent spawn to exit, includes import/context/setup/teardown','costs_not_additive':'first call part of total_loop; setup and final cleanup part of session; session part of worker wall'},'memory':{'enabled_if_remaining_seconds_at_least':30,'history':'installed torch.cuda.memory._record_memory_history(state, all stacks,max_entries512), public memory_snapshot(include_traces=False)','owner_rule':'stage correlation only unless stack identifies allocation owner; otherwise UNATTRIBUTED','main_latency_shared_residency':'not method-isolated memory; never merge with P2 peaks'},'decision':{'core':'median C/B H block-median ratios<=0.90 and>=5/6 faster -> REPEATABLE_IN_THIS_CORE_TEST candidate','width':'same threshold and3/3 faster -> EXPLORATORY_WIDTH_SIGNAL','small_difference':'median ratio within(0.90,1.10); descriptive not significance or implementation gate','host_event_common_win_required':False},'retry_limit':1}
 dump(ROOT/'PROTOCOL.json',protocol);dump(ROOT/'provenance/cpu_environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'affinity':sorted(os.sched_getaffinity(0)),'numpy':np.__version__,'sys_path':sys.path,'CUDA_context':'No Torch imported','history_API_inspection':'Installed Torch2.13 exposes state/all stack history max_entries and memory_snapshot; CPU inspection did not initialize CUDA.','stable_docs':'stable redirects2.14; installed2.13 retained; version2.13 web pages unavailable, actual installed docstrings consulted.'})
 print('prepared',len(ordered),'jobs;30timing/96method-block/27lifecycle')
if __name__=='__main__':legacy();prepare()
