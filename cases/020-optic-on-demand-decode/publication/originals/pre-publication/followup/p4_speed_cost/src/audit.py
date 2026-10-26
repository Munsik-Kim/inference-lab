"""Independent CPU audit and aggregation of P4 evidence. Never imports Torch/worker/base."""
import pathlib,json,csv,hashlib,math,statistics,argparse,sys,datetime,itertools,os
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];CASE=ROOT.parents[1]
def read(p):return json.loads(pathlib.Path(p).read_text())
def h(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def check(ok,msg):
 if not ok:raise ValueError(msg)
def public_manifest(root):
 root=pathlib.Path(root);p=root/'PUBLIC_MANIFEST_P4.json'
 if not p.exists():return
 pub=read(p);files={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p.name!='PUBLIC_MANIFEST_P4.json'};check(files==set(pub['files']),'P4 manifest coverage mismatch')
 for rel,entry in pub['files'].items():check(h(root/rel)==entry['sha256'] and (root/rel).stat().st_size==entry['bytes'],'P4 published file changed '+rel)
def quantile(a,q):
 s=sorted(a);i=(len(s)-1)*q;j=math.floor(i);return s[j]+(s[min(j+1,len(s)-1)]-s[j])*(i-j)
def statistics_(a):
 a=list(a);check(bool(a),'empty sample');check(all(math.isfinite(x) and x>=0 for x in a),'invalid sample');return {'n':len(a),'mean_us':sum(a)/len(a),'median_us':statistics.median(a),'p95_us':quantile(a,.95),'min_us':min(a),'q25_us':quantile(a,.25),'q75_us':quantile(a,.75),'IQR_us':quantile(a,.75)-quantile(a,.25),'IQR_over_median':(quantile(a,.75)-quantile(a,.25))/max(statistics.median(a),1e-12)}
def write_csv(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for row in rows for k in row))
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def recorded_path(p,root):
 original=pathlib.Path(read(root/'provenance/baseline_inventory.json')['case_root']);p=pathlib.Path(p)
 return root.parents[1]/p.relative_to(original) if p.is_relative_to(original) else p
def storage(rows):
 seen=set();total=0
 for r in rows:
  check(math.prod(r['shape'])*r['element_size']==r['payload_bytes'],'inventory payload units');k=tuple(r['key']);check(r['unique']==(k not in seen),'inventory alias flag');check(k[2]==r['storage_bytes'],'storage key size')
  if k not in seen:total+=r['storage_bytes'];seen.add(k)
 return total
def numerical(er):
 es=er['error_sumsq'];rs=er['reference_sumsq'];ma=er['max_abs'];fi=er['finite'];check(len(es)==len(rs)==len(ma)==len(fi)==16,'all layers required')
 for a,b,c,d in zip(es,rs,ma,fi):check(d and (c<=1e-6 if b==0 else math.sqrt(a)/max(math.sqrt(b),1e-12)<=.01),'per-layer numeric gate')
 check(max(ma)<=1e-6 if sum(rs)==0 else math.sqrt(sum(es))/max(math.sqrt(sum(rs)),1e-12)<=.01,'overall numeric gate')
def clock_row(x,pas):
 check(x['K']==(16 if pas=='E' else 1),'K contract')
 if pas in ['H','D']:check(x['fence']==('device_synchronize' if pas=='H' else 'end_event_synchronize') and x['end_ns']>=x['start_ns'] and math.isclose((x['end_ns']-x['start_ns'])/1000,x['host_us'],abs_tol=1e-9),'host clock/fence/units')
 if pas=='E':check(x['host_us'] is None and math.isclose(x['event_us']*16,x['bracket_stream_us'],rel_tol=1e-12),'event bracket/K units')
def preserve(root,portable=False):
 case=root.parents[1];inv=read(root/'provenance/baseline_inventory.json');after={}
 for rel,entry in inv['files'].items():
  p=case/rel;check(p.exists() and p.stat().st_size==entry['bytes'] and h(p)==entry['sha256'],'old file changed: '+rel);after[rel]=True
 extra=[str(p.relative_to(case)) for p in case.rglob('*') if p.is_file() and str(p.relative_to(case)) not in inv['files'] and not str(p.relative_to(case)).startswith('followup/p4_speed_cost/')];check(not extra,'new file outside P4 allowlist: '+str(extra))
 external_verified=0;external_missing=[]
 for folder,files in inv['external_files'].items():
  for rel,entry in files.items():
   p=pathlib.Path(folder)/rel
   if not p.exists() and portable:external_missing.append(str(p));continue
   check(p.exists() and p.stat().st_size==entry['bytes'] and h(p)==entry['sha256'],'external original modified');external_verified+=1
 return {'original_files_verified':len(after),'extra_files_outside_allowlist':extra,'external_originals_verified':external_verified,'external_missing_portable':external_missing}
def collect(root=ROOT,portable=False):
 root=pathlib.Path(root);case=root.parents[1];protocol=read(root/'PROTOCOL.json');freeze=read(root/'provenance/freeze.json');plan=read(root/'provenance/plans.json');plans={j['id']:j for j in plan};run=root/'results'/protocol['run_id'];artifacts=read(root/'provenance/artifacts.json');inputs=read(root/'provenance/inputs.json');preservation=preserve(root,portable)
 public_manifest(root)
 for rel,v in freeze['files'].items():check((root/rel).stat().st_size==v['bytes'] and h(root/rel)==v['sha256'],'frozen source/input mismatch '+rel)
 check(hashlib.sha256(json.dumps(freeze['files'],sort_keys=True).encode()).hexdigest()==freeze['bundle_sha256'],'source bundle identity')
 for e in inputs.values():
  p=recorded_path(e['path'],root);ids=np.load(p,allow_pickle=False);check(h(p)==e['sha256'] and hashlib.sha256(ids.tobytes()).hexdigest()==e['payload_sha256'],'ID identity');check(ids.dtype==np.int64 and len(ids)==e['M'] and len(np.unique(ids))==e['U'] and ids.min()>=0 and ids.max()<32768,'ID contract')
 for width,e in artifacts.items():
  artifact_file=recorded_path(e['path'],root);check(h(artifact_file)==e['sha256'],'artifact hash');meta=e['metadata'];check(meta['latent_dim']==int(width) and meta['n_groups']==int(width)//4 and meta['group_dim']==4,'width geometry')
  check(meta['compressed_ple_format_id']=='optic_npz' and meta['latent_mode']=='discrete' and meta['code_bits']==8 and meta['codebook_size']==256 and meta['ple_shape']==[32768,4096] and meta['d_ple']==256,'artifact format contract')
  with np.load(artifact_file,allow_pickle=False) as z:
   for name,ai in e['arrays'].items():check(str(z[name].dtype)==ai['dtype'] and list(z[name].shape)==ai['shape'] and hashlib.sha256(z[name].tobytes()).hexdigest()==ai['sha256'],'array identity')
   check(z['token_codes_packed_uint8'].dtype==np.uint8,'integer code dtype')
   for name in ['codebooks_fp16','decoder_weight_fp16','decoder_bias_fp16']:check(z[name].dtype==np.float16 and np.isfinite(z[name]).all(),'stored FP16 finite values')
 # Original CPU evidence: recompute all statistics and paired comparisons, not saved booleans.
 legacy=[];legacy_index={};legacy_memory=[]
 oldrun=case/'results/20261026-1235';oldledger=read(oldrun/'worker_ledger.json')
 for w in oldledger['workers']:
  if w['kind']!='cell':continue
  r=read(oldrun/w['raw']);c=r['cell'];k=(c['shape'],c['M'],c['U'],c['block'],c['method']);legacy_index[k]={}
  for metric,key in [('host','host_us'),('event','event_bracket_per_call_us')]:
   a=r['timing'][key];check(len(a)==100,'legacy sample denominator');ss=statistics_(a);legacy_index[k][metric]=ss;legacy.append({**{x:c[x] for x in ['shape','M','U','block','method']},'metric':metric,'K':1 if metric=='host' else r['timing']['event_K'],**ss})
  unique={tuple(x['storage_key']):x['storage_bytes'] for x in r['resident_inventory']};module=sum(unique.values());ids=sum(x['payload_bytes'] for x in r['id_inventory']);legacy_memory.append({**{x:c[x] for x in ['shape','M','U','block','method']},'baseline_allocated':r['baseline']['allocated'],'warm_allocated':r['resident_after_warmup']['allocated'],'module_bytes':module,'ID_bytes':ids,'residual_after_baseline':r['resident_after_warmup']['allocated']-module-ids-r['baseline']['allocated'],'timed_allocated_peak':r['timed_memory']['allocated_peak'],'timed_reserved_peak':r['timed_memory']['reserved_peak'],'after_load_endpoint_saved':any(k in r for k in ['after_load','after_artifact_load']),'after_output_release_endpoint_saved':any(k in r for k in ['after_output_release','after_output_del'])})
 legacy_pairs=[]
 for s,M,U,b,m in legacy_index:
  if m!='A':continue
  for numerator,denominator in [('C','B'),('B','A')]:
   a=legacy_index[(s,M,U,b,numerator)];d=legacy_index[(s,M,U,b,denominator)];row={'shape':s,'M':M,'U':U,'block':b,'comparison':numerator+'/'+denominator}
   for met in ['host','event']:
    for stat in ['median_us','mean_us','p95_us','min_us','IQR_us']:row[met+'_'+stat+'_ratio']=a[met][stat]/max(d[met][stat],1e-12)
   legacy_pairs.append(row)
 legacy_cb=[x for x in legacy_pairs if x['comparison']=='C/B'];legacy_check={'host_median_wins':sum(x['host_median_us_ratio']<1 for x in legacy_cb),'event_mean_wins':sum(x['event_mean_us_ratio']<1 for x in legacy_cb),'common_wins':sum(x['host_median_us_ratio']<1 and x['event_mean_us_ratio']<1 for x in legacy_cb),'direction_disagreements':sum((x['host_median_us_ratio']<1)!=(x['event_mean_us_ratio']<1) for x in legacy_cb),'host_wins_U_equals_M':sum(x['host_median_us_ratio']<1 and x['U']==x['M'] for x in legacy_cb),'all3_winning_conditions':sum(all(x['host_median_us_ratio']<1 for x in legacy_cb if (x['shape'],x['M'],x['U'])==(s,M,U)) for s,M,U in {(x['shape'],x['M'],x['U']) for x in legacy_cb})}
 old_budget=read(case/'results/gpu_worker_budget.json');old_spent=sum(w['wall_seconds'] for w in old_budget['workers']);check(len(old_budget['workers'])==protocol['prior_gpu_workers'] and abs(old_spent-protocol['prior_gpu_wall_exact'])<1e-8,'original budget debit changed')
 records=[json.loads(s) for s in (run/'worker_ledger.jsonl').read_text().splitlines()] if (run/'worker_ledger.jsonl').exists() else [];check(len({r['job_id'] for r in records})==len(records),'duplicate GPU worker ID');new_spent=sum(r['wall_seconds'] for r in records);check(new_spent<=protocol['P4_wall_cap_seconds'] and old_spent+new_spent<=1800,'GPU budget exceeded')
 timings=[];dual=[];samples=[];life=[];memstages=[];work=[];byraw={};failed=[];preflight=[];actual_methodblocks=0
 for receipt in records:
  p=root/receipt['raw'];check(h(p)==receipt['raw_sha256'],'raw worker changed');r=read(p);j=plans[receipt['job_id']];check(receipt['source_hash']==freeze['bundle_sha256'] and receipt['config_sha256']==h(root/'PROTOCOL.json'),'worker receipt source/config');check(r.get('job',j)==j,'job/order changed')
  if receipt['exit_code'] or 'samples' not in r and j['kind'] in ['timing','smoke'] or 'lifecycle' not in r and j['kind']=='lifecycle':failed.append(j['id']);continue
  byraw[j['id']]=r;check(r['source_hash']==freeze['bundle_sha256'] and r['config_sha256']==h(root/'PROTOCOL.json'),'raw source/config identity');check(r['device']['sm']==[12,0],'SM mismatch');check(r['device']['current_stream']==r['device']['default_stream'],'default stream');check(r['policy_after']=={'TF32':False,'fp16_reduction':False,'bf16_reduction':False,'fp16_accumulation':False},'arithmetic policy');check(r['allocated_delta_peak']<=2*1024**3,'memory cap')
  if j['kind']=='preflight':
   for co in r['correctness']:
    e=inputs[co['condition']];check(co['ID_payload_sha256']==e['payload_sha256'],'preflight input identity')
    for er in [*co['methods'].values(),*co['pairs'].values()]:numerical(er)
    for m,oc in co['output_contract'].items():check(oc['dtype']=='torch.bfloat16' and oc['shape']==[e['M'],4096] and oc['finite'],'preflight output contract')
    preflight.append({'Z':j['Z'],'condition':co['condition'],'methods':list(co['methods']),'max_relative_L2':max(er['relative_L2'] for er in co['methods'].values())})
   sp=root/r['sample_output']['path'];check(h(sp)==r['sample_output']['sha256'],'preflight saved sample hash')
   # Independent per-layer FP64 expression, matching sampled BF16 output values.
   with np.load(recorded_path(artifacts[str(j['Z'])]['path'],root),allow_pickle=False) as z:
    G=j['Z']//4;codes=z['token_codes_packed_uint8'].reshape(32768,G);cb=z['codebooks_fp16'].astype(np.float64);W=z['decoder_weight_fp16'].astype(np.float64);bias=z['decoder_bias_fp16'].astype(np.float64)
   with np.load(sp,allow_pickle=False) as z:
    for co in r['correctness']:
     ids=z[co['condition']+'_ids'];latent=np.concatenate([cb[g,codes[ids,g]] for g in range(G)],axis=1);ref=np.stack([latent@W[l].T+bias[l] for l in range(16)],axis=1)
     for m in co['methods']:
      x=(z[co['condition']+'_'+m+'_words'].astype(np.uint32)<<16).view(np.float32).reshape(len(ids),16,256)
      for l in range(16):check(np.isfinite(x[:,l]).all() and np.linalg.norm(x[:,l]-ref[:,l])/max(np.linalg.norm(ref[:,l]),1e-12)<=.01,'independent sampled FP64 check')
  else:
   pf=root/r['preflight_identity']['path'];check(h(pf)==r['preflight_identity']['sha256'],'timing preflight binding');check(next(x for x in records if x['raw']==r['preflight_identity']['path'])['ended_at']<receipt['started_at'],'preflight precedes main')
  if j['kind'] in ['timing','smoke']:
   expected_warm=[{'phase':phase,'round':ri,'rank':rank,'method':m} for phase,key in [('initial','initial')]+[(p,p+'_warm') for p in j['pass_order']] for ri,order in enumerate(j['orders'][key]) for rank,m in enumerate(order)];check(r['warm_calls']==expected_warm,'warm-up count/order');storage(r['shared_inventory']);count=0
   for pas in j['pass_order']:
    rr=[x for x in r['samples'] if x['pass']==pas];expected=[(ri,rank,m) for ri,order in enumerate(j['orders'][pas]) for rank,m in enumerate(order)];check([(x['round'],x['rank'],x['method']) for x in rr]==expected,'round/method ordering');last_start=0
    for x in rr:
     check(x['start_ns']>=last_start,'chronological order');last_start=x['start_ns'];clock_row(x,pas)
     samples.append({'job_id':j['id'],'phase':j['phase'],'Z':j['Z'],'condition':j['condition'],'block':j['block'],**x});count+=x['K']
    for m in j['methods']:
     mr=[x for x in rr if x['method']==m]
     for clock in (['host'] if pas=='H' else ['host','event'] if pas=='D' else ['event']):
      ss=statistics_([x[clock+'_us'] for x in mr]);M=inputs[j['condition']]['M'];U=inputs[j['condition']]['U'];decoded=0 if m=='A' else U if m=='C' else M;entry={'job_id':j['id'],'kind':j['kind'],'phase':j['phase'],'Z':j['Z'],'condition':j['condition'],'M':M,'U':U,'block':j['block'],'method':m,'pass':pas,'clock':clock,'K':16 if pas=='E' else 1,**ss,'output_positions_per_s_from_mean_timed_call':M/ss['mean_us']*1e6,'output_layer_rows_per_s_from_mean_timed_call':16*M/ss['mean_us']*1e6,'reconstructed_positions_per_s_from_mean_timed_call':decoded/ss['mean_us']*1e6,'rate_scope':'summed synchronized timed calls' if clock=='host' else 'CUDA stream bracket per-call elapsed','parent_worker_wall_seconds':receipt['wall_seconds'],'shared_residency_peak_bytes':r['global_allocated_peak'],'method_count':len(j['methods'])};timings.append(entry)
     if pas=='D':
      for x in mr:dual.append({'job_id':j['id'],'phase':j['phase'],'Z':j['Z'],'condition':j['condition'],'block':j['block'],**x,'host_minus_event_us':x['host_us']-x['event_us'],'host_over_event':x['host_us']/max(x['event_us'],1e-12)})
   if j['kind']=='timing':actual_methodblocks+=len(j['methods'])
   for m in j['methods']:
    fn_calls=sum(x['K'] for x in r['samples'] if x['method']==m)+sum(x['method']==m for x in r['warm_calls']);M=inputs[j['condition']]['M'];U=inputs[j['condition']]['U'];decoded=0 if m=='A' else U if m=='C' else M;work.append({'job_id':j['id'],'kind':j['kind'],'method':m,'actual_fn_calls':fn_calls,'output_positions':M*fn_calls,'decoded_positions':decoded*fn_calls,'decoded_layer_rows':16*decoded*fn_calls,'nominal_GEMM_FLOPs':2*decoded*4096*j['Z']*fn_calls,'setup_official_decoder_calls':128 if m=='A' else 0})
  if j['kind']=='lifecycle':
   lc=r['lifecycle'];N=j['N'];check(lc['actual_fn_calls']==N and lc['completion_synchronizations']==N and lc['warmup_calls']==0,'lifecycle N/warm-up');check(lc['session_start_ns']<=lc['artifact_start_ns']<=lc['artifact_end_ns']<=lc['ID_start_ns']<=lc['ID_end_ns']<=lc['dense_start_ns']<=lc['dense_end_ns']<=lc['loop_start_ns']<=lc['first_start_ns']<=lc['first_end_ns']<=lc['loop_end_ns']<=lc['session_end_ns'],'lifecycle clock order')
   for key,start,end in [('session_wall_seconds','session_start_ns','session_end_ns'),('setup_seconds','session_start_ns','loop_start_ns'),('total_loop_seconds','loop_start_ns','loop_end_ns'),('first_call_seconds','first_start_ns','first_end_ns'),('final_cleanup_seconds','loop_end_ns','session_end_ns')]:check(math.isclose(lc[key],(lc[end]-lc[start])/1e9,abs_tol=1e-9),'lifecycle units '+key)
   for key,start,end in [('artifact_load_H2D_seconds','artifact_start_ns','artifact_end_ns'),('ID_read_H2D_seconds','ID_start_ns','ID_end_ns')]:check(math.isclose(lc[key],(lc[end]-lc[start])/1e9,abs_tol=1e-9),'setup component clock')
   check(lc['artifact_load_H2D_seconds']+lc['ID_read_H2D_seconds']+lc['dense_build_cleanup_seconds']<=lc['setup_seconds'],'nested setup overcount');check(math.isclose(lc['session_per_call_us'],lc['session_wall_seconds']/N*1e6,rel_tol=1e-12),'session/N units')
   check(math.isclose(lc['setup_seconds']+lc['total_loop_seconds']+lc['final_cleanup_seconds'],lc['session_wall_seconds'],abs_tol=1e-9),'lifecycle non-overlap sum');check(lc['first_call_seconds']<=lc['total_loop_seconds']<=lc['session_wall_seconds']<=receipt['wall_seconds'],'lifecycle inclusion');storage(r['method_inventory']);row={'job_id':j['id'],'method':j['method'],'N':N,'block':j['block'],'worker_wall_seconds':receipt['wall_seconds'],'torch_import_seconds':r['torch_import_seconds'],'common_runtime_init_seconds':r['common_runtime_init_seconds'],**lc,**r['process_cost'],'allocated_peak_bytes':r['peak']['allocated_peak'],'reserved_peak_bytes':r['peak']['reserved_peak']};life.append(row);decoded=0 if j['method']=='A' else 512 if j['method']=='C' else 2048;work.append({'job_id':j['id'],'kind':'lifecycle','method':j['method'],'actual_fn_calls':N,'output_positions':2048*N,'decoded_positions':decoded*N,'decoded_layer_rows':16*decoded*N,'nominal_GEMM_FLOPs':2*decoded*4096*32*N,'setup_official_decoder_calls':128 if j['method']=='A' else 0})
  if j['kind']=='memory':
   for stage in r['memory_stages']:
    payload=storage(stage['inventory']);memstages.append({'job_id':j['id'],'method':j['method'],'stage':stage['stage'],'allocated_bytes':stage['allocated'],'reserved_bytes':stage['reserved'],'inventory_storage_bytes':payload,'allocated_minus_inventory_bytes':stage['allocated']-payload,'allocated_peak_bytes':stage['allocated_peak']})
   for snap in r['snapshot_files']:check(h(root/snap['path'])==snap['sha256'],'memory snapshot hash')
 # Same-round differences are supplemental; block summaries remain the repeat unit.
 round_pairs=[]
 for j in [x for x in plan if x['kind']=='timing' and x['id'] in byraw]:
  raw=byraw[j['id']];ri={(x['pass'],x['round'],x['method']):x for x in raw['samples']}
  for pas in 'HDE':
   for roundid in range(len(j['orders'][pas])):
    for a,b in [('C','B'),('A','B')]+([('S','B'),('C','S')] if 'S' in j['methods'] else []):
     for clock in (['host'] if pas=='H' else ['host','event'] if pas=='D' else ['event']):
      aa=ri[(pas,roundid,a)];bb=ri[(pas,roundid,b)];round_pairs.append({'job_id':j['id'],'Z':j['Z'],'condition':j['condition'],'block':j['block'],'pass':pas,'clock':clock,'round':roundid,'comparison':a+'/'+b,'difference_us':aa[clock+'_us']-bb[clock+'_us'],'ratio':aa[clock+'_us']/max(bb[clock+'_us'],1e-12),'numerator_start_ns':aa['start_ns'],'denominator_start_ns':bb['start_ns']})
 # Reconstructed actual semantic work counts; not hardware instruction counts.
 for row in work:
  row['request_official_decoder_calls']=row['actual_fn_calls'] if row['method']!='A' else 0;row['nominal_setup_GEMM_FLOPs']=row['setup_official_decoder_calls']*256*4096*2*plans[row['job_id']]['Z'];row['basis']='raw sample/warm_call rows or explicit actual_fn_calls'
 for j in [x for x in plan if x['id'] in byraw and x['kind'] in ['preflight','memory']]:
  r=byraw[j['id']]
  if j['kind']=='preflight':
   for co in r['correctness']:
    for m in co['methods']:
     M=co['M'];U=co['U'];decoded=0 if m=='A' else U if m=='C' else M;work.append({'job_id':j['id'],'kind':'preflight','method':m,'actual_fn_calls':1,'output_positions':M,'decoded_positions':decoded,'decoded_layer_rows':16*decoded,'nominal_GEMM_FLOPs':2*decoded*4096*j['Z'],'setup_official_decoder_calls':0,'request_official_decoder_calls':0 if m=='A' else 1,'nominal_setup_GEMM_FLOPs':0,'basis':'one output contract per condition/method'})
   work.append({'job_id':j['id'],'kind':'preflight_setup','method':'SETUP','actual_fn_calls':0,'output_positions':0,'decoded_positions':32768,'decoded_layer_rows':16*32768,'nominal_GEMM_FLOPs':0,'setup_official_decoder_calls':128,'request_official_decoder_calls':0,'nominal_setup_GEMM_FLOPs':2*32768*4096*j['Z'],'basis':'verified frozen256-token full-table loop once per preflight'})
  else:
   m=j['method'];decoded=0 if m=='A' else 512 if m=='C' else 2048;work.append({'job_id':j['id'],'kind':'memory','method':m,'actual_fn_calls':21,'output_positions':2048*21,'decoded_positions':decoded*21,'decoded_layer_rows':16*decoded*21,'nominal_GEMM_FLOPs':2*decoded*4096*32*21,'setup_official_decoder_calls':128 if m=='A' else 0,'request_official_decoder_calls':0 if m=='A' else 21,'nominal_setup_GEMM_FLOPs':2*32768*4096*32 if m=='A' else 0,'basis':'verified frozen first call plus20warm-up loop'})
 # Main paired block stats from H; D/E kept in their own scope/statistic.
 main=[r for r in timings if r['kind']=='timing'];idx={(r['Z'],r['condition'],r['block'],r['method'],r['pass'],r['clock']):r for r in main};paired=[];width=[];decisions=[]
 for j in [x for x in plan if x['kind']=='timing' and x['id'] in byraw]:
  z,cond,b=j['Z'],j['condition'],j['block']
  for numerator,denominator in [('C','B'),('A','B')]+([('S','B'),('C','S')] if 'S' in j['methods'] else []):
   for pas,clock in [('H','host'),('D','host'),('D','event'),('E','event')]:
    a=idx[(z,cond,b,numerator,pas,clock)];bb=idx[(z,cond,b,denominator,pas,clock)];row={'Z':z,'condition':cond,'M':inputs[cond]['M'],'U':inputs[cond]['U'],'block':b,'comparison':numerator+'/'+denominator,'pass':pas,'clock':clock,'median_ratio':a['median_us']/max(bb['median_us'],1e-12),'mean_ratio':a['mean_us']/max(bb['mean_us'],1e-12),'median_difference_us':a['median_us']-bb['median_us'],'mean_difference_us':a['mean_us']-bb['mean_us']};paired.append(row)
    if pas=='H':width.append(row)
 for z in [32,128,644]:
  for cond in (list(inputs) if z==32 else ['unique2048','repeat2048']):
   rs=[r for r in width if r['Z']==z and r['condition']==cond and r['comparison']=='C/B'];ratios=[r['median_ratio'] for r in rs];expected=6 if z==32 else 3
   if not ratios:continue
   med=statistics.median(ratios);wins=sum(x<1 for x in ratios);label=('REPEATABLE_IN_THIS_CORE_TEST' if z==32 else 'EXPLORATORY_WIDTH_SIGNAL') if len(rs)==expected and med<=.90 and wins>=(5 if z==32 else 3) else 'SMALL_DIFFERENCE' if .90<med<1.10 else 'NO_REPEATABLE_BENEFIT_UNDER_RULE';decisions.append({'Z':z,'condition':cond,'blocks':len(rs),'planned_blocks':expected,'median_C_B_host_median_ratio':med,'C_faster_blocks':wins,'ratios':ratios,'label':label,'no_row_reduction':inputs[cond]['M']==inputs[cond]['U']})
 attribution=[]
 for j in [x for x in plan if x['kind']=='memory' and x['id'] in byraw]:
  r=byraw[j['id']];blocks=r['active_after_removal'];identified=[b for b in blocks if b['size']==33554432 and any('setWorkspaceForHandle' in f.get('name','') for f in b['frames']) and any('getCurrentCUDABlasHandle' in f.get('name','') for f in b['frames'])];attribution.append({'method':j['method'],'before_library_allocated':r['memory_stages'][0]['allocated'],'after_library_release_allocated':next(s['allocated'] for s in r['memory_stages'] if s['stage']=='library_probe_released_baseline'),'after_module_removal_allocated':r['memory_stages'][-1]['allocated'],'classification':'STACK_IDENTIFIED_TORCH_CUBLAS_HANDLE_WORKSPACE' if identified else 'UNATTRIBUTED','identified_bytes':sum(b['size'] for b in identified),'stack_names':sorted({f['name'] for b in identified for f in b['frames'] if 'WorkspaceForHandle' in f.get('name','') or 'CUDABlasHandle' in f.get('name','')})})
 legacy_match=[]
 for hs,es in [('median_us','mean_us'),('median_us','median_us'),('mean_us','mean_us')]:legacy_match.append({'host_statistic':hs,'event_statistic':es,'host_C_faster':sum(x['host_'+hs+'_ratio']<1 for x in legacy_cb),'event_C_faster':sum(x['event_'+es+'_ratio']<1 for x in legacy_cb),'common_C_faster':sum(x['host_'+hs+'_ratio']<1 and x['event_'+es+'_ratio']<1 for x in legacy_cb),'direction_disagreement':sum((x['host_'+hs+'_ratio']<1)!=(x['event_'+es+'_ratio']<1) for x in legacy_cb),'separate_passes':True})
 main_ids={j['id'] for j in plan if j['kind']=='timing'};life_ids={j['id'] for j in plan if j['kind']=='lifecycle'};valid=set(byraw);missing_main=sorted(main_ids-valid);missing_life=sorted(life_ids-valid);kindcounts={k:sum(r['kind']==k for r in records) for k in ['preflight','smoke','timing','lifecycle','memory']};kindwalls={k:sum(r['wall_seconds'] for r in records if r['kind']==k) for k in kindcounts};budget={'prior_workers':len(old_budget['workers']),'prior_wall_seconds':old_spent,'P4_workers':len(records),'P4_wall_seconds':new_spent,'P4_cap_seconds':protocol['P4_wall_cap_seconds'],'P4_remaining_seconds':protocol['P4_wall_cap_seconds']-new_spent,'global_workers':len(old_budget['workers'])+len(records),'global_wall_seconds':old_spent+new_spent,'global_remaining_seconds':1800-old_spent-new_spent,'kind_counts':kindcounts,'kind_wall_seconds':kindwalls,'GPU_worker_wall_is_not_GPU_busy_time':True,'common_small_GEMM_probe_calls':len(records),'PLE_method_fn_calls':sum(x['actual_fn_calls'] for x in work),'official_decoder_full_table_setup_calls':sum(x['setup_official_decoder_calls'] for x in work)}
 decision={'status':'COMPLETE' if not missing_main and not missing_life and not failed else 'COMPLETE_PARTIAL','planned_timing_workers':30,'actual_timing_workers':sum(j in valid for j in main_ids),'planned_method_blocks':96,'actual_method_blocks':actual_methodblocks,'planned_lifecycle_workers':27,'actual_lifecycle_workers':len(life),'failed_workers':failed,'missing_timing_workers':missing_main,'missing_lifecycle_workers':missing_life,'descriptive_decisions':decisions,'legacy_recomputed':legacy_check,'preservation':preservation,'P2_performance_samples_merged_with_P4':False,'torch_imported': 'torch' in sys.modules}
 return {'legacy_statistics':legacy,'legacy_paired_statistics':legacy_pairs,'legacy_matched_statistics':legacy_match,'legacy_memory':legacy_memory,'controlled_latency':[r for r in timings if r['kind']=='timing'],'smoke_latency':[r for r in timings if r['kind']=='smoke'],'same_call_dual_timer':[r for r in dual if r['job_id'] in main_ids],'round_samples':[r for r in samples if r['job_id'] in main_ids],'paired_round_comparison':round_pairs,'paired_block_comparison':paired,'width_comparison':width,'lifecycle':life,'memory_stages':memstages,'memory_attribution':attribution,'work_counts':work,'preflight_audit':preflight,'budget':budget,'decision':decision,'raw':byraw}
def save(out,data):
 out=pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
 for k,v in data.items():
  if k in ['budget','decision']:dump(out/(k+'.json'),v)
  elif k!='raw':write_csv(out/(k+'.csv'),v)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['verify','summarize','figures']);p.add_argument('--root',type=pathlib.Path,default=ROOT);p.add_argument('--out',type=pathlib.Path);p.add_argument('--portable',action='store_true',help='Map recorded case paths to current case; explicitly report unavailable outside-case originals rather than claiming their verification');a=p.parse_args();data=collect(a.root,a.portable)
 if a.command=='summarize':save(a.out or a.root/'summary',data)
 if a.command=='figures':
  from figures import render
  render(a.root,a.out or a.root/'figures',data)
 print(json.dumps({'decision':data['decision'],'budget':data['budget']},indent=2))
if __name__=='__main__':main()
