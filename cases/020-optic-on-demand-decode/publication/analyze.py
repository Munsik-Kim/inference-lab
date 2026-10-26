"""Publication-only CPU reanalysis. Uses saved raw, never imports Torch/worker code."""
from pathlib import Path
import json,csv,math,statistics,hashlib,itertools
import numpy as np
from verify import CASE,ARCHIVE,read,sha,check

def stat(values):
 a=np.asarray(values,dtype=np.float64);check(a.size and np.isfinite(a).all() and (a>=0).all(),'INVALID_TIMING');q=np.quantile(a,[.25,.5,.75,.95],method='linear')
 return dict(n=len(a),mean_us=float(a.mean()),median_us=float(q[1]),p95_us=float(q[3]),min_us=float(a.min()),IQR_us=float(q[2]-q[0]))
def storage(rows,key='key'):
 seen=set();total=0;sizes={'torch.uint8':1,'torch.int64':8,'torch.float16':2,'torch.bfloat16':2,'torch.float32':4,'torch.uint16':2}
 for r in rows:
  check(r['element_size']==sizes[r['dtype']] and math.prod(r['shape'])*r['element_size']==r['payload_bytes'],'STORAGE_DTYPE_PAYLOAD');k=tuple(r[key]);check(k[2]==r['storage_bytes'],'STORAGE_KEY');unique=k not in seen;check(r.get('unique',r.get('unique_storage'))==unique,'ALIAS_FLAG')
  if unique:total+=r['storage_bytes'];seen.add(k)
 return total

def tiny_reference(hist):
 # Independent scalar group/layer formula and einsum. Stored FP16 values only.
 path=hist/'inputs/p1_seed0_random.npz'
 with np.load(path,allow_pickle=False) as z:
  meta=json.loads(z['metadata'].item());a={k:z[k] for k in z.files if k!='metadata'}
 V=meta['ple_shape'][0];G=meta['n_groups'];Z=meta['latent_dim'];codes=a['token_codes_packed_uint8'].reshape(V,G);cb=a['codebooks_fp16'].astype(np.float64);W=a['decoder_weight_fp16'].astype(np.float64);bias=a['decoder_bias_fp16'].astype(np.float64);ids=np.array([0,256,17,17],np.int64);lat=np.concatenate([cb[g,codes[ids,g]] for g in range(G)],axis=1);vector=np.einsum('tz,lhz->tlh',lat,W)+bias;loop=np.zeros_like(vector)
 for t,token in enumerate(ids):
  for l in range(3):
   for h in range(16):loop[t,l,h]=float(bias[l,h])+sum(float(cb[g,codes[token,g],q])*float(W[l,h,4*g+q]) for g in range(G) for q in range(4))
 check(np.allclose(loop,vector,rtol=1e-10,atol=1e-12),'FP64_LOOP_VECTOR')
 with np.load(hist/'results/20261026-1215/p1_with_raw/cpu_seed0_random_bf16_words.npz',allow_pickle=False) as z:
  output=(z['B'].astype(np.uint32)<<16).view(np.float32).reshape(257,3,16)[ids]
 errors=[]
 for l in [-1,0,1,2]:
  x=output if l<0 else output[:,l];r=vector if l<0 else vector[:,l];rel=float(np.linalg.norm(x-r)/max(np.linalg.norm(r),1e-12));check(np.isfinite(x).all() and rel<=.01,'BF16_TINY_REFERENCE');errors.append({'layer':l,'relative_L2':rel})
 return {'tokens':ids.tolist(),'layers':3,'loop_vector_max_abs':float(np.abs(loop-vector).max()),'rtol':1e-10,'atol':1e-12,'BF16_error':errors,'scope':'Small stored P1 slice only. Full mode additionally audits all preserved P1/P2 outputs.'}

def collect(case=CASE):
 case=Path(case);hist=case/ARCHIVE;p4=hist/'followup/p4_speed_cost';protocol=read(p4/'PROTOCOL.json');plans=read(p4/'provenance/plans.json');plan_by={x['id']:x for x in plans};input_map=read(p4/'provenance/inputs.json');run=p4/'results'/protocol['run_id'];ledger=[json.loads(s) for s in (run/'worker_ledger.jsonl').read_text().splitlines()];check(len({x['job_id'] for x in ledger})==len(ledger),'DUPLICATE_WORKER');stats=[];lifes=[];inventories={};raw_main=[];stages=[]
 for w in ledger:
  r=read(p4/w['raw']);check(sha(p4/w['raw'])==w['raw_sha256'] and w['exit_code']==0,'WORKER_RAW');j=plan_by[w['job_id']];check(r['job']==j and r['source_hash']==read(p4/'provenance/freeze.json')['bundle_sha256'],'JOB_SOURCE');check(r['config_sha256']==sha(p4/'PROTOCOL.json'),'CONFIG_SOURCE')
  if j['kind']=='timing':
   check(r['preflight_identity']['sha256']==sha(p4/r['preflight_identity']['path']),'PREFLIGHT_BINDING');raw_main.append((j,r));compact=[x for x in r['shared_inventory'] if x['name'] not in ['A_table','IDs']];module=storage(compact);inventories.setdefault(j['Z'],[]).append(module)
   for pas,rounds,K in [('H',120,1),('D',48,1),('E',24,16)]:
    rr=[x for x in r['samples'] if x['pass']==pas];expected=[(i,k,m) for i,o in enumerate(j['orders'][pas]) for k,m in enumerate(o)];check([(x['round'],x['rank'],x['method']) for x in rr]==expected,'METHOD_ORDER');check(all(x['K']==K for x in rr),'K');counts={v:0 for v in itertools.permutations(j['methods'])}
    for order in j['orders'][pas]:counts[tuple(order)]+=1
    check(len(set(counts.values()))==1,'BALANCED_ORDER')
    for x in rr:
     if pas in ['H','D']:check(math.isclose(x['host_us'],(x['end_ns']-x['start_ns'])/1000,abs_tol=1e-9),'HOST_NS_US')
     else:check(math.isclose(x['event_us']*16,x['bracket_stream_us'],rel_tol=1e-12),'EVENT_BRACKET_K')
    for m in j['methods']:
     mr=[x for x in rr if x['method']==m];check(len(mr)==rounds,'ROUND_COUNT')
     for clock in (['host'] if pas=='H' else ['host','event'] if pas=='D' else ['event']):stats.append({'Z':j['Z'],'input':j['condition'],'M':input_map[j['condition']]['M'],'U':input_map[j['condition']]['U'],'block':j['block'],'method':m,'pass':pas,'clock':clock,'K':K,**stat([x[clock+'_us'] for x in mr])})
  if j['kind']=='lifecycle':
   x=r['lifecycle'];N=j['N'];check(x['actual_fn_calls']==N and x['completion_synchronizations']==N and x['warmup_calls']==0,'LIFECYCLE_N')
   for key,start,end in [('session_wall_seconds','session_start_ns','session_end_ns'),('setup_seconds','session_start_ns','loop_start_ns'),('total_loop_seconds','loop_start_ns','loop_end_ns'),('first_call_seconds','first_start_ns','first_end_ns'),('final_cleanup_seconds','loop_end_ns','session_end_ns')]:check(math.isclose(x[key],(x[end]-x[start])/1e9,abs_tol=1e-9),'LIFECYCLE_NS')
   check(math.isclose(x['setup_seconds']+x['total_loop_seconds']+x['final_cleanup_seconds'],x['session_wall_seconds'],abs_tol=1e-9),'NESTED_INTERVALS');check(x['first_call_seconds']<=x['total_loop_seconds']<=x['session_wall_seconds']<=w['wall_seconds'],'INCLUSION');lifes.append({'method':j['method'],'N':N,'block':j['block'],'worker_wall_seconds':w['wall_seconds'],**x,**r['process_cost']})
  if j['kind']=='memory':
   for s in r['memory_stages']:stages.append({'method':j['method'],'stage':s['stage'],'allocated_bytes':s['allocated'],'reserved_bytes':s['reserved']})
 check(len(raw_main)==30 and len(stats)==384 and len(lifes)==27,'P4_DENOMINATORS')
 h={(x['Z'],x['input'],x['block'],x['method']):x for x in stats if x['pass']=='H'};paired=[]
 for j,r in raw_main:
  for n,d in [('B','A'),('C','B')]+([('S','B'),('C','S')] if 'S' in j['methods'] else []):
   a=h[j['Z'],j['condition'],j['block'],n];b=h[j['Z'],j['condition'],j['block'],d];paired.append({'Z':j['Z'],'input':j['condition'],'M':a['M'],'U':a['U'],'block':j['block'],'comparison':n+'/'+d,'median_ratio':a['median_us']/b['median_us'],'median_difference_us':a['median_us']-b['median_us']})
 representative={(32,1,1),(32,2048,512),(644,2048,512)};direct=[]
 for x in stats:
  if x['pass']=='H' and x['method'] in 'ABC' and (x['Z'],x['M'],x['U']) in representative:direct.append(x)
 directagg=[]
 for z,M,U in sorted(representative):
  for m in 'ABC':
   rr=sorted([x for x in direct if (x['Z'],x['M'],x['U'],x['method'])==(z,M,U,m)],key=lambda x:x['block']);row={'Z':z,'M':M,'U':U,'method':m,'blocks':len(rr)}
   for met in ['median_us','p95_us']:v=[x[met] for x in rr];row.update({met+'_block_median':statistics.median(v),met+'_block_min':min(v),met+'_block_max':max(v)})
   directagg.append(row)
 representative_pairs=[]
 for z,M,U in sorted(representative):
  for comparison in ['B/A','C/B']:
   rr=[x for x in paired if (x['Z'],x['M'],x['U'],x['comparison'])==(z,M,U,comparison)];ratios=[x['median_ratio'] for x in rr];diffs=[x['median_difference_us'] for x in rr];representative_pairs.append({'Z':z,'M':M,'U':U,'comparison':comparison,'blocks':len(rr),'median_of_block_ratios':statistics.median(ratios),'min_block_ratio':min(ratios),'max_block_ratio':max(ratios),'median_of_block_difference_us':statistics.median(diffs),'min_block_difference_us':min(diffs),'max_block_difference_us':max(diffs)})
 width=[]
 for z in [32,128,644]:
  path=hist/'inputs/LARGE.npz' if z==32 else p4/f'fixtures/Z{z}.npz'
  with np.load(path,allow_pickle=False) as f:
   meta=json.loads(f['metadata'].item());payload=sum(f[n].nbytes for n in f.files if n!='metadata');check(f['token_codes_packed_uint8'].dtype==np.uint8,'CODE_DTYPE');check(all(f[n].dtype==np.float16 for n in ['codebooks_fp16','decoder_weight_fp16','decoder_bias_fp16']),'FP16_ARRAYS')
  actuals=inventories[z];check(len(set(actuals))==1,'MODULE_INVENTORY_VARIATION');actual=actuals[0];G=z//4;theory=32768*G+2*256*z+2*16*256*z+2*16*256+8*G;check(actual==theory,'INVENTORY_FORMULA');width.append({'Z':z,'groups':G,'artifact_npz_bytes':path.stat().st_size,'array_payload_bytes':payload,'observed_unique_module_storage_bytes':actual,'inventory_observations':len(actuals),'theoretical_storage_bytes':theory,'module_MiB':actual/2**20,'dense_BF16_table_bytes':268435456,'dense_over_observed_module':268435456/actual,'storage_source':'compact subset of saved shared_inventory; excludes A table and common IDs'})
 oldledger=read(hist/'results/20261026-1235/worker_ledger.json');legacy=[];memory=[];oldidx={}
 for w in oldledger['workers']:
  if w['kind']!='cell':continue
  r=read(hist/'results/20261026-1235'/w['raw']);check(sha(hist/'results/20261026-1235'/w['raw'])==w['raw_sha256'],'P2_RAW_HASH');c=r['cell'];oldidx[c['shape'],c['M'],c['U'],c['block'],c['method']]=r
  for metric,key in [('host','host_us'),('event','event_bracket_per_call_us')]:check(len(r['timing'][key])==100,'P2_SAMPLES');legacy.append({**{k:c[k] for k in ['shape','M','U','block','method']},'clock':metric,'K':1 if metric=='host' else r['timing']['event_K'],**stat(r['timing'][key])})
  resident=storage(r['resident_inventory'],'storage_key');memory.append({**{k:c[k] for k in ['shape','M','U','block','method']},'NPZ_bytes':c['artifact']['bytes'],'resident_bytes':resident,'baseline_allocated_bytes':r['baseline']['allocated'],'total_allocated_peak_bytes':r['timed_memory']['allocated_peak'],'total_reserved_peak_bytes':r['timed_memory']['reserved_peak'],'incremental_peak_above_common_baseline_bytes':r['timed_memory']['allocated_peak']-r['baseline']['allocated']})
 check(len(legacy)==180 and len(memory)==90,'P2_FINAL_DENOMINATOR');hostwins=eventwins=common=0
 for k,r in oldidx.items():
  if k[-1]!='C':continue
  b=oldidx[*k[:-1],'B'];hw=statistics.median(r['timing']['host_us'])<statistics.median(b['timing']['host_us']);ew=statistics.mean(r['timing']['event_bracket_per_call_us'])<statistics.mean(b['timing']['event_bracket_per_call_us']);hostwins+=hw;eventwins+=ew;common+=hw and ew
 lifeagg=[]
 for N in [128,1024,4096]:
  for m in 'ABC':
   rr=sorted([x for x in lifes if x['N']==N and x['method']==m],key=lambda x:x['block']);row={'N':N,'method':m,'blocks':len(rr)}
   for key in ['session_wall_seconds','worker_wall_seconds','first_call_seconds','setup_seconds','total_loop_seconds','final_cleanup_seconds']:
    vals=[x[key] for x in rr];row[key+'_median']=statistics.median(vals);row[key+'_min']=min(vals);row[key+'_max']=max(vals)
   lifeagg.append(row)
 life_pairs=[];life_counts=[]
 for N in [128,1024,4096]:
  entry={'N':N,'blocks':3}
  for clock in ['session_wall_seconds','worker_wall_seconds']:
   entry[clock+'_A_faster_than_B_blocks']=sum(next(x[clock] for x in lifes if (x['N'],x['method'],x['block'])==(N,'A',b))<next(x[clock] for x in lifes if (x['N'],x['method'],x['block'])==(N,'B',b)) for b in range(3))
   for b in range(3):
    for numerator,denominator in [('A','B'),('C','B')]:
     nv=next(x[clock] for x in lifes if (x['N'],x['method'],x['block'])==(N,numerator,b));dv=next(x[clock] for x in lifes if (x['N'],x['method'],x['block'])==(N,denominator,b));life_pairs.append({'N':N,'block':b,'clock':clock,'comparison':numerator+'/'+denominator,'ratio':nv/dv,'difference_seconds':nv-dv})
  life_counts.append(entry)
 cbs=[x for x in paired if x['comparison']=='C/B'];check(len(cbs)==30,'CB_COMPARISON_DENOMINATOR');lmap={(x['N'],x['method']):x for x in lifeagg};benefits={key:1-lmap[4096,'A'][key+'_median']/lmap[4096,'B'][key+'_median'] for key in ['session_wall_seconds','worker_wall_seconds']}
 ratios=[]
 for z,inp in sorted({(x['Z'],x['input']) for x in cbs}):
  rr=[x for x in cbs if (x['Z'],x['input'])==(z,inp)];ratios.append({'Z':z,'input':inp,'blocks':len(rr),'median_of_block_C_B_ratios':statistics.median(x['median_ratio'] for x in rr),'min':min(x['median_ratio'] for x in rr),'max':max(x['median_ratio'] for x in rr),'median_of_block_C_minus_B_us':statistics.median(x['median_difference_us'] for x in rr),'C_faster_blocks':sum(x['median_ratio']<1 for x in rr)})
 selected=[x for x in memory if x['shape']=='LARGE' and x['M']==2048 and x['U']==512];mem0={m:next(x for x in selected if x['method']==m and x['block']==0) for m in 'ABC'}
 for m in 'ABC':
  for key in ['resident_bytes','total_allocated_peak_bytes','total_reserved_peak_bytes']:check(len({x[key] for x in selected if x['method']==m})==1,'P2_MEMORY_BLOCK_AGREEMENT')
 memory_facts={'A_to_B_allocated_peak_reduction':1-mem0['B']['total_allocated_peak_bytes']/mem0['A']['total_allocated_peak_bytes'],'A_to_B_module_ratio':mem0['A']['resident_bytes']/mem0['B']['resident_bytes'],'C_vs_B_allocated_peak_reduction':1-mem0['C']['total_allocated_peak_bytes']/mem0['B']['total_allocated_peak_bytes'],'observations':selected}
 diagnostic_count=sum(x['kind']=='cell' for x in read(hist/'results/20261026-1215/worker_ledger.json')['workers']);check(diagnostic_count==90,'P2_DIAGNOSTIC_DENOMINATOR')
 oldbudget=read(hist/'results/gpu_worker_budget.json');oldwall=sum(x['wall_seconds'] for x in oldbudget['workers']);newwall=sum(x['wall_seconds'] for x in ledger);check(len(oldbudget['workers'])==196 and len(ledger)==64 and oldwall+newwall<=1800,'BUDGET');kind={k:{'workers':sum(x['kind']==k for x in ledger),'wall_seconds':sum(x['wall_seconds'] for x in ledger if x['kind']==k)} for k in sorted({x['kind'] for x in ledger})};budget={'original_workers':196,'original_wall_seconds':oldwall,'P4_workers':len(ledger),'P4_wall_seconds':newwall,'total_workers':260,'total_wall_seconds':oldwall+newwall,'remaining_original_1800s':1800-oldwall-newwall,'this_publication_GPU_workers':0,'P4_kind':kind}
 facts={'question':'Does smaller embedding storage also run faster?','memory_condition':'P2 LARGE Z32 M2048 U512; method-isolated','P2_final_cell_blocks':90,'P2_diagnostic_cell_blocks':diagnostic_count,'memory':memory_facts,'P4_timing_workers':len(raw_main),'P4_method_blocks':sum(len(j['methods']) for j,r in raw_main),'P4_lifecycle_workers':len(lifes),'P4_C_B_block_comparisons':len(cbs),'P4_C_faster_blocks':sum(x['median_ratio']<1 for x in cbs),'P2_host_C_wins':hostwins,'P2_event_mean_C_wins':eventwins,'P2_common_C_wins':common,'N4096_A_reduction_ratio_of_method_medians':benefits,'lifecycle_direction_counts':life_counts,'ratios':ratios,'width_storage':width,'statistical_unit':'fresh process block; inner samples are repeated operator calls','memory_baseline_attribution':'New replay stack identifies Torch cuBLAS handle workspace; historical stack not captured','budget':budget}
 return {'all_P4_clock_statistics':stats,'representative_direct_block':direct,'representative_direct_summary':directagg,'representative_paired_summary':representative_pairs,'P4_paired_H':paired,'width_storage':width,'lifecycle_block':lifes,'lifecycle_summary':lifeagg,'lifecycle_paired':life_pairs,'P2_memory':memory,'P2_clock_statistics':legacy,'memory_stages':stages,'facts':facts,'budget':budget,'tiny_reference':tiny_reference(hist)}
def save(data,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 for k,v in data.items():
  if isinstance(v,dict):(out/(k+'.json')).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
  else:
   fields=list(dict.fromkeys(k for x in v for k in x))
   with (out/(k+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fields,lineterminator="\n");w.writeheader();w.writerows(v)
