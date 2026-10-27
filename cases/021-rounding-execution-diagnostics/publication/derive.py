"""Public scalar reaggregation. No model, Torch or private runtime imports."""
from pathlib import Path
import csv,json,math,statistics

def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open(newline="") as f:return list(csv.DictReader(f))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def close(a,b,label):require(math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12),label)
def mean(xs):xs=list(xs);return math.fsum(xs)/len(xs)
def projection(case):
 c=Path(case);old=read(c/'evidence/p0p3/decision.json');p5=read(c/'history/p5/facts.json');primary=[r for r in rows(c/'evidence/p0p3/primary.csv') if int(r['length'])==512];require(len(primary)==24,'PRIMARY_DENOMINATOR')
 result={'original_primary_verdict':old['rounding_effect'],'E0':mean(float(x['E0']) for x in primary),'E1':mean(float(x['E1']) for x in primary),'delta':mean(float(x['delta']) for x in primary),'primary_documents':24,'positions_per_document':16,'original_ci':[old['primary512']['ci_low'],old['primary512']['ci_high']],'primary_seed':61027023,'posthoc_seed':61027025,'tau_abs':old['tau_abs']}
 for row in rows(c/'evidence/p0p3/primary.csv'):close(float(row['E0'])-float(row['E1']),row['delta'],'PRIMARY_ROW_DELTA')
 close(result['E0'],old['mean_E0'],'E0');close(result['E1'],old['mean_E1'],'E1');close(result['delta'],old['primary512']['mean_delta'],'delta')
 position=rows(c/'evidence/p0p3/position_metrics.csv');docs=rows(c/'evidence/p0p3/document_metrics.csv');groups={}
 for r in position:groups.setdefault(tuple(r[k] for k in ['split','document_id','length','reference','candidate']),[]).append(r)
 for r in docs:
  own=groups[tuple(r[k] for k in ['split','document_id','length','reference','candidate'])];require(len(own)==16,'POSITION_DENOMINATOR');close(mean(float(x['KL']) for x in own),r['KL'],'POSITION_TO_DOCUMENT_KL')
 cross=rows(c/'evidence/p4/cross_precision_document.csv');js=rows(c/'evidence/p4/js_cross_precision.csv');drift=rows(c/'evidence/p4/fixed_reference_drift.csv');result['cross']=[]
 for split,n in [('DEV',8),('EVAL',24)]:
  for length in [128,512]:
   own=[x for x in cross if x['original_split']==split and int(x['length'])==length];jj=[x for x in js if x['original_split']==split and int(x['length'])==length];require(len(own)==len(jj)==n,'P4_CROSS_DENOMINATOR')
   values={k:mean(float(x[k]) for x in own) for k in ['K00','K01','K10','K11','delta_P_given_D1R0','delta_P_given_D1R1','diagonal_delta']};values.update({k:mean(float(x[k]) for x in jj) for k in ['JS00','JS01','JS10','JS11']})
   values.update(split='original_'+split+'_posthoc',length=length,documents=n);values['D0_drift']={cond:mean(float(x['KL']) for x in drift if x['original_split']==split and int(x['length'])==length and x['candidate']==cond) for cond in ['D1_R0','D1_R1','P_R0','P_R1']}
   recorded=next(x for x in read(c/'evidence/p4/p4a_facts.json')['groups'] if x['split']==values['split'] and x['length']==length)
   values['posthoc_intervals']={key:[recorded[key]['ci_low'],recorded[key]['ci_high']] for key in ['delta_P_given_D1R0','delta_P_given_D1R1','diagonal_delta']};result['cross'].append(values)
 tensors=rows(c/'evidence/p4/rounding_cancellation_tensors.csv');gaps=rows(c/'history/p5/summary/local_gap_components.csv');counts={};neg={};observed=[]
 for x in gaps:
  match=[t for t in tensors if t['analysis_split']==x['split'] and t['document_id']==x['document_id'] and t['layer']==x['layer'] and t['tensor']==x['tensor']];require(len(match)==2,'LOCAL_PAIR')
  rs={t['rounding']:t for t in match}
  for mode in ['R0','R1']:
   for a,b in [('G','pair_gap_SSE'),('N','n_energy'),('C','signed_cross')]:close(x[mode+'_'+a],rs[mode][b],'LOCAL_SOURCE_MATCH')
  dg,dn,dc=[float(x[k]) for k in ['delta_G','delta_N','delta_C']];close(dg,dn+dc,'LOCAL_DELTA_IDENTITY');tau=float(x['sign_tolerance'])
  kind='NOT_INCREASE_OR_NEAR_TOLERANCE' if dg<=tau else 'DN_NONPOS_DC_POS' if dn<=-tau and dc>tau else 'DN_POS_DC_POS' if dn>tau and dc>tau else 'DN_POS_DC_NONPOS' if dn>tau and dc<=-tau else 'NEAR_TOLERANCE_OR_OTHER';require(kind==x['classification'],'LOCAL_SIGN_CLASS')
  if x['split']=='original_EVAL_posthoc':
   observed.append(x);counts[kind]=counts.get(kind,0)+1
   if dg>tau and float(x['R0_C'])<0 and float(x['R1_C'])>float(x['R0_C']):neg[kind]=neg.get(kind,0)+1
 require(len(observed)==96,'LOCAL_24x2x2');result['gap_classes']=counts;result['weaker_negative_cross_by_class']=neg;result['local_denominator']='24 original EVAL documents x2 layers x Q/K =96 derived pairs, not96 documents'
 fits=rows(c/'history/p5/summary/candidate_fit.csv');result['single_term']={k:sum(int(x[k]) for x in fits if x['scope']=='ALL_VALID' and x['method']=='P') for k in ['coordinates','discriminating_coordinates','A_only','B_only','both','neither','undefined']};require(result['single_term']==p5['B']['candidate_fit_totals_nonoverlapping_all_valid_P'],'P5_FIT_RECOUNT');result['multiplicity']=read(c/'evidence/p5_multiplicity.json')
 sparse=[x for x in rows(c/'history/p5/summary/sparse_metrics.csv') if x['comparison']=='P_vs_D1'];oldmatch=rows(c/'history/p5/summary/sparse_P4_comparison.csv');result['sparse']=[]
 for layer,doc in dict.fromkeys((x['layer'],x['document_id']) for x in sparse):
  own=[x for x in sparse if (x['layer'],x['document_id'])==(layer,doc)];require([int(x['K']) for x in own]==[1,2,8,64,256,1024],'K_GRID');last=own[-1]
  for x in own:close(float(x['relative_RMSE']),math.sqrt(float(x['SSE'])/float(x['energy'])),'SPARSE_RMSE');close(float(x['mismatch_fraction']),int(x['numeric_mismatch'])/int(x['elements']),'SPARSE_FRACTION')
  matches=[x for x in oldmatch if (x['layer'],x['document_id'])==(layer,doc)];require(len(matches)==2 and all(x['same_input_hash']==x['same_output_bits']=='True' for x in matches),'P4_ALIGNMENT')
  result['sparse'].append({'layer':int(layer),'document_id':doc,'first_observed_K':next((int(x['K']) for x in own if int(x['numeric_mismatch'])),None),'K':[int(x['K']) for x in own],'actual_nonzero':[int(x['actual_nonzero']) for x in own],'input_norm':[float(x['norm64']) for x in own],'mismatches':[int(x['numeric_mismatch']) for x in own],'K1024_relative_RMSE':float(last['relative_RMSE']),'K1024_mismatch_fraction':float(last['mismatch_fraction']),'K1024_max_steps':int(last['max_steps']),'P4_bits_match':True})
 timing=rows(c/'evidence/p0p3/timing.csv');raw=rows(c/'evidence/p0p3/timing_raw.csv');cost=[]
 for path in ['D0','D1','P']:
  for mode in ['R0','R1']:
   own=[x for x in timing if x['path']==path and x['rounding']==mode];require(len(own)==3,'TIMING_BLOCKS')
   for x in own:
    rr=[v for v in raw if (v['path'],v['rounding'],v['block'])==(path,mode,x['block'])];require(len(rr)==10,'TIMING_SAMPLES');close(statistics.median(float(v['host_ns'])/1e6 for v in rr),x['host_median_ms'],'HOST_MEDIAN');close(mean(float(v['event_ms']) for v in rr),x['event_mean_ms'],'EVENT_MEAN');close(statistics.median(float(v['event_ms']) for v in rr),x['event_median_ms'],'EVENT_MEDIAN')
   cost.append({'path':path,'mode':mode,'blocks':3,'calls_per_block':10,'host_block_median_ms':statistics.median(float(x['host_median_ms']) for x in own),'host_block_range_ms':[min(float(x['host_median_ms']) for x in own),max(float(x['host_median_ms']) for x in own)],'event_block_median_ms':statistics.median(float(x['event_median_ms']) for x in own)})
 result['cost']=cost;result['budgets']={'P0_P3_GPU':old['budget']['worker_wall_seconds'],'P4_GPU':read(c/'evidence/p4/budget.json')['GPU']['parent_wall_seconds'],'P4_CPU':read(c/'evidence/p4/budget.json')['CPU']['parent_wall_seconds'],'P5_GPU':p5['budget']['GPU']['parent_wall_seconds'],'P5_CPU':p5['budget']['CPU']['parent_wall_seconds'],'campaign_GPU':p5['budget']['campaign_GPU_parent_seconds']};close(sum(result['budgets'][k] for k in ['P0_P3_GPU','P4_GPU','P5_GPU']),result['budgets']['campaign_GPU'],'GPU_LEDGER_SUM')
 ledger=read(c/'evidence/worker_accounting.json')
 for phase in ledger:
  if phase['phase']=='P0_P3':close(math.fsum(x['worker_wall_seconds'] for x in phase['entries']),result['budgets']['P0_P3_GPU'],'P0_P3_WORKER_DEBIT')
  else:
   for kind in ['GPU','CPU']:close(math.fsum(x['parent_wall_seconds'] for x in phase['entries'] if x['kind']==kind),result['budgets']['P4_'+kind],'P4_WORKER_DEBIT')
 p5ledger=read(c/'history/p5/records/ledger.json')['entries']
 for kind in ['GPU','CPU']:close(math.fsum(x['parent_wall_seconds'] for x in p5ledger if x['kind']==kind),result['budgets']['P5_'+kind],'P5_WORKER_DEBIT')
 result['execution_counts']={'original':old['counts'],'P5':p5['counts']};result['identities']={'Git_base':'bb2ab39ba5e9f6cea7e4c7d122fa6b26e143e688','original_GPU_source':old['gpu_source_bundle_sha256'],'P5_GPU_source':p5['source_GPU_bundle'],'P5_protocol':p5['protocol_identity']};return result

def numerical(case):
 import numpy as np
 c=Path(case);p=[r for r in rows(c/'evidence/p0p3/primary.csv') if int(r['length'])==512];delta=np.array([float(x['delta']) for x in p]);rng=np.random.default_rng(61027023);means=delta[rng.integers(0,24,(10000,24))].mean(1);ci=np.quantile(means,[.025,.975],method='linear');old=read(c/'evidence/p0p3/decision.json')['primary512'];close(ci[0],old['ci_low'],'BOOTSTRAP_LO');close(ci[1],old['ci_high'],'BOOTSTRAP_HI')
 # Independently validate full-vocabulary-derived per-position scalar means, not absent logits.
 cross=rows(c/'evidence/p4/cross_precision_document.csv');positions=rows(c/'evidence/p4/cross_precision_positions.csv')
 for x in cross:
  own=[v for v in positions if v['document_id']==x['document_id'] and v['length']==x['length']];require(len(own)==16,'P4_16_POSITIONS')
  for k in ['K00','K01','K10','K11']:close(mean(float(v[k]) for v in own),x[k],'P4_POSITION_MEAN')
 for group in read(c/'evidence/p4/p4a_facts.json')['groups']:
  own=[x for x in cross if x['analysis_split']==group['split'] and int(x['length'])==group['length']]
  for key in ['delta_P_given_D1R0','delta_P_given_D1R1','diagonal_delta']:
   values=np.array([float(x[key]) for x in own]);rng=np.random.default_rng(61027025);boot=values[rng.integers(0,len(values),(10000,len(values)))].mean(1);ci2=np.quantile(boot,[.025,.975],method='linear');close(ci2[0],group[key]['ci_low'],'P4_BOOTSTRAP_LO');close(ci2[1],group[key]['ci_high'],'P4_BOOTSTRAP_HI')
 return {'original_bootstrap_interval':list(map(float,ci)),'replicates':10000,'seed':61027023,'scope':'Saved per-position/document scalars. No hidden logits KL recalculation.','P4_position_rows':len(positions)}
