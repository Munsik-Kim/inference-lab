"""Post-hoc CPU review of supplied Case 008 records. Does not edit source data."""
from pathlib import Path
import json,math,collections,numpy as np
# Adapted only the path/output interface of the supplied review script.
# Review byte arithmetic below is independently checked against frozen metadata.
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True,help='New external directory')
args=parser.parse_args()
CASE=args.case.resolve();BASE=args.output.resolve()
if BASE.exists() or BASE.is_relative_to(CASE):raise ValueError('Use a new external output directory')
BASE.mkdir(parents=True)
read=lambda p:json.loads(p.read_text())
result={'scope':'Post-hoc descriptive calculations from supplied scalar records; no new GPU runs or fresh observations','tracks':{}}
for track,baseline in [('Q','Q-BF16'),('R','R-B')]:
 rows=read(CASE/f'results/raw/{track}/model_records.json')
 inputs=read(CASE/f'inputs/{track}/heldout.json');imap={r['id']:r for r in inputs}
 assert len(imap)==192
 lookup={(r['arm'],r['id']):r for r in rows}
 assert len(lookup)==len(rows)
 arms=sorted({r['arm'] for r in rows})
 assert len(rows)==192*len(arms)
 report={}
 for arm in arms:
  report[arm]={}
  for task in ['all','retrieval','comparison','code']:
   rr=[lookup[arm,i] for i in sorted(imap) if task=='all' or imap[i]['task']==task]
   bb=[lookup[baseline,r['id']] for r in rr]
   sc=[r['score'] for r in rr];bc=[r['score'] for r in bb]
   outside=np.array([s['full_log_normalizer']-s['choice_log_normalizer'] for s in sc])
   f=np.array([s['full_gold_nll'] for s in sc]);c=np.array([s['choice_nll'] for s in sc])
   b_out=np.array([s['full_log_normalizer']-s['choice_log_normalizer'] for s in bc])
   df=np.array([s['full_gold_nll']-b['full_gold_nll'] for s,b in zip(sc,bc)])
   dc=np.array([s['choice_nll']-b['choice_nll'] for s,b in zip(sc,bc)])
   assert np.max(np.abs(f-c-outside))<1e-12
   ent=np.array([-sum(p*math.log(p) for p in s['choice_probabilities'] if p>0) for s in sc])
   v={'n':len(sc),'correct_forced_choice':sum(s['correct'] for s in sc),'predicted_label_counts':dict(collections.Counter('ABCD'[s['prediction']] for s in sc)),
      'full_argmax_allowed':sum(s['full_argmax_allowed'] for s in sc),'mass_mean':float(np.mean([s['allowed_mass'] for s in sc])),
      'mass_geometric':float(np.exp(-outside.mean())),'mass_quantiles':np.quantile([s['allowed_mass'] for s in sc],[0,.5,1]).tolist(),
      'mean_choice_nll':float(c.mean()),'mean_full_nll':float(f.mean()),'mean_outside_label_penalty':float(outside.mean()),
      'delta_full_nll':float(df.mean()),'delta_choice_nll':float(dc.mean()),'delta_outside_label_penalty':float((outside-b_out).mean()),
      'choice_entropy_mean':float(ent.mean()),'full_nll_improved_items':int((df<0).sum()),'choice_nll_improved_items':int((dc<0).sum()),
      'regressions_vs_baseline':sum(b['correct'] and not s['correct'] for b,s in zip(bc,sc)),
      'gains_vs_baseline':sum(not b['correct'] and s['correct'] for b,s in zip(bc,sc)),
      'flips_vs_baseline':sum(b['prediction']!=s['prediction'] for b,s in zip(bc,sc)),
      'length_min_median_max':np.quantile([imap[r['id']]['length'] for r in rr],[0,.5,1]).tolist()}
   if arm!=baseline and 'local' in rr[0]:
    v['mean_local_relative_error']=float(np.mean([r['local']['relative_error'] for r in rr]))
    v['mean_recorded_KL']=float(np.mean([r['full_kl_B_candidate'] for r in rr]))
   report[arm][task]=v
 result['tracks'][track]=report
 if track=='R':
  result['repair_diagnostics']={}
  for a in ['I25','P25','S50']:
   out={}
   for task in ['all','retrieval','comparison','code']:
    ids=[i for i in sorted(imap) if task=='all' or imap[i]['task']==task]
    un=[lookup[a,i] for i in ids];rp=[lookup[a+'-R',i] for i in ids]
    base=[lookup[baseline,i] for i in ids]
    v={}
    for region,inds in [('all32',list(range(32))),('last_token',[31]),('last4_positions',[28,29,30,31])]:
     u=np.array([sum(r['sampled_local'][j]['squared_error'] for j in inds) for r in un])
     e=np.array([sum(r['sampled_local'][j]['squared_error'] for j in inds) for r in rp])
     for r in un+rp: assert r['sampled_positions'][-1]==imap[r['id']]['length']-1
     v[region]={'pooled_energy_recovery':float(1-e.sum()/u.sum()),'n_improved_prompts':int((e<u).sum()),'n_worse_prompts':int((e>u).sum()),'n':len(ids)}
    v['gold_regressions_repair_vs_pruned']=sum(x['score']['correct'] and not y['score']['correct'] for x,y in zip(un,rp))
    v['gold_gains_repair_vs_pruned']=sum(not x['score']['correct'] and y['score']['correct'] for x,y in zip(un,rp))
    v['choice_flips_repair_vs_pruned']=sum(x['score']['prediction']!=y['score']['prediction'] for x,y in zip(un,rp))
    out[task]=v
   result['repair_diagnostics'][a]=out
# Exact byte arithmetic from supplied records.
bf=8044982000;w4=2651839568
result['byte_arithmetic']={'Q_safetensors_reduction':1-w4/bf,'Q_safetensors_ratio':bf/w4,'Q_file_GiB':[bf/2**30,w4/2**30],
'Q_named_parameters_GiB':[8044936192/2**30,2651735728/2**30],
'Q_allocator_peak_GiB':[11260111360/2**30,5795133952/2**30],
'Q_allocator_peak_reduction':1-5795133952/11260111360,
'R_complete_parameter_reduction_25':1-1187381248/1192099840,'R_complete_parameter_reduction_50':1-1182662656/1192099840}
(BASE/'posthoc_review_metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
for track in ['Q','R']:
 print('\nTRACK',track)
 for arm in result['tracks'][track]:
  print(arm)
  for task,v in result['tracks'][track][arm].items():
   print(' ',task, 'correct',v['correct_forced_choice'],'in_labels',v['full_argmax_allowed'],'mass_mean/geomean',round(v['mass_mean'],6),f"{v['mass_geometric']:.5g}",'delta_full/choice/mass',*[round(v[x],6) for x in ['delta_full_nll','delta_choice_nll','delta_outside_label_penalty']])
print('\nREPAIR POSITIONS')
for name,ds in result['repair_diagnostics'].items():
 for task,v in ds.items(): print(name,task,{k:x for k,x in v.items() if k in ['all32','last_token','gold_regressions_repair_vs_pruned','gold_gains_repair_vs_pruned']})
print('BYTE_ARITHMETIC',result['byte_arithmetic'])
