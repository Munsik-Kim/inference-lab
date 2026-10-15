"""Portable saved-record audit: stdlib core, optional NumPy bootstrap. No model/GPU."""
import argparse,ast,json,math,tempfile,subprocess,sys,collections
from pathlib import Path

def close(a,b,tol=1e-10):assert abs(a-b)<=tol*max(1,abs(a),abs(b)),(a,b)
def audit(case,bootstrap=False):
 case=Path(case).resolve();run=case/'results/p1_r2'
 with tempfile.TemporaryDirectory(prefix='case014-scalar-') as td:
  view=Path(td);(view/'p1').symlink_to(run,target_is_directory=True)
  for n in ['p0.json','p0_full.json']:(view/n).symlink_to(case/'results/local_r1'/n)
  output=view/'audited.json';subprocess.run([sys.executable,str(case/'scripts/audit.py'),'--results',str(view),'--output',str(output)],check=True);basic=json.loads(output.read_text())
 raw=[json.loads(s) for p in sorted((run/'cells').glob('*.jsonl')) for s in p.read_text().splitlines()];groups=collections.defaultdict(list)
 for r in raw:groups[r['block'],r['projection'],r['split'],r['alpha']].append(r)
 summary=json.loads((run/'derived_summary.json').read_text());assert len(summary['rows'])==120
 terms=['SSE_after','SSE_before','reference_SSE','A_norm2','T_inherited','T_cancellation','T_cont','T_requant','arithmetic_residual_norm2','cast_residual_norm2']
 classified=collections.Counter();newclass={};bootchecks=0
 for q in summary['rows']:
  key=q['block'],q['projection'],q['split'],q['alpha'];rr=sorted(groups[key],key=lambda r:r['document_id_hash']);base=sorted(groups[key[:3]+(0,)],key=lambda r:r['document_id_hash']);n=sum(r['valid_token_count'] for r in rr);s=math.fsum(r['NLL_sum'] for r in rr)
  assert n==q['valid_tokens'];close(s,q['NLL_sum']);close(s/n,q['NLL_mean']);close(math.exp(s/n),q['PPL'])
  for k in terms:close(math.fsum(r[k] for r in rr),q[k])
  close(math.fsum(r['sampled_full_vocab_KL_sum'] for r in rr)/sum(r['KL_position_count'] for r in rr),q['mean_full_vocab_KL']);close(sum(r['flip_count'] for r in rr)/n,q['flip_rate'])
  diffs=[(r['NLL_sum']-b['NLL_sum'])/255 for r,b in zip(rr,base)];close(math.fsum(diffs)/len(diffs),q['DeltaNLL_vs_alpha0_doc_mean']);close((s-math.fsum(r['NLL_sum'] for r in base))/n,q['DeltaNLL_vs_alpha0_token_weighted'])
  if bootstrap:
   import numpy as np
   rng=np.random.default_rng(6101516);ix=rng.integers(0,len(diffs),size=(10000,len(diffs)));ci=np.quantile(np.asarray(diffs)[ix].mean(axis=1),[.025,.975])
   for a,b in zip(ci,q['DeltaNLL_pointwise_CI95']):close(float(a),b)
   bootchecks+=1
  for r,b in zip(rr,base):
   tol=1e-9*max(1,r['SSE_before'],r['SSE_after']);labels=[]
   if r['T_inherited'] < -tol and r['T_cont'] > tol:labels.append('A_CANCELLATION_LOSS')
   if r['T_cont'] < -tol and r['Delta'] > tol:labels.append('B_REQUANTIZATION_REVERSAL')
   cal=groups[(r['block'],r['projection'],'CAL-FIT',r['alpha'])]
   if r['split']!='CAL-FIT' and math.fsum(v['T_inherited'] for v in cal)<=tol and r['T_inherited']>tol:labels.append('C_CONTINUOUS_TRANSFER_LOSS')
   if r['Delta'] < -tol and (r['NLL_sum']-b['NLL_sum'])/255>1e-9:labels.append('D_LOCAL_NLL_DISAGREEMENT')
   classified.update(labels);newclass[(r['block'],r['projection'],r['split'],r['alpha'],r['document_id_hash'])]=labels
 for r in json.loads((run/'classifications.json').read_text()):assert r['labels']==newclass[(r['block'],r['projection'],r['split'],r['alpha'],r['document_id_hash'])]
 assert dict(classified)==summary['classification_counts']
 calls=[json.loads(s) for s in (run/'forward_calls.jsonl').read_text().splitlines()];counts=dict(collections.Counter(r['stage'] for r in calls));assert counts=={'SMOKE':5,'CAL_TEACHER':32,'CAL_PREFIX':96,'HELDOUT_TEACHER':64,'HELDOUT_PREFIX':192,'GRID':2880,'CONTINUOUS':576}
 selected=json.loads((run/'fit_frozen.json').read_text())['selected']
 for r in selected:
  b=int(r['target'].split('.')[2]);p=r['target'].split('.')[-1];a={alpha:math.fsum(x['SSE_after'] for x in groups[(b,p,'CAL-FIT',alpha)]) for alpha in [0,.25,.5,.75,1]};assert min(a,key=lambda x:(a[x],x))==r['selected_alpha']
 artifacts=json.loads((run/'artifact_manifest.json').read_text())['candidates'];assert len(artifacts)==30
 for p in ['o_proj','down_proj']:assert {r['artifact_bytes'] for r in artifacts if r['target'].endswith(p)}=={1179684 if p=='o_proj' else 1769508}
 return {'status':'PASS','raw_rows':len(raw),'summary_rows':120,'CAL_selection_checked':6,'full_model_forward_records':len(calls),'NLL_KL_flip_payload_and_classification_aggregates':'PASS','bootstrap_intervals_recomputed':bootchecks,'CPU_unique_unit_tests':'separate15 historical tests; auditor is not added to that count','P0_MC_rerun':False,'limits':'No model weights/activation arrays/logits in bundle; no GPU/NLL/logit re-evaluation or external weight hash verification.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',default=str(Path(__file__).resolve().parents[2]));p.add_argument('--output',required=True);p.add_argument('--bootstrap',action='store_true');a=p.parse_args();v=audit(a.case,a.bootstrap);Path(a.output).write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
