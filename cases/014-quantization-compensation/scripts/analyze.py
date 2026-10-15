"""Derive all-target tables and document paired exploratory uncertainty from raw rows."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from common import save,sha

def run(root):
 root=Path(root);p1=root/'p1';files=sorted((p1/'cells').glob('*.jsonl'))
 if not files:return {'P1':'NOT_RUN'}
 rows=[json.loads(s) for p in files for s in p.read_text().splitlines()];lookup={(r['block'],r['projection'],r['split'],r['document_id_hash'],r['alpha']):r for r in rows}
 selected=json.loads((p1/'fit_frozen.json').read_text())['selected'];sel={(int(r['target'].split('.')[2]),r['target'].split('.')[-1]):r['selected_alpha'] for r in selected}
 groups={}
 for r in rows:groups.setdefault((r['block'],r['projection'],r['split'],r['alpha']),[]).append(r)
 summary=[];classifications=[]
 for key,rs in sorted(groups.items()):
  l,p,split,a=key;rs=sorted(rs,key=lambda r:r['document_id_hash']);base=[lookup[(l,p,split,r['document_id_hash'],0)] for r in rs]
  nll=np.array([(r['NLL_sum']-b['NLL_sum'])/r['valid_token_count'] for r,b in zip(rs,base)]);local=np.array([r['SSE_after']-r['SSE_before'] for r in rs]);rng=np.random.default_rng(6101516);ix=rng.integers(0,len(rs),size=(10000,len(rs)));ci=np.quantile(nll[ix].mean(axis=1),[.025,.975]);s=lambda k:math.fsum(r[k] for r in rs)
  tokens=sum(r['valid_token_count'] for r in rs);base_tokens=sum(r['valid_token_count'] for r in base)
  row={'block':l,'projection':p,'split':split,'alpha':a,'selected_CAL':a==sel[(l,p)],'documents':len(rs),'valid_tokens':tokens,'NLL_sum':s('NLL_sum'),'NLL_mean':s('NLL_sum')/tokens,'PPL':math.exp(s('NLL_sum')/tokens),'DeltaNLL_vs_alpha0_doc_mean':float(nll.mean()),'DeltaNLL_vs_alpha0_token_weighted':(s('NLL_sum')-math.fsum(b['NLL_sum'] for b in base))/tokens,'DeltaNLL_pointwise_CI95':ci.tolist(),'SSE_after':s('SSE_after'),'SSE_before':s('SSE_before'),'DeltaSSE':float(local.sum()),'DeltaSSE_percent':float(local.sum()/s('SSE_before')*100) if s('SSE_before') else None,'reference_SSE':s('reference_SSE'),'normalized_SSE':s('SSE_after')/s('reference_SSE'),'fixedB_SSE':s('fixedB_SSE'),'A_norm2':s('A_norm2'),'T_inherited':s('T_inherited'),'T_cancellation':s('T_cancellation'),'T_cont':s('T_cont'),'T_requant':s('T_requant'),'mean_full_vocab_KL':s('sampled_full_vocab_KL_sum')/sum(r['KL_position_count'] for r in rs),'flip_rate':sum(r['flip_count'] for r in rs)/tokens,'arithmetic_residual_norm2':s('arithmetic_residual_norm2'),'cast_residual_norm2':s('cast_residual_norm2'),'max_identity_relative_gap':max(r['identity_relative_gap'] for r in rs),'artifact_bytes':rs[0]['artifact_bytes'],'code_change_rate':rs[0]['code_change_rate'],'scale_change_mean':rs[0]['scale_change_mean'],'scale_change_max':rs[0]['scale_change_max'],'endpoint_rate':rs[0]['endpoint_rate'],'clipping_rate':rs[0]['clipping_rate']}
  summary.append(row)
  for r,b in zip(rs,base):
   tol=1e-9*max(1,r['SSE_before'],r['SSE_after']);labels=[]
   if r['T_inherited'] < -tol and r['T_cont'] > tol:labels.append('A_CANCELLATION_LOSS')
   if r['T_cont'] < -tol and r['Delta'] > tol:labels.append('B_REQUANTIZATION_REVERSAL')
   cal_inherit=math.fsum(q['T_inherited'] for q in groups[(l,p,'CAL-FIT',a)])
   if split!='CAL-FIT' and cal_inherit<=tol and r['T_inherited']>tol:labels.append('C_CONTINUOUS_TRANSFER_LOSS')
   if r['Delta'] < -tol and (r['NLL_sum']-b['NLL_sum'])/255>1e-9:labels.append('D_LOCAL_NLL_DISAGREEMENT')
   classifications.append({'block':l,'projection':p,'split':split,'alpha':a,'document_id_hash':r['document_id_hash'],'labels':labels,'numerical_sign_tolerance':tol,'near_zero_Delta':abs(r['Delta'])<=tol,'arithmetic_residual_relative_to_R':(r['arithmetic_residual_norm2']/r['SSE_after'])**.5 if r['SSE_after']>0 else None})
 ranks=[]
 for l,p in sel:
  cal=sorted([r for r in summary if (r['block'],r['projection'],r['split'])==(l,p,'CAL-FIT')],key=lambda r:(r['SSE_after'],r['alpha']));ev=sorted([r for r in summary if (r['block'],r['projection'],r['split'])==(l,p,'EVAL-ID')],key=lambda r:(r['SSE_after'],r['alpha']))
  cr=[next(i+1 for i,r in enumerate(cal) if r['alpha']==a) for a in [0,.25,.5,.75,1]];er=[next(i+1 for i,r in enumerate(ev) if r['alpha']==a) for a in [0,.25,.5,.75,1]]
  ranks.append({'block':l,'projection':p,'CAL_order':[r['alpha'] for r in cal],'EVAL_order_posthoc':[r['alpha'] for r in ev],'selected_CAL':sel[(l,p)],'Spearman_descriptive':float(np.corrcoef(cr,er)[0,1]),'CAL_exact_ties':len({r['SSE_after'] for r in cal})<5,'EVAL_exact_ties':len({r['SSE_after'] for r in ev})<5,'ties_policy':'smaller alpha for display; correlation ranks require caution if ties exist'})
 result={'rows':summary,'rankings':ranks,'classification_counts':{k:sum(k in r['labels'] for r in classifications) for k in ['A_CANCELLATION_LOSS','B_REQUANTIZATION_REVERSAL','C_CONTINUOUS_TRANSFER_LOSS','D_LOCAL_NLL_DISAGREEMENT']},'classification_counts_unit':'document × target × nonzero alpha observations; labels overlap, repeated documents not independent','unique_EVAL_documents':32,'uncertainty':'10000 document-paired bootstrap, seed6101516, 95% exploratory pointwise; no familywise or generalization guarantee','raw_file_hashes':{str(p.relative_to(root)):sha(p) for p in files}}
 save(root/'derived_summary.json',result);save(root/'classifications.json',classifications)
 with (root/'summary.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args();r=run(a.results);print({k:v for k,v in r.items() if k not in ['rows','raw_file_hashes']})
