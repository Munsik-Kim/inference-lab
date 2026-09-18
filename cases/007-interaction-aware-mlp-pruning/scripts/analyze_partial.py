"""CPU summary for completed calibration/development when HELD_OUT is blocked."""
import argparse,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import load,write_new,select,calibration_q,bootstrap_indices,interval

def summarize(raw):
 status=load(raw/'heldout/status.json')
 if status['status']!='BLOCKED_RESOURCE' or status['model_forwards']!=0:raise ValueError('Not the blocked pre-heldout state')
 if list((raw/'heldout').glob('c007-*.json')):raise ValueError('Unexpected heldout measurements')
 sels=load(raw/'selection.json');cal=[load(f) for f in sorted((raw/'calibrate').glob('c007-*.json'))];q=calibration_q(cal);rows=[load(f) for f in sorted((raw/'development').glob('c007-*.json'))]
 if any(r.get('evidence_kind')!='gpu_measurement' or r.get('valid') is not True for r in [*cal,*rows]):raise ValueError('Mock/invalid measurement forbidden')
 summary={'status':'BLOCKED_RESOURCE','deployment':'NOT_ASSESSED','heldout_prompts_measured':0,'heldout_model_calls':0,'timing':'NOT_RUN','selection':{},'splits':{}}
 for name,s in sels['selections'].items():
  again=select(q,s['budget'],s['method'])
  if again['removed']!=s['removed']:raise ValueError('Selection not reproducible')
  ps=s['removed'];summary['selection'][name]={**s,'J_ind_common':float(np.trace(q[np.ix_(ps,ps)])),'J_pair_common':float(q[np.ix_(ps,ps)].sum())}
 for split,n in [('CALIBRATION',96),('DEVELOPMENT',48)]:
  rs=[r for r in rows if r['split']==split]
  if len(rs)!=n or len({r['id'] for r in rs})!=n or any(not r['valid'] for r in rs):raise ValueError('Invalid split')
  per={};names=[u['method'] for u in rs[0]['local']]
  for method in names:
   units=[next(u for u in r['local'] if u['method']==method) for r in rs];metrics={}
   for metric in ['relative_error','absolute_rms','cosine']:
    a=np.array([u[metric] for u in units],float)
    if not np.isfinite(a).all():raise ValueError('Undefined metric')
    metrics[metric]={'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}
   metrics['worst_prompt']=rs[int(np.argmax([u['relative_error'] for u in units]))]['id'];per[method]=metrics
  differences={};idx=bootstrap_indices([r['task'] for r in rs])
  for b in [4,8]:
   diff=[]
   for r in rs:
    u={x['method']:x for x in r['local']};diff.append(u[f'PAIRWISE_{b}']['relative_error']-u[f'INDEPENDENT_{b}']['relative_error'])
   differences[str(b)]={'mean_pairwise_minus_independent':float(np.mean(diff)),'ci95':interval(diff,idx),'pairwise_better_prompts':int(np.sum(np.asarray(diff)<0)),'n':n,'interpretation':'DEVELOPMENT_DESCRIPTIVE_NOT_HELDOUT' if split=='DEVELOPMENT' else 'CALIBRATION_IN_SAMPLE'}
  summary['splits'][split]={'prompts':n,'methods':per,'paired_differences':differences}
 off=q[np.triu_indices(16,1)];summary['calibration_Q']={'Q':q.tolist(),'negative_pairs':int((off<0).sum()),'positive_pairs':int((off>0).sum()),'zero_pairs':int((off==0).sum()),'eigenvalues':np.linalg.eigvalsh(q).tolist(),'pre_symmetry_max_abs':max(r['Q_diagnostics']['pre_symmetry_max_abs'] for r in cal),'diagonal_max_abs':max(r['Q_diagnostics']['diagonal_max_abs'] for r in cal)}
 return summary

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();write_new(a.output,summarize(a.raw));print('BLOCKED_RESOURCE summary regenerated')
if __name__=='__main__':main()
