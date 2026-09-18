"""Regenerate CPU summaries from immutable scalar records into a NEW directory."""
import argparse,sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import *

def describe(a):
 a=np.asarray(a,dtype=float)
 if not len(a) or not np.isfinite(a).all():raise ValueError('Missing/undefined observations')
 return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}
def timing_summary(rows):
 result={};rng=np.random.default_rng(SEED)
 for boundary in ['MLP','MODEL_PREFILL']:
  cell=[r for r in rows if r['boundary']==boundary];lookup={(r['id'],r['round'],r['block'],r['method']):r for r in cell}
  if len(lookup)!=len(cell):raise ValueError('Duplicate timing')
  ids=sorted({r['id'] for r in cell});methods=sorted({r['method'] for r in cell});tasks=[next(r['task'] for r in cell if r['id']==i) for i in ids];inds=bootstrap_indices(tasks)
  # Shared global process draws across documents, same draws across methods.
  rd=rng.integers(0,3,(5000,3));bi=rng.integers(0,5,(5000,len(ids),3,5))
  per={}
  for m in methods:
   ratios=np.array([[[lookup[i,rr,b,'B']['wall_ms']/lookup[i,rr,b,m]['wall_ms'] for b in range(5)] for rr in range(3)] for i in ids]);lat=[r['wall_ms'] for r in cell if r['method']==m];draws=[]
   for j in range(5000):draws.append(float(np.median(ratios[inds[j,:,None,None],rd[j,None,:,None],bi[j]])))
   per[m]={'speedup_median':float(np.median(ratios)),'ci95':np.quantile(draws,[.025,.975]).tolist(),'wall_ms':describe(lat),'event_ms':describe([r['event_ms'] for r in cell if r['method']==m])}
  result[boundary]=per
 return result

def analyze(raw):
 sels=load(raw/'selection.json')['selections'];local=[];model=[]
 for stage in ['development','heldout']:
  for f in sorted((raw/stage).glob('c007-*.json')):
   r=load(f)
   if r.get('evidence_kind')!='gpu_measurement' or r.get('valid') is not True:raise ValueError('Mock/invalid measurement')
   if 'local' in r:local.append(r)
   if r['split']=='HELD_OUT':model.append(r)
 lookup,ids=join(model,['B',*sels]);out={'deployment':'NOT_ASSESSED','scope':'Qwen3-0.6B layer13 BF16 MLP, 512 tokens, 32 sampled positions; synthetic English tasks','local':{},'transfer':{},'model':{},'timing':{}};intervals=[]
 for split,n in COUNTS.items():
  rs=[r for r in local if r['split']==split]
  if len(rs)!=n or len({r['id'] for r in rs})!=n:raise ValueError('Wrong local split coverage')
  table={}
  for name in [*sels,*[f'RANDOM_{b}_{i:02d}' for b in [4,8] for i in range(20)]]:
   vals=[next(u for u in r['local'] if u['method']==name) for r in rs]
   table[name]={'relative_error':describe([u['relative_error'] for u in vals]),'absolute_rms':describe([u['absolute_rms'] for u in vals]),'cosine':describe([u['cosine'] for u in vals]),'worst_prompt':rs[int(np.argmax([u['relative_error'] for u in vals]))]['id']}
  out['local'][split]=table
 held=[r for r in local if r['split']=='HELD_OUT'];inds=bootstrap_indices([r['task'] for r in held]);paired=[]
 for b in [4,8]:
  dif=[]
  for r in held:
   u={x['method']:x for x in r['local']};i=u[f'INDEPENDENT_{b}'];p=u[f'PAIRWISE_{b}'];d=p['relative_error']-i['relative_error'];dif.append(d);paired.append({'id':r['id'],'task':r['task'],'budget':b,'independent_error':i['relative_error'],'pairwise_error':p['relative_error'],'difference':d})
  ci=interval(dif,inds);intervals.append(ci);out['transfer'][str(b)]={'mean_paired_relative_error_delta':float(np.mean(dif)),'ci95':ci,'pairwise_better_prompts':int(np.sum(np.asarray(dif)<0)),'n':len(dif)}
 for name in ['B',*sels]:
  rs=[lookup[i,name] for i in ids];bs=[lookup[i,'B'] for i in ids];tasks=[r['task'] for r in rs];idx=bootstrap_indices(tasks);cells=[transition(b['score'],r['score']) for b,r in zip(bs,rs)]
  out['model'][name]={'n':len(rs),'correct':sum(r['score']['correct'] for r in rs),'per_task_correct':{t:sum(r['score']['correct'] for r in rs if r['task']==t) for t in TASKS},'transitions':{k:sum(c['cell']==k for c in cells) for k in ['both_correct','regression','gain','both_wrong']},'choice_flips':sum(c['flip'] for c in cells),'wrong_to_wrong':sum(c['wrong_to_wrong'] for c in cells),'vocab_top1_changes':sum(r['score']['full_argmax']!=b['score']['full_argmax'] for r,b in zip(rs,bs)),'mean_label_mass':float(np.mean([r['score']['label_mass'] for r in rs])),'mean_full_vocab_KL':float(np.mean([r.get('full_vocab_kl_B_to_candidate',0) for r in rs]))}
  for metric in ['nll','brier','margin']:
   diff=[r['score'][metric]-b['score'][metric] for r,b in zip(rs,bs)];out['model'][name]['delta_'+metric]={'mean':float(np.mean(diff)),'ci95':interval(diff,idx)}
 out['status']=decision(intervals,all(r['valid'] for r in model))
 trows=[]
 for i in range(3):trows+=load(raw/f'timing{i}/blocks.json')
 out['timing']=timing_summary(trows)
 return out,paired

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 summary,pairs=analyze(a.raw);a.output.mkdir(parents=True);write_new(a.output/'summary.json',summary);write_new(a.output/'pairs.json',pairs);print(summary['status'])
if __name__=='__main__':main()
