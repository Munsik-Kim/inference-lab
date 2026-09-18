"""Independent CPU arithmetic check. No GPU imports and no analysis helper reuse."""
import argparse,itertools,json,hashlib,math,sys
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def close(a,b):
 if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-11):raise ValueError(f'Numerical mismatch {a} != {b}')
def scalar_score(s,g,full):
 m=max(s);ln=m+math.log(math.fsum(math.exp(x-m) for x in s));p=[math.exp(x-ln) for x in s]
 return {'nll':ln-s[g],'brier':math.fsum((x-(j==g))**2 for j,x in enumerate(p)),'margin':s[g]-max(x for j,x in enumerate(s) if j!=g),'prediction':max(range(4),key=lambda i:s[i]),'label_mass':math.exp(ln-full)}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();raw=a.raw;summary=read(a.summary);sel=read(raw/'selection.json');rows=[read(f) for f in sorted((raw/'calibrate').glob('c007-*.json'))];assert len(rows)==96 and all(r['split']=='CALIBRATION' for r in rows)
 q=[[math.fsum(r['Q'][i][j] for r in rows)/96 for j in range(16)] for i in range(16)]
 for name,s in sel['selections'].items():
  best=None
  for ps in itertools.combinations(range(16),s['budget']):
   cost=math.fsum(q[i][i] for i in ps) if s['method']=='INDEPENDENT' else math.fsum(q[i][j] for i in ps for j in ps)
   if best is None or (cost,ps)<best:best=(cost,ps)
  assert list(best[1])==s['removed'];close(best[0],s['objective'])
 local={};scalars=0;models={}
 for stage in ['development','heldout']:
  for f in sorted((raw/stage).glob('c007-*.json')):
   r=read(f);assert r['valid'] is True and r['evidence_kind']=='gpu_measurement';assert r['selection_sha256']==hashlib.sha256((raw/'selection.json').read_bytes()).hexdigest()
   for u in r.get('local',[]):
    for v in [u,*u['token_metrics']]:
     close(v['relative_error'],math.sqrt(v['e2']/v['r2']));close(v['absolute_rms'],math.sqrt(v['e2']/v['n']));close(v['cosine'],v['dot']/math.sqrt(v['a2']*v['r2']));scalars+=1
    key=(r['split'],u['method']);local.setdefault(key,[]).append((r['id'],r['task'],math.sqrt(u['e2']/u['r2'])))
   if stage=='heldout':
    key=(r['id'],r['method']);assert key not in models;models[key]=r
    v=scalar_score(r['score']['option_logits'],r['score']['gold'],r['score']['full_lse'])
    for k in v:close(v[k],r['score'][k])
 for (split,m),vals in local.items():
  x=np.array([v[2] for v in vals]);s=summary['local'][split][m]['relative_error'];close(s['mean'],math.fsum(x)/len(x));close(s['median'],float(np.quantile(x,.5)));close(s['p95'],float(np.quantile(x,.95)))
 cis=[]
 for b in [4,8]:
  iv=sorted(local['HELD_OUT',f'INDEPENDENT_{b}']);pv=sorted(local['HELD_OUT',f'PAIRWISE_{b}']);assert [r[0] for r in iv]==[r[0] for r in pv];diff=np.array([p[2]-i[2] for p,i in zip(pv,iv)]);rng=np.random.default_rng(707180);pieces=[]
  for t in ['RETRIEVAL','COMPARISON','CODE']:
   ix=[i for i,r in enumerate(iv) if r[1]==t];pieces.append(rng.choice(ix,(5000,len(ix)),replace=True))
  # Independently generated indices, same seeded sampling design.
  ix=np.concatenate(pieces,axis=1);ci=np.quantile(np.mean(diff[ix],axis=1),[.025,.975]);cis.append(ci)
  close(float(np.mean(diff)),summary['transfer'][str(b)]['mean_paired_relative_error_delta'])
  for actual,expected in zip(ci,summary['transfer'][str(b)]['ci95']):close(actual,expected)
 ids=sorted({k[0] for k in models});assert len(ids)==192
 for m in ['B',*sel['selections']]:
  counts=dict(both_correct=0,regression=0,gain=0,both_wrong=0);flips=0
  for i in ids:
   br=models[i,'B'];cr=models[i,m];assert br['token_hash']==cr['token_hash'] and br['mlp_input_hash']==cr['mlp_input_hash'];bs=br['score'];cs=cr['score'];bk=bs['prediction']==bs['gold'];ck=cs['prediction']==cs['gold'];cell='both_correct' if bk and ck else 'regression' if bk else 'gain' if ck else 'both_wrong';counts[cell]+=1;flips+=bs['prediction']!=cs['prediction']
  assert counts==summary['model'][m]['transitions'] and flips==summary['model'][m]['choice_flips']
 tr=[]
 for rr in range(3):tr+=read(raw/f'timing{rr}/blocks.json')
 for boundary in ['MLP','MODEL_PREFILL']:
  lut={(r['id'],r['round'],r['block'],r['method']):r for r in tr if r['boundary']==boundary};assert len(lut)==450
  for m in ['B',*sel['selections']]:
   ratios=[v['wall_ms']/lut[(i,rr,b,m)]['wall_ms'] for (i,rr,b,n),v in lut.items() if n=='B'];close(float(np.median(ratios)),summary['timing'][boundary][m]['speedup_median'])
 decision='COMPLETED_POSITIVE_TRANSFER' if all(c[1]<0 for c in cis) else 'COMPLETED_NEGATIVE_TRANSFER' if all(c[0]>0 for c in cis) else 'COMPLETED_NO_CLEAR_TRANSFER';assert summary['status']==decision
 result={'status':'PASS','local_norm_records_recomputed':scalars,'model_cells':len(models),'independent_selectors':4,'paired_transfer_intervals':2,'timing_ratios':'independent scalar medians checked; analysis rerun checks full timing CI determinism','limits':['Calculation consistency, not independent GPU replication','Excluded vectors needed to verify full-vocabulary KL and native output squared sums','Option scores rederived from stored logits; local errors from stored squared norms'],'decision':decision}
 with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
