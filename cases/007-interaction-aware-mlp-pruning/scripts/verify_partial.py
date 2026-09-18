"""Separate scalar path for partial evidence; no analysis/core helper reuse."""
import argparse,json,itertools,math,hashlib
from pathlib import Path
import numpy as np

def read(p):return json.loads(p.read_text())
def close(a,b):
 if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-11):raise ValueError(f'{a} != {b}')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();summary=read(a.summary);sel=read(a.raw/'selection.json');cal=[read(f) for f in sorted((a.raw/'calibrate').glob('c007-*.json'))];assert len(cal)==96 and all(r['split']=='CALIBRATION' for r in cal)
 q=[[math.fsum(r['Q'][i][j] for r in cal)/96 for j in range(16)] for i in range(16)]
 for name,s in sel['selections'].items():
  scores=[]
  for ps in itertools.combinations(range(16),s['budget']):
   cost=math.fsum(q[i][i] for i in ps) if s['method']=='INDEPENDENT' else math.fsum(q[i][j] for i in ps for j in ps);scores.append((cost,ps))
  v=min(scores);assert list(v[1])==s['removed'];close(v[0],s['objective'])
 rows=[read(f) for f in sorted((a.raw/'development').glob('c007-*.json'))];count=0
 for split,n in [('CALIBRATION',96),('DEVELOPMENT',48)]:
  rs=[r for r in rows if r['split']==split];assert len(rs)==n and len({r['id'] for r in rs})==n
  for r in rs:
   assert r['valid'] is True and r['evidence_kind']=='gpu_measurement';assert r['selection_sha256']==hashlib.sha256((a.raw/'selection.json').read_bytes()).hexdigest()
   for u in r['local']:
    for v in [u,*u['token_metrics']]:close(v['relative_error'],math.sqrt(v['e2']/v['r2']));close(v['absolute_rms'],math.sqrt(v['e2']/v['n']));close(v['cosine'],v['dot']/math.sqrt(v['a2']*v['r2']));count+=1
  for method in summary['splits'][split]['methods']:
   vals=[math.sqrt(next(u for u in r['local'] if u['method']==method)['e2']/next(u for u in r['local'] if u['method']==method)['r2']) for r in rs];d=summary['splits'][split]['methods'][method]['relative_error'];close(math.fsum(vals)/n,d['mean']);close(float(np.quantile(vals,.5)),d['median']);close(float(np.quantile(vals,.95)),d['p95'])
  rng=np.random.default_rng(707180);parts=[]
  for task in ['RETRIEVAL','COMPARISON','CODE']:
   ids=[i for i,r in enumerate(rs) if r['task']==task];parts.append(rng.choice(ids,(5000,len(ids)),replace=True))
  ix=np.concatenate(parts,axis=1)
  for b in [4,8]:
   dif=[]
   for r in rs:
    u={v['method']:v for v in r['local']};dif.append(math.sqrt(u[f'PAIRWISE_{b}']['e2']/u[f'PAIRWISE_{b}']['r2'])-math.sqrt(u[f'INDEPENDENT_{b}']['e2']/u[f'INDEPENDENT_{b}']['r2']))
   ci=np.quantile(np.array(dif)[ix].mean(axis=1),[.025,.975]);expected=summary['splits'][split]['paired_differences'][str(b)];close(math.fsum(dif)/n,expected['mean_pairwise_minus_independent'])
   for x,y in zip(ci,expected['ci95']):close(x,y)
 assert summary['status']=='BLOCKED_RESOURCE' and not list((a.raw/'heldout').glob('c007-*.json')) and summary['heldout_model_calls']==0
 result={'status':'PASS','prompts':144,'methods_per_prompt':44,'norm_records_recomputed':count,'selectors_recomputed':4,'paired_intervals_recomputed':4,'heldout':'NOT_RUN','timing':'NOT_RUN','scope':'Independent scalar arithmetic consistency; cannot reproduce excluded tensors or GPU outputs'}
 with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
