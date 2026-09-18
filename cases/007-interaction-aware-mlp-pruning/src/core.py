"""CPU accounting, exhaustive selectors, scoring and paired statistics."""
from __future__ import annotations
import hashlib,itertools,json,math,random
from pathlib import Path
import numpy as np

SEED=707180
TASKS=('RETRIEVAL','COMPARISON','CODE')
COUNTS={'CALIBRATION':96,'DEVELOPMENT':48,'HELD_OUT':192}

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def file_hash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write_new(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def groups(width:int,n:int=16)->list[list[int]]:
 if width<n:raise ValueError('Too few channels')
 return [list(map(int,g)) for g in np.array_split(np.arange(width),n)]
def kept(mapping,removed):
 if len(set(removed))!=len(removed) or any(i not in range(len(mapping)) for i in removed):raise ValueError('Invalid groups')
 return [j for i,g in enumerate(mapping) if i not in removed for j in g]
def positions(length):return np.floor(np.linspace(0,length-1,min(32,length))).astype(int).tolist()
def candidates(budget):return list(itertools.combinations(range(16),budget))
def random_sets(budget):
 rng=random.Random(SEED+budget);return [list(p) for p in rng.sample(candidates(budget),20)]
def select(q,budget,method):
 q=np.asarray(q,dtype=np.float64)
 if q.shape!=(16,16) or not np.isfinite(q).all():raise ValueError('Invalid Q')
 if method not in ('INDEPENDENT','PAIRWISE'):raise ValueError('Unknown method')
 sets=candidates(budget)
 costs=[float(np.trace(q[np.ix_(p,p)])) if method=='INDEPENDENT' else float(q[np.ix_(p,p)].sum()) for p in sets]
 i=min(range(len(sets)),key=lambda i:(costs[i],sets[i]))
 return {'removed':list(sets[i]),'objective':costs[i],'enumerated':len(sets)}
def calibration_q(records):
 if len(records)!=96 or any(r['split']!='CALIBRATION' for r in records):raise ValueError('Calibration-only selection requires 96 prompts')
 if len({r['id'] for r in records})!=96:raise ValueError('Duplicate calibration prompt')
 return np.mean([r['Q'] for r in records],axis=0)
def norm_stats(a,r):
 a=np.asarray(a,dtype=np.float64);r=np.asarray(r,dtype=np.float64)
 if a.shape!=r.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(r).all():raise ValueError('Invalid full local output')
 e2=float(np.square(a-r).sum());r2=float(np.square(r).sum());a2=float(np.square(a).sum());dot=float((a*r).sum());n=r.size
 return {'e2':e2,'r2':r2,'a2':a2,'dot':dot,'n':n,'relative_error':math.sqrt(e2/r2) if math.sqrt(r2/n)>1e-6 else None,'absolute_rms':math.sqrt(e2/n),'reference_rms':math.sqrt(r2/n),'cosine':dot/math.sqrt(a2*r2) if a2*r2>0 else None}
def lse(z):
 z=np.asarray(z,dtype=np.float64)
 if not np.isfinite(z).all():raise ValueError('Nonfinite logits')
 m=z.max();return float(m+np.log(np.exp(z-m).sum()))
def score(z,label_ids,gold):
 z=np.asarray(z,dtype=np.float64);s=z[label_ids];l=lse(s);full=lse(z);q=np.exp(s-l);t=np.eye(4)[gold];pred=int(np.argmax(s))
 return {'option_logits':s.tolist(),'gold':gold,'prediction':pred,'correct':pred==gold,'nll':l-float(s[gold]),'brier':float(((q-t)**2).sum()),'margin':float(s[gold]-np.max(np.delete(s,gold))),'label_mass':math.exp(l-full),'full_lse':full,'full_argmax':int(z.argmax()),'full_gold_nll':full-float(s[gold]),'tie':int(np.sum(s==s.max()))>1}
def kl(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 if a.shape!=b.shape:raise ValueError('KL shape')
 la=a-lse(a);lb=b-lse(b);return float((np.exp(la)*(la-lb)).sum())
def transition(b,c):
 if b['gold']!=c['gold']:raise ValueError('Gold mismatch')
 cell='both_correct' if b['correct'] and c['correct'] else 'regression' if b['correct'] else 'gain' if c['correct'] else 'both_wrong'
 return {'cell':cell,'flip':b['prediction']!=c['prediction'],'wrong_to_wrong':cell=='both_wrong' and b['prediction']!=c['prediction']}
def join(rows,methods):
 out={}
 for r in rows:
  key=(r['id'],r['method'])
  if key in out:raise ValueError('Duplicate pair cell')
  if r.get('evidence_kind')!='gpu_measurement' or not r.get('valid'):raise ValueError('Unverified record')
  out[key]=r
 ids=sorted({r['id'] for r in rows})
 for i in ids:
  if any((i,m) not in out for m in methods):raise ValueError('Missing pair')
  if len({out[i,m]['token_hash'] for m in methods})!=1:raise ValueError('Token mismatch')
 return out,ids

def bootstrap_indices(tasks,repeats=5000,seed=SEED):
 tasks=np.asarray(tasks);rng=np.random.default_rng(seed);parts=[]
 for task in TASKS:
  ids=np.flatnonzero(tasks==task)
  if len(ids)==0:raise ValueError('Missing task')
  parts.append(rng.choice(ids,(repeats,len(ids)),replace=True))
 return np.concatenate(parts,axis=1)
def interval(values,indices):
 a=np.asarray(values,dtype=np.float64)
 if not np.isfinite(a).all():raise ValueError('Undefined metric')
 return np.quantile(a[indices].mean(axis=1),[.025,.975],method='linear').tolist()
def decision(intervals,valid):
 if not valid:return 'BLOCKED_IMPLEMENTATION'
 if len(intervals)!=2:return 'BLOCKED_IMPLEMENTATION'
 if all(ci[1]<0 for ci in intervals):return 'COMPLETED_POSITIVE_TRANSFER'
 if all(ci[0]>0 for ci in intervals):return 'COMPLETED_NEGATIVE_TRANSFER'
 return 'COMPLETED_NO_CLEAR_TRANSFER'
