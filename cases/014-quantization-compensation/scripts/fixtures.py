"""Exact independent examples and paired own-trajectory infinite-grid dither MC."""
from fractions import Fraction as F
import numpy as np
from common import save

def exact():
 fixed=[]
 for eps in [F(1,1000),F(1,100),F(1,10)]:
  for sign in [-1,1]:
   base=(1+eps)*(1+sign*eps);comp=(1+eps)*(1/(1+eps)+sign*eps)
   fixed.append({'epsilon':str(eps),'second_error_sign':sign,'BASE':str(base),'COMP':str(comp),'BASE_abs_error':str(abs(1-base)),'COMP_abs_error':str(abs(1-comp))})
 w=[F(4,5),F(29,10),F(13,10)];teacher=F(1);zbase=zcomp=F(1);rows=[]
 for i,wi in enumerate(w):
  teacher*=wi;bc=min(range(4),key=lambda q:abs(F(q)-wi));loss=[(teacher-q*zcomp)**2 for q in range(4)];cc=min(range(4),key=lambda q:loss[q]);target=teacher/zcomp
  rows.append({'layer':i,'teacher_target':str(teacher),'BASE_input':str(zbase),'COMP_input':str(zcomp),'continuous_COMP_target':str(target),'COMP_losses':[str(a) for a in loss],'BASE_code':bc,'COMP_code':cc,'unique':loss.count(loss[cc])==1,'overload':not F(0)<=target<=F(3)})
  zbase*=bc;zcomp*=cc
 return {'fixed_errors':fixed,'finite_codebook':{'teacher':str(teacher),'BASE_output':str(zbase),'COMP_output':str(zcomp),'BASE_abs_error':str(abs(teacher-zbase)),'COMP_abs_error':str(abs(teacher-zcomp)),'layers':rows,'origin':'independent user-supplied 2-bit scalar fixture; not paper Theorem22 numbers'}}

def dither(n=2000):
 rng=np.random.default_rng(6101501);x=rng.normal(size=(4,16));allrows=[]
 for step in [.01,.1]:
  for depth in [2,8,16]:
   errors={a:[] for a in [0,.5,1]};failures=[];maxcond={a:0. for a in errors}
   for seed in range(n):
    u=np.random.default_rng(61015000+seed).uniform(-step/2,step/2,size=(depth,4,4))
    for alpha in errors:
     z=x.copy()
     try:
      for layer in range(depth):
       gram=z@z.T;cond=float(np.linalg.cond(gram));maxcond[alpha]=max(maxcond[alpha],cond)
       if np.linalg.matrix_rank(gram)!=4:raise ValueError('rank failure')
       target=np.eye(4)
       if alpha:target=target+alpha*np.linalg.solve(gram,((x-z)@z.T).T).T
       q=step*np.rint((target+u[layer])/step)-u[layer];z=q@z
       if not np.isfinite(z).all():raise ValueError('nonfinite')
      errors[alpha].append(float(((x-z)**2).sum()))
     except Exception as e:failures.append({'seed':seed,'alpha':alpha,'layer':layer,'error':str(e)});errors[alpha].append(None)
   base=np.array([np.nan if v is None else v for v in errors[0]])
   for alpha,values in errors.items():
    a=np.array([np.nan if v is None else v for v in values]);v=a-base;valid=bool(np.isfinite(v).all())
    se=float(v.std(ddof=1)/n**.5) if valid else None;m=float(v.mean()) if valid else None
    allrows.append({'depth':depth,'step':step,'alpha':alpha,'n_seeds':n,'seed_SSE':values,'mean_SSE':float(a.mean()) if valid else None,'paired_delta':m,'SE':se,'MC95':[m-1.96*se,m+1.96*se] if valid else None,'max_Gram_condition':maxcond[alpha],'failures':failures,'status':'VALID' if valid else 'ASSUMPTION_OR_NUMERICAL_FAILURE','seeds_start':61015000,'sampling_unit':'independent dither seed; common random numbers across own trajectories'})
 return {'X':x.tolist(),'rows':allrows,'raw_policy':'all failure seeds retained; no clipping/ridge/selection','dither_real_not_INT4':True}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seeds',type=int,default=2000);a=p.parse_args()
 save(a.output,{'exact':exact(),'dither':dither(a.seeds)})
