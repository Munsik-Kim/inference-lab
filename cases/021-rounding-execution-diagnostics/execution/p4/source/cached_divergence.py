"""Same formulas, bounded per-document normalization reuse.

Vector and math.fsum paths maintain separate caches. At most five original
arrays are retained in each; no result from a different array is substituted.
"""
import math
import numpy as np

vector_cache={}
scalar_cache={}

def normalized(a,cache,scalar):
    key=id(a)
    if key in cache and cache[key][0] is a:return cache[key][1:]
    if len(cache)>=5:cache.clear()
    if scalar:
        logs=[]
        for row in np.atleast_2d(a):
            shift=np.asarray(row,np.float64)-float(max(row))
            log=shift-math.log(math.fsum(np.exp(shift)))
            if abs(math.fsum(np.exp(log))-1)>2e-12:raise ValueError('NORMALIZATION')
            logs.append(log)
        lp=np.array(logs)
    else:
        x=np.asarray(a,np.float64);z=x-x.max(-1,keepdims=True);lp=z-np.log(np.exp(z).sum(-1,keepdims=True))
    prob=np.exp(lp);cache[key]=(a,lp,prob);return lp,prob

def vector_pair(a,b):
    la,pa=normalized(a,vector_cache,False);lb,pb=normalized(b,vector_cache,False)
    lm=np.logaddexp(la,lb)-math.log(2.)
    return (pa*(la-lb)).sum(-1),.5*((pa*(la-lm)).sum(-1)+(pb*(lb-lm)).sum(-1))

def scalar_pair(a,b,include_js=True):
    la,pa=normalized(a,scalar_cache,True);lb,pb=normalized(b,scalar_cache,True);rows=[]
    for x,p,y,q in zip(la,pa,lb,pb):
        k=math.fsum(p*(x-y))
        js=None
        if include_js:
            high=np.maximum(x,y);low=np.minimum(x,y);m=high+np.log1p(np.exp(low-high))-math.log(2.)
            js=.5*(math.fsum(p*(x-m))+math.fsum(q*(y-m)))
        rows.append((k,0. if js is None else js))
    return np.array(rows)
