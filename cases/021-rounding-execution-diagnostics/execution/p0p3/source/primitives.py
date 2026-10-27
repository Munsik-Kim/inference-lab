"""CPU FP64 audit formulas, independent of Torch and model execution."""
import math
import numpy as np

def log_prob(x):
    x = np.asarray(x, np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))

def kl(reference, candidate):
    p, q = log_prob(reference), log_prob(candidate)
    return np.sum(np.exp(p) * (p - q), axis=-1)

def kl_loop(reference, candidate):
    result = []
    for a, b in zip(np.asarray(reference, np.float64), np.asarray(candidate, np.float64)):
        ma, mb = max(a), max(b)
        za = math.log(math.fsum(math.exp(float(v-ma)) for v in a)) + ma
        zb = math.log(math.fsum(math.exp(float(v-mb)) for v in b)) + mb
        result.append(math.fsum(math.exp(float(v)-za) * (float(v)-za-float(w)+zb) for v,w in zip(a,b)))
    return np.asarray(result)

def norm_rope(x, weight, epsilon, cos, sin):
    x, weight, cos, sin = [np.asarray(a, np.float64) for a in (x,weight,cos,sin)]
    y = x / np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+epsilon) * weight
    h = y.shape[-1]//2
    rotated = np.concatenate((-y[...,h:],y[...,:h]),axis=-1)
    return y*cos[:,None,:,:] + rotated*sin[:,None,:,:]

def norm_rope_loop(x, weight, epsilon, cos, sin):
    x=np.asarray(x,np.float64); out=np.empty_like(x); d=x.shape[-1]
    for batch in range(x.shape[0]):
        for head in range(x.shape[1]):
            for token in range(x.shape[2]):
                v=x[batch,head,token]; den=math.sqrt(math.fsum(float(z*z) for z in v)/d+epsilon)
                for i in range(d):
                    j=(i+d//2)%d; sign=-1 if i<d//2 else 1
                    out[batch,head,token,i]=(v[i]*weight[i]/den*cos[batch,token,i] + sign*v[j]*weight[j]/den*sin[batch,token,i])
    return out

def vector_decomposition(d0,d1,p):
    d0,d1,p=[np.asarray(x,np.float64) for x in (d0,d1,p)]
    a,b=d1-d0,p-d1
    aa,bb,cross,total=[float(np.sum(x)) for x in (a*a,b*b,2*a*b,(p-d0)**2)]
    return {'format_sse':aa,'execution_sse':bb,'signed_cross':cross,'total_sse':total,'identity_residual':total-aa-bb-cross}

def paired_bootstrap(delta, seed=61027023, replicates=10000):
    x=np.asarray(delta,np.float64); rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),(replicates,len(x)))].mean(axis=1)
    lo,hi=np.quantile(means,[.025,.975],method='linear')
    return {'document_count':len(x),'mean_delta':float(x.mean()),'ci_low':float(lo),'ci_high':float(hi),'replicates':replicates,'seed':seed,'quantile':'linear'}
