"""New FP64 formulas, independent scalar paths and BF16 serialization."""
import math
import struct
import numpy as np

def bfdecode(bits):
    a=np.asarray(bits)
    if a.dtype!=np.dtype('uint16'):raise ValueError('BF16_BITS_DTYPE')
    return (a.astype(np.uint32)<<16).view(np.float32)
def bfbits32(values):
    a=np.asarray(values,dtype=np.float32)
    if not np.isfinite(a).all():raise ValueError('NONFINITE')
    u=a.view(np.uint32)
    return ((u+np.uint32(32767)+((u>>16)&1))>>16).astype(np.uint16)
def bfround64(values):
    a=np.asarray(values,dtype=np.float64)
    if not np.isfinite(a).all():raise ValueError('NONFINITE')
    negative=np.signbit(a);x=np.abs(a)
    # BF16 normal spacing 2^(floor(log2(x))-7); subnormal spacing 2^-133.
    _,exponent=np.frexp(x);step=np.exp2(np.maximum(exponent-8,-133))
    out=np.rint(x/step)*step;out=np.copysign(out,a)
    return out
def logprob(x):
    x=np.asarray(x,np.float64);m=x.max(-1,keepdims=True);z=x-m
    return z-np.log(np.exp(z).sum(-1,keepdims=True))
def distribution_pair(a,b):
    la,lb=logprob(a),logprob(b);pa,pb=np.exp(la),np.exp(lb)
    lm=np.logaddexp(la,lb)-math.log(2.)
    return (pa*(la-lb)).sum(-1),.5*((pa*(la-lm)).sum(-1)+(pb*(lb-lm)).sum(-1))
def independent_pair(a,b):
    # math.fsum scalar normalization and sums; no calls to vector path.
    results=[]
    for va,vb in zip(np.atleast_2d(a),np.atleast_2d(b)):
        ma,mb=float(max(va)),float(max(vb))
        za=np.asarray(va,np.float64)-ma;zb=np.asarray(vb,np.float64)-mb
        la=za-math.log(math.fsum(np.exp(za)));lb=zb-math.log(math.fsum(np.exp(zb)))
        pa,pb=np.exp(la),np.exp(lb)
        if abs(math.fsum(pa)-1)>2e-12 or abs(math.fsum(pb)-1)>2e-12:raise ValueError('NORMALIZATION')
        # Stable scalar log mixture, independent from np.logaddexp.
        high=np.maximum(la,lb);low=np.minimum(la,lb)
        lm=high+np.log1p(np.exp(low-high))-math.log(2.)
        k=math.fsum(pa*(la-lb))
        js=.5*(math.fsum(pa*(la-lm))+math.fsum(pb*(lb-lm)))
        results.append((k,js))
    return np.array(results)
def norm_rope(q,w,eps,cos,sin):
    q=np.asarray(q,np.float64);w=np.asarray(w,np.float64);cos=np.asarray(cos,np.float64);sin=np.asarray(sin,np.float64)
    if q.ndim!=4 or w.shape!=(q.shape[-1],) or q.shape[-1]%2 or cos.shape!=(q.shape[0],q.shape[2],q.shape[3]) or sin.shape!=cos.shape:raise ValueError('HEAD_POSITION_SHAPE')
    n=q/np.sqrt(np.mean(q*q,axis=-1,keepdims=True)+eps)*w
    h=n.shape[-1]//2;rotate=np.concatenate([-n[...,h:],n[...,:h]],axis=-1)
    return n*cos[:,None]+rotate*sin[:,None]
def norm_rope_scalar(q,w,eps,cos,sin):
    q=np.asarray(q,np.float64);out=np.empty_like(q);B,H,T,D=q.shape
    if np.shape(cos)!=(B,T,D) or np.shape(sin)!=(B,T,D) or np.shape(w)!=(D,):raise ValueError('HEAD_POSITION_SHAPE')
    for b in range(B):
        for h in range(H):
            for t in range(T):
                den=math.sqrt(math.fsum(float(x)*float(x) for x in q[b,h,t])/D+eps)
                n=[float(q[b,h,t,j])/den*float(w[j]) for j in range(D)]
                for j in range(D):
                    rot=-n[j+D//2] if j<D//2 else n[j-D//2]
                    out[b,h,t,j]=n[j]*float(cos[b,t,j])+rot*float(sin[b,t,j])
    return out
def metrics(ref,actual):
    x=np.asarray(ref,np.float64);y=np.asarray(actual,np.float64)
    if x.shape!=y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):raise ValueError('SHAPE_FINITE')
    diff=y-x;sse=float(np.sum(diff*diff));energy=float(np.sum(x*x))
    return {'SSE':sse,'reference_energy':energy,'relative_RMSE':None if energy==0 else math.sqrt(sse/energy),'max_abs':float(np.max(np.abs(diff),initial=0)),'elements':x.size,'energy_status':'UNDEFINED' if energy==0 else 'DEFINED'}
def decomposition(uD,uP,yD,yP):
    uD,uP,yD,yP=[np.asarray(x,np.float64) for x in (uD,uP,yD,yP)]
    if not(uD.shape==uP.shape==yD.shape==yP.shape):raise ValueError('COORDINATE_SHAPE')
    d=uP-uD;eD=yD-uD;eP=yP-uP;n=eP-eD;gap=yP-yD
    de=float(np.sum(d*d));ne=float(np.sum(n*n));cross=2*float(np.sum(d*n));sse=float(np.sum(gap*gap));scale=max(de,ne,abs(cross),sse)
    residual=sse-de-ne-cross
    if (scale==0 and residual!=0) or (scale>0 and abs(residual)>1e-10*scale):raise ValueError('ENERGY_IDENTITY')
    mD,mP=metrics(uD,yD),metrics(uP,yP)
    return {'d_energy':de,'n_energy':ne,'signed_cross':cross,'pair_gap_SSE':sse,'identity_residual':residual,'identity_scale':scale,'cosine_d_n':None if de==0 or ne==0 else cross/(2*math.sqrt(de*ne)),'individual_error_sum':mD['SSE']+mP['SSE'],**{'D1_'+k:v for k,v in mD.items()},**{'P_'+k:v for k,v in mP.items()}}
def bootstrap(values,seed=61027025):
    values=np.asarray(values,np.float64);rng=np.random.default_rng(seed)
    draw=values[rng.integers(0,len(values),(10000,len(values)))].mean(1);ci=np.quantile(draw,[.025,.975],method='linear')
    return {'documents':len(values),'mean':float(values.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1]),'replicates':10000,'seed':seed,'scope':'POSTHOC descriptive paired-document interval'}
def weight_vector(path,key,length):
    import json,hashlib
    with open(path,'rb') as f:
        n=int.from_bytes(f.read(8),'little');header=json.loads(f.read(n));row=header[key]
        if row['dtype']!='BF16' or row['shape']!=[length]:raise ValueError('NORM_WEIGHT_SHAPE')
        begin,end=row['data_offsets'];f.seek(8+n+begin);raw=f.read(end-begin)
    return bfdecode(np.frombuffer(raw,dtype='<u2')).astype(np.float64),hashlib.sha256(raw).hexdigest()
