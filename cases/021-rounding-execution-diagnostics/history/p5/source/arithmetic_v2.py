"""CPU numerical primitives and an independent exact rational BF16 oracle."""
import math
from fractions import Fraction
import numpy as np

def decode(bits):
    b=np.asarray(bits)
    if b.dtype!=np.uint16:raise ValueError('BF16_BIT_DTYPE')
    return (b.astype(np.uint32)<<np.uint32(16)).view(np.float32)
def finite(bits):return ((np.asarray(bits,dtype=np.uint16)&0x7f80)!=0x7f80)
def round_bits(x):
    x=np.asarray(x,np.float64)
    if not np.isfinite(x).all():raise ValueError('NONFINITE_INPUT')
    magnitude=np.abs(x);_,e=np.frexp(magnitude);step=np.ldexp(np.ones_like(x),np.maximum(e-8,-133));y=np.copysign(np.rint(magnitude/step)*step,x)
    with np.errstate(over='ignore'):return (y.astype(np.float32).view(np.uint32)>>16).astype(np.uint16)
def rational_bits(q):
    """Integer quotient/remainder nearest-even. No float32 double rounding."""
    q=Fraction(q);sign=0x8000 if q<0 else 0;q=abs(q)
    if not q:return sign
    n,d=q.numerator,q.denominator;e=n.bit_length()-d.bit_length()
    if (n < (d<<e)) if e>=0 else ((n<<(-e))<d):e-=1
    step=max(e-7,-133)
    top,bot=(n,d<<step) if step>=0 else (n<<(-step),d)
    m,rem=divmod(top,bot)
    if 2*rem>bot or (2*rem==bot and m%2):m+=1
    if m==0:return sign
    qround=Fraction(m*(1<<step),1) if step>=0 else Fraction(m,1<<(-step))
    if qround>=Fraction(1<<128):return sign|0x7f80
    # qround is now exactly representable in BF16, so this is serialization only.
    return sign|int(np.asarray(float(qround),np.float32).view(np.uint32)>>16)
def exact_candidates(a,c,s):
    av=Fraction.from_float(float(a));sv=Fraction.from_float(float(s));code=int(c)
    wb=rational_bits(code*sv);w=Fraction.from_float(float(decode(np.array(wb,np.uint16))))
    return rational_bits(av*w),rational_bits(av*code*sv)
def candidates(a,c,s):
    c=np.asarray(c,np.float64);s=np.asarray(s,np.float64);a=np.asarray(a,np.float64)
    exact=c*s;A=round_bits(a*decode(round_bits(exact)).astype(np.float64));B=round_bits(a*exact)
    return A,B
def rank(bits):
    b=np.asarray(bits,np.uint16)
    if not finite(b).all():raise ValueError('NONFINITE_BF16')
    mag=(b&0x7fff).astype(np.int32);return np.where((b&0x8000)!=0,32768-mag,32768+mag)
def steps(a,b):return np.abs(rank(a).astype(np.int64)-rank(b).astype(np.int64))
def numeric_equal(a,b):return rank(a)==rank(b)
def neighbors(bit):
    k=int(rank(np.array(bit,np.uint16)));out=[]
    for q in [k-1,k+1]:
        v=32768-q if q<32768 else q-32768;v|=0x8000 if q<32768 else 0
        out.append(None if v&0x7f80==0x7f80 else float(decode(np.array(v,np.uint16))))
    return out
def metrics(ref,y):
    ref=np.asarray(ref,np.float64);y=np.asarray(y,np.float64)
    if ref.shape!=y.shape or not (np.isfinite(ref).all() and np.isfinite(y).all()):raise ValueError('SHAPE_FINITE')
    err=y-ref;sse=float(np.sum(err*err));energy=float(np.sum(ref*ref))
    return {'SSE':sse,'energy':energy,'relative_RMSE':None if energy==0 else math.sqrt(sse/energy),'MAE':float(np.mean(np.abs(err))),'max_abs':float(np.max(np.abs(err),initial=0)),'elements':ref.size,'energy_status':'UNDEFINED' if energy==0 else 'DEFINED'}
def ulp_metrics(a,b):
    d=steps(a,b);nz=d[d>0];x=decode(a).astype(np.float64);y=decode(b).astype(np.float64)
    return {'elements':d.size,'numeric_mismatch':int(nz.size),'mismatch_fraction':float(nz.size/d.size),'steps1':int(np.sum(nz==1)),'steps2':int(np.sum(nz==2)),'steps_gt2':int(np.sum(nz>2)),'all_median_steps':float(np.median(d)),'all_p95_steps':float(np.quantile(d,.95,method='linear')),'mismatch_median_steps':None if not nz.size else float(np.median(nz)),'mismatch_p95_steps':None if not nz.size else float(np.quantile(nz,.95,method='linear')),'max_steps':int(d.max(initial=0)),'reference_abs_max':float(np.max(np.abs(x),initial=0)),'candidate_abs_max':float(np.max(np.abs(y),initial=0)),**metrics(x,y)}
def onehot(rows,I=1024,M=512):
    if len(rows)>M:raise ValueError('ROW_CAP')
    z=np.zeros((M,I),np.uint16)
    for i,row in enumerate(rows):
        axis=int(row['axis']);bit=int(row['coefficient_bits'])
        if not 0<=axis<I or not bool(finite(np.array(bit,np.uint16))) or (bit&0x7fff)==0:raise ValueError('ONEHOT_INPUT')
        z[i,axis]=bit
    if not np.all(np.count_nonzero(decode(z),axis=1)<=1):raise ValueError('NOT_ONEHOT')
    return z
def nested(z,layer,doc_index,seed=61027027):
    if z.shape!=(512,1024) or z.dtype!=np.uint16:raise ValueError('ACTUAL_Z_SHAPE')
    rng=np.random.default_rng(np.random.SeedSequence([seed,layer,doc_index,512,1024]));order=np.stack([rng.permutation(1024) for _ in range(512)])
    for K in [1,2,8,64,256,1024]:
        out=np.zeros_like(z);np.put_along_axis(out,order[:,:K],np.take_along_axis(z,order[:,:K],1),1)
        if K==1024 and not np.array_equal(out,z):raise ValueError('FULL_MASK_NOT_EXACT')
        yield K,out,order[:,:K]
