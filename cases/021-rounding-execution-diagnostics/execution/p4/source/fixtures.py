"""Fixed structured inputs. No artifact values participate in selecting axes."""
import numpy as np
from numerics import bfbits32

def structured(I=1024,M=512):
    if I%64 or M!=512:raise ValueError('SHAPE_CONTRACT')
    result=[]
    for sign,label in [(1,'ONE_HOT_PLUS'),(-1,'ONE_HOT_MINUS')]:
        for start in range(0,I,M):
            n=min(M,I-start);z=np.zeros((M,I),np.float32);z[np.arange(n),start+np.arange(n)]=sign
            result.append((f'{label}_{start:04d}',label,z,n,{'axis_start':start,'axis_end_exclusive':start+n,'zero_control_rows':M-n}))
    for across,label in [(False,'PAIR_WITHIN_GROUP'),(True,'PAIR_ACROSS_GROUP')]:
        z=np.zeros((M,I),np.float32);G=I//64
        for r in range(M):
            g=r%G;a=(r//G)%64;h=(g+1+(r//G)%(G-1))%G if across else g;b=(a+17)%64
            z[r,g*64+a]=1;z[r,h*64+b]=-1
        result.append((label,label,z,M,{'rule':'g=r%G, a=(r//G)%64; within h=g; across h=(g+1+(r//G)%(G-1))%G; b=(a+17)%64; signs+1,-1'}))
    rng=np.random.default_rng(np.random.SeedSequence([61027024,64,I,M]));z=np.zeros((M,I),np.float32)
    for r in range(M):z[r,(r%(I//64))*64:((r%(I//64))+1)*64]=rng.choice([-0.125,0.125],64)
    result.append(('GROUP64_SIGNS','GROUP64_SIGNS',z,M,{'group_rule':'r%G','seed_sequence':[61027024,64,I,M],'rows_per_group':M//(I//64)}))
    return result

def inverse_scales(permuted):
    a=np.asarray(permuted)
    if a.ndim!=2 or a.size%64:raise ValueError('SCALE_SHAPE')
    out=np.empty_like(a).reshape(-1);flat=a.reshape(-1)
    for offset in range(64):
        # packed offset=col*8+row, original offset=row*8+col
        row,col=divmod(offset,8);out[offset::64]=flat[(col*8+row)::64]
    return out.reshape(a.shape)
