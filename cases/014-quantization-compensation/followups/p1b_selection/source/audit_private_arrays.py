"""Additional local CPU audit using private matrices, no model forward/GPU."""
import argparse
import json
import math
import struct
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from p1b_common import save,sha,utc

def independent_decode(path):
    b=Path(path).read_bytes();magic,v,o,i,g,p,ns,nc=struct.unpack('<8s7I',b[:36])
    assert magic==b'C014INT4' and v==1 and g==64 and len(b)==36+ns*4+nc
    s=np.frombuffer(b,dtype='<f4',offset=36,count=ns).reshape(o,p//g)
    x=np.frombuffer(b,dtype=np.uint8,offset=36+ns*4)
    n=np.stack([x&15,x>>4],axis=1).reshape(o,p).astype('i2');n=np.where(n>7,n-16,n)
    return (n.reshape(o,p//g,g).astype('f4')*s[...,None]).reshape(o,p)[:,:i].astype('f8')

def main():
    ap=argparse.ArgumentParser()
    for k in ['run','private','old-private','model','output']:ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(4);run=Path(a.run);private=Path(a.private);old=Path(a.old_private)
    rr=[json.loads(s) for p in sorted((run/'cells/FRESH-EVAL-ID').glob('*.jsonl')) for s in p.read_text().splitlines()]
    chosen=[]
    for t in ['model.layers.4.self_attn.o_proj','model.layers.12.mlp.down_proj','model.layers.20.mlp.down_proj']:
        doc=min(r['document_id_hash'] for r in rr if r['target']==t)
        chosen.extend(next(r for r in rr if r['target']==t and r['document_id_hash']==doc and r['alpha']==alpha) for alpha in [0,.25,1])
    checks=[]
    with safe_open(str(Path(a.model)/'model.safetensors'),framework='pt',device='cpu') as f:
        for r in chosen:
            t=r['target'];d=r['document_id_hash'];n=t.replace('.','_')
            W=f.get_tensor(t+'.weight').float().double()
            X=torch.from_numpy(np.load(private/'traces/teacher'/d/(n+'_X.npy')).astype('f8'))
            Z=torch.from_numpy(np.load(private/'traces/prefix'/d/(n+'_Z.npy')).astype('f8'))
            K=torch.from_numpy(np.load(old/f'artifacts/{t}_K.npy'));D=r['alpha']*K
            p=old/f'artifacts/candidates/{t}.alpha{r["alpha"]:g}.int4';assert sha(p)==r['artifact_sha256']
            Q=torch.from_numpy(independent_decode(p));Q0=torch.from_numpy(independent_decode(old/f'artifacts/candidates/{t}.alpha0.int4'))
            A=W@(X-Z);C=D@Z;B=(Q-W-D)@Z;B0=(Q0-W)@Z;DB=B-B0;R=W@X-Q@Z;R0=W@X-Q0@Z
            ss=lambda x:float((x*x).sum());dot=lambda x,y:float((x*y).sum())
            got={'SSE_after':ss(R),'SSE_before':ss(R0),'A_norm2':ss(A),'C_norm2':ss(C),'B_norm2':ss(B),'B0_norm2':ss(B0),
                 'T_inherited':ss(C)-2*dot(A,C),'T_cancellation':2*dot(B0,C),'T_requant':ss(DB)-2*dot(R0-C,DB)}
            worst=max(abs(v-r[k])/max(1,abs(v),abs(r[k])) for k,v in got.items());assert worst<=1e-10,(t,r['alpha'],worst)
            checks.append({'target':t,'alpha':r['alpha'],'document_id_hash':d,'scalar_fields':len(got),'max_scaled_difference':worst,'status':'PASS'})
    cropped=np.load(private/'smoke_arrays.private.npz');w,x,z,k,q,q0=[cropped[n].astype('f8') for n in ['W','X','Z','K','Q','Q0']]
    A=w@(x-z);C=.25*k@z;B=(q-w-.25*k)@z;B0=(q0-w)@z;DB=B-B0;R0=A-B0
    direct=float(np.sum((w@x-q@z)**2)-np.sum((w@x-q0@z)**2));terms=float(np.sum(C*C)-2*np.sum(A*C)+2*np.sum(B0*C)+np.sum(DB*DB)-2*np.sum((R0-C)*DB))
    gap=abs(direct-terms)/max(1,abs(direct),abs(terms));assert gap<=1e-10
    save(a.output,{'status':'PASS','audited_utc':utc(),'full_private_matrix_rows':checks,'cropped_actual_SMOKE_array_identity_scaled_gap':gap,
         'cropped_SMOKE_scope':'8×32 weight /32×8 input slices, validates algebra on actual small slices; not full module output',
         'method':'independent struct/nibble decoder and Torch CPU FP64 direct products; no decompose.py import, no fitting, no model forward',
         'limits':'Does not re-evaluate GPU NLL/KL/argmax, candidate BF16 output arrays or all 3360 matrix rows. Matrices/model remain external to review ZIP.'})
    print('Private-matrix CPU audit PASS: 9 full input/weight row checks + cropped SMOKE identity.')

if __name__=='__main__':main()
