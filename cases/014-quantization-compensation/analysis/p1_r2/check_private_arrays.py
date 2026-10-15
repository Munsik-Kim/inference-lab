"""Independent Torch CPU FP64 checks of a fixed EVAL document per target plus first A/B examples.
Uses private W/X/Z/K and saved payloads. No model forward/GPU and no new fit.
"""
import argparse,json,struct,math
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open

def decode(p):
 b=Path(p).read_bytes();magic,v,o,i,g,padded,ns,nc=struct.unpack_from('<8s7I',b);assert magic==b'C014INT4' and v==1
 s=np.frombuffer(b,dtype='<f4',count=ns,offset=36).copy().reshape(o,-1);b=np.frombuffer(b,dtype='u1',offset=36+4*ns);q=np.stack([b&15,b>>4],axis=-1).reshape(o,padded).astype('i2');q[q>7]-=16
 return (q.reshape(o,-1,64).astype('f4')*s[...,None]).reshape(o,padded)[:,:i].copy()
def main(case,private,model,out):
 case=Path(case);private=Path(private);run=case/'results/p1_r2';torch.set_num_threads(2);allrows=[json.loads(s) for p in sorted((run/'cells').glob('*.jsonl')) for s in p.read_text().splitlines()];cl=json.loads((run/'classifications.json').read_text());by={(r['block'],r['projection'],r['split'],r['alpha'],r['document_id_hash']):r for r in allrows};selection=json.loads((run/'fit_frozen.json').read_text())['selected'];keys=[]
 for s in selection:
  n=s['target'];b=int(n.split('.')[2]);p=n.split('.')[-1];doc=min(r['document_id_hash'] for r in allrows if r['split']=='EVAL-ID');keys.append((b,p,'EVAL-ID',s['selected_alpha'],doc))
 for label in ['A_CANCELLATION_LOSS','B_REQUANTIZATION_REVERSAL']:
  cand=sorted((r['block'],r['projection'],r['split'],r['alpha'],r['document_id_hash']) for r in cl if label in r['labels'])
  if cand:keys.append(cand[0])
 records=[]
 for key in sorted(set(keys)):
  r=by[key];b,p,split,alpha,doc=key;n=f'model.layers.{b}.'+('self_attn.' if p=='o_proj' else 'mlp.')+p;tag=n.replace('.','_')
  with safe_open(str(Path(model)/'model.safetensors'),framework='pt',device='cpu') as f:W=f.get_tensor(n+'.weight').double()
  X=torch.from_numpy(np.load(private/'traces/teacher'/doc/(tag+'_X.npy'))).double();Z=torch.from_numpy(np.load(private/'traces/prefix'/doc/(tag+'_Z.npy'))).double();K=torch.from_numpy(np.load(private/'artifacts'/(n+'_K.npy')));Q=torch.from_numpy(decode(private/'artifacts/candidates'/f'{n}.alpha{alpha:g}.int4')).double();Q0=torch.from_numpy(decode(private/'artifacts/candidates'/f'{n}.alpha0.int4')).double()
  A=W@(X-Z);C=(alpha*K)@Z;B0=(Q0-W)@Z;B=(Q-W-alpha*K)@Z;D=B-B0;R=W@X-Q@Z;R0=W@X-Q0@Z
  ss=lambda t:float((t*t).sum());values={'A_norm2':ss(A),'C_norm2':ss(C),'B0_norm2':ss(B0),'B_norm2':ss(B),'SSE_after':ss(R),'SSE_before':ss(R0),'T_inherited':ss(C)-2*float((A*C).sum()),'T_cancellation':2*float((B0*C).sum()),'T_requant':ss(D)-2*float(((R0-C)*D).sum()),'cast_residual_norm2':ss((Q.float().bfloat16().float().double()-Q)@Z)}
  gaps={k:abs(v-r[k])/max(1,abs(v),abs(r[k])) for k,v in values.items()};assert max(gaps.values())<=1e-10,gaps
  records.append({'block':b,'projection':p,'split':split,'alpha':alpha,'document_id_hash':doc,'max_relative_gap':max(gaps.values()),'checked_scalars':values})
 Path(out).write_text(json.dumps({'status':'PASS','records':records,'selection':'lexicographically first EVAL document for every CAL-selected target; first raw A and B examples, no refit','scope':'private algebra and cast check with independently decoded artifacts and Torch CPU FP64; not model output/NLL/KL rerun'},indent=2));print('PASS',len(records))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--private',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.case,a.private,a.model,a.output)
