"""Fresh CPU process: independently decode all saved INT4 artifacts and test small outputs."""
import argparse,json,struct,hashlib
from pathlib import Path
import numpy as np
import torch

def main(private,manifest,out):
 private=Path(private);metadata=json.loads(Path(manifest).read_text())['candidates'];records=[]
 for r in metadata:
  p=private/'artifacts/candidates'/f"{r['target']}.alpha{r['alpha']:g}.int4";b=p.read_bytes();magic,version,o,i,g,padded,ns,nc=struct.unpack_from('<8s7I',b);assert magic==b'C014INT4' and version==1
  scales=np.frombuffer(b,dtype='<f4',offset=36,count=ns).copy().reshape(o,-1);packed=np.frombuffer(b,dtype='u1',offset=36+4*ns);n=np.column_stack([packed&15,packed>>4]).reshape(o,padded).astype('i2');n=np.where(n>=8,n-16,n).astype('i1');q=(n.reshape(o,-1,64).astype('f4')*scales[...,None]).reshape(o,padded)[:,:i].copy()
  # Original codec decoded through a different implementation, checked bit-exact.
  import sys;sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
  from codec import decode
  expected,c,s=decode(b);np.testing.assert_array_equal(q,expected);np.testing.assert_array_equal(c,n[:,:i])
  x=np.random.default_rng(61015).normal(size=(4,i)).astype('f4');y=x.astype('f8')@q.astype('f8').T
  records.append({'target':r['target'],'alpha':r['alpha'],'artifact_bytes':len(b),'artifact_sha256':hashlib.sha256(b).hexdigest(),'independent_codes_exact':True,'independent_FP32_dequant_exact':True,'BF16_weight_FP32_view_sha256':hashlib.sha256(torch.from_numpy(q).bfloat16().float().numpy().tobytes()).hexdigest(),'fixed_synthetic_FP64_linear_output_sha256':hashlib.sha256(y.tobytes()).hexdigest(),'scope':'fresh-process CPU serialization/linear check; not full-model GPU rerun'})
 Path(out).write_text(json.dumps({'status':'PASS','artifacts':records},indent=2));print('PASS',len(records))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.private,a.manifest,a.output)
