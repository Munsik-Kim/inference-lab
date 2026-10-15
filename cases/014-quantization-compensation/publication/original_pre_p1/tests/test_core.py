import unittest,sys,tempfile,subprocess,os,struct
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from fractions import Fraction as F
from codec import encode,decode,write_reload,HEADER,MAGIC,VERSION
from fit import fit_numpy,fit_torch
from decompose import measure
from fixtures import exact
class Core(unittest.TestCase):
 def setUp(self):
  r=np.random.default_rng(31);self.w=r.normal(size=(5,7));self.x=r.normal(size=(7,20));self.z=self.x+.1*r.normal(size=self.x.shape)
 def test_fraction_both_signs(self):
  for a in exact()['fixed_errors']:
   e=F(a['epsilon']);self.assertEqual(F(a['COMP_abs_error']),e+e*e);self.assertEqual(F(a['BASE_abs_error']),e*e if a['second_error_sign']==-1 else 2*e+e*e)
 def test_finite_all_losses_and_path(self):
  a=exact()['finite_codebook'];self.assertEqual(a['teacher'],'377/125');self.assertEqual(a['BASE_abs_error'],'2/125');self.assertEqual(a['COMP_abs_error'],'123/125');self.assertTrue(all(r['unique'] and not r['overload'] for r in a['layers']));self.assertEqual([r['COMP_code'] for r in a['layers']],[1,2,2])
 def test_all_nibbles_independent_decoder(self):
  codes=np.tile(np.arange(-8,8,dtype=np.int8),4);n=codes.astype(np.int16)&15;p=(n[::2]|n[1::2]<<4).astype('u1')
  b=HEADER.pack(MAGIC,VERSION,1,64,64,64,1,32)+np.array([1],'<f4').tobytes()+p.tobytes();w,q,_=decode(b);np.testing.assert_array_equal(q[0],codes)
  independent=np.array([((int(v)>>(4*k))&15) for v in p for k in [0,1]]);independent=np.where(independent>7,independent-16,independent);np.testing.assert_array_equal(independent,codes)
 def test_ties_zero_odd_padding(self):
  w=np.zeros((2,65),'f4');w[0,:5]=[7,.5,1.5,-.5,-1.5];b,_=encode(w);deq,q,s=decode(b);np.testing.assert_array_equal(q[0,:5],[7,0,2,0,-2]);self.assertEqual(s[1,0],1);self.assertEqual(len(b),36+2*2*4+2*128//2)
 def test_reject_version_shape_length_padding(self):
  b,_=encode(self.w)
  for c in [b[:10],b+b'1',b[:8]+struct.pack('<I',99)+b[12:]]:
   with self.assertRaises(ValueError):decode(c)
 def test_reject_scale_nan_underflow(self):
  for v in [float('nan'),float('inf'),np.nextafter(np.float32(0),np.float32(1))]:
   with self.assertRaises(ValueError):encode(np.array([[v]],'f4'))
 def test_numpy_torch_full_gram(self):
  for ridge in [0,1e-3]:
   k,l,d=fit_numpy(self.w,self.x,self.z,ridge);kt,lt,dt=fit_torch(self.w,self.x,self.z,ridge);np.testing.assert_allclose(k,kt,rtol=1e-10,atol=1e-12);self.assertLess(dt['relative_solve_residual'],1e-10)
 def test_decomposition_all_alphas_saved_matrix(self):
  k,_,_=fit_numpy(self.w,self.x,self.z);q0=decode(encode(self.w)[0])[0]
  for a in [0,.25,.5,.75,1]:
   q=decode(encode(self.w+a*k)[0])[0];m=measure(self.w,self.x,self.z,a*k,q,q0);self.assertLess(m['identity_relative_gap'],1e-10);self.assertLess(m['residual_identity_relative'],1e-10)
 def test_independent_torch_numpy_decomposition(self):
  import torch
  k,_,_=fit_numpy(self.w,self.x,self.z);q0=decode(encode(self.w)[0])[0];q=decode(encode(self.w+.75*k)[0])[0]
  ref=measure(self.w,self.x,self.z,.75*k,q,q0)
  W,X,Z,D,Q,Q0=[torch.tensor(a,dtype=torch.float64) for a in [self.w,self.x,self.z,.75*k,q,q0]]
  A=W@(X-Z);C=D@Z;B=(Q-W-D)@Z;B0=(Q0-W)@Z;DB=B-B0;R0=A-B0
  terms=[float((C*C).sum()-2*(A*C).sum()),float(2*(B0*C).sum()),float((DB*DB).sum()-2*((R0-C)*DB).sum())]
  for name,v in zip(['T_inherited','T_cancellation','T_requant'],terms):self.assertLess(abs(ref[name]-v)/max(1,abs(v)),1e-10)
  direct=float(((W@X-Q@Z)**2).sum()-((W@X-Q0@Z)**2).sum());self.assertLess(abs(direct-sum(terms))/max(1,abs(direct)),1e-10)
 def test_fixedB_special_formula(self):
  k,_,_=fit_numpy(self.w,self.x,self.z,0);AP=k@self.z;B=np.random.default_rng(2).normal(size=AP.shape);A=self.w@(self.x-self.z)
  for a in [0,.25,1]:
   lhs=np.sum((A-a*AP-B)**2)-np.sum((A-B)**2);rhs=-a*(2-a)*np.sum(AP**2)+2*a*np.sum(AP*B);self.assertLess(abs(lhs-rhs)/max(1,abs(lhs),abs(rhs)),1e-10)
 def test_ridge_not_projector_and_cal_contraction(self):
  k,l,_=fit_numpy(self.w,self.x,self.z);H=self.z.T@np.linalg.solve(self.z@self.z.T+l*np.eye(7),self.z);self.assertGreater(np.linalg.norm(H@H-H),1e-6)
  A=self.w@(self.x-self.z)
  self.assertTrue(all(np.sum((A-a*k@self.z)**2)<=np.sum(A*A)+1e-10 for a in [0,.25,.5,.75,1]))
 def test_evaluation_cannot_refit_and_eval_equal_not_zero(self):
  k,_,_=fit_numpy(self.w,self.x,self.z);b=encode(self.w+k)[0];m=measure(self.w,self.x,self.x,k,decode(b)[0],decode(encode(self.w)[0])[0]);self.assertGreater(m['C_norm2'],0);self.assertEqual(b,encode(self.w+k)[0])
 def test_no_prefix_fit_all_artifacts_equal(self):
  k,_,_=fit_numpy(self.w,self.x,self.x);np.testing.assert_array_equal(k,np.zeros_like(k));self.assertEqual(len({encode(self.w+a*k)[0] for a in [0,.25,.5,.75,1]}),1)
 def test_new_process_reload_linear(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'tiny.int4';w,_,_,_=write_reload(p,self.w)
   env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/codec.py'),str(p),'--output',str(Path(d)/'reloaded.npy')],check=True,env=env)
   np.testing.assert_array_equal(w@self.z,np.load(Path(d)/'reloaded.npy')@self.z)
 def test_undefined_normalized_not_epsilon(self):
  z=np.zeros_like(self.z);m=measure(self.w,z,z,np.zeros_like(self.w),self.w,self.w);self.assertIsNone(m['normalized_SSE']);self.assertIsNone(m['cosine_A_B0'])
if __name__=='__main__':unittest.main()
