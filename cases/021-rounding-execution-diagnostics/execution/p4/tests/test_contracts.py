import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from common import write,freeze,verify
from supervise import gate_then_start
from numerics import bfdecode,bfbits32,bfround64,distribution_pair,independent_pair,norm_rope,norm_rope_scalar,decomposition,metrics
from fixtures import structured,inverse_scales

class Contracts(unittest.TestCase):
    def test_guard_nonzero_exception_timeout_has_no_start_or_spawn(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def nonzero():
                r=subprocess.run([sys.executable,'-c','raise SystemExit(7)'])
                return r.returncode==0
            def exception():raise ValueError('fake gate exception')
            def timeout():subprocess.run([sys.executable,'-c','import time;time.sleep(1)'],timeout=.02,check=True);return True
            def journal():(root/'start.json').write_text('START')
            def spawn():return subprocess.run([sys.executable,'-c','from pathlib import Path;Path('+repr(str(root/'sentinel'))+').write_text("SPAWNED")'])
            for gate in [nonzero,exception,timeout]:
                with self.assertRaises(Exception):gate_then_start(gate,journal,spawn)
                self.assertFalse((root/'start.json').exists());self.assertFalse((root/'sentinel').exists())
            gate_then_start(lambda:True,journal,spawn)
            self.assertTrue((root/'sentinel').exists())
    def test_output_and_identity_refuse_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);f=root/'fixed';f.write_bytes(b'old');rec=freeze(root,[f]);verify(root,rec)
            f.write_bytes(b'new')
            with self.assertRaises(ValueError):verify(root,rec)
            out=root/'result.json';write(out,{'old':1})
            with self.assertRaises(FileExistsError):write(out,{'new':2})
            self.assertEqual(json.loads(out.read_text()),{'old':1})
    def test_bf16_bits_ties_sign_zero_and_double_round(self):
        bits=np.array([0,32768,0x3f80,0xbf80,1,0x7f7f],np.uint16)
        np.testing.assert_array_equal(bfbits32(bfdecode(bits)),bits)
        x=np.array([1+1/256,1+3/256,-1-1/256,-1-3/256,0.,-0.])
        np.testing.assert_array_equal(bfround64(x),[1.,1+4/256,-1.,-1-4/256,0.,-0.])
        v=np.array([1+1/256+2**-30])
        self.assertNotEqual(float(bfround64(v)[0]),float(bfdecode(bfbits32(v))[0]))
    def test_signed_offset_and_group_coordinates(self):
        c=np.array([-8,-7,-1,0,1,7,-8,7])
        signed=[int(x)&15 for x in c];offset=[int(x)+8 for x in c]
        self.assertEqual(signed,[8,9,15,0,1,7,8,7]);self.assertEqual(offset,[0,1,7,8,9,15,0,15])
        self.assertNotEqual(signed,offset)
        plain=np.arange(256).reshape(2,128);packed=plain.reshape(-1,8,8).transpose(0,2,1).reshape(plain.shape)
        np.testing.assert_array_equal(inverse_scales(packed),plain);self.assertFalse(np.array_equal(packed,plain))
    def test_beta_once_and_two_oracles(self):
        x=np.eye(4);beta=np.array([.5,2,4,.25]);z=bfdecode(bfbits32(x*beta))
        wexact=np.array([[1.00390625,2,-1,.25],[.5,1,1.0078125,-2.]])
        wdense=bfround64(wexact)
        for w in [wexact,wdense]:
            y=z@w.T
            for i in range(4):
                for j in range(2):self.assertAlmostEqual(y[i,j],math.fsum(float(z[i,k])*float(w[j,k]) for k in range(4)))
        self.assertFalse(np.array_equal(z*beta,z));self.assertFalse(np.array_equal(z@wexact.T,z@wdense.T))
    def test_kl_js_direction_symmetry_zero_and_offset(self):
        a=np.array([[1001,1002,999],[1,-9,3]],float);b=np.array([[1003,999,1002],[-3,2,5]],float)
        k,js=distribution_pair(a,b);ind=independent_pair(a,b)
        np.testing.assert_allclose(k,ind[:,0],rtol=1e-9,atol=1e-12);np.testing.assert_allclose(js,ind[:,1],rtol=1e-9,atol=1e-12)
        kr,jr=distribution_pair(b,a);self.assertFalse(np.allclose(k,kr));np.testing.assert_allclose(js,jr,atol=1e-12)
        z=distribution_pair(a,a);np.testing.assert_allclose(z,0,atol=1e-12)
        n=128;positions=[(j+1)*n//17 for j in range(16)];tokens=np.arange(n)+300
        self.assertTrue(np.all(tokens[np.array(positions)+1]!=tokens[positions]))
    def test_norm_rope_scalar_and_wrong_mapping(self):
        rng=np.random.default_rng(61027024);q=rng.normal(size=(2,3,5,8));w=np.linspace(.2,1.7,8);c=rng.normal(size=(2,5,8));s=rng.normal(size=c.shape)
        np.testing.assert_allclose(norm_rope(q,w,1e-5,c,s),norm_rope_scalar(q,w,1e-5,c,s),rtol=1e-10,atol=1e-12)
        with self.assertRaises(ValueError):norm_rope(q.transpose(0,2,1,3),w,1e-5,c,s)
        with self.assertRaises(ValueError):norm_rope(q,w,1e-5,c[:,:4],s)
        self.assertGreater(np.max(np.abs(norm_rope(q,w,1e-5,c,s)-norm_rope(q,w,1e-5,np.roll(c,1,axis=1),s))),.01)
    def test_decomposition_cancellation_and_zero(self):
        r=decomposition(np.array([0.,1.]),np.array([1.,2.]),np.array([.2,1.2]),np.array([.8,1.8]))
        self.assertLess(r['signed_cross'],0);self.assertAlmostEqual(r['identity_residual'],0)
        z=decomposition(np.zeros(2),np.zeros(2),np.zeros(2),np.zeros(2));self.assertIsNone(z['cosine_d_n']);self.assertIsNone(z['D1_relative_RMSE'])
        with self.assertRaises(ValueError):decomposition(np.zeros(2),np.zeros(3),np.zeros(2),np.zeros(2))
    def test_structured_coverage_and_shared_bytes(self):
        fixtures=structured();plus=[f for f in fixtures if f[1]=='ONE_HOT_PLUS'];minus=[f for f in fixtures if f[1]=='ONE_HOT_MINUS']
        np.testing.assert_array_equal(np.concatenate([f[2] for f in plus]),np.eye(1024))
        np.testing.assert_array_equal(np.concatenate([f[2] for f in minus]),-np.eye(1024))
        self.assertEqual(sum(f[3] for f in fixtures),3584)
        for _,_,z,_,_ in fixtures:np.testing.assert_array_equal(bfdecode(bfbits32(z)),z)
    def test_cli_blocked_and_cpu_help_without_torch(self):
        sup=Path(__file__).resolve().parents[1]/'source/supervise.py'
        r=subprocess.run([sys.executable,'-B',str(sup),'--help'],capture_output=True,text=True);self.assertEqual(r.returncode,0)
        r=subprocess.run([sys.executable,'-B',str(sup),'--root','/definitely/missing','--kind','CPU','--label','blocked','--timeout','1','--','true'],capture_output=True,text=True)
        self.assertNotEqual(r.returncode,0);self.assertNotIn('CUDA',r.stderr)
        self.assertNotIn('torch',sys.modules)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    write(a.output/'result.json',{'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'gate_regression_passed':result.wasSuccessful(),'torch_imported':'torch' in sys.modules})
    sys.exit(0 if result.wasSuccessful() else 2)
