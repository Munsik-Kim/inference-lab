import argparse,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from fractions import Fraction
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from common import write,freeze,verify
from arithmetic_v2 import *
from supervise_v2 import gate_then_start
class Contracts(unittest.TestCase):
    def test_gate_has_no_spawn_or_start_on_nonzero_exception_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            def bad():return subprocess.run([sys.executable,'-B','-c','raise SystemExit(7)']).returncode==0
            def error():raise ValueError('fake')
            def timeout():subprocess.run([sys.executable,'-B','-c','import time;time.sleep(2)'],timeout=.02,check=True)
            def start():(p/'journal').write_text('start')
            def spawn():return subprocess.run([sys.executable,'-B','-c','from pathlib import Path;Path('+repr(str(p/'sentinel'))+').write_text("spawned")'])
            for gate in [bad,error,timeout]:
                with self.assertRaises(Exception):gate_then_start(gate,start,spawn)
                self.assertFalse((p/'journal').exists());self.assertFalse((p/'sentinel').exists())
            gate_then_start(lambda:True,start,spawn);self.assertTrue((p/'sentinel').exists())
    def test_output_and_source_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);f=p/'source';f.write_text('before');s=freeze(p,[f]);verify(p,s);f.write_text('after')
            with self.assertRaises(ValueError):verify(p,s)
            write(p/'x.json',{'a':1})
            with self.assertRaises(FileExistsError):write(p/'x.json',{'a':2})
    def test_bf16_exact_rounding(self):
        values=[Fraction(1)+Fraction(1,256),Fraction(1)+Fraction(3,256),-Fraction(1)-Fraction(1,256),Fraction(1,2**134),Fraction(3,2**134),Fraction(1,2**126)-Fraction(1,2**134),Fraction(1<<128),Fraction(0)]
        for q in values:self.assertEqual(int(round_bits(float(q))),rational_bits(q))
        self.assertEqual(int(round_bits(-0.)),0x8000)
        q=Fraction(1)+Fraction(1,256)+Fraction(1,2**30)
        self.assertEqual(rational_bits(q),0x3f81)
        self.assertNotEqual(int(round_bits(float(np.float32(float(q))))),rational_bits(q))
        self.assertEqual(rational_bits(Fraction(1,2**134)),0);self.assertEqual(rational_bits(Fraction(3,2**134)),2)
    def test_candidates_example_and_scalar(self):
        for a,expected in [(1,(3.03125,3.03125)),(.125,(.37890625,.37890625)),(1.5,(4.5625,4.53125))]:
            bits=candidates(a,3,1.0078125);self.assertEqual(tuple(float(decode(x)) for x in bits),expected);self.assertEqual(tuple(int(x) for x in bits),exact_candidates(a,3,1.0078125))
        rng=np.random.default_rng(61027028)
        for _ in range(100):
            aa=float(decode(np.array(rng.integers(0x3000,0x4500),np.uint16)));ss=float(decode(np.array(rng.integers(0x3000,0x4200),np.uint16)));c=int(rng.integers(-8,8))
            self.assertEqual(tuple(int(x) for x in candidates(aa,c,ss)),exact_candidates(aa,c,ss))
    def test_ulp_signed_zero_order_finite(self):
        v=np.array([0xbf81,0xbf80,0x8001,0x8000,0,1,0x3f80,0x3f81],np.uint16);self.assertTrue(np.all(np.diff(rank(v))>=0));self.assertEqual(int(steps(v[2],v[5])),2);self.assertEqual(int(steps(v[3],v[4])),0)
        with self.assertRaises(ValueError):rank(np.array([0x7fc0],np.uint16))
    def test_onehot_and_nested(self):
        r=[{'axis':i,'coefficient_bits':int(round_bits((-1 if i%2 else 1)*1.5))} for i in range(512)];z=onehot(r);self.assertTrue(np.all(np.count_nonzero(decode(z),1)==1))
        prev=np.zeros_like(z,dtype=bool)
        for K,x,order in nested(z,0,0):
            mask=np.zeros_like(z,dtype=bool);np.put_along_axis(mask,order,True,1);self.assertTrue(np.all(~prev|mask));prev=mask;self.assertTrue(np.all(np.count_nonzero(decode(x),1)<=K))
            if K==1024:np.testing.assert_array_equal(x,z)
        with self.assertRaises(ValueError):onehot([{'axis':0,'coefficient_bits':0}])
    def test_layout_beta_and_energy(self):
        c=np.array([-8,-7,-1,0,1,7]);self.assertFalse(np.array_equal(c&15,c+8));self.assertTrue(np.array_equal(((c+8)&15)-8,c))
        scales=np.arange(128).reshape(2,64);permuted=scales.reshape(-1,8,8).transpose(0,2,1).reshape(2,64);self.assertFalse(np.array_equal(scales,permuted));self.assertTrue(np.array_equal(scales,permuted.reshape(-1,8,8).transpose(0,2,1).reshape(2,64)))
        x=np.ones(4);beta=np.array([.5,2,4,.25]);z=decode(round_bits(x*beta));self.assertFalse(np.array_equal(z,z*beta))
        d=np.array([1.,-2.]);n0=np.array([-.9,1.8]);n1=np.array([-.1,.2])
        G=lambda n:np.dot(d+n,d+n);N=lambda n:np.dot(n,n);C=lambda n:2*np.dot(d,n)
        self.assertAlmostEqual(G(n1)-G(n0),N(n1)-N(n0)+C(n1)-C(n0));self.assertIsNone(metrics(np.zeros(2),np.zeros(2))['relative_RMSE'])
    def test_help_and_blocked_exit(self):
        f=Path(__file__).resolve().parents[1]/'source/supervise.py';q=subprocess.run([sys.executable,'-B',str(f),'--help'],capture_output=True,text=True);self.assertEqual(q.returncode,0)
        q=subprocess.run([sys.executable,'-B',str(f),'--root','/nonexistent-case021-p5','--kind','CPU','--label','blocked','--timeout','1','--','true'],capture_output=True,text=True);self.assertNotEqual(q.returncode,0);self.assertNotIn('torch',sys.modules)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts));write(a.output/'result.json',{'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'gate_regression_passed':result.wasSuccessful(),'torch_imported':'torch' in sys.modules});sys.exit(0 if result.wasSuccessful() else 2)
