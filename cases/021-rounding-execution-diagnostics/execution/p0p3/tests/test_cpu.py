import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import unittest
import tempfile
import subprocess
import json
import numpy as np
from layout import pack_signed,unpack_signed,pack_marlin_input,marlin_scale_permute,bf16,metrics
from primitives import kl,kl_loop,norm_rope,norm_rope_loop,vector_decomposition
from common import freeze_files,verify_files,write

class Contracts(unittest.TestCase):
    def test_signed_encoding_and_offset_are_different(self):
        c=np.array([-8,-7,-1,0,1,7,-8,7],np.int8)
        self.assertEqual(pack_signed(c),bytes([0x98,0x0f,0x71,0x78]))
        np.testing.assert_array_equal(unpack_signed(bytes([0x98,0x0f,0x71,0x78]),(8,)),c)
        packed=pack_marlin_input(c[None,:]); n=int(packed.view(np.uint32)[0,0])
        self.assertEqual([(n>>(4*i))&15 for i in range(8)],[0,1,7,8,9,15,0,15])
        self.assertNotEqual(pack_signed(c),packed.tobytes())
        with self.assertRaises(ValueError):pack_signed(np.array([-9,0]))
    def test_scale_coordinates(self):
        s=np.arange(256).reshape(2,128); p=marlin_scale_permute(s)
        for n in range(s.size):
            tile,offset=divmod(n,64); r,c=divmod(offset,8)
            self.assertEqual(p.ravel()[n],s.ravel()[tile*64+c*8+r])
        self.assertFalse(np.array_equal(p,s))
    def test_beta_once_and_axes(self):
        x=np.eye(4,dtype=np.float32); beta=np.array([.5,2,4,.25]); w=np.arange(12).reshape(3,4)
        y=(x*beta)@w.T
        for i in range(4):np.testing.assert_array_equal(y[i],beta[i]*w[:,i])
        self.assertFalse(np.array_equal(y,((x*beta)*beta)@w.T))
    def test_bf16_rounding_ties(self):
        np.testing.assert_array_equal(bf16(np.array([1.,1+1/256,1+3/256,-2.])),[1.,1.,1+4/256,-2.])
    def test_norm_rope_independent(self):
        rng=np.random.default_rng(61027021); q=rng.normal(size=(2,3,5,8)); w=np.linspace(.3,1.7,8)
        angle=np.arange(2*5*8).reshape(2,5,8)*.07; c,s=np.cos(angle),np.sin(angle)
        a=norm_rope(q,w,1e-5,c,s); b=norm_rope_loop(q,w,1e-5,c,s)
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)
        self.assertGreater(np.max(np.abs(a-norm_rope(q,w,1e-5,np.roll(c,1,axis=1),s))),.01)
    def test_kl_direction_and_stability(self):
        a=np.array([[1001,1002,999],[1,-9,3]],float); b=np.array([[1003,999,1002],[-3,2,5]],float)
        np.testing.assert_allclose(kl(a,b),kl_loop(a,b),rtol=1e-10,atol=1e-12)
        self.assertFalse(np.allclose(kl(a,b),kl(b,a)))
        np.testing.assert_allclose(kl(a,a),0,atol=1e-12)
    def test_energy_zero_and_vector_identity(self):
        self.assertIsNone(metrics(np.zeros(2),np.ones(2))['relative_RMSE'])
        r=vector_decomposition([1,2],[2,3],[1,4]); self.assertAlmostEqual(r['identity_residual'],0)
    def test_identity_mutation_and_output_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'input.bin';p.write_bytes(b'fixed input')
            f=freeze_files([p],root);verify_files(f,root)
            p.write_bytes(b'changed input')
            with self.assertRaisesRegex(ValueError,'SOURCE_INPUT_CONFIG_MISMATCH'):verify_files(f,root)
            dest=root/'result.json';write(dest,{'value':1})
            with self.assertRaises(FileExistsError):write(dest,{'value':2})
            self.assertEqual(json.loads(dest.read_text()),{'value':1})
    def test_blocked_cli_and_help_do_not_need_torch(self):
        worker=Path(__file__).resolve().parents[1]/'source/worker.py'
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json';p.write_text(json.dumps({'files':{},'bundle_sha256':'invalid'}))
            r=subprocess.run([sys.executable,str(worker),'--private',d,'--output',d,'--freeze',str(p),'--mode','full'],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('FREEZE_BUNDLE_MISMATCH',r.stderr)
        r=subprocess.run([sys.executable,str(worker),'--help'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0);self.assertIn('--mode',r.stdout)
    def test_cpu_entry_import_has_no_torch(self):
        source=Path(__file__).resolve().parents[1]/'source'
        code='import sys;sys.path.insert(0,'+repr(str(source))+');import common,layout,primitives,worker;assert "torch" not in sys.modules'
        subprocess.run([sys.executable,'-c',code],check=True)
    def test_large_vocabulary_independent_normalization(self):
        import importlib.util
        path=Path(__file__).resolve().parents[1]/'analysis/audit.py'
        spec=importlib.util.spec_from_file_location('case021_cpu_audit',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rng=np.random.default_rng(21);a=rng.normal(0,4,(2,151936));b=a+rng.normal(0,.03,a.shape)
        np.testing.assert_allclose(module.independent_kl(a,b),kl(a,b),rtol=1e-9,atol=2e-11)

if __name__=='__main__':unittest.main()
