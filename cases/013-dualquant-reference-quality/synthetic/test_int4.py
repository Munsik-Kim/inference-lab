"""Small CPU-only tests of the exact published core; no checkpoint or vendor code."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import torch
from reference_int4 import QuantizedWeight, quantize_element_int4, pack_nibbles, unpack_nibbles
from paper_int4 import solve, serialize_fast, fast_ledger, activation_int4

class Int4Contracts(unittest.TestCase):
    def test_all_signed_codes_pack_bit_exact(self):
        x=torch.arange(-8,8,dtype=torch.int8).reshape(2,8)
        p=pack_nibbles(x)
        self.assertTrue(torch.equal(unpack_nibbles(p,x.shape),x))
    def test_rounding_and_saturation(self):
        x=torch.tensor([-9.,-.5,.5,1.5,8.])
        self.assertEqual(quantize_element_int4(x).tolist(),[-8,0,0,2,7])
    def test_serialize_reload_and_actual_ledger(self):
        torch.manual_seed(58);q,_=solve(torch.randn(7,128),iterations=1)
        b=serialize_fast(q);loaded=QuantizedWeight.deserialize(b)
        self.assertEqual(b,q.serialize());self.assertEqual(fast_ledger(q,b)['serialized_bytes'],len(b))
        self.assertTrue(torch.equal(loaded.codes,q.codes));self.assertTrue(torch.equal(loaded.beta_inv,q.beta_inv))
    def test_group_chunk_full_row_reduction(self):
        torch.manual_seed(55);w=torch.randn(31,256)
        a,_=solve(w,iterations=5,group_chunk=1);b,_=solve(w,iterations=5,group_chunk=4)
        torch.testing.assert_close(a.weight_hat(),b.weight_hat(),atol=1e-5,rtol=1e-4)
    def test_beta_transport_and_omission_control(self):
        torch.manual_seed(56);w=torch.randn(16,128);w[:,7]*=50;x=torch.randn(5,128)
        q,_=solve(w,iterations=5)
        y=torch.nn.functional.linear(x*q.beta_inv,q.weight_pre())
        torch.testing.assert_close(y,torch.nn.functional.linear(x,q.weight_hat()),atol=2e-4,rtol=2e-4)
        self.assertGreater(float((y-torch.nn.functional.linear(x,q.weight_pre())).abs().max()),.1)
    def test_zero_tiny_and_invalid_failure(self):
        q,_=solve(torch.zeros(7,128));self.assertTrue(torch.equal(q.weight_hat(),torch.zeros(7,128)))
        x=torch.full((7,128),1e-15);x[0]=0;x[:,0]=0
        with self.assertRaisesRegex(ValueError,'zero/nonpositive beta'):solve(x)
        x[1,1]=float('nan')
        with self.assertRaises(ValueError):solve(x)
    def test_activation_zero_and_ties(self):
        x=torch.zeros(2,128);x[0,1]=7;x[0,2]=.5;x[0,3]=-.5
        y=activation_int4(x);self.assertTrue(torch.equal(y[1],x[1]));self.assertEqual(y[0,2].item(),0)

if __name__=='__main__':unittest.main()
