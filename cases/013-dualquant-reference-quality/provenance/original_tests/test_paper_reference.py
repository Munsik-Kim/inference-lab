import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import torch
from torch import nn
from reference_int4 import quantize_weight,QuantizedWeight
from paper_int4 import solve,activation_int4,ReferenceLinear,replace_projection,serialize_fast,fast_ledger
from paper_model_runner import json_result,score_long

class PaperReferenceTests(unittest.TestCase):
    def test_fast_serializer_is_identical_to_original_format(self):
        torch.manual_seed(54)
        for method in ['rtn','row_only','row_column']:
            q,_=solve(torch.randn(5,128),method,1);data=serialize_fast(q)
            self.assertEqual(data,q.serialize());self.assertEqual(fast_ledger(q,data)['serialized_bytes'],len(data))
    def test_vectorized_axes_match_existing_golden(self):
        torch.manual_seed(53);w=torch.randn(19,128)*0.1
        for method in ['rtn','row_only','row_column']:
            a,_=solve(w,method,5,1);b=quantize_weight(w,method,num_iter=5)
            torch.testing.assert_close(a.weight_hat(),b.weight_hat(),atol=1e-6,rtol=1e-5)
    def test_group_chunks_preserve_full_row_reduction(self):
        torch.manual_seed(55);w=torch.randn(31,256)
        a,_=solve(w,'row_column',5,1);b,_=solve(w,'row_column',5,4)
        torch.testing.assert_close(a.weight_hat(),b.weight_hat(),atol=1e-5,rtol=1e-4)
    def test_full_beta_transport_and_negative_control(self):
        torch.manual_seed(56);w=torch.randn(16,128);w[:,7]*=50;x=torch.randn(5,128)
        q,_=solve(w,'row_column',5)
        layer=nn.Linear(128,16);layer.bias.data.zero_();ref=ReferenceLinear(layer,q)
        y=ref(x);expected=torch.nn.functional.linear(x,q.weight_hat())
        torch.testing.assert_close(y,expected,atol=2e-4,rtol=2e-4)
        self.assertGreater(float((y-torch.nn.functional.linear(x,q.weight_pre())).abs().max()),0.1)
    def test_row_scale_cancellation_fp64(self):
        torch.manual_seed(57);w=torch.randn(13,128,dtype=torch.float64)
        a=quantize_weight(w,'rtn');b=quantize_weight(w,'row_only',num_iter=15)
        torch.testing.assert_close(a.weight_hat(),b.weight_hat(),atol=1e-12,rtol=1e-12)
    def test_zero_tiny_and_nonfinite(self):
        w=torch.zeros(7,128)
        q,_=solve(w);self.assertTrue(torch.equal(q.weight_hat(),w))
        tiny=torch.full((7,128),1e-15);tiny[0]=0;tiny[:,0]=0
        t,_=solve(tiny,'rtn',0);self.assertTrue(torch.equal(t.weight_hat(),tiny))
        with self.assertRaisesRegex(ValueError,'zero/nonpositive beta'):
            solve(tiny,'row_column',15)
        bad=torch.ones(7,128);bad[0,0]=float('nan')
        with self.assertRaises(ValueError):solve(bad)
    def test_pack_load_all_scales(self):
        torch.manual_seed(58);q,_=solve(torch.randn(7,128),'row_column',1)
        b=q.serialize();r=QuantizedWeight.deserialize(b)
        self.assertTrue(torch.equal(q.codes,r.codes));self.assertTrue(torch.equal(q.beta_inv,r.beta_inv));self.assertEqual(q.storage_ledger()['serialized_bytes'],len(b))
    def test_activation_zero_group_outlier_and_rounding(self):
        x=torch.zeros(2,128);x[0,1]=7;x[0,2]=0.5;x[0,3]=-0.5
        y=activation_int4(x);self.assertEqual(float(y[0,2]),0);self.assertEqual(float(y[0,3]),0);self.assertTrue(torch.equal(y[1],x[1]))
    def test_only_allowed_projection_is_replaced(self):
        m=nn.Module();m.block=nn.Module();m.block.q_proj=nn.Linear(64,4);m.lm_head=nn.Linear(64,4)
        q,_=solve(m.block.q_proj.weight);replace_projection(m,'block.q_proj',q)
        self.assertIsInstance(m.block.q_proj,ReferenceLinear)
        with self.assertRaises(ValueError):replace_projection(m,'lm_head',q)
    def test_parser_and_exact_evidence_are_distinct(self):
        task={'answer':'22','evidence':['D1']}
        self.assertTrue(score_long('{"answer":"22","evidence":["D1"]}',task)['exact_success'])
        r=score_long('{"answer":"22","evidence":["D2"]}',task);self.assertTrue(r['parse_success']);self.assertFalse(r['exact_success'])
        self.assertIsNone(json_result('not json'))

if __name__=='__main__':unittest.main()
