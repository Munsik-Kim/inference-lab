import copy,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
SRC=Path(__file__).resolve().parents[1]/'source';CASE=SRC.parents[2];sys.path.insert(0,str(SRC));sys.path.insert(0,str(CASE/'scripts'))
from p2_common import groups,all_names
from p2_fit import factor,solve
from model_state import WeightBank
from independent_codec import read
from codec import encode,decode
from fit import fit_numpy
from artifact_io import write_new

class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for n in ['q_proj','k_proj','v_proj','o_proj']:setattr(self,n,torch.nn.Linear(8,8,bias=False,dtype=torch.float64))
    def forward(self,x):
        q,k,v=[getattr(self,n)(x).view(1,-1,2,4).transpose(1,2) for n in ['q_proj','k_proj','v_proj']]
        y=torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)
        return self.o_proj(y.transpose(1,2).reshape(1,-1,8))
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__();self.gate_proj=torch.nn.Linear(8,16,bias=False,dtype=torch.float64);self.up_proj=torch.nn.Linear(8,16,bias=False,dtype=torch.float64);self.down_proj=torch.nn.Linear(16,8,bias=False,dtype=torch.float64)
    def forward(self,x):return self.down_proj(torch.nn.functional.silu(self.gate_proj(x))*self.up_proj(x))
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.self_attn=Attention();self.mlp=MLP();self.n1=torch.nn.LayerNorm(8,dtype=torch.float64);self.n2=torch.nn.LayerNorm(8,dtype=torch.float64)
    def forward(self,x):
        y=x+self.self_attn(self.n1(x));return y+self.mlp(self.n2(y))
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([Block(),Block()])
    def forward(self,x):
        for b in self.model.layers:x=b(x)
        return x
def capture(model,x):
    out={};hooks=[]
    def make(n):
        def h(m,inp,y):out[n]=inp[0].detach().clone()
        return h
    try:
        for n in all_names(2):hooks.append(model.get_submodule(n).register_forward_hook(make(n)))
        with torch.no_grad():y=model(x)
        return y,out
    finally:
        for h in hooks:h.remove()

class P2(unittest.TestCase):
    def setUp(self):torch.manual_seed(6101522);self.model=Tiny().eval();self.x=torch.randn(1,6,8,dtype=torch.float64)
    def test_topology_common_inputs_and_same_block_recompute(self):
        _,a=capture(self.model,self.x)
        for b in [0,1]:
            g=groups(b);self.assertTrue(torch.equal(a[g[0][1][0]],a[g[0][1][1]]));self.assertTrue(torch.equal(a[g[0][1][0]],a[g[0][1][2]]));self.assertTrue(torch.equal(a[g[2][1][0]],a[g[2][1][1]]))
        bank=WeightBank(self.model,all_names(2));n=groups(0)[0][1][0];bank.install(n,bank.originals[n]*3)
        _,b=capture(self.model,self.x);self.assertTrue(torch.equal(a[n],b[n]));self.assertFalse(torch.equal(a[groups(0)[1][1][0]],b[groups(0)[1][1][0]]))
        n=groups(0)[2][1][0];bank.restore();bank.install(n,bank.originals[n]*3);_,c=capture(self.model,self.x)
        self.assertFalse(torch.equal(a[groups(0)[3][1][0]],c[groups(0)[3][1][0]]))
    def test_candidates_have_independent_paths_and_original_W(self):
        bank=WeightBank(self.model,all_names(2));n=groups(0)[1][1][0];original=bank.originals[n].clone();paths=[]
        for alpha in [.25,.5]:
            bank.restore();bank.install(n,original+alpha*torch.ones_like(original));paths.append(capture(self.model,self.x)[1][groups(1)[0][1][0]])
        self.assertFalse(torch.equal(*paths));self.assertTrue(torch.equal(bank.originals[n],original));bank.restore();self.assertTrue(bank.unchanged())
    def test_future_block_cannot_change_prior_input(self):
        _,a=capture(self.model,self.x);bank=WeightBank(self.model,all_names(2));n=groups(1)[3][1][0];bank.install(n,bank.originals[n]*10);_,b=capture(self.model,self.x)
        for _,names in groups(0):
            for name in names:self.assertTrue(torch.equal(a[name],b[name]))
    def test_CAL_only_and_factor_shared_matches_independent_numpy(self):
        r=np.random.default_rng(5);X=r.normal(size=(4,16));Z=X+.1*r.normal(size=X.shape);docs=[{'split':'CAL-FIT','document_id_hash':'a','input_ids':list(range(16))}]
        f=factor(X,Z,docs)
        for rows in [3,7]:
            W=r.normal(size=(rows,4));K,d=solve(W,f,row_chunk=2);N,lam,_=fit_numpy(W,X,Z);np.testing.assert_allclose(K,N,rtol=1e-10,atol=1e-12);self.assertAlmostEqual(f['lambda'],lam)
        docs[0]['split']='EVAL-ID'
        with self.assertRaises(ValueError):factor(X,Z,docs)
    def test_X_equals_Z_zero_K_and_alpha0_same_RTN(self):
        r=np.random.default_rng(6);X=r.normal(size=(4,16));W=r.normal(size=(3,4));docs=[{'split':'CAL-FIT','document_id_hash':'a','input_ids':list(range(16))}]
        K,_=solve(W,factor(X,X,docs));self.assertEqual(np.count_nonzero(K),0)
        self.assertEqual(encode(W)[0],encode(W+0*K)[0]);self.assertEqual(encode(W)[0],encode(W+.5*K)[0])
    def test_independent_decoder_codes_scales_weights_and_reload(self):
        r=np.random.default_rng(9);w=r.normal(size=(5,65)).astype('f4');w[0]=0
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tiny.int4';p.write_bytes(encode(w)[0]);a=read(p);b=decode(p.read_bytes())
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
            self.assertEqual(p.stat().st_size,36+5*2*4+5*128//2)
    def test_exception_restores_weights_biases_and_hooks(self):
        bank=WeightBank(self.model,all_names(2));n=all_names(2)[0]
        with self.assertRaises(RuntimeError):
            with bank.restored_on_exit():bank.install(n,bank.originals[n]*9);raise RuntimeError('injected')
        self.assertTrue(bank.unchanged());self.assertTrue(all(not m._forward_hooks for m in self.model.modules()))
    def test_all_projection_count_unique_and_group_order(self):
        self.assertEqual(len(all_names()),196);self.assertEqual(len(set(all_names())),196)
        self.assertEqual([g for g,_ in groups(7)],['qkv','o','gate_up','down'])
    def test_atomic_payload_validation_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'w.int4';w=np.arange(65,dtype='f4').reshape(1,65)
            out=write_new(p,w);self.assertTrue(p.is_file());self.assertFalse(p.with_name('w.int4.partial').exists())
            np.testing.assert_array_equal(out[0],read(p)[0])
            with self.assertRaises(FileExistsError):write_new(p,w)
            bad=Path(d)/'bad.int4'
            with self.assertRaises(ValueError):write_new(bad,np.array([[np.nan]],dtype='f4'))
            self.assertFalse(bad.exists())
if __name__=='__main__':unittest.main()
