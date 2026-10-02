"""Synthetic arithmetic contracts; not model quality or SM120 evidence."""
import torch
from source.adapter import make_noise,sample_from_noise
from looped_dit.diffusion import euler_sample
class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.calls=[]
    def forward(self,x,text,mask,num_loops=None):
        self.calls.append((num_loops,mask.sum().item(),str(x.dtype)))
        return x.float()*.2+mask.sum().float()*.01

def test_dedicated_rng_unaffected_by_global_draws():
    x=make_noise(11,'cpu',torch.float32,8);torch.randn(300)
    assert torch.equal(x,make_noise(11,'cpu',torch.float32,8))
def test_noise_different_seeds():assert not torch.equal(make_noise(1,'cpu',torch.float32,8),make_noise(2,'cpu',torch.float32,8))
def test_official_cpu_parity_and_calls():
    m=Model();text=torch.ones(1,2,3);mask=torch.ones(1,2,dtype=torch.long)
    torch.manual_seed(42);old=euler_sample(m,text,mask,8,3,6,2,2)
    m.calls=[];new,raw,trace=sample_from_noise(m,text,mask,make_noise(42,'cpu',torch.float32,8),3,2,True)
    assert torch.equal(old,new) and len(m.calls)==6 and len(trace)==3
    assert all(m.calls[i][1]==(2 if i%2==0 else 0) for i in range(6))
    assert all(t['finite'] for t in trace)
def test_nonfinite_pre_clamp_caught():
    class Bad(Model):
        def forward(self,*a,**k):return torch.full_like(a[0],float('inf'))
    import pytest
    with pytest.raises(FloatingPointError):sample_from_noise(Bad(),torch.ones(1,2,3),torch.ones(1,2),torch.ones(1,3,8,8),2,1,True)
def test_training_flag_restored_on_error():
    class Bad(Model):
        def forward(self,*a,**k):raise RuntimeError('synthetic_test')
    m=Bad().train();import pytest
    with pytest.raises(RuntimeError):sample_from_noise(m,torch.ones(1,2,3),torch.ones(1,2),torch.ones(1,3,8,8),2,1)
    assert m.training
