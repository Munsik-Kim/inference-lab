"""Restore original projection weights and biases after any local intervention."""
import contextlib
import torch

class WeightBank:
    def __init__(self,model,names):
        self.model=model;self.names=list(names)
        self.originals={n:model.get_submodule(n).weight.detach().cpu().clone() for n in names}
        self.biases={n:None if model.get_submodule(n).bias is None else model.get_submodule(n).bias.detach().cpu().clone() for n in names}
    @torch.no_grad()
    def restore(self):
        for n,w in self.originals.items():
            m=self.model.get_submodule(n);m.weight.copy_(w.to(device=m.weight.device,dtype=m.weight.dtype))
            b=self.biases[n]
            if b is not None:m.bias.copy_(b.to(device=m.bias.device,dtype=m.bias.dtype))
    @torch.no_grad()
    def install(self,n,weight):
        m=self.model.get_submodule(n)
        q=torch.as_tensor(weight,device=m.weight.device,dtype=m.weight.dtype)
        if q.shape!=m.weight.shape or not torch.isfinite(q).all():raise ValueError('installed weight shape/nonfinite')
        m.weight.copy_(q)
    def unchanged(self):
        return all(torch.equal(self.model.get_submodule(n).weight.detach().cpu(),w) and
                   (b is None or torch.equal(self.model.get_submodule(n).bias.detach().cpu(),b)) for n,w in self.originals.items() for b in [self.biases[n]])
    @contextlib.contextmanager
    def restored_on_exit(self):
        try:yield self
        finally:self.restore()

