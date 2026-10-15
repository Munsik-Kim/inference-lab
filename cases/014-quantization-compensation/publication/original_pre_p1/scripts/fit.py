"""Full-input CPU FP64 ridge, Cholesky, reusable factor and row chunks."""
import numpy as np

def fit_numpy(w,x,z,ridge=1e-3):
 w,x,z=(np.asarray(a,np.float64) for a in (w,x,z));g=z@z.T
 lam=float(ridge*np.trace(g)/g.shape[0])
 if not np.isfinite(g).all() or np.trace(g)<=0:raise ValueError('zero/nonfinite Gram')
 h=g+lam*np.eye(len(g));l=np.linalg.cholesky(h);rhs=(w@(x-z))@z.T
 k=np.linalg.solve(l.T,np.linalg.solve(l,rhs.T)).T
 return k,lam,{'relative_solve_residual':float(np.linalg.norm(k@h-rhs)/max(np.linalg.norm(rhs),1)), 'condition':float(np.linalg.cond(h))}

def fit_torch(w,x,z,ridge=1e-3,row_chunk=128):
 import torch
 w,x,z=(torch.as_tensor(np.asarray(a,dtype=np.float64),device='cpu') for a in (w,x,z))
 g=z@z.T;trace=g.trace();lam=float(ridge*trace/g.shape[0])
 if not torch.isfinite(g).all() or trace<=0:raise ValueError('zero/nonfinite Gram')
 h=g+lam*torch.eye(len(g),dtype=torch.float64);l=torch.linalg.cholesky(h)
 k=torch.empty_like(w);num=den=0.
 for start in range(0,len(w),row_chunk):
  rhs=(w[start:start+row_chunk]@(x-z))@z.T
  v=torch.linalg.solve_triangular(l,rhs.T,upper=False)
  out=torch.linalg.solve_triangular(l.T,v,upper=True).T;k[start:start+row_chunk]=out
  num+=float(((out@h-rhs)**2).sum());den+=float((rhs**2).sum())
 # eigenvalues diagnose full regularized Gram, no approximate/block-diagonal fit.
 e=torch.linalg.eigvalsh(h)
 return k.numpy(),lam,{'relative_solve_residual':(num/max(den,1))**.5,'condition':float(e[-1]/e[0]),'min_eigenvalue':float(e[0]),'max_eigenvalue':float(e[-1]),'lambda':lam,'trace':float(trace),'dtype':'CPU FP64','row_chunk':row_chunk}
