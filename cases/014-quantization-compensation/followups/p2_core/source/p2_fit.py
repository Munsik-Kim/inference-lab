"""Full-input CPU FP64 ridge factor shared within one actual common-input group."""
import math,time
import numpy as np
import torch

def factor(x,z,documents,ridge=.001):
    if not documents or any(d['split']!='CAL-FIT' for d in documents):raise ValueError('fitting accepts CAL-FIT only')
    if len({d['document_id_hash'] for d in documents})!=len(documents):raise ValueError('duplicate CAL document')
    t=time.monotonic();X=torch.as_tensor(np.asarray(x,dtype=np.float64),device='cpu');Z=torch.as_tensor(np.asarray(z,dtype=np.float64),device='cpu')
    if X.shape!=Z.shape or X.ndim!=2 or X.shape[1]!=sum(len(d['input_ids']) for d in documents):raise ValueError('CAL shape/token mismatch')
    if not torch.isfinite(X).all() or not torch.isfinite(Z).all():raise ValueError('nonfinite CAL')
    G=Z@Z.T;trace=float(G.trace())
    if not math.isfinite(trace) or trace<=0:raise ValueError('zero/nonfinite Gram trace')
    lam=ridge*trace/G.shape[0];H=G+lam*torch.eye(len(G),dtype=torch.float64)
    L=torch.linalg.cholesky(H);e=torch.linalg.eigvalsh(H)
    return {'X':X,'Z':Z,'D':X-Z,'H':H,'L':L,'lambda':lam,'diagnostics':{'trace':trace,'lambda':lam,'dimension':len(G),'tokens':Z.shape[1],
        'min_eigenvalue':float(e[0]),'max_eigenvalue':float(e[-1]),'condition':float(e[-1]/e[0]),'factorization_seconds':time.monotonic()-t,'X_equals_Z':bool(torch.equal(X,Z))}}

def solve(w,f,row_chunk=128):
    t=time.monotonic();W=torch.as_tensor(np.asarray(w,dtype=np.float64),device='cpu')
    if W.ndim!=2 or W.shape[1]!=f['Z'].shape[0] or not torch.isfinite(W).all():raise ValueError('weight shape/finite')
    K=torch.empty_like(W);num=den=0.;chunks=0
    for lo in range(0,len(W),row_chunk):
        rhs=(W[lo:lo+row_chunk]@f['D'])@f['Z'].T
        v=torch.linalg.solve_triangular(f['L'],rhs.T,upper=False)
        k=torch.linalg.solve_triangular(f['L'].T,v,upper=True).T
        K[lo:lo+row_chunk]=k;num+=float(((k@f['H']-rhs)**2).sum());den+=float((rhs**2).sum());chunks+=1
    residual=math.sqrt(num/max(den,1.))
    if not torch.isfinite(K).all() or residual>1e-10:raise ValueError('nonfinite K / solve residual gate')
    if f['diagnostics']['X_equals_Z'] and torch.count_nonzero(K):raise ValueError('X=Z did not produce zero K')
    return K.numpy(),{'solve_seconds':time.monotonic()-t,'relative_solve_residual':residual,'RHS_squared_energy':den,'row_chunks':chunks,'row_chunk':row_chunk,'K_norm2':float((K*K).sum())}

