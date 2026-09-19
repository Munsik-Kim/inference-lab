"""Optional independent ridge check. Requires excluded private H/Y/FP64 solutions."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.modelpack.common import read,write,require,sha


def check(calibration,development,output):
    eta=read(development/'selection.json')['eta'];y=np.load(calibration/'teacher.npy',allow_pickle=False);result={}
    for name in ['I25','P25','S50']:
        path=calibration/(name+'-statistics.npz');x=np.load(path,allow_pickle=False);h=x['H'];g=x['G'];b=x['B'];w0=x['W0']
        require(np.isfinite(y).all() and np.isfinite(h).all(),'Invalid private vectors')
        np.testing.assert_allclose(h.T@h/len(h),g,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(y.T@h/len(h),b,rtol=1e-12,atol=1e-12)
        lam=eta*np.trace(g)/len(g);a=g+lam*np.eye(len(g));rhs=b+lam*w0
        # Direct dense solve is separate from the primary two-step Cholesky solve.
        alternate=np.linalg.solve(a,rhs.T).T
        saved=np.load(calibration/f'{name}-eta-{eta}-fp64.npy',allow_pickle=False)
        residual=float(np.linalg.norm(saved@a-rhs)/np.linalg.norm(rhs))
        relative=float(np.linalg.norm(alternate-saved)/np.linalg.norm(saved))
        require(residual<1e-9 and relative<1e-9,'Independent ridge mismatch')
        result[name]={'eta':eta,'lambda':float(lam),'relative_solution_difference':relative,
           'normal_equation_relative_residual':residual,'statistics_sha256':sha(path),
           'sampled_vectors':len(h),'status':'PASS'}
    write(output,{'status':'PASS','checks':result,'scope':'private stored H/Y and selected FP64 solutions; not fresh GPU replication'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--development',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();check(a.calibration,a.development,a.output)
