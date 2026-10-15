"""CPU-only local audit using private saved SMOKE matrices and selected CAL rows."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from p2_common import ALPHA,save,sha,utc,jsonl
from independent_codec import read

def main():
    ap=argparse.ArgumentParser()
    for k in ['private','run','output']:ap.add_argument('--'+k,required=True)
    a=ap.parse_args();start=time.monotonic();pr=Path(a.private);run=Path(a.run);identity=json.loads((pr/'identity.private.json').read_text());torch.set_num_threads(4)
    probe=json.loads((run/'smoke_matrix_verification.json').read_text());dmap={(r['method'],r['name']):r for r in jsonl(run/'CAL_diagnostics.jsonl')};docs=json.loads((pr/'inputs/documents.private.json').read_text());cal=[d for d in docs if d['split']=='CAL-FIT'];smokechecks=[];solvechecks=[]
    with safe_open(str(Path(identity['snapshot']['model_path'])/'model.safetensors'),framework='pt',device='cpu') as f:
        # Full first-o SMOKE matrices are independently multiplied with Torch CPU FP64.
        n='model.layers.0.self_attn.o_proj';W=f.get_tensor(n+'.weight').float().double();X=torch.from_numpy(np.load(pr/'smoke_matrix/teacher_X.npy').astype('f8'))
        for r in probe['rows']:
            method=r['method'];Z=torch.from_numpy(np.load(pr/f'smoke_matrix/{method}_Z.npy').astype('f8'));K=torch.zeros_like(W) if method=='RTN' else torch.from_numpy(np.load(pr/'artifacts'/method/'K'/(n+'.npy')))
            Q=torch.from_numpy(read(pr/'artifacts'/method/(n+'.int4'))[0].astype('f8'));Q0=torch.from_numpy(read(pr/'artifacts/RTN'/(n+'.int4'))[0].astype('f8'));D=ALPHA[method]*K
            A=W@(X-Z);C=D@Z;B=(Q-W-D)@Z;B0=(Q0-W)@Z;DB=B-B0;R=W@X-Q@Z;R0=W@X-Q0@Z
            ss=lambda v:float((v*v).sum());dot=lambda x,y:float((x*y).sum())
            values={'SSE_after':ss(R),'SSE_before':ss(R0),'A_norm2':ss(A),'C_norm2':ss(C),'B_norm2':ss(B),'B0_norm2':ss(B0),'T_inherited':ss(C)-2*dot(A,C),'T_cancellation':2*dot(B0,C),'T_requant':ss(DB)-2*dot(R0-C,DB)}
            scale=max(1,r['SSE_before'],r['SSE_after'],abs(r['T_cont']),abs(r['T_requant']));worst=max(abs(v-r[k])/scale for k,v in values.items());assert worst<=1e-10
            smokechecks.append({'method':method,'module':n,'max_scaled_scalar_difference':worst,'fields':len(values),'status':'PASS'})
        # Verify the ridge normal equations on eight fixed output rows, retaining all CAL tokens/input dimensions.
        fixed=[(0,'qkv','self_attn.q_proj'),(0,'o','self_attn.o_proj'),(12,'gate_up','mlp.up_proj'),(27,'down','mlp.down_proj')]
        for method in ['COMP_QUARTER','COMP_HALF']:
            for b,g,suffix in fixed:
                n=f'model.layers.{b}.'+suffix;W=f.get_tensor(n+'.weight')[:8].float().numpy().astype('f8');K=np.load(pr/'artifacts'/method/'K'/(n+'.npy'),mmap_mode='r')[:8].copy()
                X=np.concatenate([np.load(pr/'traces/teacher'/f'b{b}_{g}'/(d['document_id_hash']+'.npy')).astype('f8') for d in cal],axis=1)
                Z=np.concatenate([np.load(pr/'traces'/method/f'b{b}_{g}'/(d['document_id_hash']+'.npy')).astype('f8') for d in cal],axis=1);diag=dmap[method,n];lam=.001*float(np.sum(Z*Z))/Z.shape[0]
                assert abs(lam-diag['lambda'])<=1e-10*max(1,abs(lam))
                rhs=(W@(X-Z))@Z.T;left=(K@Z)@Z.T+lam*K
                residual=float(np.linalg.norm(left-rhs)/max(1,np.linalg.norm(rhs)));assert residual<=1e-10,(method,n,residual)
                solvechecks.append({'method':method,'module':n,'output_rows_checked':8,'all_input_dimensions':Z.shape[0],'CAL_tokens':Z.shape[1],'normal_equation_relative_residual':residual,'status':'PASS'})
    save(a.output,{'status':'PASS','recorded_utc':utc(),'full_SMOKE_matrix_checks':smokechecks,'selected_CAL_normal_equation_checks':solvechecks,'wall_seconds':time.monotonic()-start,'audit_source_sha256':sha(__file__),
       'scope':'independent Torch CPU FP64 SMOKE products and NumPy normal-equation check on fixed8 output rows with all input dimensions/CAL tokens; no GPU/model forward, refit or new quantization',
       'limits':'Not all392 K matrices audited row-by-row; GPU NLL/KL/logit evaluation not independently rerun by this CPU check; private arrays excluded from review ZIP.'})
    print('Private CPU audit PASS:3 full SMOKE matrix comparisons and8 fixed CAL normal-equation checks.')
if __name__=='__main__':main()
