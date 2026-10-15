"""Registered reused-SMOKE matrix check, using saved K/INT4 without refitting."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from p2_common import all_names,ALPHA,sha,save,utc
from run_p2 import configure,tensor_sha,putarray
from model_state import WeightBank
from independent_codec import read

def main():
    ap=argparse.ArgumentParser()
    for k in ['case','private','output']:ap.add_argument('--'+k,required=True)
    a=ap.parse_args();start=time.monotonic();case=Path(a.case);pr=Path(a.private);out=Path(a.output)
    registration=json.loads((out/'supplemental_smoke_registration.json').read_text());identity=json.loads((pr/'identity.private.json').read_text());n=registration['module']
    sys.path.insert(0,str(case/'scripts'));import decompose,codec,p2_fit
    def deny(*a,**k):raise RuntimeError('SMOKE matrix probe forbids fitting/encoding')
    codec.encode=codec.write_reload=p2_fit.factor=p2_fit.solve=deny
    d=next(x for x in json.loads((pr/'inputs/documents.private.json').read_text()) if x['split']=='SMOKE')
    configure();model=AutoModelForCausalLM.from_pretrained(identity['snapshot']['model_path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();names=all_names();bank=WeightBank(model,names)
    w=bank.originals[n].float().numpy().astype('f8');base={};rows=[];counts=0
    with bank.restored_on_exit(),torch.inference_mode():
        for method in ['BF16','RTN','COMP_QUARTER','COMP_HALF']:
            bank.restore()
            if method!='BF16':
                for name in names:bank.install(name,read(pr/'artifacts'/method/(name+'.int4'))[0])
            buf={}
            def hook(m,inp,y):buf.update(X=inp[0].detach().clone().float().cpu().numpy()[0].T,Y=y.detach().clone().float().cpu().numpy()[0].T)
            h=model.get_submodule(n).register_forward_hook(hook)
            try:
                ids=torch.tensor([d['input_ids']],device='cuda',dtype=torch.long);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
                result=model(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(256,device='cuda').unsqueeze(0),use_cache=False,return_dict=True)
                assert torch.isfinite(result.logits).all();counted_sha=tensor_sha(result.logits[0]);expected=json.loads((pr/f'smoke/{method}_stats.json').read_text())['full_logits_BF16_sha256'];assert counted_sha==expected
                counts+=1
            finally:h.remove()
            if method=='BF16':base=buf
            else:
                q=read(pr/'artifacts'/method/(n+'.int4'))[0];q0=read(pr/'artifacts/RTN'/(n+'.int4'))[0]
                K=np.zeros_like(w) if method=='RTN' else np.load(pr/'artifacts'/method/'K'/(n+'.npy'))
                alpha=ALPHA[method];m=decompose.measure(w,base['X'],buf['X'],alpha*K,q,q0,base['Y'],buf['Y'],model.get_submodule(n).weight.detach().float().cpu().numpy())
                assert m['identity_relative_gap']<=1e-10 and m['residual_identity_relative']<=1e-10
                putarray(pr/f'smoke_matrix/{method}_Z.npy',buf['X']);putarray(pr/f'smoke_matrix/{method}_Y.npy',buf['Y'])
                rows.append({'method':method,'module':n,'alpha':alpha,'token_hash':d['token_hash'],'full_logits_match_built_model':True,**m,
                             'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'wall_seconds':time.monotonic()-tick})
        putarray(pr/'smoke_matrix/teacher_X.npy',base['X']);putarray(pr/'smoke_matrix/teacher_Y.npy',base['Y'])
    assert bank.unchanged()
    save(out/'smoke_matrix_verification.json',{'status':'PASS','rows':rows,'full_model_forwards':counts,'refit_or_encode_calls':0,'registered_before_fresh_outputs':True,'source_sha256':sha(__file__),'worker_internal_seconds':time.monotonic()-start,
          'interpretation':'fixed first-block o projection on reused SMOKE; full stored-weight model inputs; local alpha0 counterfactual distinct from a general whole-model RTN decomposition'})
    print('Reused-SMOKE matrix identity PASS for RTN/QUARTER/HALF;4 native full-model output hashes also matched.')
if __name__=='__main__':main()
