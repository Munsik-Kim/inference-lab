"""Fresh GPU process: independent INT4 decoding, all196 installed weights, SMOKE parity."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from p2_common import all_names,sha,save,utc
from independent_codec import read
from model_state import WeightBank
from run_p2 import configure,tensor_sha

def main():
    ap=argparse.ArgumentParser()
    for k in ['case','private','output']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--remaining-seconds',type=float,required=True);a=ap.parse_args();start=time.monotonic();case=Path(a.case);pr=Path(a.private);out=Path(a.output)
    identity=json.loads((pr/'identity.private.json').read_text());frozen=json.loads((out/'artifact_frozen.json').read_text());manifest=json.loads((out/'input_manifest.json').read_text())
    assert sha(pr/'inputs/documents.private.json')==manifest['private_tokens_sha256']
    docs=json.loads((pr/'inputs/documents.private.json').read_text());d=next(d for d in docs if d['split']=='SMOKE')
    expected={(r['method'],r['name']):r for r in [json.loads(s) for s in (out/'artifacts.jsonl').read_text().splitlines()]}
    configure();model=AutoModelForCausalLM.from_pretrained(identity['snapshot']['model_path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    names=all_names(model.config.num_hidden_layers);bank=WeightBank(model,names);checked=[];count=0
    sys.path.insert(0,str(case/'scripts'));import codec,p2_fit
    def deny(*a,**k):raise RuntimeError('reload worker forbids fitting/encoding')
    codec.encode=codec.write_reload=p2_fit.factor=p2_fit.solve=deny
    with bank.restored_on_exit(),torch.inference_mode():
        for method in ['BF16','RTN','COMP_QUARTER','COMP_HALF']:
            if time.monotonic()-start>a.remaining_seconds:raise TimeoutError('remaining P2 budget')
            bank.restore();files=0
            if method!='BF16':
                for n in names:
                    p=pr/'artifacts'/method/(n+'.int4');m=expected[method,n];assert sha(p)==m['sha256'] and p.stat().st_size==m['artifact_bytes']
                    w,q,s=read(p);assert list(w.shape)==m['shape'];bank.install(n,w)
                    assert tensor_sha(model.get_submodule(n).weight)==m['installed_BF16_sha256'];files+=1
            assert model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr()
            ids=torch.tensor([d['input_ids']],device='cuda',dtype=torch.long);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
            result=model(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.arange(256,device='cuda').unsqueeze(0),use_cache=False,return_dict=True)
            logits=result.logits[0];assert logits.shape==(256,model.config.vocab_size) and torch.isfinite(logits).all()
            got=tensor_sha(logits);old=json.loads((pr/f'smoke/{method}_stats.json').read_text());assert got==old['full_logits_BF16_sha256'],'fresh-process full-model logit SHA mismatch'
            nll=float(torch.nn.functional.cross_entropy(logits[:-1].float(),ids[0,1:],reduction='sum'));assert nll==old['NLL_sum']
            torch.cuda.synchronize();count+=1
            checked.append({'method':method,'status':'PASS','independent_decoded_files':files,'installed_BF16_weight_hashes':files,'full_logits_BF16_sha256':got,'NLL_sum':nll,
                            'NLL_exact':True,'full_logits_exact':True,'input_token_hash':d['token_hash'],'wall_seconds':time.monotonic()-tick,
                            'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
    assert bank.unchanged()
    save(out/'fresh_process_reload.json',{'status':'PASS','recorded_utc':utc(),'checks':checked,'full_model_forwards':count,'independently_decoded_artifacts':588,
         'all196_installed_weights_per_method':'PASS','model_restore':'PASS','fit_encode_calls':0,'source_sha256':sha(__file__),'worker_internal_seconds':time.monotonic()-start,
         'scope':'new GPU process full model SMOKE parity, not fresh evaluation rerun; scalar auditor is separate'})
    print('Fresh-process reload PASS:588 independent decodes/installed weight checks +4 full-model SMOKE output hashes.',flush=True)
if __name__=='__main__':main()
