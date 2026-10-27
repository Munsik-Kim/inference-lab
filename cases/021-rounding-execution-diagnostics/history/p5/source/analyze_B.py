"""Exact single-term audit from saved output bits; branch C only from evidence."""
import argparse
from pathlib import Path
import numpy as np
from common import read,write,identity,csvwrite,array_hash
from arithmetic_v2 import decode,candidates,numeric_equal,ulp_metrics,finite
from prepare_v2 import load_artifact
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;a.output.mkdir(parents=True,exist_ok=False);data=read(r/'inputs_manifest/prepared.json');metrics=[];fits=[];examples=[];repeat=[];control_bad=0;dense_bad=0;actual_gap=0
    for layer in [0,25]:
        q,s,_,_=load_artifact(r,layer)
        for f in [x for x in data['fixtures'] if x['phase']=='B' and x['layer']==layer]:
            assert identity(f['path'])==f['identity'];z=np.load(f['path'],allow_pickle=False);assert array_hash(z)==f['array_sha256'] and np.all(np.count_nonzero(decode(z),1)<=1)
            n=f['valid_rows'];A=np.zeros((512,2048),np.uint16);B=np.zeros_like(A)
            for start in range(0,n,64):
                selection=f['rows'][start:start+64];axes=np.array([x['axis'] for x in selection]);coef=decode(np.array([x['coefficient_bits'] for x in selection],np.uint16)).astype(np.float64);A[start:start+len(selection)],B[start:start+len(selection)]=candidates(coef[:,None],q.codes[:,axes].T,s[:,axes//64].T)
            assert finite(A).all() and finite(B).all()
            out={}
            for method in ['D1','P']:
                for block in [0,1,2]:
                    receipt=read(r/f'records/gpu-B-l{layer}-b{block}/data/receipt.json');item=next(x for x in receipt['rows'] if x['fixture']==f['fixture'] and x['method']==method);assert item['input_sha256']==array_hash(z);assert item['method_order']==data['method_orders'][f"B:{layer}:{block}:{f['fixture']}"]
                    if block==0:
                        assert identity(item['output_path'])==item['output_identity'];out[method]=np.load(item['output_path'],allow_pickle=False);assert array_hash(out[method])==item['output_sha256']
                    else:repeat.append({'layer':layer,'block':block,'fixture':f['fixture'],'method':method,'same_bits':item['output_sha256']==array_hash(out[method]),'hash':item['output_sha256']})
                assert not np.any(decode(out[method][n:])),'ZERO_CONTROL'
            for label,ref,value in [('D1_vs_A',A,out['D1']),('P_vs_A',A,out['P']),('P_vs_B',B,out['P']),('P_vs_D1',out['D1'],out['P'])]:metrics.append({'layer':layer,'fixture':f['fixture'],'comparison':label,'valid_rows':n,'zero_control_rows':512-n,**ulp_metrics(ref[:n],value[:n])})
            db=~numeric_equal(out['D1'][:n],A[:n]);gap=~numeric_equal(out['D1'][:n],out['P'][:n]);dense_bad+=int(db.sum())
            if f['fixture'] in ['POWER1','POWER_EIGHTH']:control_bad+=int(gap.sum())
            actual=np.array([x['origin']=='ACTUAL_DEV' for x in f['rows']]);actual_gap+=int(gap[actual].sum())
            scopes={'ALL_VALID':np.ones((n,2048),bool),'ACTUAL_COEFFICIENT':np.broadcast_to(actual[:,None],(n,2048)),'FIXED_COEFFICIENT':np.broadcast_to(~actual[:,None],(n,2048))}
            if f['fixture']=='CHALLENGE':
                chosen=np.zeros((n,2048),bool);chosen[np.arange(n),[x['selected_output'] for x in f['rows']]]=True;scopes.update(SELECTED_CHALLENGE_COORDINATES=chosen,OTHER_CHALLENGE_OUTPUTS=~chosen)
            for scope,mask in scopes.items():
                if not mask.any():continue
                for method in ['D1','P']:
                    eqA=numeric_equal(out[method][:n],A[:n]);eqB=numeric_equal(out[method][:n],B[:n]);different=~numeric_equal(A[:n],B[:n]);fits.append({'layer':layer,'fixture':f['fixture'],'scope':scope,'method':method,'coordinates':int(mask.sum()),'discriminating_coordinates':int((different&mask).sum()),'A_only':int((eqA&~eqB&mask).sum()),'B_only':int((~eqA&eqB&mask).sum()),'both':int((eqA&eqB&mask).sum()),'neither':int((~eqA&~eqB&mask).sum()),'undefined':0})
            for kind,mask in [('D1_P_difference',gap),('A_B_discriminator',~numeric_equal(A[:n],B[:n]))]:
                for row,col in np.argwhere(mask)[:16]:examples.append({'layer':layer,'fixture':f['fixture'],'kind':kind,'row':int(row),'output_axis':int(col),'coefficient_origin':f['rows'][row]['origin'],'D1':float(decode(out['D1'][row,col])),'P':float(decode(out['P'][row,col])),'A':float(decode(A[row,col])),'B':float(decode(B[row,col])),'scope':'bounded first row-major examples; not independent workloads'})
    stable=all(x['same_bits'] for x in repeat)
    if control_bad:branch='A';status='CONTROL_MISMATCH'
    elif dense_bad:branch='B';status='IMPLEMENTATION_UNRESOLVED'
    elif not stable:branch='E';status='REPEAT_UNRESOLVED'
    elif actual_gap:branch='C';status='SINGLE_TERM_ACTUAL_DIFFERENCE_OBSERVED'
    else:branch='D';status='ACTUAL_SINGLE_TERM_NOT_OBSERVED'
    csvwrite(a.output/'single_term_metrics.csv',metrics);csvwrite(a.output/'candidate_fit.csv',fits);csvwrite(a.output/'single_term_examples.csv',examples);csvwrite(a.output/'single_term_repeats.csv',repeat)
    write(a.output/'B_decision.json',{'branch':branch,'status':status,'control_numeric_mismatch':control_bad,'D1_A_numeric_mismatch':dense_bad,'actual_coefficient_D1_P_mismatch':actual_gap,'repeat_comparisons':len(repeat),'all_repeat_hashes_equal':stable,'conditional_C':'RUN_FROZEN_GRID' if branch=='D' else 'NOT_RUN_NOT_NEEDED' if branch=='C' else 'NOT_RUN_BLOCKED','scope':'Same inputs repeated across processes, no independent model samples','output_zero_padding_not_in_denominator':True,'candidate_fit_totals_nonoverlapping_all_valid_P':{k:sum(x[k] for x in fits if x['scope']=='ALL_VALID' and x['method']=='P') for k in ['coordinates','discriminating_coordinates','A_only','B_only','both','neither','undefined']}})
    print(read(a.output/'B_decision.json'))
if __name__=='__main__':main()
