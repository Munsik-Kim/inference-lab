"""Conditional nested replay: FP64 matrices, exact dyadic subset and ambiguity."""
import argparse,math
from fractions import Fraction
from pathlib import Path
import numpy as np
from common import read,write,identity,csvwrite,array_hash
from arithmetic_v2 import decode,round_bits,rank,numeric_equal,metrics,ulp_metrics,rational_bits,nested,neighbors
from prepare_v2 import load_artifact
def unrank(k):
    k=np.asarray(k,np.int64);return np.where(k<32768,(32768-k)|0x8000,k-32768).astype(np.uint16)
def oracle(z,w):
    y=np.empty((512,2048),np.float64);bound=np.empty_like(y);eps=np.finfo(float).eps;gamma=1024*eps/(1-1024*eps)
    for row in range(0,512,64):
        y[row:row+64]=z[row:row+64]@w.T;bound[row:row+64]=gamma*(np.abs(z[row:row+64])@np.abs(w).T)+4*eps*np.abs(y[row:row+64])
    bits=round_bits(y);k=rank(bits);v=decode(bits).astype(float);low=(decode(unrank(k-1)).astype(float)+v)/2;high=(decode(unrank(k+1)).astype(float)+v)/2;margin=np.minimum(np.abs(y-low),np.abs(high-y));return y,bits,margin>bound,bound
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;a.output.mkdir(parents=True,exist_ok=False);data=read(r/'inputs_manifest/prepared.json');assert read(r/'summary/B_decision.json')['branch']=='D';p4=Path(read(r/'references/locations.json')['p4_root']);metrics_rows=[];rounded=[];audits=[];repeats=[];cross_old=[];inputstats=[]
    for layer in [0,25]:
        q,s,_,prior=load_artifact(r,layer);wexact=(q.codes.astype(float).reshape(2048,16,64)*s[...,None]).reshape(2048,1024);wdense=decode(round_bits(wexact)).astype(float)
        for f in [x for x in data['fixtures'] if x['phase']=='C' and x['layer']==layer]:
            assert identity(f['path'])==f['identity'];zb=np.load(f['path'],allow_pickle=False);assert array_hash(zb)==f['array_sha256'];original=np.load(f['original_Z_path'],allow_pickle=False);expected=next(x for x in nested(original,layer,f['document_index']) if x[0]==f['K']);assert np.array_equal(zb,expected[1]) and array_hash(expected[2])==f['mask_axes_sha256'];z=decode(zb).astype(float);output={}
            for method in ['D1','P']:
                for block in [0,1,2]:
                    rec=read(r/f'records/gpu-C-l{layer}-b{block}/data/receipt.json');item=next(x for x in rec['rows'] if x['fixture']==f['fixture'] and x['method']==method);assert item['input_sha256']==array_hash(zb);assert item['method_order']==data['method_orders'][f"C:{layer}:{block}:{f['fixture']}"]
                    if block==0:assert identity(item['output_path'])==item['output_identity'];output[method]=np.load(item['output_path'],allow_pickle=False);assert array_hash(output[method])==item['output_sha256']
                    else:repeats.append({'layer':layer,'document_id':f['document_id'],'K':f['K'],'method':method,'block':block,'bit_equal':item['output_sha256']==array_hash(output[method])})
                if f['K']==1024:
                    old=read(p4/f'records/gpu-q{layer}-b0/data/operator_receipt.json');item=next(x for x in old['rows'] if x['fixture']=='ACTUAL_DEV_'+f['document_id'] and x['method']==method);arr=np.load(item['output_path'],allow_pickle=False);assert identity(item['output_path'])==item['output_file'];cross_old.append({'layer':layer,'document_id':f['document_id'],'method':method,'same_input_hash':array_hash(zb)==item['input_sha256'],'same_output_bits':np.array_equal(arr,output[method]),'scope':'Across P4/P5 time and process; not one-worker controlled comparison'})
            values={k:decode(v).astype(float) for k,v in output.items()};common={'layer':layer,'document_id':f['document_id'],'K':f['K'],'actual_nonzero':int(np.count_nonzero(z)),'norm64':f['norm64'],'input_min':f['min'],'input_max':f['max'],'nonzero_exponent_min':f['nonzero_exponent_min'],'nonzero_exponent_max':f['nonzero_exponent_max']};inputstats.append(common)
            metrics_rows.append({**common,'comparison':'P_vs_D1',**ulp_metrics(output['D1'],output['P'])})
            for name,w in [('dense64',wdense),('exact64',wexact)]:
                y,pred,known,bound=oracle(z,w);ambiguous_before=int((~known).sum());known_bits=pred.copy();rng=np.random.default_rng(np.random.SeedSequence([61027028,layer,f['document_index'],f['K']]))
                coords=[(int(i),int(j),'PRESELECTED') for i,j in zip(rng.integers(0,512,8),rng.integers(0,2048,8))];pool=(~numeric_equal(output['D1'],output['P']))|~known
                coords += [(int(i),int(j),'POSTHOC_FIRST_MISMATCH_OR_UNCERTAIN') for i,j in np.argwhere(pool)[:8]]
                for i,j,selection in coords:
                    dot=sum((Fraction.from_float(float(z[i,k]))*Fraction.from_float(float(w[j,k])) for k in np.flatnonzero(z[i])),Fraction(0));exactbit=rational_bits(dot);error=float(Fraction.from_float(float(y[i,j]))-dot);assert abs(error)<=1e-12+1e-10*abs(float(dot));beforebit=int(pred[i,j]);known[i,j]=True;known_bits[i,j]=exactbit
                    center=float(decode(np.array(exactbit,np.uint16)));prev,nxt=neighbors(exactbit);mids=[(Fraction.from_float(center)+Fraction.from_float(v))/2 for v in [prev,nxt] if v is not None];tie=any(dot==m for m in mids)
                    audits.append({**common,'oracle':name,'row':i,'column':j,'selection':selection,'FP64_value':float(y[i,j]),'exact_dyadic_as_FP64':float(dot),'FP64_absolute_error':abs(error),'error_bound':float(bound[i,j]),'FP64_round_bits':beforebit,'exact_round_bits':exactbit,'exact_midpoint_tie':tie,'round_numeric_equal':bool(numeric_equal(np.array(beforebit,np.uint16),np.array(exactbit,np.uint16)))})
                for method in ['D1','P']:
                    met=metrics(y,values[method]);# independent scalar reduction across all coordinates
                    delta=values[method]-y;ss=math.fsum(np.square(delta).ravel());energy=math.fsum(np.square(y).ravel());assert math.isclose(ss,met['SSE'],rel_tol=1e-10,abs_tol=1e-12) and math.isclose(energy,met['energy'],rel_tol=1e-10,abs_tol=1e-12)
                    metrics_rows.append({**common,'comparison':method+'_vs_'+name,**met})
                    match=numeric_equal(output[method],known_bits);rounded.append({**common,'oracle':name,'method':method,'scope':'BF16 nearest-even candidate, FP64 error-bound-qualified or exact-audited subset','coordinates':known.size,'known':int(known.sum()),'UNRESOLVED':int((~known).sum()),'matches_known':int((match&known).sum()),'mismatch_known':int((~match&known).sum()),'FP64_unqualified_before_subset':ambiguous_before,'raw_FP64_cast_mismatch':int((~numeric_equal(output[method],pred)).sum())})
                del y,pred,known,bound
    csvwrite(a.output/'sparse_metrics.csv',metrics_rows);csvwrite(a.output/'sparse_candidate_fit.csv',rounded);csvwrite(a.output/'sparse_exact_audit.csv',audits);csvwrite(a.output/'sparse_repeats.csv',repeats);csvwrite(a.output/'sparse_P4_comparison.csv',cross_old);csvwrite(a.output/'sparse_input_stats.csv',inputstats)
    gaps=[x for x in metrics_rows if x['comparison']=='P_vs_D1'];first=[]
    for layer in [0,25]:
        for doc in data['DEV_documents_sorted_first_two']:
            own=[x for x in gaps if x['layer']==layer and x['document_id']==doc];first.append({'layer':layer,'document_id':doc,'first_observed_K':next((x['K'] for x in own if x['numeric_mismatch']>0),None),'conditions':[{'K':x['K'],'numeric_mismatch':x['numeric_mismatch'],'max_steps':x['max_steps'],'norm64':x['norm64']} for x in own]})
    write(a.output/'C_facts.json',{'first_observation':first,'actual_input_conditions':len(gaps),'repeats':len(repeats),'all_repeat_bits_equal':all(x['bit_equal'] for x in repeats),'P4_K1024_comparisons':cross_old,'oracle_exact_audit_rows':len(audits),'exact_midpoint_ties':sum(x['exact_midpoint_tie'] for x in audits),'FP64_round_disagreements_on_audit':sum(not x['round_numeric_equal'] for x in audits),'remaining_UNRESOLVED_per_oracle_condition':sum(x['UNRESOLVED'] for x in rounded[::2]),'scope':'Fixed nested masks; K and input norm vary together. No universal threshold or internal rounding-order proof.'});print(read(a.output/'C_facts.json'))
if __name__=='__main__':main()
