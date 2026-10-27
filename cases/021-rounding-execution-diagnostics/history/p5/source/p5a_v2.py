"""Reaggregate P4 energies and BF16 steps, without new GPU/model calls."""
import argparse,csv,math
from pathlib import Path
import numpy as np
from common import read,write,identity,csvwrite
from arithmetic_v2 import decode,ulp_metrics,steps,neighbors
def table(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);r=a.root;p4=Path(read(r/'references/locations.json')['p4_root']);proto=read(r/'P5_PROTOCOL.json');src=table(p4/'outputs/p4b_v1/rounding_cancellation_tensors.csv');components=[];maxres=0
    for split,doc,layer,kind in dict.fromkeys((x['analysis_split'],x['document_id'],x['layer'],x['tensor']) for x in src):
        rs={x['rounding']:x for x in src if (x['analysis_split'],x['document_id'],x['layer'],x['tensor'])==(split,doc,layer,kind)};x,y=rs['R0'],rs['R1'];row={'split':split,'document_id':doc,'layer':int(layer),'tensor':kind,'heads':int(x['heads']),'positions':int(x['positions']),'elements':int(x['elements'])}
        assert x['elements']==y['elements'] and x['norm_weight_sha256']==y['norm_weight_sha256'] and x['epsilon']==y['epsilon']
        d=float(x['d_energy']);assert d==float(y['d_energy'])
        for mode,v in rs.items():
            g,n,c=[float(v[k]) for k in ['pair_gap_SSE','n_energy','signed_cross']];scale=max(abs(g),abs(n),abs(c),abs(d));res=g-d-n-c;assert abs(res)<=1e-10*scale if scale else res==0;maxres=max(maxres,abs(res));row.update({mode+'_G':g,mode+'_N':n,mode+'_C':c})
        dg=row['R1_G']-row['R0_G'];dn=row['R1_N']-row['R0_N'];dc=row['R1_C']-row['R0_C'];scale=max(abs(v) for k,v in row.items() if k.startswith('R'));tau=max(proto['sign_tolerance']['atol'],proto['sign_tolerance']['rtol']*scale);res=dg-dn-dc;assert abs(res)<=1e-10*max(scale,abs(dg),abs(dn),abs(dc)) if scale else res==0
        if dg<=tau:group='NOT_INCREASE_OR_NEAR_TOLERANCE'
        elif dn<=-tau and dc>tau:group='DN_NONPOS_DC_POS'
        elif dn>tau and dc>tau:group='DN_POS_DC_POS'
        elif dn>tau and dc<=-tau:group='DN_POS_DC_NONPOS'
        else:group='NEAR_TOLERANCE_OR_OTHER'
        row.update(d_energy=d,delta_G=dg,delta_N=dn,delta_C=dc,delta_identity_residual=res,sign_tolerance=tau,classification=group,P4_exact_sign_gap_increase=dg>0,P4_negative_cross_weakening=row['R0_C']<0 and row['R0_C']<row['R1_C']);components.append(row)
    groups=[];documents=[]
    for split in ['original_DEV_posthoc','original_EVAL_posthoc']:
        for group in ['DN_NONPOS_DC_POS','DN_POS_DC_POS','DN_POS_DC_NONPOS','NEAR_TOLERANCE_OR_OTHER','NOT_INCREASE_OR_NEAR_TOLERANCE']:
            own=[x for x in components if x['split']==split and x['classification']==group];groups.append({'split':split,'classification':group,'cells':len(own),'sum_delta_G':math.fsum(x['delta_G'] for x in own),'sum_delta_N':math.fsum(x['delta_N'] for x in own),'sum_delta_C':math.fsum(x['delta_C'] for x in own),'P4_negative_cross_weakening_count':sum(x['P4_negative_cross_weakening'] for x in own)})
        for doc in dict.fromkeys(x['document_id'] for x in components if x['split']==split):
            own=[x for x in components if x['document_id']==doc];documents.append({'split':split,'document_id':doc,'tensor_cells':len(own),'elements_per_path':sum(x['elements'] for x in own),**{k:math.fsum(x[k] for x in own) for k in ['delta_G','delta_N','delta_C']}})
    documents_ulp=[];hist=[];coordinates=[];inputs=read(p4/'inputs/projection_input_manifest.json')['inputs']
    for layer in [0,25,27]:
        rec=read(p4/f'records/gpu-q{layer}-b0/data/operator_receipt.json')
        for entry in [x for x in inputs if x['kind']=='ACTUAL_DEV' and layer in x['layers']]:
            arrays={}
            for method in ['D1','P']:
                item=next(x for x in rec['rows'] if x['fixture']==entry['fixture'] and x['method']==method);assert identity(item['output_path'])==item['output_file'];arrays[method]=np.load(item['output_path'],allow_pickle=False)
            x,y=arrays['D1'],arrays['P'];met=ulp_metrics(x,y);documents_ulp.append({'layer':layer,'document_id':entry['document_id'],'original_split':'DEV','M':512,'O':2048,**met})
            dd=steps(x,y)
            for value,count in zip(*np.unique(dd,return_counts=True)):hist.append({'layer':layer,'document_id':entry['document_id'],'representable_steps':int(value),'elements':int(count)})
            indices=np.argwhere(dd>0)[:proto['p5a_coordinate_limit_per_document']]
            for row,col in indices:
                xb,yb=int(x[row,col]),int(y[row,col]);xn,yn=neighbors(xb),neighbors(yb)
                coordinates.append({'layer':layer,'document_id':entry['document_id'],'row':int(row),'column':int(col),'D1_bits':xb,'P_bits':yb,'D1':float(decode(x[row,col])),'P':float(decode(y[row,col])),'steps':int(dd[row,col]),'D1_prev':xn[0],'D1_next':xn[1],'P_prev':yn[0],'P_next':yn[1],'selection':'first mismatch coordinates in row-major order, bounded; not a random workload sample'})
    csvwrite(a.output/'local_gap_change_components.csv',components);csvwrite(a.output/'local_gap_change_groups.csv',groups);csvwrite(a.output/'local_gap_documents.csv',documents);csvwrite(a.output/'projection_ulp_documents.csv',documents_ulp);csvwrite(a.output/'projection_ulp_histogram.csv',hist);csvwrite(a.output/'projection_ulp_coordinates.csv',coordinates)
    facts={'scope':'Existing P4 posthoc reaggregation','local_tensor_pairs':len(components),'local_documents':len(documents),'component_groups':groups,'ULP_documents':len(documents_ulp),'ULP_elements':sum(x['elements'] for x in documents_ulp),'ULP_numeric_mismatches':sum(x['numeric_mismatch'] for x in documents_ulp),'ULP_step1':sum(x['steps1'] for x in documents_ulp),'ULP_step2':sum(x['steps2'] for x in documents_ulp),'ULP_step_gt2':sum(x['steps_gt2'] for x in documents_ulp),'ULP_max':max(x['max_steps'] for x in documents_ulp)}
    write(a.output/'p5a_facts.json',facts);write(a.output/'p5a_verification.json',{'energy_identity_relative_tolerance':1e-10,'max_absolute_energy_residual':maxres,'source_tensor_rows':len(src),'numeric_zero_collapsed':True,'nonfinite':'rejected','coordinate_subset_rows':len(coordinates),'FP64_midpoint_diagnosis':'NOT_RUN in P5A; no kernel-cause inference from ULP steps','P4_saved_scalar_oracle_coordinates':{'rows':[0,1,255,511],'columns':[0,1,127,1023,2047],'scope':'Existing subset can be cross-referenced; P5A does not recompute projection oracles'},'new_GPU_calls':0});print(facts)
if __name__=='__main__':main()
