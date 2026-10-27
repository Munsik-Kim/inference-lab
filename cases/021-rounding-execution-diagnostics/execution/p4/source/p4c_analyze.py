"""CPU FP64 two-oracle analysis of saved first-block GPU outputs."""
import argparse
import importlib.util
import math
from pathlib import Path
import sys
import time
import numpy as np
from common import read,write,identity,csvwrite,array_hash
from numerics import bfdecode,bfbits32,bfround64,metrics

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    start=time.process_time();loc=read(a.root/'references/locations.json');case=Path(loc['original_case'])
    spec=importlib.util.spec_from_file_location('rpi4_schema_only',case/'source/layout.py');schema=importlib.util.module_from_spec(spec);sys.modules[spec.name]=schema;spec.loader.exec_module(schema)
    inputs=read(a.root/'inputs/projection_input_manifest.json');external=read(a.root/'references/external_inputs.json');rows=[];repeat=[];audit=[];contracts=[];kernels=[];counts={'fresh_process_blocks':0,'main_core_calls':0,'control_core_or_module_calls':0,'profile_calls':0,'full_model_forwards':0};max_scalar=0
    audit_rows=[0,1,255,511];audit_columns=[0,1,127,1023,2047]
    for layer in [0,25,27]:
        first=read(a.root/'records'/f'gpu-q{layer}-b0/data/operator_receipt.json');contract=first['contract'];contracts.append(contract);q=schema.load_rpi4(contract['projection']['path'])
        if identity(contract['projection']['path'])!={k:contract['projection'][k] for k in ['bytes','sha256']}:raise ValueError('ARTIFACT_CHANGED')
        O,I=q.shape;scale_bits=bfbits32(q.alpha_inv.astype(np.float32)*q.gamma.astype(np.float32))
        if array_hash(scale_bits.T.copy())!=contract['actual_unpermuted_scale_array_sha256'] or array_hash(q.codes)!=contract['codes_array_sha256']:raise ValueError('ACTUAL_CODE_SCALE_IDENTITY')
        wexact=(q.codes.astype(np.float64).reshape(O,I//64,64)*bfdecode(scale_bits).astype(np.float64)[...,None]).reshape(O,I)
        if not np.array_equal(wexact,wexact.astype(np.float32).astype(np.float64)):raise ValueError('FP32_PRODUCT_EXACTNESS')
        dense_bits=bfbits32(bfround64(wexact))
        if array_hash(dense_bits)!=contract['actual_D1_weight_bits_sha256']:raise ValueError('ACTUAL_D1_WEIGHT_IDENTITY')
        wdense=bfdecode(dense_bits).astype(np.float64)
        for block in [0,1,2]:
            rec=read(a.root/'records'/f'gpu-q{layer}-b{block}/data/operator_receipt.json')
            for key in counts:
                if key=='fresh_process_blocks':counts[key]+=1
                else:counts[key]+=rec[key]
            if rec['kernel_evidence']:kernels.append({'layer':layer,'block':block,**rec['kernel_evidence']})
            for row in rec['rows']:
                expected=inputs['method_orders'][f'{layer}:{block}:{row["fixture"]}']
                if row['method_order']!=expected:raise ValueError('METHOD_ORDER')
                if block:
                    firstrow=next(x for x in first['rows'] if x['fixture']==row['fixture'] and x['method']==row['method'])
                    equal=row['output_array_sha256']==firstrow['output_array_sha256']
                    if row['bit_identical_to_first']!=equal:raise ValueError('REPEAT_HASH_STATUS')
                    repeat.append({k:row[k] for k in ['layer','block','fixture','kind','method','output_array_sha256','first_block_output_sha256','bit_identical_to_first','different_elements','repeat_SSE','repeat_max_abs']})
        selected=[r for r in inputs['inputs'] if layer in r['layers']]
        for entry in selected:
            if identity(entry['path'])!=entry['identity']:raise ValueError('INPUT_IDENTITY')
            zbits=np.load(entry['path'],allow_pickle=False);z=bfdecode(zbits).astype(np.float64);values={};saved={}
            for method in ['D1','P']:
                row=next(x for x in first['rows'] if x['fixture']==entry['fixture'] and x['method']==method)
                if row['input_sha256']!=array_hash(zbits) or identity(row['output_path'])!=row['output_file']:raise ValueError('SAME_INPUT_OUTPUT_IDENTITY')
                bits=np.load(row['output_path'],allow_pickle=False)
                if array_hash(bits)!=row['output_array_sha256']:raise ValueError('OUTPUT_ARRAY_HASH')
                values[method]=bfdecode(bits).astype(np.float64);saved[method]=row
            # Memory bounded one fixture at a time, no 4D broadcast.
            exact=np.empty((512,O),np.float64);dense=np.empty_like(exact)
            for begin in range(0,512,64):
                exact[begin:begin+64]=z[begin:begin+64]@wexact.T
                dense[begin:begin+64]=z[begin:begin+64]@wdense.T
            for rowidx in audit_rows:
                for col in audit_columns:
                    se=math.fsum(float(z[rowidx,k])*float(wexact[col,k]) for k in range(I))
                    sd=math.fsum(float(z[rowidx,k])*float(wdense[col,k]) for k in range(I))
                    if not np.allclose([se,sd],[exact[rowidx,col],dense[rowidx,col]],rtol=1e-10,atol=1e-12):raise ValueError('SCALAR_DOT_ORACLE')
                    max_scalar=max(max_scalar,abs(se-exact[rowidx,col]),abs(sd-dense[rowidx,col]))
                    audit.append({'layer':layer,'fixture':entry['fixture'],'row':rowidx,'column':col,'exact_scalar':se,'exact_matmul':float(exact[rowidx,col]),'dense_scalar':sd,'dense_matmul':float(dense[rowidx,col]),'D1':float(values['D1'][rowidx,col]),'P':float(values['P'][rowidx,col])})
            n=entry['valid_rows']
            for name,ref,candidate in [('D1_vs_dense64',dense,values['D1']),('P_vs_dense64',dense,values['P']),('D1_vs_exact64',exact,values['D1']),('P_vs_exact64',exact,values['P']),('P_vs_D1',values['D1'],values['P'])]:
                result=metrics(ref[:n],candidate[:n]);diff=candidate[:n]-ref[:n]
                # Independent flatten/fsum aggregate checks, not a saved PASS flag.
                sse=math.fsum(np.square(diff).ravel());energy=math.fsum(np.square(ref[:n]).ravel())
                if not np.allclose([sse,energy],[result['SSE'],result['reference_energy']],rtol=1e-10,atol=1e-12):raise ValueError('METRIC_REDUCTION_AUDIT')
                rows.append({'layer':layer,'fixture':entry['fixture'],'kind':entry['kind'],'comparison':name,'M':512,'valid_rows':n,'zero_control_rows':512-n,'I':I,'O':O,'input_norm64':entry['norm64'],'input_min':entry['min'],'input_max':entry['max'],**result})
            if n<512:
                if np.any(values['D1'][n:]) or np.any(values['P'][n:]):raise ValueError('ZERO_CONTROL_OUTPUT')
            del exact,dense,z,values
        del wexact,wdense
    grouped=[]
    for layer in [0,25,27]:
        for kind in ['ONE_HOT_PLUS','ONE_HOT_MINUS','PAIR_WITHIN_GROUP','PAIR_ACROSS_GROUP','GROUP64_SIGNS','ACTUAL_DEV']:
            for comparison in ['D1_vs_dense64','P_vs_dense64','D1_vs_exact64','P_vs_exact64','P_vs_D1']:
                subset=[r for r in rows if r['layer']==layer and r['kind']==kind and r['comparison']==comparison];sse=math.fsum(r['SSE'] for r in subset);energy=math.fsum(r['reference_energy'] for r in subset)
                grouped.append({'layer':layer,'kind':kind,'comparison':comparison,'fixtures':len(subset),'valid_rows':sum(r['valid_rows'] for r in subset),'SSE':sse,'reference_energy':energy,'relative_RMSE':None if energy==0 else math.sqrt(sse/energy),'max_abs':max(r['max_abs'] for r in subset)})
    observations=[]
    for layer in [0,25,27]:
        own=[r for r in rows if r['layer']==layer];single=[r for r in own if r['comparison']=='P_vs_D1' and r['kind'].startswith('ONE_HOT')];multi=[r for r in own if r['comparison']=='P_vs_D1' and not r['kind'].startswith('ONE_HOT')]
        label='SINGLE_PRODUCT_DIFFERENCE_OBSERVED' if any(r['SSE']>0 for r in single) else 'MULTI_TERM_DIFFERENCE_OBSERVED' if any(r['SSE']>0 for r in multi) else 'NO_DIFFERENCE_OBSERVED_WITHIN_THIS_PROBE'
        closer=[];opposite=[];equal=[]
        for fixture in dict.fromkeys(r['fixture'] for r in own):
            d=next(r for r in own if r['fixture']==fixture and r['comparison']=='D1_vs_exact64');p=next(r for r in own if r['fixture']==fixture and r['comparison']=='P_vs_exact64')
            (closer if p['SSE']<d['SSE'] else opposite if p['SSE']>d['SSE'] else equal).append(fixture)
        observations.append({'layer':layer,'observation':label,'onehot_nonzero_gap_fixtures':sum(r['SSE']>0 for r in single),'onehot_fixtures':len(single),'multi_term_nonzero_gap_fixtures':sum(r['SSE']>0 for r in multi),'multi_term_fixtures':len(multi),'P_closer_to_exact64':closer,'D1_closer_to_exact64':opposite,'equal_exact64_error':equal})
    csvwrite(a.output/'operator_metrics.csv',rows);csvwrite(a.output/'operator_grouped.csv',grouped);csvwrite(a.output/'operator_repeats.csv',repeat);csvwrite(a.output/'oracle_audit_subset.csv',audit)
    np.savez(a.output/'oracle_audit_subset.npz',layer=np.array([r['layer'] for r in audit]),row=np.array([r['row'] for r in audit]),column=np.array([r['column'] for r in audit]),exact_scalar=np.array([r['exact_scalar'] for r in audit]),dense_scalar=np.array([r['dense_scalar'] for r in audit]),D1=np.array([r['D1'] for r in audit]),P=np.array([r['P'] for r in audit]))
    write(a.output/'projection_contracts.json',{'contracts':contracts,'weight_matrices_stored':False,'oracles':'Full FP64 matrices ephemeral; actual D1 identity checked from all weight bits before GPU calls'})
    write(a.output/'kernel_evidence.json',{'probes':kernels})
    write(a.output/'p4c_facts.json',{'scope':'POSTHOC three projections and fixed input families only','observations':observations,'counts':counts,'valid_probe_rows_per_block_per_projection':7680,'independent_model_samples':0,'repeat_equal_rows':sum(r['bit_identical_to_first'] for r in repeat),'repeat_rows':len(repeat)})
    write(a.output/'p4c_verification.json',{'status':'COMPLETE','oracle_scalar_cells':len(audit),'max_scalar_matmul_absolute_difference':max_scalar,'scalar_atol':1e-12,'scalar_rtol':1e-10,'metric_fsum_audited':len(rows),'two_oracles_retained':True,'finite_layout_and_identity_checked':True,'projection_difference_threshold':'NONE; historical full-logit0.02 not used','CPU_process_seconds':time.process_time()-start,'CUDA_context':'NOT_INITIALIZED','shared_code':'Serialization/metadata and generic scalar metric helper; independent math.fsum dot and reduction paths validate oracle/output metrics'})
    print(str({'status':'COMPLETE','counts':counts,'observations':observations,'max_scalar':max_scalar}))

if __name__=='__main__':main()
