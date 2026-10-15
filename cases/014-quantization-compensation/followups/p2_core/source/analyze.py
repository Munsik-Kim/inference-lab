"""CPU saved-record P2 summaries and paired document bootstrap. No model imports."""
import argparse,csv,json,math,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from p2_common import save,sha,jsonl,utc,METHODS

def writecsv(path,rows):
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run);start=time.monotonic()
    rows=[r for p in sorted((run/'metrics').glob('*.jsonl')) for r in jsonl(p)];inp=json.loads((run/'input_manifest.json').read_text());model=json.loads((run/'model_manifest.json').read_text());frozen=json.loads((run/'artifact_frozen.json').read_text())
    manifest={d['document_id_hash']:d for d in inp['documents'] if d['split'] in ['EVAL-ID','SHIFT']};groups=defaultdict(list);lookup={}
    for r in rows:
        key=r['method'],r['split'],r['document_id_hash']
        if key in lookup:raise ValueError('duplicate evaluation row')
        if r['status']!='COMPLETE' or r['valid_token_count']!=255 or r['KL_position_count']!=16 or r['token_hash']!=manifest[r['document_id_hash']]['token_hash']:raise ValueError('invalid/mismatched row')
        lookup[key]=r;groups[key[:2]].append(r)
    if len(rows)!=4*len(manifest):raise ValueError('missing method/document evaluation')
    summaries=[];paired=[];differences=[]
    for split in ['EVAL-ID','SHIFT']:
        ids=sorted(d for d,v in manifest.items() if v['split']==split);N=len(ids)
        if N==0:continue
        ix=np.random.default_rng(6101523).integers(0,N,size=(10000,N));nll={}
        for method in METHODS:
            rr=[lookup[method,split,d] for d in ids];nll[method]=np.array([r['NLL_sum']/r['valid_token_count'] for r in rr])
        gap=float(nll['RTN'].mean()-nll['BF16'].mean())
        for method in METHODS:
            rr=[lookup[method,split,d] for d in ids];loss=math.fsum(r['NLL_sum'] for r in rr);valid=sum(r['valid_token_count'] for r in rr);mean=loss/valid
            diff=nll[method]-nll['RTN'];delta=float(diff.mean());denom_ok=gap>1e-6
            summaries.append({'method':method,'split':split,'documents':N,'NLL_sum':loss,'valid_token_count':valid,'NLL_token_weighted':mean,'NLL_document_mean':float(nll[method].mean()),'PPL':math.exp(mean),
              'NLL_vs_BF16':float(nll[method].mean()-nll['BF16'].mean()),'NLL_vs_RTN':delta,'PPL_relative_change_vs_RTN':math.expm1(mean-float(nll['RTN'].mean())),
              'RTN_BF16_NLL_gap':gap,'recovered_RTN_loss_fraction':float((nll['RTN'].mean()-nll[method].mean())/gap) if denom_ok else None,'recovery_undefined_reason':None if denom_ok else 'RTN minus BF16 NLL gap <=1e-6',
              'mean_full_vocab_KL':math.fsum(r['sampled_full_vocab_KL_sum'] for r in rr)/sum(r['KL_position_count'] for r in rr),'argmax_flip_rate':sum(r['flip_count'] for r in rr)/valid,
              'lower_NLL_documents_vs_RTN':int((diff<0).sum()),'higher_NLL_documents_vs_RTN':int((diff>0).sum()),'exact_tie_documents_vs_RTN':int((diff==0).sum()),'median_document_delta_NLL_vs_RTN':float(np.median(diff)),
              'projection_core_bytes':rr[0]['projection_core_bytes'],'core_accounting':'BF16 original tensor payload' if method=='BF16' else 'actual INT4 files incl header/scales/padding'})
        comparisons=[('COMP_QUARTER','RTN'),('COMP_HALF','RTN')]+[(m,'BF16') for m in ['RTN','COMP_QUARTER','COMP_HALF']]
        for candidate,baseline in comparisons:
            diff=nll[candidate]-nll[baseline];draws=diff[ix].mean(axis=1);ci=np.quantile(draws,[.025,.975]);point=float(diff.mean());verdict='LOWER_NLL' if ci[1]<0 else 'HIGHER_NLL' if ci[0]>0 else 'INCONCLUSIVE'
            paired.append({'split':split,'candidate':candidate,'baseline':baseline,'documents':N,'mean_document_delta_NLL':point,'CI95':ci.tolist(),'PPL_relative_change':math.expm1(point),'PPL_relative_change_CI95':[math.expm1(float(v)) for v in ci],
               'verdict':verdict,'primary':split=='EVAL-ID' and candidate=='COMP_QUARTER' and baseline=='RTN','lower_documents':int((diff<0).sum()),'higher_documents':int((diff>0).sum()),'exact_ties':int((diff==0).sum()),'median':float(np.median(diff)),
               'project_reference_PPL_1pct_reduction_reached':math.expm1(point)<=-.01,'scope':'paired documents; auxiliary pointwise except registered primary; fixed model/CAL and methods'})
            for d,v in zip(ids,diff):differences.append({'split':split,'document_id_hash':d,'candidate':candidate,'baseline':baseline,'delta_NLL':float(v)})
    artifacts=jsonl(run/'artifacts.jsonl');ledger=jsonl(run/'build_ledger.jsonl');complete=json.loads((run/'completion.json').read_text());worker=json.loads((run/'worker_receipt.json').read_text())
    core={m:sum(r['artifact_bytes'] for r in artifacts if r['method']==m) for m in ['RTN','COMP_QUARTER','COMP_HALF']}
    assert len(artifacts)==588 and all(v==247733136 for v in core.values())
    calls=jsonl(run/'forward_calls.jsonl');reload=json.loads((run/'fresh_process_reload.json').read_text());probe=json.loads((run/'smoke_matrix_verification.json').read_text())
    resources={'worker_wall_seconds_base':worker['cumulative_worker_wall_seconds'],'supplemental_smoke_worker_seconds':json.loads((run/'smoke_matrix_receipt.json').read_text())['wall_seconds'],
       'all_full_model_forwards':len(calls)+reload['full_model_forwards']+probe['full_model_forwards'],'main_full_model_forwards':len(calls),'fresh_process_reload_forwards':reload['full_model_forwards'],'supplemental_SMOKE_forwards':probe['full_model_forwards'],
       'stage_seconds_inclusive':complete['stage_seconds_inclusive'],'metric_seconds_inside_stages':complete['metric_seconds_inside_build_evaluation'],
       'fit_factorization_seconds':math.fsum(r['factor_diagnostics']['factorization_seconds'] for r in ledger),'fit_projection_solve_seconds':math.fsum(r['fit']['solve_seconds'] for r in jsonl(run/'CAL_diagnostics.jsonl')),
       'convert_pack_independent_reload_seconds':math.fsum(r['convert_pack_reload_seconds'] for r in artifacts),'evaluation_reload_seconds':math.fsum(r['reload_seconds'] for r in jsonl(run/'evaluation_reload.jsonl')),
       'peak_Torch_allocated_bytes':max([r['peak_allocated'] for r in calls]+[r['peak_allocated'] for r in reload['checks']]+[r['peak_allocated'] for r in probe['rows']]),
       'peak_Torch_reserved_bytes':max([r['peak_reserved'] for r in calls]+[r['peak_reserved'] for r in reload['checks']]+[r['peak_reserved'] for r in probe['rows']]),
       'max_observed_post_forward_device_used_bytes':max(r['device_total_bytes']-r['device_free_bytes'] for r in calls),'device_sample_scope':'post-forward samples, not continuous device/Windows peak; background jobs untouched',
       'projection_core_bytes':core,'BF16_projection_tensor_bytes':model['original_projection_BF16_tensor_bytes'],'common_unchanged_parameter_bytes':model['unchanged_common_parameter_bytes_unique'],
       'GPU_busy_TTFT_TPOT_serving_throughput':'NOT_MEASURED','runtime':'stored INT4 decoded to dense BF16; offline capture/fit costs are not serving speed'}
    resources['cumulative_worker_wall_seconds']=resources['worker_wall_seconds_base']+resources['supplemental_smoke_worker_seconds'];assert resources['cumulative_worker_wall_seconds']<=10800
    writecsv(run/'method_summary.csv',summaries);writecsv(run/'paired_comparisons.csv',paired);writecsv(run/'paired_document_differences.csv',differences)
    result={'status':'ANALYZED','created_utc':utc(),'raw_rows':len(rows),'method_summaries':summaries,'paired_comparisons':paired,'resources':resources,'bootstrap_seed':6101523,'bootstrap_resamples':10000,
       'confidence_scope':'conditional on existing model/original CAL32 and fixed two strengths; paired document sample uncertainty, no calibration/model-population guarantee','analysis_source_sha256':sha(__file__),'analysis_seconds':time.monotonic()-start}
    save(run/'summary.json',result);save(run/'resource_summary.json',resources)
    primary=next(r for r in paired if r['primary']);shift=[r for r in paired if r['split']=='SHIFT' and r['baseline']=='RTN']
    save(run/'decision.json',{'status':'COMPLETED_P2_CORE','primary':primary,'auxiliary_SHIFT':shift,'methods':['BF16','RTN','COMP_QUARTER','COMP_HALF'],'core_files_per_quantized_method':196,'K_fits_per_comp_method':196,'factorizations_per_comp_method':112,
        'same_INT4_core_bytes':True,'new_selector_or_alpha_search':False,'P0_P1_P1b':json.loads((run/'preservation.json').read_text())['status'] if (run/'preservation.json').exists() else 'NOT_CHECKED_IN_THIS_REANALYSIS','packed_serving_benefit':'NOT_ESTABLISHED','P3_4B_agent':'NOT_RUN_SCOPE','remote_publication':False})
    print(json.dumps({'primary':primary,'resources':resources},indent=2))
if __name__=='__main__':main()
