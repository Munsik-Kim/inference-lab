"""Create review metadata from receipts. No model evaluation or artifact writes."""
import argparse
import json
import re
import shutil
from pathlib import Path
from collections import Counter
import numpy as np
from p1b_common import save,sha,utc,jsonl

def main():
    ap=argparse.ArgumentParser()
    for k in ['case','run','private','first-private','old-inputs']:ap.add_argument('--'+k,required=True)
    a=ap.parse_args();case=Path(a.case).resolve();run=Path(a.run);pr=Path(a.private);first=Path(a.first_private)
    summary=json.loads((run/'summary.json').read_text());complete=json.loads((run/'completion.json').read_text());receipt=json.loads((run/'worker_receipt.json').read_text())
    data=json.loads((run/'input_manifest.json').read_text());new=json.loads((pr/'inputs/documents.private.json').read_text());old=json.loads(Path(a.old_inputs).read_text());actual=[r for r in new if r['split']!='REUSED_SMOKE']
    def canon(d):
        if 'PG' not in d['source']:return d['source_id']
        v=d['source_id'];m=re.search(r'/ebooks/(\d+)',v);return 'gutenberg-ebook-'+str(int(m.group(1))) if m else v
    assert len({canon(r) for r in actual})==112 and not ({canon(r) for r in actual}&{canon(r) for r in old})
    assert not ({r['content_sha256'] for r in actual}&{r['content_sha256'] for r in old})
    assert not ({r['token_hash'] for r in actual}&{r['token_hash'] for r in old})
    for r in actual:
        assert len(r['input_ids'])==256
        assert __import__('hashlib').sha256(np.array(r['input_ids'],dtype='<i8').tobytes()).hexdigest()==r['token_hash']
    assert sha(pr/'inputs/documents.private.json')==data['private_tokens_sha256']
    save(run/'input_independent_validation.json',{'status':'PASS','recorded_utc':utc(),'fresh_documents':112,'prior_documents_excluded':97,'canonical_ID_content_token_hash_disjoint':True,'private_token_hashes_recomputed':112,
        'counts':dict(Counter(r['split'] for r in actual)),'old_new_5gram_check':data['near_duplicate_check'],'source_scope':'confirmed Case014 P0/P1 manifests + bounded current repo cases/research/configs search; other history and pretraining UNKNOWN'})
    # Preserve failed source bytes and redact only copies of logs, never the original logs.
    archive=case/'followups/p1b_selection/attempts/smoke01_source';archive.mkdir(parents=True,exist_ok=False)
    for p in (first/'failed_source_01').glob('*.py'):shutil.copyfile(p,archive/p.name)
    def redact(text):
        identity=json.loads((pr/'identity.private.json').read_text())
        replacements=[(str(case),'${CASE}'),(identity['old_private'],'${P1_PRIVATE}'),(str(first),'${P1B_FIRST_PRIVATE}'),(str(pr),'${P1B_PRIVATE}')]
        for src,dest in replacements:text=text.replace(src,dest)
        return re.sub(r'/home/[^/\s]+/', '${HOME}/', text)
    logdir=run/'receipts';logdir.mkdir(exist_ok=False)
    logs=[first/'attempts/data-01.log',first/'attempts/old-15-tests-rerun.log',first/'attempts/p1b-tests-01.log',first/'attempts/p1b-tests-02.log',first/'attempts/worker-01.log',first/'attempts/original-p1-audit.log',pr/'attempts/p1b-tests-03.log',pr/'attempts/worker-01.log',pr/'attempts/analyze-01.log',pr/'attempts/audit-01.log',pr/'attempts/private-array-audit.log',pr/'attempts/plot-01.log',pr/'attempts/preservation.log']
    exports=[]
    for i,p in enumerate(logs):
        dest=logdir/f'{i:02d}_{p.name}'
        dest.write_text(redact(p.read_text()))
        exports.append({'label':f'ATTEMPT_LOG_{i:02d}','external_raw_sha256':sha(p),'export':str(dest.relative_to(run)),'export_sha256':sha(dest),'rule':'literal absolute path to semantic label mapping; source log unchanged'})
    save(run/'log_export_manifest.json',{'files':exports,'no_model_text_or_credentials':True})
    first_out=case/'results/p1b_r1'
    save(first_out/'decision.json',{'status':'FAILED_SMOKE_HISTORICAL_ARRAY_MISSING','exit_code':1,'full_model_forwards':len(jsonl(first_out/'forward_calls.jsonl')),
         'SELECT_EVAL_SHIFT_forwards':0,'worker_wall_seconds':receipt['previous_failed_worker_seconds'],'reason':'P1 SMOKE prefix Z array was not saved; new worker originally assumed its presence. Historical teacher logits matched exactly. No quantizer/model/data changes.',
         'resolution':'fresh r2 worker keeps teacher historical parity, checks prefix hashes/current repeat/common Z; historical prefix-array comparison is null/NOT_SAVED. Source snapshots and failure preserved.'})
    oldaudit=json.loads((first/'original_p1_audit.json').read_text());save(run/'original_p1_reaudit.json',oldaudit)
    prim=next(m for m in summary['macro_contrasts'] if m['primary'])
    decision={'status':'COMPLETED_P1B','run_id':run.name,'candidate_rows':3360,'target_alpha_split_cells':90,'failed_core_cells':0,'missing_core_cells':0,
        'core_targets':6,'alpha_grid':[0,.25,.5,.75,1],'selection_frozen_before_EVAL':True,
        'A_OLD_FIT_LOCAL_TRANSFER':{'verdict':'LOWER_FRESH_ID_LOCAL_SSE','D_macro':prim['D_macro'],'CI95':prim['D_macro_CI95'],'unit':'difference / alpha0 SSE; six fixed independent interventions'},
        'B_ADDITIONAL_VS_SIMPLE_FIXED':'ID local SSE lower than fixed .25/.5; NLL gain over .25 inconclusive, over .5 auxiliary pointwise CI below zero.',
        'C_SSE_VS_NLL':'Selections differ for 5/6 targets; ID NLL-selector minus SSE-selector NLL CI includes zero; NLL selection can worsen local SSE.',
        'D_SHIFT_COST_BYTES':'Both selectors worsen SHIFT SSE and NLL vs alpha0; selected core bytes unchanged; SELECT phase shares full forward evaluation cost.',
        'P2':'NOT_RUN_SCOPE','P2_RECOMMENDATION':'Do not adopt these six independent alpha choices as a full-model default. Any P2 requires separate approval and own-trajectory/fresh-evaluation registration.',
        'P3':'NOT_RUN_SCOPE','FORMAL_SERVING_BENEFIT':'NOT_ESTABLISHED','REMOTE_PUBLICATION':False,'backbone_training_or_fit_or_requantize':False}
    save(run/'decision.json',decision)
    calls=jsonl(run/'forward_calls.jsonl');dev=max(c['device_total_bytes']-c['device_free_bytes'] for c in calls)
    save(run/'resource_summary.json',{'worker_wall_seconds_success':receipt['worker_wall_seconds'],'failed_worker_wall_seconds':receipt['previous_failed_worker_seconds'],'cumulative_worker_wall_seconds':receipt['cumulative_worker_seconds'],
        'successful_full_model_forwards':3813,'failed_attempt_full_model_forwards':3,'all_attempt_full_model_forwards':3816,'candidate_rows':3360,'policy_alias_rows':4704,
        'phase_forward_counts':complete['full_model_forward_counts'],'data_prepare_seconds':data['data_prepare_seconds'],'validation_seconds':json.loads((run/'reused_artifact_manifest.json').read_text())['validation_seconds'],
        'stages_inclusive_seconds':complete['stage_seconds_inclusive'],'metric_seconds_already_inside_stages':complete['metric_seconds_within_stage_totals'],
        'audit_seconds':json.loads((run/'independent_audit.json').read_text())['wall_seconds'],'analysis_seconds':summary['analysis_seconds'],
        'peak_Torch_allocated_bytes':max(c['peak_allocated'] for c in calls),'peak_Torch_reserved_bytes':max(c['peak_reserved'] for c in calls),'max_observed_post_forward_device_used_bytes':dev,
        'device_memory_scope':'post-forward CUDA mem_get_info samples include other GPU use; not a continuous Windows/device peak; device background processes untouched',
        'GPU_busy_TTFT_TPOT_serving_throughput':'NOT_MEASURED','new_dataset_transfer_bytes':0,
        'selection_cost':'SELECT teacher32 + prefix96 + candidate960 =1088 full-model forwards; SSE and NLL selectors share the recorded full forwards. SSE criterion needs local traces only in principle; NLL requires suffix/head.',
        'target_artifact_bytes':{'o_proj':1179684,'down_proj':1769508,'all_30_existing_target_files':44237880},'runtime':'saved INT4 decoded BF16 dense quality reference; no packed serving speed/VRAM saving measurement'})
    save(run/'verification.json',{'status':'PASS','existing_unique_tests_rerun':15,'new_unique_tests':10,'new_test_runs':'initial false positive on str.encode guard, corrected scope;10 PASS before first worker and10 PASS after runner availability fix; repeated runs are not new unique tests',
        'original_P1_reaudit':oldaudit,'fresh_saved_scalar_audit':json.loads((run/'independent_audit.json').read_text()),'private_array_audit':json.loads((run/'private_array_audit.json').read_text()),
        'preservation':json.loads((run/'preservation.json').read_text())['status'],'P0_MC_rerun':False,'original_P1_GPU_sweep_rerun':False,'new_GPU_scope':'P1b only; first failed SMOKE3 + completed3813 forwards',
        'figure_visual_inspection':'PNG figures1/4 visually checked; no browser UI test claimed','review_ZIP_validation':'separate final ZIP receipt after extraction; see external review directory',
        'limits':'CPU saved-scalar recomputation is not GPU/model re-evaluation; private matrix check covers9 rows and cropped SMOKE, not all model computations.'})
    save(case/'followups/p1b_selection/AMENDMENT.json',{'recorded_utc':utc(),'record_type':'POSTHOC_RECORD_OF_PRE_SELECT_FIX; source/attempt receipts establish actual chronology','before_new_SELECT_forward':True,'spec_or_inputs_changed':False,'changes':[{'reason':'Historical P1 SMOKE prefix array not retained','patch':'Check availability of historical prefix Z comparison; keep exact teacher parity, immutable prefix hashes and current common-Z/repeat checks.','failed_worker_seconds':receipt['previous_failed_worker_seconds']},{'reason':'Conservative budget accounting','patch':'carry first failed worker time into second-worker remaining budget'}],
         'first_source_sha256':json.loads((first_out/'source_freeze.json').read_text())['execution_source_sha256'],'resumed_source_sha256':json.loads((run/'source_freeze.json').read_text())['execution_source_sha256']})
    print('Metadata, failed receipts, input validation and four-axis decision written.')

if __name__=='__main__':main()
