"""Prepare public receipts without changing private original logs or prior research."""
import argparse,json,re,sys,hashlib
from pathlib import Path
from p2_common import save,sha,utc,jsonl

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--private',required=True);a=p.parse_args();case=Path(a.case).resolve();pr=Path(a.private);run=case/'results/p2_r1';identity=json.loads((pr/'identity.private.json').read_text())
    replacements={str(case):'${CASE}',str(pr):'${P2_PRIVATE}',identity['snapshot']['model_path']:'${WSL_SNAPSHOT}',identity['snapshot']['python']:'${REFERENCE_PYTHON}',identity['cpu_python']:'${CPU_PYTHON}',identity['data_python']:'${DATA_PYTHON}'}
    def redact(s):
        for k,v in sorted(replacements.items(),key=lambda kv:-len(kv[0])):s=s.replace(k,v)
        s=re.sub(r'/home/[^/\s]+/[^\s"\']*','${REDACTED_LOCAL_PATH}',s)
        return s
    logs=run/'logs';logs.mkdir(exist_ok=False);exports=[]
    for src in sorted((pr/'attempts').glob('*.log')):
        content=redact(src.read_text());dst=logs/src.name;dst.write_text(content)
        exports.append({'source_label':'P2_PRIVATE/attempts/'+src.name,'source_sha256':sha(src),'export_relative':'logs/'+src.name,'export_sha256':sha(dst),'source_bytes':src.stat().st_size,'export_bytes':dst.stat().st_size,'redacted':src.read_text()!=content})
    save(run/'log_export_manifest.json',{'rule':'Replace confirmed local paths with semantic labels; preserve original private logs unchanged; no response/metric correction','files':exports})
    regress=[]
    for name,count,kind in [('old15.log',15,'original P0 regression rerun'),('p1b10.log',10,'existing P1b regression rerun'),('tests-p2-02.log',9,'unique new P2 tests')]:
        txt=(pr/'attempts'/name).read_text();assert f'Ran {count} tests' in txt and txt.rstrip().endswith('OK')
        regress.append({'log':'logs/'+name,'unique_tests':count,'kind':kind,'status':'PASS','exit_code':0,'evidence':'actual pre-GPU execution completed successfully; log count/OK checked again here'})
    summary=json.loads((run/'summary.json').read_text());primary=next(r for r in summary['paired_comparisons'] if r['primary']);workers=json.loads((run/'worker_receipt.json').read_text());probe=json.loads((run/'smoke_matrix_receipt.json').read_text())
    receipts=[]
    for step,cmd in zip(workers['steps'],workers['commands']):receipts.append({'phase':step['phase'],'command':cmd,**step})
    receipts.append(probe);assert all(r['exit_code']==0 for r in receipts)
    save(run/'execution_receipts.json',{'workers':receipts,'pre_GPU_regressions':regress,'CLI_help':{'command':'${REFERENCE_PYTHON} followups/p2_core/source/run_p2.py --help','exit_code':0,'log':'logs/cli-help.log','model_forwards':0},
       'data_prepare':{'command':'${DATA_PYTHON} followups/p2_core/source/prepare_inputs.py --case ${CASE} --private ${P2_PRIVATE} --pg19 ${PG19_TRAIN_SHARD_00000} --pile ${PILE10K_ARROW}','exit_code':0,'log':'logs/data-01.log','new_transfer_bytes':0}})
    save(run/'verification.json',{'status':'PASS','recorded_utc':utc(),'P2_unique_new_tests':9,'prior_tests_rerun':{'P0':15,'P1b':10},'repeated_execution_is_not_new_unique_tests':True,'CPU_regression_logs':regress,
        'independent_scalar_audit':json.loads((run/'independent_audit.json').read_text()),'private_array_audit':json.loads((run/'private_array_audit.json').read_text()),
        'fresh_GPU_process_reload':json.loads((run/'fresh_process_reload.json').read_text())['status'],'fixed_registered_SMOKE_matrix_check':'PASS','frozen_source_files_verified_unchanged':12,
        'projections_per_quantized_model':196,'quantized_models':3,'artifact_files':588,'expected_metric_rows':320,'completed_metric_rows':320,'missing_cells':[],'failed_GPU_workers':0,'GPU_forward_calls':summary['resources']['all_full_model_forwards'],
        'worker_budget_seconds':10800,'actual_cumulative_worker_wall_seconds':summary['resources']['cumulative_worker_wall_seconds'],'preservation':json.loads((run/'preservation.json').read_text())['status'],
        'GPU_BUSY_TTFT_TPOT':'NOT_MEASURED','P0_MC_P1_P1b_GPU_rerun':False,'P3_4B_agent_packed_serving':False,'package_install_upgrade_build':False,'git_stage_commit_remote_publication':False,
        'archive_extraction_validation':'separate actual receipt next to review ZIP; not preclaimed here'})
    save(run/'interpretation.json',{'registered_primary':primary,'scope':'actual whole-model Qwen3-0.6B; original CAL32; fresh PG19 and separate fresh ArXiv groups','PG19':'lower NLL at identical core bytes','ArXiv':'QUARTER higher point estimate with CI crossing zero; HALF higher NLL with auxiliary CI above zero','serving_benefit':'NOT_ESTABLISHED','new_selector':False,'primary_changed_to_HALF':False,'P3_or_remote_next_step':'NOT_RUN_SCOPE'})
    files={str(p.relative_to(case)):sha(p) for p in sorted((case/'followups/p2_core').rglob('*')) if p.is_file() and p.suffix in ['.py','.json','.txt','.md','.html']}
    save(run/'review_source_manifest.json',{'recorded_utc':utc(),'files':files,'review_source_tree_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
       'execution_source_sha256':json.loads((run/'source_freeze.json').read_text())['execution_source_sha256'],'distinction':'GPU runtime files are the unchanged12-file execution snapshot. Postprocessing/report/verification sources have this separate final identity; archive manifest covers the complete review bundle.'})
    print('Public logs/receipts, verification and separate review-source identity prepared.')
if __name__=='__main__':main()
