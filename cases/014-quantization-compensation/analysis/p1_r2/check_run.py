"""Independent run-level auditor augmenting the existing scalar auditor.
No GPU; verifies raw denominators, stage counts, complete artifacts and pairing.
"""
import argparse,json,math,hashlib,subprocess,sys
from pathlib import Path

def loadrows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def close(a,b):assert abs(a-b)<=1e-10*max(1,abs(a),abs(b)),(a,b)
def main(case,private,view,output):
 case=Path(case);private=Path(private);run=case/'results/p1_r2';view=Path(view);view.mkdir(exist_ok=True,parents=True)
 for n,src in [('p1',run),('p0.json',case/'results/local_r1/p0.json'),('p0_full.json',case/'results/local_r1/p0_full.json')]:
  p=view/n
  if not p.exists():p.symlink_to(src,target_is_directory=src.is_dir())
 subprocess.run([sys.executable,str(case/'scripts/audit.py'),'--results',str(view),'--output',str(view/'scalar_audit.json')],check=True)
 rows=[r for p in sorted((run/'cells').glob('*.jsonl')) for r in loadrows(p)];calls=loadrows(run/'forward_calls.jsonl');counts={s:sum(r['stage']==s for r in calls) for s in set(r['stage'] for r in calls)}
 expected={'SMOKE':5,'CAL_TEACHER':32,'CAL_PREFIX':96,'HELDOUT_TEACHER':64,'HELDOUT_PREFIX':192,'GRID':2880,'CONTINUOUS':576}
 assert counts==expected,(counts,expected)
 teacher=[r for r in loadrows(run/'teacher.jsonl') if r['split']!='SMOKE'];prefix=loadrows(run/'prefix_only.jsonl');continuous=loadrows(run/'continuous.jsonl');assert len(teacher)==96 and len(prefix)==288 and len(continuous)==576
 # Same original documents and token hashes across every target/alpha, same per-doc NLL reference.
 ref={r['document_id_hash']:r for r in teacher};manifest=json.loads((private/'inputs/input_manifest.json').read_text());inputs={r['document_id_hash']:r for r in manifest['documents']}
 for r in rows:
  assert r['token_hash']==inputs[r['document_id_hash']]['token_hash'];assert r['offset']==inputs[r['document_id_hash']]['offset'];assert r['NLL_teacher_sum']==ref[r['document_id_hash']]['NLL_sum']
  assert r['run_id']=='p1_r2';assert r['clipping_rate']==0;assert r['status']=='COMPLETE'
 # Every frozen artifact hash/actual byte size is rechecked locally; artifacts not exported.
 artifact=json.loads((run/'artifact_manifest.json').read_text())['candidates'];assert len(artifact)==30
 for r in artifact:
  p=private/'artifacts/candidates'/f"{r['target']}.alpha{r['alpha']:g}.int4";assert p.stat().st_size==r['artifact_bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];assert r['artifact_bytes']==(1179684 if r['target'].endswith('o_proj') else 1769508)
 for p in run.glob('*.fit.json'):
  r=json.loads(p.read_text());assert r['diagnostics']['relative_solve_residual']<1e-10;assert r['no_prefix']['delta_nonzero_count']==0;assert len(set(r['no_prefix']['alpha_artifact_hashes']))==1
 events=loadrows(run/'events.jsonl');stages=[r['stage'] for r in events];assert stages.index('FIT_FROZEN')<stages.index('EVAL_RUNNING');assert sum(r['stage']=='FIT_TARGET_FROZEN' for r in events)==6
 # In calls file the first held-out model evaluation follows all CAL calls. Event receipt carries frozen selection.
 firstheld=next(i for i,r in enumerate(calls) if r['stage']=='HELDOUT_TEACHER');lastcal=max(i for i,r in enumerate(calls) if r['stage'] in ['CAL_TEACHER','CAL_PREFIX']);assert firstheld>lastcal
 result={'status':'PASS','core_cells':30,'core_document_rows':len(rows),'unique_documents_excluding_smoke':len(ref),'forward_stage_counts':counts,'total_forward_calls':len(calls),'expected_actual_forward_count':3845,'prepared_rough_forward_estimate':3865,'count_note':'Prepared smoke metadata used a rough count; enumerated loop cardinality and actual3845 calls agree. No required cell omitted.','controls':{'teacher_documents_including_smoke':97,'prefix_only_rows':len(prefix),'continuous_rows':len(continuous)},'actual_artifact_files':30,'artifact_bytes_by_type':{'o_proj':1179684,'down_proj':1769508},'max_identity_relative_gap':max(r['identity_relative_gap'] for r in rows),'max_residual_identity_relative':max(r['residual_identity_relative'] for r in rows),'peak_allocated_bytes':max(r['peak_allocated'] for r in calls),'peak_reserved_bytes':max(r['peak_reserved'] for r in calls),'max_device_used_sample_bytes':max(r['device_total_bytes']-r['device_free_bytes'] for r in calls),'device_memory_scope':'post-forward synchronized CUDA free/total samples; not a continuously monitored Windows peak','limitations':'Raw scalar/identity/pairing/count checks and local artifact hashes do not reproduce the model logits. P0 MC was not rerun.'}
 Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--private',required=True);p.add_argument('--view',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.case,a.private,a.view,a.output)
