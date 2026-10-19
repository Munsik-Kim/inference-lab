import json,hashlib,shutil,datetime,argparse,sys,subprocess,os
from pathlib import Path
SOURCE_P2=Path(__file__).resolve().parents[1]
repo=SOURCE_P2.parents[3];case=SOURCE_P2.parents[1]
parser=argparse.ArgumentParser(description="Explicit public allowlist and path normalization; no model loading")
parser.add_argument('--private-root',type=Path,required=True)
parser.add_argument('--runtime',type=Path,required=True)
parser.add_argument('--gpu-python',type=Path,required=True)
parser.add_argument('--output-p2',type=Path,required=True)
args=parser.parse_args()
private=args.private_root.resolve();p2=args.output_p2.resolve()
p2.mkdir(parents=True,exist_ok=True);(p2/'publication').mkdir(exist_ok=True)
run=private/'run_r2';pub=p2/'results/gpu_r2';pub.mkdir(parents=True,exist_ok=False)
runtime=json.loads(args.runtime.read_text())
original=json.loads((Path(runtime['original_result_root'])/'experiment_manifest.json').read_text())
environment_prefix=subprocess.check_output([str(args.gpu_python),'-B','-c','import sys; print(sys.prefix)'],text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''}).strip()
package_prefix=subprocess.check_output([str(args.gpu_python),'-B','-c',"import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec('torch').origin).resolve().parents[4])"],text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''}).strip()
replacements={str(repo):'${SOURCE_WORKTREE}',str(private):'${PRIVATE_P2_ROOT}',runtime['model']['path']:'${MODEL_SNAPSHOT}',original['models']['4b']['path']:'${MODEL_SNAPSHOT}',runtime['original_result_root']:'${ORIGINAL_RESULT_ROOT}',runtime['artifact_root']:'${EXISTING_ARTIFACT_ROOT}',
str(args.gpu_python.parent.parent.resolve()):'${EXISTING_GPU_ENV}',str(args.gpu_python.parent.parent):'${EXISTING_GPU_ENV}',environment_prefix:'${EXISTING_GPU_ENV}',package_prefix:'${EXISTING_GPU_ENV}'}
replacements=dict(sorted(replacements.items(),key=lambda x:-len(x[0])))
def sha(data):return hashlib.sha256(data).hexdigest()
copy_map=[]
def copy_file(source,dest,raw=False):
 data=source.read_bytes();clean=data
 if raw:
  if any(k.encode() in data for k in replacements):raise ValueError('personal path in measured raw; explicit separate sanitization needed')
 else:
  for before,after in replacements.items():clean=clean.replace(before.encode(),after.encode())
 if b'/home/' in clean:raise ValueError('unmapped personal path: '+source.name)
 dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():raise FileExistsError(dest)
 dest.write_bytes(clean)
 copy_map.append({'private_label':str(source.relative_to(private)),'public_relative_path':dest.relative_to(p2).as_posix(),'original_bytes':len(data),'original_sha256':sha(data),'public_bytes':len(clean),'public_sha256':sha(clean),'byte_identical':clean==data,'transformation':'none' if clean==data else 'known local path prefixes -> semantic labels; raw responses untouched'})
for name in ['identity.json','planned_cells.json','prior_attempts.json','receipt.json','attempt_journal.jsonl','requests.jsonl','reproduction.json']:
 copy_file(run/name,pub/name,raw=name in ['requests.jsonl'])
for folder in ['cells','not_applicable']:
 for source in sorted((run/folder).glob('*.json')):copy_file(source,pub/folder/source.name,raw=True)
for folder in ['preflight','C0-BF16','C0-RTN','C0-D16','RECOVERY-BF16','RECOVERY-RTN','RECOVERY-D16']:
 for source in sorted((run/folder).glob('*.json')):copy_file(source,pub/folder/source.name)
for source in sorted(run.glob('*.stderr.log')):copy_file(source,pub/'logs'/source.name)
for folder,newfolder in [('audit_r2','analysis'),('diagnostics_r2','diagnostics')]:
 for source in sorted((private/folder).glob('*')):
  if source.is_file():copy_file(source,pub/newfolder/source.name)
for name in ['token_audit_r2.json','post_gpu_model_identity.json']:
 copy_file(private/name,pub/name)
failure=p2/'results/gpu_r1_infra';failure.mkdir(exist_ok=False)
for name in ['identity.json','receipt.json','attempt_journal.jsonl','requests.jsonl','C0-BF16.stderr.log']:
 copy_file(private/'run_r1'/name,failure/name)
copy_file(private/'cuda_visibility_probe.json',failure/'cuda_visibility_probe.json')
pre=private/'pre_execution_cpu';target=p2/'results/gpu_pre_execution_checks';target.mkdir(exist_ok=False)
for source in pre.glob('*'):
 if source.is_file():copy_file(source,target/source.name)
copy_file(pre/'baseline/verification.json',target/'baseline_verification.json')
(p2/'publication/sanitization.json').write_text(json.dumps({'schema_version':1,'raw_measurement_response_files_byte_identical':True,'rules':[{'replacement':v,'reason':'private local input/execution path'} for v in sorted(set(replacements.values()))],'files':copy_map,'note':'original private files remain unchanged; path-normalized metadata/logs are not claimed byte-identical'},indent=2)+'\n')
requests=[json.loads(x) for x in (pub/'requests.jsonl').read_text().splitlines() if x]
costs=[]
for method in ['BF16','RTN','D16']:
 for arm in ['C0','G','N','E']:
  rs=[r for r in requests if r['method']==method and r['arm']==arm]
  costs.append({'method':method,'arm':arm,'unique_model_requests':len(rs),'focal':sum(r['request_role']=='FOCAL' for r in rs),'continuation':sum(r['request_role']=='CONTINUATION' for r in rs),'generated_tokens':sum(r['output_tokens'] for r in rs),'generation_seconds_sum':sum(r['seconds'] for r in rs)})
summary=json.loads((pub/'analysis/summary.json').read_text());receipt=json.loads((pub/'receipt.json').read_text())
mem={'torch_peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in requests),'torch_peak_reserved_bytes':max(r['peak_reserved_bytes'] for r in requests),'scope':'per-request peaks including the resident dense BF16 model, not packed-serving memory savings','device_wide_max_used_at_request_end_bytes':max(r['device_free_total_after_request'][1]-r['device_free_total_after_request'][0] for r in requests),'device_wide_sampling':'CUDA driver free/total after each request; includes other GPU/Windows usage, not continuous NVML sampling'}
(pub/'costs_by_arm.json').write_text(json.dumps({'rows':costs,'memory':mem,'worker_wall':{k:receipt[k] for k in ['total_worker_wall_seconds','prior_measured_worker_wall_seconds','cumulative_measured_worker_wall_seconds','budget_debited_worker_wall_seconds','worker_wall_limit_seconds']},'initial_CUDA_load_failure_attempts':1,'failed_generation_requests':0,'smoke_model_requests':0},indent=2)+'\n')
# Physical source files were checked before and after GPU, using the same saved identities.
a=json.loads((pub/'preflight/local_file_identity.json').read_text());b=json.loads((pub/'post_gpu_model_identity.json').read_text())
for key in ['snapshot_files','saved_artifacts','runtime_configuration_sha256']:
 if a[key]!=b[key]:raise ValueError('model/artifact identity changed: '+key)
(pub/'model_preservation.json').write_text(json.dumps({'status':'PASS','snapshot_files_checked':len(a['snapshot_files']),'RTN_projection_files':252,'D16_projection_files':252,'before_after_hash_bytes_match':True,'before':'preflight/local_file_identity.json','after':'post_gpu_model_identity.json','loaded_module_and_tensor_readback':'C0-*/loaded_gate.json and RECOVERY-*/loaded_gate.json','note':'CPU hash-check module-coverage NOT_RUN is local to that CPU check; actual loaded GPU coverage is independently recorded in six loaded_gate files','official_external_weight_digest_comparison':'NOT_PERFORMED'},indent=2)+'\n')
sys.path.insert(0,str(SOURCE_P2/'publication'))
from derive import derive
import tempfile
with tempfile.TemporaryDirectory(prefix='case015-p2-derived-') as tmp:
 generated=Path(tmp)/'derived';derive(pub,generated)
 shutil.copyfile(generated/'decision.json',pub/'decision.json')
 (pub/'tables').mkdir()
 for name in ['method_api_arm.csv','paired_outcomes.csv','C0_reproduction.csv','request_costs.csv','display.json']:
  shutil.copyfile(generated/name,pub/'tables'/name)
print(json.dumps({'public_copies':len(copy_map),'byte_identical':sum(x['byte_identical'] for x in copy_map),'measured_cells':summary['measured_cells'],'requests':len(requests)}))
