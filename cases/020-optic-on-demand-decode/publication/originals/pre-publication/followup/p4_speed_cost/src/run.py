"""Parent wall accounting and serial launch; immutable original ledger is never edited."""
import argparse,subprocess,time,datetime,os,sys
from base import *
def journal(event):
 event={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),**event}
 with (ROOT/'provenance/journal.jsonl').open('a') as f:f.write(json.dumps(event,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def freeze():
 files={rel:info(ROOT/rel) for rel in ['src/base.py','src/prepare.py','src/worker.py','src/run.py','PROTOCOL.json','provenance/plans.json','provenance/inputs.json','provenance/artifacts.json','fixtures/Z128.npz','fixtures/Z644.npz']}
 data={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'bundle_sha256':sha(json.dumps(files,sort_keys=True).encode()),'already_seen':['Original P2 final/diagnostic runs and outside report','P4A historical raw statistics and32MiB at initialized baseline','No P4 GPU timing or new-width GPU correctness yet'],'scope':'initial benchmark freeze before preflight; new CPU auditor/report files will be identified separately by final P4 manifest'}
 if (ROOT/'provenance/freeze.json').exists():raise ValueError('freeze already exists; do not reset')
 dump(ROOT/'provenance/freeze.json',data);journal({'event':'SOURCE_CONFIG_INPUT_FREEZE','source_hash':data['bundle_sha256']})
def disk_bytes():return sum(p.stat().st_size for folder in [ROOT,PRIVATE] for p in folder.rglob('*') if p.is_file())
def gpu_idle():
 r=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],capture_output=True,text=True)
 if r.returncode or len([x for x in r.stdout.splitlines()[1:] if x.strip()])>0:raise RuntimeError('GPU_BUSY_OR_UNVERIFIED: '+r.stdout+r.stderr)
 return {'stdout':r.stdout,'stderr':r.stderr,'exit_code':r.returncode}
def run_all():
 source_identity();protocol=read(ROOT/'PROTOCOL.json');plans=read(ROOT/'provenance/plans.json');run=ROOT/'results'/protocol['run_id'];run.mkdir(parents=True,exist_ok=True);ledger_path=run/'worker_ledger.jsonl'
 if ledger_path.exists() and ledger_path.stat().st_size:raise ValueError('append-only run already has workers; review and use a new run ID for a correction')
 completed=[];spent=0.;stopped=None;preflight_index={}
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_CACHE_PATH=str(PRIVATE/'cuda_cache'),CUDA_CACHE_MAXSIZE=str(128*1024**2),XDG_CACHE_HOME=str(PRIVATE/'cache'),MPLCONFIGDIR=str(PRIVATE/'mpl_cache'))
 def worker(job):
  nonlocal spent,stopped
  remaining=protocol['P4_wall_cap_seconds']-spent
  if remaining<5:stopped='BUDGET_EXHAUSTED';return False
  if datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()>protocol['stop_GPU_before_KST']:stopped='DOCUMENTATION_RESERVE_STOP';return False
  if disk_bytes()>protocol['disk_hard_cap_bytes']:stopped='DISK_CAP';return False
  try:idle=gpu_idle()
  except Exception as e:stopped='GPU_BUSY';journal({'event':'GPU_BUSY','reason':repr(e),'job':job['id']});return False
  out=run/(job['id']+'.json');assert not out.exists();cmd=[sys.executable,'-B',str(ROOT/'src/worker.py'),'--job',job['id'],'--out',str(out)];start=datetime.datetime.now(datetime.timezone.utc).isoformat();journal({'event':'WORKER_START','job':job['id'],'phase':job['phase'],'remaining_P4_seconds':remaining,'global_remaining_seconds':1800-protocol['prior_gpu_wall_exact']-spent,'idle_check':idle})
  t=time.perf_counter();timeout=False
  try:r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=min(120,remaining));code=r.returncode;stdout=r.stdout;stderr=r.stderr
  except subprocess.TimeoutExpired as e:code=124;stdout=str(e.stdout);stderr=str(e.stderr);timeout=True
  wall=time.perf_counter()-t;spent+=wall;(run/(job['id']+'.stdout.txt')).write_text(stdout);(run/(job['id']+'.stderr.txt')).write_text(stderr)
  if not out.exists():dump(out,{'status':'BUDGET_EXHAUSTED' if timeout and remaining<120 else 'WORKER_FAILED','job_id':job['id'],'exit_code':code})
  record={'job_id':job['id'],'kind':job['kind'],'phase':job['phase'],'started_at':start,'ended_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wall_seconds':wall,'exit_code':code,'raw':str(out.relative_to(ROOT)),'raw_sha256':info(out)['sha256'],'source_hash':source_identity(),'config_sha256':info(ROOT/'PROTOCOL.json')['sha256'],'P4_cumulative_seconds':spent,'global_cumulative_seconds':protocol['prior_gpu_wall_exact']+spent}
  with ledger_path.open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
  completed.append(record);raw=read(out);journal({'event':'WORKER_END',**record,'worker_status':raw.get('status'),'owned_disk_bytes':disk_bytes()});print(job['id'],raw.get('status'),f'{wall:.3f}s',f'P4remaining={protocol["P4_wall_cap_seconds"]-spent:.3f}s',flush=True)
  if code or raw.get('status')!='PASS':stopped=raw.get('status','WORKER_FAILED');return False
  if disk_bytes()>protocol['disk_hard_cap_bytes']:stopped='DISK_CAP';return False
  return True
 for phase in ['preflight','smoke','core','width','lifecycle','memory']:
  if stopped:break
  selected=[j for j in plans if j['phase']==phase]
  if phase=='memory' and protocol['P4_wall_cap_seconds']-spent<30:journal({'event':'OPTIONAL_MEMORY_SKIPPED','reason':'remaining P4 budget below30seconds'});continue
  journal({'event':'PHASE_START','phase':phase,'planned_workers':len(selected)})
  for j in selected:
   if not worker(j):break
   if phase=='preflight':
    p=run/(j['id']+'.json');preflight_index[str(j['Z'])]={'path':str(p.relative_to(ROOT)),**info(p)}
  if phase=='preflight' and not stopped:dump(ROOT/'provenance/preflight_index.json',preflight_index)
  if phase=='smoke' and not stopped:
   smoke=read(run/'smoke_Z644.json');dump(ROOT/'provenance/main_freeze.json',{'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_hash':source_identity(),'config_sha256':info(ROOT/'PROTOCOL.json')['sha256'],'preflight':preflight_index,'already_seen_P4':['three width preflights with actual GPU FP16/FP32 probe and correctness','Z644 unique smoke:6 H rounds,6 D rounds,6 E brackets(K16), warm20+20per pass'],'smoke_raw':info(run/'smoke_Z644.json'),'smoke_memory_delta':smoke['allocated_delta_peak'],'amendments':'None. No design/source/input/threshold changes after seeing P4 GPU results.'});journal({'event':'MAIN_FREEZE','identity':info(ROOT/'provenance/main_freeze.json')})
  journal({'event':'PHASE_END','phase':phase,'stop_reason':stopped,'P4_spent_seconds':spent})
 status='COMPLETE' if not stopped else 'COMPLETE_PARTIAL';dump(run/'execution_receipt.json',{'status':status,'stop_reason':stopped,'workers':len(completed),'P4_spent_seconds':spent,'prior_workers':protocol['prior_gpu_workers'],'prior_spent_seconds':protocol['prior_gpu_wall_exact'],'global_spent_seconds':protocol['prior_gpu_wall_exact']+spent,'P4_remaining_seconds':protocol['P4_wall_cap_seconds']-spent,'global_remaining_seconds':1800-protocol['prior_gpu_wall_exact']-spent,'performance_success_not_required':True});journal({'event':'GPU_PHASE_FINISHED','status':status,'stop_reason':stopped,'workers':len(completed),'P4_wall_seconds':spent})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','all']);a=p.parse_args();freeze() if a.command=='freeze' else run_all()
