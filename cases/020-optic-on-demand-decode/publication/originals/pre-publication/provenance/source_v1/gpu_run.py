"""Serial isolated workers and cumulative wall ledger. Parent never imports Torch."""
import argparse,subprocess,time,datetime,os,sys,pathlib
from common import *
PRIVATE=pathlib.Path('/home/mskim/Codex_main/case020-optic-private-20261026-1215')
RUN=ROOT/'results/20261026-1215'
def freeze():
 fs={str(p.relative_to(ROOT)):file_info(p) for d in ['src','configs','inputs','vendor'] for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
 original=load(ROOT/'provenance/freeze.json');original['gpu_source_frozen_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();original['gpu_files']=fs;original['gpu_bundle_sha256']=sha(json.dumps(fs,sort_keys=True).encode());original['amendments_before_cuda']=['Added GPU worker and orchestration before first CUDA execution. No observed GPU numerics/latency. Shapes, inputs, seeds, gates and loops unchanged.'];dump(ROOT/'provenance/freeze.json',original)
def run(mode):
 RUN.mkdir(exist_ok=True,parents=True);ledger_path=RUN/'worker_ledger.json';ledger=load(ledger_path) if ledger_path.exists() else {'limit_seconds':1800,'workers':[],'spent_seconds':0,'status':'RUNNING'}
 if mode=='probe':freeze()
 def worker(kind,index,label):
  remaining=1800-sum(w['wall_seconds'] for w in ledger['workers'])
  if remaining<10:ledger['status']='BUDGET_EXHAUSTED';dump(ledger_path,ledger);return False
  outfile=RUN/f'{label}.json';cmd=[sys.executable,'-B',str(ROOT/'src/gpu_worker.py'),'--kind',kind,'--index',str(index),'--out',str(outfile),'--pair-dir',str(PRIVATE/'paired_outputs')]
  env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_CACHE_PATH=str(PRIVATE/'cuda_cache'),MPLCONFIGDIR=str(PRIVATE/'mpl_cache'),XDG_CACHE_HOME=str(PRIVATE/'cache'))
  started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.perf_counter()
  try:r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=min(remaining,120));exitcode=r.returncode;stdout=r.stdout;stderr=r.stderr
  except subprocess.TimeoutExpired as e:exitcode=124;stdout=str(e.stdout);stderr=str(e.stderr)
  wall=time.perf_counter()-t;(RUN/f'{label}.stdout.txt').write_text(stdout);(RUN/f'{label}.stderr.txt').write_text(stderr)
  receipt={'label':label,'kind':kind,'index':index,'started_at':started,'wall_seconds':wall,'exit_code':exitcode,'command':cmd,'raw':outfile.name,'raw_sha256':file_info(outfile)['sha256'] if outfile.exists() else None};ledger['workers'].append(receipt);ledger['spent_seconds']=sum(w['wall_seconds'] for w in ledger['workers']);ledger['remaining_seconds']=1800-ledger['spent_seconds'];dump(ledger_path,ledger)
  status=load(outfile).get('status') if outfile.exists() else 'NO_RESULT';print(label,status,round(wall,3),'s; remaining',round(ledger['remaining_seconds'],1),flush=True)
  return exitcode==0 and status=='PASS'
 if mode=='probe':
  ok=worker('probe',0,'probe');ledger['status']='PROBE_PASS' if ok else 'BLOCKED_ENV'
 elif mode=='smoke':
  cells=load(ROOT/'provenance/planned_cells.json');indices=[next(i for i,c in enumerate(cells) if c['block']==0 and c['shape']==s and c['M']==2048 and c['condition']=='unique' and c['method']==m) for s in SHAPES for m in METHODS]
  ok=True
  for i in indices:
   if not worker('smoke',i,'smoke_'+cells[i]['cell_id']):ok=False;break
  ledger['status']='SMOKE_PASS' if ok else 'BLOCKED_ENV'
 else:
  if ledger['status']!='SMOKE_PASS':raise RuntimeError('probe/smoke gate not PASS')
  cells=load(ROOT/'provenance/planned_cells.json');ok=True
  for i,c in enumerate(cells):
   if not worker('cell',i,c['cell_id']):ok=False;break
  ledger['status']='COMPLETE' if ok else 'COMPLETE_PARTIAL'
  pairs=[]
  # Pair same shape/input/block in CPU memory; remove only owned requested-output files after audit.
  for block in range(3):
   for inp in load(ROOT/'provenance/input_manifest.json'):
    base=f"b{block}_{inp['shape']}_M{inp['M']}_{inp['condition']}";paths={m:PRIVATE/'paired_outputs'/f'{base}_{m}.npy' for m in METHODS}
    if not all(p.exists() for p in paths.values()):continue
    outs={m:np.load(p,allow_pickle=False).reshape(inp['M'],SHAPES[inp['shape']]['L'],SHAPES[inp['shape']]['H']) for m,p in paths.items()}
    for a,b in [('A','B'),('B','C'),('A','C')]:pairs.append({'block':block,'shape':inp['shape'],'M':inp['M'],'U':inp['U'],'condition':inp['condition'],'methods':a+'/'+b,**metrics(outs[a],outs[b])})
    del outs
    for p in paths.values():p.unlink()
  dump(RUN/'gpu_pair_correctness.json',pairs)
 dump(ledger_path,ledger)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['probe','smoke','sweep']);a=p.parse_args();run(a.mode)
