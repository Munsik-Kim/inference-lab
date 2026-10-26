import datetime,subprocess,sys,os,platform,shutil
from common import *
def cmd(xs):
 r=subprocess.run(xs,capture_output=True,text=True);return {'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def prepare():
 protocol={'run_id':'20261026-1215','upstream_sha':UPSTREAM,'shapes':SHAPES,'methods':{'A':'DENSE_RECON_LOOKUP','B':'COMPACT_REQUEST_DECODE','C':'COMPACT_UNIQUE_DECODE'},'artifact_seed':61026020,'input_seed':61026021,'order_seed':61026022,'mapping':'SeedSequence([root_seed,shape_index,condition_index]); shape_index SMALL=0,LARGE=1; condition index order M1unique,M128unique,M128repeat75,M2048unique,M2048repeat75','warmup':20,'host_samples':100,'event_brackets':100,'event_calls':{'1':32,'128':8,'2048':1},'blocks':3,'dense_chunk':256,'rtol_fp64':1e-10,'atol_fp64':1e-12,'relative_L2_limit':1e-2,'zero_max_abs_limit':1e-6,'quantile':'numpy.quantile(method=linear)','max_gpu_worker_wall_seconds':1800,'max_allocated_delta_bytes':2*1024**3,'planned_cells':30,'planned_cell_blocks':90,'output':'raw unscaled BF16 [*ids.shape,L*H], interpreted [M,L,H]','timing':'warm-cache eager operator, GPU-resident IDs/output; synchronized host wall includes completion synchronize; event bracket stream elapsed/K','autograd':False,'cuda_graph':False,'torch_compile':False,'reduced_precision_reduction':False,'tf32':False,'source_freeze':'after P1 observed; before CUDA probe/smoke and sweep; not a global preregistration','budgets':{'initial_start_kst':'2026-10-26T09:57:00+09:00','environment_switch_interruption':'User interrupted and switched Windows native to WSL; resumed 12:12 KST. Preserve elapsed calendar time; aim final before13:57 KST.','documentation_minimum_seconds':1800},'resource_estimate':{'worktree_bytes_observed':1160000000,'new_dependencies_download_bytes':0,'max_new_bytes':8*1024**3,'public_case_target_bytes':128*1024**2,'dense_table_bytes':{'SMALL':16777216,'LARGE':268435456},'estimated_max_operator_extra_bytes':600*1024**2}}
 inputs=[]
 for si,(name,s) in enumerate(SHAPES.items()):
  meta,a=fixture(s,np.random.SeedSequence([61026020,si,0]));path=ROOT/'inputs'/f'{name}.npz';save_fixture(path,meta,a)
  artifact={'path':str(path.relative_to(ROOT)),**file_info(path),'metadata':meta,'arrays':array_inventory(a),'payload_bytes':sum(v.nbytes for v in a.values())}
  for ci,(M,kind) in enumerate([(1,'unique'),(128,'unique'),(128,'repeat75'),(2048,'unique'),(2048,'repeat75')]):
   U=M if kind=='unique' else M//4;r=np.random.default_rng(np.random.SeedSequence([61026021,si,ci]));ids=r.choice(s['V'],size=U,replace=False).astype(np.int64)
   if kind=='repeat75':ids=np.repeat(ids,4);r.shuffle(ids)
   validate_ids(ids,s['V']);p=ROOT/'inputs'/f'{name}_M{M}_{kind}.npy';np.save(p,ids,allow_pickle=False)
   inputs.append({'shape':name,'M':M,'U':len(np.unique(ids)),'condition':kind,'condition_index':ci,'path':str(p.relative_to(ROOT)),**file_info(p),'ids_bytes_sha256':sha(ids.tobytes()),'artifact':artifact})
 cells=[]
 for block in range(3):
  these=[dict(x,method=m,block=block,cell_id=f"b{block}_{x['shape']}_M{x['M']}_{x['condition']}_{m}") for x in inputs for m in METHODS]
  np.random.default_rng(np.random.SeedSequence([61026022,block])).shuffle(these);cells+=these
 dump(ROOT/'configs/protocol.json',protocol);dump(ROOT/'provenance/input_manifest.json',inputs);dump(ROOT/'provenance/planned_cells.json',cells)
 import torch
 env={'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'platform':platform.platform(),'python':sys.version,'executable':sys.executable,'sys_path':sys.path,'numpy':np.__version__,'numpy_path':np.__file__,'torch':torch.__version__,'torch_path':torch.__file__,'torch_cuda_runtime':torch.version.cuda,'torch_arch_list':torch.cuda.get_arch_list(),'runtime_policy':'read-only reuse of pre-existing baseline runtime; no pip install/upgrade; not clean-environment validation','torch_config':torch.__config__.show(),'os_release':pathlib.Path('/etc/os-release').read_text(),'memory':pathlib.Path('/proc/meminfo').read_text(),'cpu':cmd(['lscpu']),'disk':dict(zip(['total','used','free'],shutil.disk_usage(ROOT))),'child_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE','PYTORCH_CUDA_ALLOC_CONF','CUDA_MODULE_LOADING']},'CUDA_context_initialized':torch.cuda.is_initialized()}
 dump(ROOT/'provenance/environment.json',env)
 files={str(p.relative_to(ROOT)):file_info(p) for d in ['src','configs','inputs','vendor'] for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
 dump(ROOT/'provenance/freeze.json',{'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base_head':'3e6f2c6b36fb2a3853cc3677bee84904d28567cf','upstream_sha':UPSTREAM,'already_observed':['P1 official example and387 CPU arithmetic records PASS','Windows/WSL location and nvidia-smi device state; no GPU compute executed'],'files':files,'bundle_sha256':sha(json.dumps(files,sort_keys=True).encode()),'note':'Uncommitted local source bundle identity; distinct from Git base. GPU code must be added to amendment freeze before first CUDA execution.'})
 print('prepared',len(inputs),'inputs',len(cells),'cell-blocks; CUDA initialized:',torch.cuda.is_initialized())
if __name__=='__main__':prepare()
