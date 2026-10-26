"""One isolated GPU operator worker. No fallback for upstream CUDA arithmetic."""
import argparse,time,gc,os,sys,traceback,subprocess,datetime
from common import *
def device_state():
 r=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader'],capture_output=True,text=True);return {'exit_code':r.returncode,'sample':r.stdout.strip(),'stderr':r.stderr.strip()}
def main(args):
 import torch
 torch.set_num_threads(1)
 policy_before={'tf32':torch.backends.cuda.matmul.allow_tf32,'fp16_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,'bf16_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 # Explicit FP32 accumulation policy; upstream _matmul_fp32 itself remains unchanged.
 if hasattr(torch.backends.cuda.matmul,'allow_fp16_accumulation'):torch.backends.cuda.matmul.allow_fp16_accumulation=False
 torch.cuda.init();z=torch.ones((8,8),device='cuda',dtype=torch.float16);zz=torch.mm(z,z,out_dtype=torch.float32);torch.cuda.synchronize()
 assert zz.dtype==torch.float32 and bool(torch.isfinite(zz).all())
 del z,zz;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
 baseline={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'device':device_state()}
 result={'kind':args.kind,'baseline':baseline,'device_name':torch.cuda.get_device_name(),'sm':list(torch.cuda.get_device_capability()),'policy_before':policy_before,'policy_after':{'tf32':torch.backends.cuda.matmul.allow_tf32,'fp16_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,'bf16_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'fp16_accumulation':getattr(torch.backends.cuda.matmul,'allow_fp16_accumulation',None)},'cuda_path_probe':'FP16 torch.mm(out_dtype=FP32) finite PASS','allocator':torch.cuda.memory.get_allocator_backend(),'upstream_sha':UPSTREAM}
 loader,resbytes=upstream()
 if args.kind=='preflight':
  shape=list(SHAPES)[args.index];s=SHAPES[shape];V,L,H=s['V'],s['L'],s['H'];inp=[i for i in load(ROOT/'provenance/input_manifest.json') if i['shape']==shape];decoder=loader(ROOT/inp[0]['artifact']['path'],device='cuda');table=torch.empty((V,L*H),device='cuda',dtype=torch.bfloat16);torch.cuda.reset_peak_memory_stats()
  records=[]
  with torch.inference_mode():
   for start in range(0,V,256):
    ix=torch.arange(start,min(start+256,V),device='cuda');v=decoder(ix);table[start:start+len(ix)].copy_(v);del ix,v
   fs=functions(torch,decoder,table)
   for item in inp:
    ids_np=np.load(ROOT/item['path'],allow_pickle=False);validate_ids(ids_np,V);ids=torch.from_numpy(ids_np).cuda();meta,a=read_fixture(ROOT/item['artifact']['path']);ref=reference(meta,a,ids_np);outs={};checks={}
    for method,f in fs.items():
     x=f(ids);assert x.dtype==torch.bfloat16 and x.device.type=='cuda' and list(x.shape)==[item['M'],L*H];outs[method]=x.float().cpu().numpy().reshape(item['M'],L,H);checks[method]=metrics(outs[method],ref);del x
    pairs={aa+'/'+bb:metrics(outs[aa],outs[bb]) for aa,bb in [('A','B'),('B','C'),('A','C')]}
    records.append({'shape':shape,'M':item['M'],'U':item['U'],'condition':item['condition'],'ids_bytes_sha256':item['ids_bytes_sha256'],'artifact_sha256':item['artifact']['sha256'],'methods':checks,'pairs':pairs});del ids,ref,outs,a
   torch.cuda.synchronize()
  result['records']=records;result['worker_source_sha256']=file_info(ROOT/'src/gpu_worker.py')['sha256'];result['protocol_sha256']=file_info(ROOT/'configs/protocol.json')['sha256'];result['allocated_delta_peak']=torch.cuda.max_memory_allocated()-baseline['allocated'];result['memory_gate_pass']=result['allocated_delta_peak']<=2*1024**3;result['status']='PASS' if result['memory_gate_pass'] and all(er['pass'] for row in records for er in [*row['methods'].values(),*row['pairs'].values()]) else 'BLOCKED_NUMERICS';result['final_device']=device_state();dump(args.out,result);return
 if args.kind=='probe':
  path=ROOT/'inputs/p1_seed0_random.npz';meta,a=read_fixture(path);d=loader(path,device='cuda');ids=np.arange(257,dtype=np.int64);tids=torch.from_numpy(ids).cuda()
  with torch.inference_mode():x=d(tids)
  result['correctness']=metrics(x.float().cpu().numpy().reshape(257,3,16),reference(meta,a,ids));result['dtype']=str(x.dtype);result['shape']=list(x.shape);result['inventory']=tensor_inventory(module_tensors(d));result['torch_profiler_path']=str((pathlib.Path(args.out).parent/'probe_trace.json').name)
  del x
  with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
   x=d(tids);torch.cuda.synchronize()
  prof.export_chrome_trace(str(pathlib.Path(args.out).parent/'probe_trace.json'))
  result['profiler_ops']=[{'name':e.key,'count':e.count} for e in prof.key_averages()];result['status']='PASS' if result['correctness']['pass'] else 'BLOCKED_NUMERICS';result['final_device']=device_state();dump(args.out,result);return
 cell=load(ROOT/'provenance/planned_cells.json')[args.index];result['cell']=cell
 protocol=load(ROOT/'configs/protocol.json');preflight_path=ROOT/'results'/protocol['run_id']/f"preflight_{cell['shape']}.json";preflight=load(preflight_path)
 assert preflight['worker_source_sha256']==file_info(ROOT/'src/gpu_worker.py')['sha256'] and preflight['protocol_sha256']==file_info(ROOT/'configs/protocol.json')['sha256']
 check=next(r for r in preflight['records'] if r['ids_bytes_sha256']==cell['ids_bytes_sha256']);assert check['artifact_sha256']==cell['artifact']['sha256'] and all(er['pass'] for er in [*check['methods'].values(),*check['pairs'].values()]);result['preflight_identity']={'path':str(preflight_path.relative_to(ROOT)),'sha256':file_info(preflight_path)['sha256'],'input_sha256':cell['ids_bytes_sha256']}
 s=SHAPES[cell['shape']];V,L,H,Z,G,D=[s[k] for k in ['V','L','H','Z','G','D']];method=cell['method']
 ids_np=np.load(ROOT/cell['path'],allow_pickle=False);validate_ids(ids_np,V);assert sha(ids_np.tobytes())==cell['ids_bytes_sha256']
 ids=torch.from_numpy(ids_np).cuda();torch.cuda.synchronize()
 torch.cuda.reset_peak_memory_stats();setup_start=time.perf_counter();decoder=loader(ROOT/cell['artifact']['path'],device='cuda');upstream_resident=resbytes(decoder);decoder_inventory=tensor_inventory(module_tensors(decoder))
 with torch.inference_mode():
  if method=='A':
   table=torch.empty((V,L*H),device='cuda',dtype=torch.bfloat16)
   for start in range(0,V,256):
    chunk_ids=torch.arange(start,min(start+256,V),device='cuda',dtype=torch.long);chunk=decoder(chunk_ids);table[start:start+len(chunk_ids)].copy_(chunk);del chunk_ids,chunk
   torch.cuda.synchronize();setup_peak=torch.cuda.max_memory_allocated();del decoder;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
   def f():return table.index_select(0,ids.reshape(-1)).reshape(*ids.shape,L*H)
   state=tensor_inventory([('dense_table',table,True)])
  else:
   torch.cuda.synchronize();setup_peak=torch.cuda.max_memory_allocated()
   if method=='B':
    def f():return decoder(ids)
   else:
    def f():
     unique,inverse=torch.unique(ids.reshape(-1),sorted=True,return_inverse=True)
     return decoder(unique).index_select(0,inverse).reshape(*ids.shape,L*H)
   state=tensor_inventory(module_tensors(decoder))
  result['setup']={'wall_seconds':time.perf_counter()-setup_start,'allocated_peak':setup_peak,'reserved_peak':torch.cuda.max_memory_reserved(),'decoder_coexist_inventory':decoder_inventory,'A_compact_removed':method=='A','dense_chunk':256}
  result['resident_inventory']=state;result['upstream_compact_resident_bytes']=upstream_resident;result['id_inventory']=tensor_inventory([('ids',ids,True)])
  torch.cuda.reset_peak_memory_stats();x=f();torch.cuda.synchronize();result['cold_peak']={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()};result['output_inventory']=tensor_inventory([('output',x,False)])
  meta,arrays=read_fixture(ROOT/cell['artifact']['path']);ref=reference(meta,arrays,ids_np);actual=x.float().cpu().numpy().reshape(len(ids_np),L,H)
  result['correctness']=metrics(actual,ref);result['output_dtype']=str(x.dtype);result['output_shape']=list(x.shape)
  # Temporary requested outputs only, for paired audit in parent; never dense tables.
  if args.kind=='cell':
   pair_path=pathlib.Path(args.pair_dir)/f"{cell['cell_id']}.npy";pair_path.parent.mkdir(parents=True,exist_ok=True);np.save(pair_path,actual,allow_pickle=False)
   if cell['block']==0 and method=='B':
    raw_output_path=pathlib.Path(args.out).with_suffix('.bf16.npz');np.savez_compressed(raw_output_path,words=x.view(torch.uint16).cpu().numpy())
    result['raw_bf16_output']={'path':raw_output_path.name,**file_info(raw_output_path),'encoding':'uint16 BF16 words, native little endian; interpret uint32(words)<<16 as float32'}
  del x,actual,ref,arrays
  if not result['correctness']['pass']:result['status']='BLOCKED_NUMERICS';dump(args.out,result);return
  for _ in range(20):
   x=f();del x
  torch.cuda.synchronize();gc.collect();result['resident_after_warmup']={'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'device':device_state()}
  torch.cuda.reset_peak_memory_stats();reserved_start=torch.cuda.memory_reserved();host=[]
  for _ in range(100):
   torch.cuda.synchronize();t=time.perf_counter_ns();x=f();torch.cuda.synchronize();host.append((time.perf_counter_ns()-t)/1000);del x
  torch.cuda.synchronize();result['timed_memory']={'allocated_peak':torch.cuda.max_memory_allocated(),'reserved_start':reserved_start,'reserved_peak':torch.cuda.max_memory_reserved(),'device_sample_after_host':device_state()}
  K={1:32,128:8,2048:1}[cell['M']];event=[]
  for _ in range(100):
   begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
   for _ in range(K):x=f();del x
   end.record();end.synchronize();event.append(begin.elapsed_time(end)*1000/K)
  result['timed_memory']['allocated_peak']=max(result['timed_memory']['allocated_peak'],torch.cuda.max_memory_allocated());result['timed_memory']['reserved_peak']=max(result['timed_memory']['reserved_peak'],torch.cuda.max_memory_reserved())
  floor_host=[];floor_event=[]
  for _ in range(100):
   torch.cuda.synchronize();t=time.perf_counter_ns();torch.cuda.synchronize();floor_host.append((time.perf_counter_ns()-t)/1000)
   e1=torch.cuda.Event(enable_timing=True);e2=torch.cuda.Event(enable_timing=True);e1.record();e2.record();e2.synchronize();floor_event.append(e1.elapsed_time(e2)*1000)
  result['timing']={'host_us':host,'event_bracket_per_call_us':event,'event_K':K,'floor_host_us':floor_host,'floor_empty_event_us':floor_event,'floor_subtracted':False}
  if method=='C':
   unique,inverse=torch.unique(ids,sorted=True,return_inverse=True);decoded=decoder(unique);final=decoded.index_select(0,inverse);result['unique_inventory']=tensor_inventory([('unique',unique,False),('inverse',inverse,False),('unique_decoded',decoded,False),('final',final,False)]);del unique,inverse,decoded,final
   parts={'unique':[],'decode':[],'inverse':[]}
   for _ in range(10):
    torch.cuda.synchronize();t=time.perf_counter_ns();u,inv=torch.unique(ids,sorted=True,return_inverse=True);torch.cuda.synchronize();parts['unique'].append((time.perf_counter_ns()-t)/1000)
    t=time.perf_counter_ns();v=decoder(u);torch.cuda.synchronize();parts['decode'].append((time.perf_counter_ns()-t)/1000)
    t=time.perf_counter_ns();v2=v.index_select(0,inv);torch.cuda.synchronize();parts['inverse'].append((time.perf_counter_ns()-t)/1000);del u,inv,v,v2
   result['diagnostic_parts_us']=parts
 result['allocated_delta_peak']=max(result['setup']['allocated_peak'],result['cold_peak']['allocated'],result['timed_memory']['allocated_peak'])-baseline['allocated']
 result['memory_gate_pass']=result['allocated_delta_peak']<=2*1024**3;result['status']='PASS' if result['memory_gate_pass'] else 'BLOCKED_MEMORY';result['final_device']=device_state();dump(args.out,result)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['probe','preflight','smoke','cell'],default='cell');p.add_argument('--index',type=int,default=0);p.add_argument('--out',required=True);p.add_argument('--pair-dir',default='/tmp/case020_pairs');a=p.parse_args()
 try:main(a)
 except Exception as e:dump(a.out,{'status':'BLOCKED_ENV','exception':repr(e),'traceback':traceback.format_exc(),'kind':a.kind,'index':a.index});raise
