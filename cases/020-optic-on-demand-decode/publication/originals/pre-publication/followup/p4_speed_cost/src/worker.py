"""One serial P4 CUDA worker. Preflight, timing, lifecycle and diagnostics are disjoint."""
import argparse,time,gc,os,sys,traceback,subprocess,resource,datetime
from base import *
def telemetry():
 cmd=['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=name,driver_version,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv,noheader'];r=subprocess.run(cmd,capture_output=True,text=True)
 cpu=[]
 try:cpu=[x.strip() for x in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('cpu MHz')]
 except Exception as e:cpu=[repr(e)]
 return {'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_command':cmd,'exit_code':r.returncode,'gpu':r.stdout.strip(),'stderr':r.stderr.strip(),'CPU_MHz_samples':cpu,'affinity':sorted(os.sched_getaffinity(0))}
def state(torch,named=()):
 torch.cuda.synchronize();return {'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'allocated_peak':torch.cuda.max_memory_allocated(),'reserved_peak':torch.cuda.max_memory_reserved(),'inventory':inventory(named)}
def build_dense(torch,dec):
 table=torch.empty((32768,4096),device='cuda',dtype=torch.bfloat16)
 for start in range(0,32768,256):
  ids=torch.arange(start,min(start+256,32768),device='cuda',dtype=torch.long);x=dec(ids);table[start:start+len(ids)].copy_(x);del ids,x
 torch.cuda.synchronize();return table
def correctness_ok(errors):
 es=np.array(errors['error_sumsq']);rs=np.array(errors['reference_sumsq']);ma=np.array(errors['max_abs']);fi=all(errors['finite']);rel=np.sqrt(es)/np.maximum(np.sqrt(rs),1e-12);return bool(fi and all(ma[i]<=1e-6 if rs[i]==0 else rel[i]<=.01 for i in range(16)) and (ma.max()<=1e-6 if rs.sum()==0 else math.sqrt(es.sum()/rs.sum())<=.01))
def guard(job,source_hash):
 if job['kind']=='preflight':return None
 index=read(ROOT/'provenance/preflight_index.json');entry=index[str(job['Z'])];p=ROOT/entry['path']
 if info(p)['sha256']!=entry['sha256']:raise ValueError('PREFLIGHT_HASH_MISMATCH')
 pre=read(p)
 if pre['source_hash']!=source_hash or pre['config_sha256']!=info(ROOT/'PROTOCOL.json')['sha256']:raise ValueError('PREFLIGHT_SOURCE_CONFIG_MISMATCH')
 found=[r for r in pre['correctness'] if r['condition']==job['condition']]
 if not found or not all(correctness_ok(e) for row in found for e in [*row['methods'].values(),*row['pairs'].values()]):raise ValueError('PREFLIGHT_NUMERICAL_GATE')
 return {'path':entry['path'],'sha256':entry['sha256']}
def main(args):
 source_hash=source_identity();jobs=read(ROOT/'provenance/plans.json');job=next(j for j in jobs if j['id']==args.job);protocol=read(ROOT/'PROTOCOL.json');inputs=read(ROOT/'provenance/inputs.json');arts=read(ROOT/'provenance/artifacts.json');width=job['Z'];art=artifact(width)
 if info(art)['sha256']!=arts[str(width)]['sha256']:raise ValueError('ARTIFACT_HASH_MISMATCH')
 for label,entry in inputs.items():
  p=pathlib.Path(entry['path']);ids_np=np.load(p,allow_pickle=False);validate_ids(ids_np)
  if info(p)['sha256']!=entry['sha256'] or sha(ids_np.tobytes())!=entry['payload_sha256'] or len(np.unique(ids_np))!=entry['U']:raise ValueError('INPUT_IDENTITY_MISMATCH')
 preflight_identity=guard(job,source_hash);import_start=time.perf_counter_ns();import torch;import_end=time.perf_counter_ns()
 result={'job':job,'source_hash':source_hash,'config_sha256':info(ROOT/'PROTOCOL.json')['sha256'],'upstream_sha':UPSTREAM,'artifact_sha256':arts[str(width)]['sha256'],'preflight_identity':preflight_identity,'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'torch_path':torch.__file__,'numpy_path':np.__file__,'torch_import_seconds':(import_end-import_start)/1e9,'telemetry_before':telemetry(),'status':'STARTED'}
 common_start=time.perf_counter_ns();torch.set_num_threads(1);before={'TF32':torch.backends.cuda.matmul.allow_tf32,'fp16_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,'bf16_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'fp16_accumulation':getattr(torch.backends.cuda.matmul,'allow_fp16_accumulation',None)}
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 if hasattr(torch.backends.cuda.matmul,'allow_fp16_accumulation'):torch.backends.cuda.matmul.allow_fp16_accumulation=False
 torch.cuda.init();assert torch.cuda.get_device_capability()==(12,0);result['device']={'name':torch.cuda.get_device_name(),'sm':list(torch.cuda.get_device_capability()),'runtime':torch.version.cuda,'allocator':torch.cuda.memory.get_allocator_backend(),'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'sys_path':sys.path,'default_stream':int(torch.cuda.default_stream().cuda_stream),'current_stream':int(torch.cuda.current_stream().cuda_stream)};result['policy_before']=before;result['policy_after']={'TF32':torch.backends.cuda.matmul.allow_tf32,'fp16_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,'bf16_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'fp16_accumulation':getattr(torch.backends.cuda.matmul,'allow_fp16_accumulation',None)}
 memory_stages=[];snapshots={}
 if job['kind']=='memory':
  memory_stages.append({'stage':'context_initialized_before_library_probe',**state(torch)})
  try:torch.cuda.memory._record_memory_history(enabled='state',context='state',stacks='all',max_entries=512);result['history']='state/all stacks/max_entries512'
  except Exception as e:result['history_error']=repr(e)
 with torch.inference_mode():
  x=torch.ones((8,8),device='cuda',dtype=torch.float16);y=torch.mm(x,x,out_dtype=torch.float32);torch.cuda.synchronize();assert y.dtype==torch.float32 and bool(torch.isfinite(y).all());result['probe']={'left_dtype':str(x.dtype),'out_dtype':str(y.dtype),'finite':True}
  if job['kind']=='memory':memory_stages.append({'stage':'library_probe_output_live',**state(torch,[('probe_input',x),('probe_output',y)])})
  del x,y;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
  result['baseline']=state(torch);result['common_runtime_init_seconds']=(time.perf_counter_ns()-common_start)/1e9;torch.cuda.reset_peak_memory_stats()
  if job['kind']=='memory':
   memory_stages.append({'stage':'library_probe_released_baseline',**state(torch)});snapshots['after_library_probe_release']=torch.cuda.memory.memory_snapshot(include_traces=False)
  loader=upstream();result['official_import_path']=str(sys.modules['ple_eval.compact_adapter'].__file__)
  if job['kind']=='preflight':
   dec=loader(art,device='cuda');table=build_dense(torch,dec);meta,a=arrays(art);records=[];sample_arrays={}
   conditions=list(inputs) if width==32 else ['unique2048','repeat2048']
   for condition in conditions:
    entry=inputs[condition];ids_np=np.load(entry['path'],allow_pickle=False);ids=torch.from_numpy(ids_np).cuda();names=list('ABCS' if width==32 and condition=='unique2048' else 'ABC');fs=methods(torch,dec,table,ids,names);outs={};parts={m:[] for m in names};indexes=np.unique(np.r_[np.arange(min(4,len(ids_np))),np.arange(max(0,len(ids_np)-4),len(ids_np))]);sample_arrays[condition+'_ids']=ids_np[indexes]
    output_contract={}
    for m in names:
     x=fs[m]();assert x.dtype==torch.bfloat16 and x.device.type=='cuda' and list(x.shape)==[len(ids_np),4096];words=x.view(torch.uint16).cpu().numpy();out=(words.astype(np.uint32)<<16).view(np.float32).reshape(len(ids_np),16,256);outs[m]=out;sample_arrays[condition+'_'+m+'_words']=words[indexes].copy();output_contract[m]={'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device),'finite':bool(np.isfinite(out).all()),'min':float(out.min()),'max':float(out.max()),'L2_norm':float(np.linalg.norm(out.astype(np.float64)))};del x,words
    for start,ref in ref_chunks(meta,a,ids_np):
     for m in names:parts[m].append(numerical_stats(outs[m][start:start+len(ref)],ref))
    errs={m:merge_errors(v) for m,v in parts.items()};pairerrs={aa+'/'+bb:merge_errors([numerical_stats(outs[aa],outs[bb].astype(np.float64))]) for aa,bb in [('A','B'),('B','C'),('A','C')]+([('C','S'),('B','S')] if 'S' in names else [])};records.append({'condition':condition,'M':entry['M'],'U':entry['U'],'ID_payload_sha256':entry['payload_sha256'],'artifact_sha256':arts[str(width)]['sha256'],'methods':errs,'pairs':pairerrs,'output_contract':output_contract});del ids,fs,outs
   sample_path=pathlib.Path(args.out).with_suffix('.samples.npz');np.savez_compressed(sample_path,**sample_arrays);result['sample_output']={'path':str(sample_path.relative_to(ROOT)),**info(sample_path)};result['correctness']=records;result['gate']=all(correctness_ok(e) for row in records for e in [*row['methods'].values(),*row['pairs'].values()]);del table,dec,a
  elif job['kind'] in ['timing','smoke']:
   t=time.perf_counter_ns();temp=loader(art,device='cuda');table=build_dense(torch,temp);del temp;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();dec=loader(art,device='cuda');ids_np=np.load(inputs[job['condition']]['path'],allow_pickle=False);ids=torch.from_numpy(ids_np).cuda();torch.cuda.synchronize();fs=methods(torch,dec,table,ids,job['methods']);result['shared_setup_seconds']=(time.perf_counter_ns()-t)/1e9;result['shared_inventory']=inventory([('A_table',table),('IDs',ids)]+[(n,x) for n,x in dec.named_buffers()]);result['after_shared_setup']=state(torch);max_alloc=result['after_shared_setup']['allocated_peak'];rows=[];warm_calls=[];pass_records=[]
   def warm(phase,orders):
    for ri,order in enumerate(orders):
     for rank,m in enumerate(order):x=fs[m]();del x;warm_calls.append({'phase':phase,'round':ri,'rank':rank,'method':m})
    torch.cuda.synchronize()
   warm('initial',job['orders']['initial']);max_alloc=max(max_alloc,torch.cuda.max_memory_allocated())
   for pas in job['pass_order']:
    warm(pas,job['orders'][pas+'_warm']);max_alloc=max(max_alloc,torch.cuda.max_memory_allocated());before_pass=state(torch);torch.cuda.reset_peak_memory_stats();pass_anchor=time.perf_counter_ns()
    for ri,order in enumerate(job['orders'][pas]):
     for rank,m in enumerate(order):
      torch.cuda.synchronize()
      if pas=='H':
       t0=time.perf_counter_ns();x=fs[m]();torch.cuda.synchronize();t1=time.perf_counter_ns();del x;rows.append({'pass':'H','round':ri,'rank':rank,'method':m,'K':1,'start_ns':t0,'end_ns':t1,'host_us':(t1-t0)/1000,'event_us':None,'fence':'device_synchronize','pass_relative_start_us':(t0-pass_anchor)/1000})
      elif pas=='D':
       e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);t0=time.perf_counter_ns();e0.record();x=fs[m]();e1.record();e1.synchronize();t1=time.perf_counter_ns();elapsed=e0.elapsed_time(e1)*1000;del x;rows.append({'pass':'D','round':ri,'rank':rank,'method':m,'K':1,'start_ns':t0,'end_ns':t1,'host_us':(t1-t0)/1000,'event_us':elapsed,'fence':'end_event_synchronize','pass_relative_start_us':(t0-pass_anchor)/1000})
      else:
       e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True);t0=time.perf_counter_ns();e0.record()
       for _ in range(16):x=fs[m]();del x
       e1.record();e1.synchronize();elapsed=e0.elapsed_time(e1)*1000;rows.append({'pass':'E','round':ri,'rank':rank,'method':m,'K':16,'start_ns':t0,'end_ns':None,'host_us':None,'event_us':elapsed/16,'bracket_stream_us':elapsed,'fence':'end_event_synchronize','pass_relative_start_us':(t0-pass_anchor)/1000})
    torch.cuda.synchronize();after_pass=state(torch);max_alloc=max(max_alloc,after_pass['allocated_peak']);pass_records.append({'pass':pas,'before':before_pass,'after':after_pass,'wall_seconds':(time.perf_counter_ns()-pass_anchor)/1e9})
   result['samples']=rows;result['warm_calls']=warm_calls;result['passes']=pass_records;result['global_allocated_peak']=max_alloc;result['gate']=all((r['host_us'] is None or np.isfinite(r['host_us']) and r['host_us']>=0) and (r['event_us'] is None or np.isfinite(r['event_us']) and r['event_us']>=0) for r in rows);result['output_positions_per_call']=len(ids_np);result['decoded_positions_per_method']={m:0 if m=='A' else inputs[job['condition']]['U'] if m=='C' else len(ids_np) for m in job['methods']};del fs,dec,table,ids
  elif job['kind']=='lifecycle':
   method=job['method'];session_start=time.perf_counter_ns();artifact_start=time.perf_counter_ns();dec=loader(art,device='cuda');torch.cuda.synchronize();artifact_end=time.perf_counter_ns();id_start=time.perf_counter_ns();ids_np=np.load(inputs[job['condition']]['path'],allow_pickle=False);validate_ids(ids_np);ids=torch.from_numpy(ids_np).cuda();torch.cuda.synchronize();id_end=time.perf_counter_ns();dense_start=time.perf_counter_ns();table=None
   if method=='A':table=build_dense(torch,dec);del dec;dec=None;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
   dense_end=time.perf_counter_ns();f=methods(torch,dec,table,ids,[method])[method];result['method_inventory']=inventory([('IDs',ids)]+([('dense_table',table)] if method=='A' else [(n,x) for n,x in dec.named_buffers()]));loop_start=time.perf_counter_ns();first_start=time.perf_counter_ns();x=f();torch.cuda.synchronize();first_end=time.perf_counter_ns();del x
   for _ in range(1,job['N']):x=f();torch.cuda.synchronize();del x
   loop_end=time.perf_counter_ns();peak=state(torch);del f,dec,table,ids;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();session_end=time.perf_counter_ns();result['lifecycle']={'session_start_ns':session_start,'artifact_start_ns':artifact_start,'artifact_end_ns':artifact_end,'ID_start_ns':id_start,'ID_end_ns':id_end,'dense_start_ns':dense_start,'dense_end_ns':dense_end,'loop_start_ns':loop_start,'first_start_ns':first_start,'first_end_ns':first_end,'loop_end_ns':loop_end,'session_end_ns':session_end,'artifact_load_H2D_seconds':(artifact_end-artifact_start)/1e9,'ID_read_H2D_seconds':(id_end-id_start)/1e9,'dense_build_cleanup_seconds':(dense_end-dense_start)/1e9 if method=='A' else 0.,'setup_seconds':(loop_start-session_start)/1e9,'first_call_seconds':(first_end-first_start)/1e9,'total_loop_seconds':(loop_end-loop_start)/1e9,'final_cleanup_seconds':(session_end-loop_end)/1e9,'session_wall_seconds':(session_end-session_start)/1e9,'actual_fn_calls':job['N'],'completion_synchronizations':job['N'],'warmup_calls':0,'session_per_call_us':(session_end-session_start)/job['N']/1000};result['peak']=peak;result['after_final_cleanup']=state(torch);result['gate']=True
  elif job['kind']=='memory':
   method=job['method'];memory_stages.append({'stage':'before_artifact',**state(torch)});dec=loader(art,device='cuda');memory_stages.append({'stage':'after_artifact_load',**state(torch,list(dec.named_buffers()))});ids_np=np.load(inputs[job['condition']]['path'],allow_pickle=False);ids=torch.from_numpy(ids_np).cuda();table=None
   if method=='A':
    table=build_dense(torch,dec);del dec;dec=None;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
   named=[('IDs',ids)]+([('dense_table',table)] if method=='A' else list(dec.named_buffers()));memory_stages.append({'stage':'before_first_fn',**state(torch,named)});f=methods(torch,dec,table,ids,[method])[method];x=f();memory_stages.append({'stage':'first_output_live',**state(torch,named+[('output',x)])});del x;torch.cuda.synchronize();memory_stages.append({'stage':'after_output_release',**state(torch,named)})
   for _ in range(20):x=f();del x
   torch.cuda.synchronize();memory_stages.append({'stage':'after_warmup20',**state(torch,named)});del f,dec,table,ids,named;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();memory_stages.append({'stage':'after_module_ID_removal',**state(torch)});snapshots['after_module_ID_removal']=torch.cuda.memory.memory_snapshot(include_traces=False);result['memory_stages']=memory_stages;result['snapshot_files']=[]
   for label,data in snapshots.items():
    p=pathlib.Path(args.out).with_name(pathlib.Path(args.out).stem+'_'+label+'.snapshot.json');dump(p,data);result['snapshot_files'].append({'label':label,'path':str(p.relative_to(ROOT)),**info(p)})
   active=[]
   for seg in snapshots['after_module_ID_removal']:
    for b in seg.get('blocks',[]):
     if b.get('state')=='active_allocated':active.append({'size':b.get('size'),'requested_size':b.get('requested_size'),'frames':b.get('frames',[])})
   result['active_after_removal']=active;result['gate']=True
  torch.cuda.synchronize();peak=max(result.get('global_allocated_peak',0),torch.cuda.max_memory_allocated(),result.get('peak',{}).get('allocated_peak',0));result['allocated_delta_peak']=peak-result['baseline']['allocated'];result['memory_gate']=result['allocated_delta_peak']<=2*1024**3;result['status']='PASS' if result['gate'] and result['memory_gate'] else 'BLOCKED_NUMERICS_OR_MEMORY'
 result['telemetry_after']=telemetry();ru=resource.getrusage(resource.RUSAGE_SELF);result['process_cost']={'user_cpu_seconds':ru.ru_utime,'system_cpu_seconds':ru.ru_stime,'peak_RSS_bytes':ru.ru_maxrss*1024};dump(args.out,result)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 try:main(a)
 except Exception as e:dump(a.out,{'status':'BLOCKED_ENV','job_id':a.job,'exception':repr(e),'traceback':traceback.format_exc()});raise
