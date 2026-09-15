"""One paired process round. No GPU work runs from the CPU analyzer."""
import argparse,gc,json,random,sys,time,traceback,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import CASE,sha,write_json,utc,fingerprint
from src.protocol import verify_phase,configure,current_environment,gpu_guard
from src.backends import adapter,tensor_hash
from src.reference import sampled_fp32
from src.base_metrics import unit_error
from src.timing import measured_block
from src.runtime import Sampler,profile_call

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['dev','fresh'],required=True);p.add_argument('--round',type=int,required=True);p.add_argument('--traces',type=Path,required=True);a=p.parse_args()
 phase='a' if a.stage=='dev' else 'b';s,digest=verify_phase(phase)
 if a.stage=='fresh':
  assert s['finalist_id'] in ['V1','V2','V3','V4'];ids=['B','A_PUBLIC',s['finalist_id']];lengths=[4096,512,2048];blocks=20;manifest=CASE/'inputs/fresh_manifest.json';cap=CASE/'provenance/fresh_capture.json';rounds=5
 else:ids=s['executable_configs'];lengths=[4096];blocks=5;manifest=CASE/'inputs/dev_manifest.json';cap=CASE/'provenance/dev_capture.json';rounds=2
 assert 0<=a.round<rounds
 out=CASE/f'results/{a.stage}/round-{a.round}.json';assert not out.exists(),'Refuse to overwrite raw results'
 import torch,numpy as np
 configure();start=gpu_guard();env=current_environment();assert fingerprint(env)==s['environment_fingerprint'],'Runtime fingerprint changed'
 result={'evidence_kind':'gpu_measurement','mock':False,'stage':a.stage,'round':a.round,'started_at_utc':utc(),'phase_sha256':digest,'environment_fingerprint':fingerprint(env),'process_start':start,'measurements':[],'errors':[],'counts':{'warmup':20,'blocks':blocks,'calls':10}}
 docs=json.loads(manifest.read_text())['documents'];caps=json.loads(cap.read_text())['traces'];sampler=Sampler().start();functions={c:adapter(c) for c in ids}
 try:
  with torch.inference_mode():
   for n in lengths:
    for di,d in enumerate(docs):
     did=d['document_id'];meta=next(t for t in caps if t['document_id']==did and t['length']==n);path=a.traces/meta['trace_file'];assert sha(path)==meta['trace_sha256']
     assert meta['all_query_positions'] and meta['query_count']==n and meta['status']=='VALID' and meta['native_replay_bitwise_equal']
     host=torch.load(path,map_location='cpu',weights_only=True);assert host['meta']['input_sha256']==d['prefix_sha256'][str(n)] and host['meta']['model_revision']==s['model_revision']
     q,k,v=[host[x].cuda() for x in ['q','k','v']];assert q.shape==(1,16,n,128) and k.shape==v.shape==(1,8,n,128)
     hashes=[tensor_hash(x) for x in [q,k,v]];pos=[i*(n-1)//31 for i in range(32)];ref=sampled_fp32(q,k,v,pos,128**-.5,True).numpy();records={}
     for cid,fn in functions.items():
      call=lambda fn=fn:fn(q,k,v,causal=True,scale=128**-.5)
      t=time.perf_counter();o=call();torch.cuda.synchronize();cold=time.perf_counter()-t
      assert o.shape==q.shape and o.dtype==torch.bfloat16 and o.is_contiguous();bad=int((~torch.isfinite(o)).sum());badrows=int((~torch.isfinite(o).all(-1)).sum())
      observed=o[:,:,pos].float().cpu().numpy();units=[]
      for h in range(16):
       m=unit_error(observed[0,h],ref[0,h]);m.update(head=h,reference_norm=float(np.linalg.norm(ref[0,h].astype('float64'))),row_error_norms=[float(x) if np.isfinite(x) else None for x in np.linalg.norm(observed[0,h].astype('float64')-ref[0,h].astype('float64'),axis=-1)],row_reference_norms=np.linalg.norm(ref[0,h].astype('float64'),axis=-1).tolist());units.append(m)
      rec=dict(config_id=cid,document_id=did,split=a.stage,length=n,layer=13,hq=16,hkv=8,dim=128,round=a.round,input_sha256=hashes,trace_sha256=meta['trace_sha256'],prefix_sha256=d['prefix_sha256'][str(n)],phase_sha256=digest,query_positions=pos,units=units,full_output_nonfinite=bad,full_invalid_rows=badrows,output_dtype='BF16',status='OK',first_call_seconds=cold,wall_ms=[],event_ms=[],order_positions=[])
      if cid=='B':
       rec['native_bitwise_equal']=bool(torch.equal(o.cpu(),host['native']));assert rec['native_bitwise_equal'],'Inherited capture replay changed'
       # One real-geometry diagnostic, outside timing; all variants were profiled on smoke.
       if di==0 and a.round==0:
        rec['real_geometry_profile']=profile_call(call);assert rec['real_geometry_profile']['fused_bf16']
      del o
      for _ in range(20):call()
      torch.cuda.synchronize();records[cid]=rec
     rng=random.Random(505100+a.round*10000+n+di*37);orders=[]
     while len(orders)<blocks:
      cycle=list(ids);rng.shuffle(cycle);rot=[cycle[i:]+cycle[:i] for i in range(len(cycle))];rng.shuffle(rot);orders.extend(rot)
     for bi,order in enumerate(orders[:blocks]):
      for oi,cid in enumerate(order):
       fn=functions[cid];torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
       wall,event=measured_block(lambda:fn(q,k,v,causal=True,scale=128**-.5),torch.cuda.synchronize,lambda:torch.cuda.Event(enable_timing=True),10)
       rec=records[cid];rec['wall_ms'].append(wall);rec['event_ms'].append(event);rec['order_positions'].append(oi)
       rec['peak_allocated_bytes']=max(rec.get('peak_allocated_bytes',0),torch.cuda.max_memory_allocated());rec['peak_reserved_bytes']=max(rec.get('peak_reserved_bytes',0),torch.cuda.max_memory_reserved());rec['incremental_peak_allocated_bytes']=max(rec.get('incremental_peak_allocated_bytes',0),torch.cuda.max_memory_allocated()-before)
       assert rec['peak_allocated_bytes']<=13*2**30
     assert hashes==[tensor_hash(x) for x in [q,k,v]],'INPUT_MUTATION'
     for rec in records.values():rec['inputs_unchanged']=True;result['measurements'].append(rec)
     print(json.dumps({'round':a.round,'document_id':did,'length':n,'configs':ids,'completed':True}),flush=True)
     del q,k,v,host,ref,observed,records,call;gc.collect();torch.cuda.empty_cache()
  result['status']='COMPLETED'
 except Exception:
  result['status']='FAILED';result['errors'].append(re.sub(r'/home/[^\s\"\']+','<local-path>',traceback.format_exc()))
 finally:
  result['gpu_samples']=sampler.stop();result['completed_at_utc']=utc();write_json(out,result)
 if result['status']!='COMPLETED':raise SystemExit(2)
if __name__=='__main__':main()
