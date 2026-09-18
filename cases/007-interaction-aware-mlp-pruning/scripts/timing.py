"""Paired complete-call timing; separate wall and CUDA-event blocks."""
import random,time,subprocess
from src.core import load,write_new,SEED
from src.surgery import replace_mlp

def run_timing(rt,case,work,outdir,spec,round_id):
 if round_id not in range(3):raise ValueError('Round outside fixed budget')
 if load(work/'heldout/summary.json')['status']!='PASS':raise ValueError('Heldout validity incomplete')
 torch=rt.torch;sel=load(work/'selection.json')['selections'];rt.add(sel);names=['B',*sel];rows=load(case/'inputs/HELD_OUT.json');rows=[r for r in rows if int(r['id'].rsplit('-',1)[-1])<2];rng=random.Random(SEED+round_id);records=[]
 def telemetry():return subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,clocks.sm,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
 before=telemetry();start_alloc=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
 with torch.inference_mode():
  for row in rows:
   _,_,_,_,x=rt.diagnose(row);ids=torch.tensor([row['token_ids']],device='cuda')
   for boundary in ['MLP','MODEL_PREFILL']:
    for name in names:
     with replace_mlp(rt.layer,rt.modules[name]):
      fn=(lambda:rt.modules[name](x)) if boundary=='MLP' else (lambda:rt.forward(ids))
      for _ in range(spec['timing']['warmup']):fn()
      torch.cuda.synchronize()
    for block in range(spec['timing']['blocks']):
     order=list(names);rng.shuffle(order)
     for name in order:
      with replace_mlp(rt.layer,rt.modules[name]):
       fn=(lambda:rt.modules[name](x)) if boundary=='MLP' else (lambda:rt.forward(ids))
       torch.cuda.synchronize();t=time.perf_counter()
       for _ in range(spec['timing']['calls']):out=fn()
       torch.cuda.synchronize();wall=(time.perf_counter()-t)*1000/spec['timing']['calls'];del out
       start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
       for _ in range(spec['timing']['calls']):out=fn()
       end.record();end.synchronize();event=start.elapsed_time(end)/spec['timing']['calls'];del out
      records.append({'id':row['id'],'task':row['task'],'token_hash':row['token_hash'],'round':round_id,'block':block,'method':name,'boundary':boundary,'wall_ms':wall,'event_ms':event,'evidence_kind':'gpu_measurement','calls':spec['timing']['calls'],'order':order})
   print(row['id'],flush=True)
 write_new(outdir/'blocks.json',records)
 write_new(outdir/'summary.json',{'status':'PASS','round':round_id,'blocks':len(records),'start_allocated_bytes':start_alloc,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'telemetry_before':before,'telemetry_after':telemetry(),'telemetry_fields':['temperature_C','SM_clock_MHz','whole_device_used_MiB','utilization_percent'],'whole_device_peak':'NOT_SAMPLED','scope':'3 process rounds same device; paired block-mean latency, warm-cache eager, no graph/compile, no hooks/reference in timed region'})
