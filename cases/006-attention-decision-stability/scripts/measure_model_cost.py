"""Paired model-prefill wall blocks; CUDA events are separate device diagnostics."""
import argparse,json,sys,time,random,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.model_runtime import Runtime,ARMS
from src.intervention import ScopedAttention
from src.storage import load,write_new,Ledger,file_hash
from scripts.measure_scores import read_inputs
from scripts.run_case006 import gpu_gate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--inputs',nargs='+',type=Path,required=True);p.add_argument('--subsets',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--process-round',type=int,choices=[0,1,2],required=True)
    p.add_argument('--work-dir',type=Path,required=True);a=p.parse_args();case=Path(__file__).resolve().parents[1]
    frozen=load(a.freeze)
    if frozen['kind']!='EVAL_MANIFEST_FREEZE':raise ValueError('No eval freeze')
    for name,sha in frozen['code_hashes'].items():
        if file_hash(case/name)!=sha:raise ValueError('Frozen implementation changed')
    selected={i for v in load(a.subsets).values() for i in v['timing_ids']}
    inputs=[x for x in read_inputs(a.inputs) if x['base_id'] in selected and x['length']==4096]
    if len(inputs)!=12 or any(frozen['input_hashes'].get(x['item_id'])!=x['token_hash'] for x in inputs):raise ValueError('Timing subset mismatch')
    a.work_dir.mkdir(parents=True,exist_ok=True);gate=gpu_gate();write_new(a.work_dir/f'gate-r{a.process_round}-{time.time_ns()}.json',gate)
    if gate['status']!='RESOURCE_GATE_PASSED':return 20
    rt=Runtime(a.snapshot);torch=rt.torch;ledger=Ledger(a.work_dir/'cells');rng=random.Random(606902+a.process_round)
    for item in inputs:
        identity={'tokens':item['token_hash'],'freeze':file_hash(a.freeze),'round':a.process_round}
        cell=f"{item['item_id']}--r{a.process_round}"
        if ledger.get(cell,identity) is not None:continue
        ids=rt.ids(item['token_ids']);rows=[]
        def call(arm,first_token=False):
            with torch.inference_mode(),ScopedAttention(rt.registry,arm,record_calls=False):
                output=rt.model(input_ids=ids,use_cache=True,logits_to_keep=1)
                if first_token:token=output.logits[0,-1].argmax()
        for arm in ARMS:
            for _ in range(5):call(arm)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        for block in range(5):
            # Same balanced Latin rotation for every cell; seed picks initial order.
            initial=list(ARMS)
            if block==0:rng.shuffle(initial);order0=initial
            offset=(block+a.process_round)%3;order=order0[offset:]+order0[:offset]
            for order_index,arm in enumerate(order):
                torch.cuda.synchronize();start=time.perf_counter()
                for _ in range(5):call(arm)
                torch.cuda.synchronize();wall=(time.perf_counter()-start)/5
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(5):call(arm)
                end.record();end.synchronize();event=begin.elapsed_time(end)/5000
                torch.cuda.synchronize();start=time.perf_counter()
                for _ in range(5):call(arm,True)
                torch.cuda.synchronize();first=(time.perf_counter()-start)/5
                rows.append({'base_id':item['base_id'],'item_id':item['item_id'],'task':item['task'],'arm':arm,
                             'token_hash':item['token_hash'],'process':a.process_round,'block':block,'order':order_index,
                             'calls':5,'seconds_per_call':wall,'cuda_event_seconds_per_call':event,'local_first_token_seconds_per_call':first,
                             'evidence_kind':'gpu_timing','mock':False})
        telemetry=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
        ledger.put(cell,identity,{'rows':rows,'telemetry_values':telemetry.stdout.strip(),'telemetry_fields':['temperature_C','SM_MHz','memory_MHz','power_W','whole_device_used_MiB','GPU_utilization_percent'],
                               'allocated_after_bytes':torch.cuda.memory_allocated(),'reserved_after_bytes':torch.cuda.memory_reserved(),
                               'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                               'load_seconds':rt.load_seconds,'scope':'Prefill + native last-position LM head; request cache reset each call; input IDs already GPU-resident; no reference/finite/profiler/capture hooks'})
        print(json.dumps({'process_round':a.process_round,'item':item['item_id'],'blocks_per_arm':5}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
