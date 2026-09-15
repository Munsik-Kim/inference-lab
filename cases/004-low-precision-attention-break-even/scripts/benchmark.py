"""Run one process round. Full operator is primary; no hidden prequantization."""
import argparse,gc,json,math,random,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import CASE,shapes,sha,write_json,verify_spec,fingerprint,utc
from src.metrics import unit_error
from src.timing import measured_block


def safe_error():
    import re
    text=traceback.format_exc()
    text=re.sub(r'/home/[^\s\"\']+', '<local-path>',text)
    return text


def finite_values(values):
    return [float(v) if math.isfinite(float(v)) else None for v in values]


def workload(shape,stage,work,manifest,torch):
    if shape['family']=='synthetic':
        seed=(400000 if stage=='dev' else 410000)+shapes().index(shape)
        g=torch.Generator(device='cpu').manual_seed(seed)
        x=[torch.randn(shape['batch'],h,shape['length'],shape['dim'],generator=g,dtype=torch.float32).to(torch.bfloat16) for h in [shape['hq'],shape['hkv'],shape['hkv']]]
        yield 'synthetic-'+str(seed),x,None,dict(seed=seed,source='independent standard Gaussian float32, rounded to BF16 on CPU',status='VALID')
    else:
        captures=json.loads((CASE/f'provenance/{stage}_capture.json').read_text())['traces']
        for d in manifest['documents']:
            if d['split']!=stage:continue
            meta=next(t for t in captures if t['document_id']==d['document_id'] and t['length']==shape['length'])
            path=work/'traces'/stage/meta['trace_file'];assert sha(path)==meta['trace_sha256']
            t=torch.load(path,map_location='cpu',weights_only=True)
            assert t['q'].shape==(1,16,shape['length'],128) and t['k'].shape==(1,8,shape['length'],128)
            assert t['meta']['all_query_positions'] and t['meta']['input_sha256']==d['prefix_sha256'][str(shape['length'])]
            yield d['document_id'],[t['q'],t['k'],t['v']],t['native'],meta


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['dev','confirmation'],required=True);p.add_argument('--round',type=int,required=True);p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--only',nargs='*');a=p.parse_args()
    if a.stage=='confirmation':spec,spec_sha=verify_spec()
    else:spec,spec_sha=None,None
    import numpy as np
    import torch
    from src.backends import Adapter,tensor_hash,assert_unmodified,quantized_kernel_call
    from src.reference import sampled_fp32
    from src.runtime import environment,profile_call,Sampler
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    out=CASE/'results'/a.stage/f'round-{a.round}.json';assert not out.exists(),'Preserve completed or failed process rounds'
    env=environment();env_sha=fingerprint(env)
    if spec:assert env_sha==spec['environment_fingerprint'],'Environment changed after freeze'
    free,total=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction(13*2**30/total)
    result=dict(evidence_kind='gpu_measurement',mock=False,stage=a.stage,round=a.round,started_at_utc=utc(),spec_sha256=spec_sha,environment_fingerprint=env_sha,environment=env,process_start=dict(free_bytes=free,total_bytes=total,allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved()),measurements=[],errors=[])
    if free<3*2**30:
        result['status']='GPU_BUSY';write_json(out,result);raise SystemExit(3)
    sampler=Sampler().start()
    manifest=json.loads((CASE/'inputs/manifest.json').read_text())
    adapters={n:Adapter(n) for n in ['default','flash','sage']}
    counts=dict(warmup=20,blocks=8,calls=10) if a.stage=='dev' else spec['timing']['counts']
    selected=spec['baseline_selection'] if spec else {}
    chosen=[s for s in shapes() if not a.only or s['id'] in a.only]
    assert all(s in shapes() for s in chosen)
    completed=True
    try:
        with torch.inference_mode():
            for shape in chosen:
                for did,host,native,meta in workload(shape,a.stage,a.work_dir,manifest,torch):
                    if meta['status']!='VALID':
                        result['errors'].append(dict(shape=shape,document_id=did,status='INVALID_CAPTURE',reason=meta['status']));continue
                    q,k,v=[t.cuda() for t in host];del host
                    hashes=[tensor_hash(t) for t in [q,k,v]]
                    n=shape['length'];positions=sorted(set(int(i*(n-1)//31) for i in range(32)))
                    ref=sampled_fp32(q,k,v,positions,shape['scale'],shape['causal']).numpy()
                    funcs={name:(lambda fn=fn:fn(q,k,v,causal=shape['causal'],scale=shape['scale'])) for name,fn in adapters.items()}
                    records={};outputs={}
                    for name,call in funcs.items():
                        rec=dict(evidence_kind='gpu_measurement',mock=False,status='OK',stage=a.stage,round=a.round,shape=shape,document_id=did,input_sha256=hashes,spec_sha256=spec_sha,environment_fingerprint=env_sha,backend=name,wall_ms=[],event_ms=[],order_positions=[],blocks=counts['blocks'],calls_per_block=counts['calls'],warmup_calls=counts['warmup'],positions=positions,errors_by_head=[])
                        try:
                            torch.cuda.reset_peak_memory_stats();cold=time.perf_counter();o=call();torch.cuda.synchronize();rec['first_call_seconds']=time.perf_counter()-cold
                            outputs[name]=o
                            assert_unmodified(hashes,[tensor_hash(t) for t in [q,k,v]])
                            observed=o[:,:,positions].float().cpu().numpy()
                            rec['errors_by_head']=[dict(head=h,**unit_error(observed[0,h],ref[0,h]),row_error_norms=finite_values(np.linalg.norm(observed[0,h].astype(np.float64)-ref[0,h].astype(np.float64),axis=-1)),row_reference_norms=finite_values(np.linalg.norm(ref[0,h].astype(np.float64),axis=-1))) for h in range(shape['hq'])]
                            if native is not None:rec['native_bf16_replay']=unit_error(o.float().cpu().numpy(),native.float().numpy())
                            if a.stage=='dev':
                                rec['profile']=profile_call(call)
                                if name in ['default','flash'] and not rec['profile']['fused_bf16']:raise RuntimeError('UNSUPPORTED: not a profiled fused BF16 backend')
                                if name=='sage' and not rec['profile']['low_precision']:raise RuntimeError('UNSUPPORTED: low-precision kernel execution not observed')
                            for _ in range(counts['warmup']):call()
                            torch.cuda.synchronize()
                            rec['warmup_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                        except Exception:
                            rec['status']='OOM' if 'out of memory' in str(sys.exc_info()[1]).lower() else 'UNSUPPORTED';rec['error']=safe_error()
                            if rec['status']=='OOM':raise
                        records[name]=rec
                    if records['default']['status']=='OK' and records['sage']['status']=='OK':
                        records['sage']['vs_default_bf16_by_head']=[dict(head=h,**unit_error(outputs['sage'][0,h,positions].float().cpu().numpy(),outputs['default'][0,h,positions].float().cpu().numpy())) for h in range(shape['hq'])]
                    outputs.clear()
                    if 'o' in locals():del o
                    gc.collect()
                    names=[name for name,rec in records.items() if rec['status']=='OK']
                    if not names:
                        result['measurements'].extend(records.values());continue
                    rng=random.Random(420000+a.round*1000+shapes().index(shape)*31+sum(did.encode()))
                    orders=[]
                    while len(orders)<counts['blocks']:
                        perm=names.copy();rng.shuffle(perm)
                        cycle=[perm[i:]+perm[:i] for i in range(len(perm))];rng.shuffle(cycle);orders.extend(cycle)
                    for bi,order in enumerate(orders[:counts['blocks']]):
                        for order_index,name in enumerate(order):
                            torch.cuda.reset_peak_memory_stats()
                            allocation_before=torch.cuda.memory_allocated()
                            wall,event=measured_block(funcs[name],torch.cuda.synchronize,lambda:torch.cuda.Event(enable_timing=True),counts['calls'])
                            rec=records[name];rec['wall_ms'].append(wall);rec['event_ms'].append(event);rec['order_positions'].append(order_index)
                            rec['peak_allocated_bytes']=max(rec.get('peak_allocated_bytes',0),torch.cuda.max_memory_allocated());rec['peak_reserved_bytes']=max(rec.get('peak_reserved_bytes',0),torch.cuda.max_memory_reserved())
                            rec['incremental_peak_allocated_bytes']=max(rec.get('incremental_peak_allocated_bytes',0),torch.cuda.max_memory_allocated()-allocation_before)
                            if rec['peak_allocated_bytes']>13*2**30:raise RuntimeError('MEMORY_LIMIT_EXCEEDED')
                    # Auxiliary kernel-only measurements follow all complete-call blocks.
                    # Their prequantized tensors never remain live during primary timing.
                    if records['sage']['status']=='OK':
                        try:
                            internal,info=quantized_kernel_call(q,k,v,shape['causal'],shape['scale'])
                            isolated=internal();complete=funcs['sage']();torch.cuda.synchronize()
                            if not torch.equal(isolated,complete):raise ValueError('NOT_ISOLATABLE: output differs from complete wrapper')
                            del isolated,complete
                            for _ in range(counts['warmup']):internal()
                            kr=dict(evidence_kind='gpu_measurement',mock=False,status='OK',stage=a.stage,round=a.round,shape=shape,document_id=did,backend='kernel_only',input_sha256=hashes,spec_sha256=spec_sha,environment_fingerprint=env_sha,wall_ms=[],event_ms=[],blocks=counts['blocks'],calls_per_block=counts['calls'],warmup_calls=counts['warmup'],public_wrapper_bitwise_equal=True,internal_details=info,order_scope='After complete-operator blocks; auxiliary, not the speedup gate')
                            for bi in range(counts['blocks']):
                                wall,event=measured_block(internal,torch.cuda.synchronize,lambda:torch.cuda.Event(enable_timing=True),counts['calls'])
                                kr['wall_ms'].append(wall);kr['event_ms'].append(event)
                            records['kernel_only']=kr
                        except Exception:
                            if 'out of memory' in str(sys.exc_info()[1]).lower():raise
                            records['sage']['kernel_only_status']='NOT_ISOLATABLE';records['sage']['kernel_only_error']=safe_error()
                    assert_unmodified(hashes,[tensor_hash(t) for t in [q,k,v]])
                    for name,rec in records.items():
                        rec['inputs_unchanged']=True;result['measurements'].append(rec)
                    print(json.dumps({'stage':a.stage,'round':a.round,'shape':shape['id'],'document':did,'status':{k:v['status'] for k,v in records.items()},'wall_median_ms':{k:float(np.median(v['wall_ms'])) for k,v in records.items() if v['wall_ms']}}),flush=True)
                    # The next workload starts without prior device tensors/outputs.
                    funcs.clear();outputs.clear();del q,k,v,ref,native,records
                    if 'internal' in locals():del internal
                    if 'isolated' in locals():del isolated
                    gc.collect();torch.cuda.empty_cache()
    except Exception:
        completed=False;result['errors'].append(dict(shape=shape if 'shape' in locals() else None,document_id=did if 'did' in locals() else None,status='OOM' if 'out of memory' in str(sys.exc_info()[1]).lower() else 'FAILED',error=safe_error()))
    finally:
        result['status']='COMPLETED' if completed else 'FAILED';result['completed_at_utc']=utc();result['gpu_samples']=sampler.stop();write_json(out,result)
    if not completed:raise SystemExit(2)
if __name__=='__main__':main()
