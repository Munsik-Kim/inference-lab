"""Bounded same-input readout diagnostic. Requires original case and private vectors.
No downloads/installations; no output may overwrite source evidence.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, importlib, json, math, sys, time
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-case',type=Path,required=True);p.add_argument('--original-run',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--work-dir',type=Path,required=True)
    a=p.parse_args();supp=Path(__file__).resolve().parents[1];case=a.original_case.resolve()
    if a.work_dir.resolve().is_relative_to(case):p.error('Private output must be outside original case')
    sys.path.insert(0,str(case))
    from src.storage import load,write_new,file_hash,digest
    from src.model_runtime import Runtime,tensor_hash
    from src.metrics import score_full,kl_logits
    from scripts.measure_scores import restore_base
    from scripts.run_case006 import gpu_gate
    protocol=load(supp/'protocol.json');assert file_hash(supp/'protocol.json')==(supp/'protocol.sha256').read_text().split()[0]
    for n,h in protocol['original_code_hashes'].items():assert file_hash(case/n)==h,n
    a.work_dir.mkdir(exist_ok=True,parents=True)
    if list(a.work_dir.glob('failure-*.json')):raise RuntimeError('Prior failure requires diagnosis, not an automatic retry')
    for d in ['cells','private','attempts','smoke']: (a.work_dir/d).mkdir(exist_ok=True)
    write_new(a.work_dir/'source-freeze.json',{'protocol_sha256':file_hash(supp/'protocol.json'),'runner_sha256':file_hash(Path(__file__)),'utc_unix':time.time()})
    gate=gpu_gate();write_new(a.work_dir/'gpu-gate.json',gate)
    if gate['status']!='RESOURCE_GATE_PASSED':raise RuntimeError('GPU_BUSY')
    import torch
    expected=load(case/'provenance/source_fingerprints.json')['files'];actual=[]
    for f in expected:
        if '/' in f['file']:
            import sageattention
            path=Path(sageattention.__file__).parent/f['file'].split('/')[-1]
        else:path=Path(importlib.import_module(f['file']).__file__)
        sha=file_hash(path);assert sha==f['sha256'],f['file'];actual.append({'file':f['file'],'sha256':sha})
    for f in load(case/'provenance/model.json')['files']:assert file_hash(a.snapshot/f['name'])==f['verified_sha256'],f['name']
    rt=Runtime(a.snapshot);head=rt.model.lm_head
    assert head.bias is None and tuple(head.weight.shape)==(151936,1024) and head.weight.dtype==torch.bfloat16
    w_hash=tensor_hash(head.weight); W=head.weight.detach().to(torch.float32).clone();W64=head.weight.detach()[[32,33,34,35]].double().cpu().numpy().copy()
    assert not W.requires_grad and W.data_ptr()!=head.weight.data_ptr()
    semantic=load(case/'provenance/integration_pass.json')
    write_new(a.work_dir/'environment.json',{'torch':torch.__version__,'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'device_capability':torch.cuda.get_device_capability(0),'head_shape':list(head.weight.shape),'head_dtype':str(head.weight.dtype),'weight_sha256':w_hash,'bias':None,'same_weight_embedding_storage':head.weight.data_ptr()==rt.model.model.embed_tokens.weight.data_ptr(),'tf32_before':torch.backends.cuda.matmul.allow_tf32,'fp32_precision_getter':torch.backends.cuda.matmul.fp32_precision,'source_files':actual,'runtime_seconds_to_load':rt.load_seconds})
    def project(h,native,gold,smoke=False):
        if h.shape!=(1,1024) or not np.isfinite(h).all() or not np.isfinite(native).all():raise ValueError('Invalid full hidden/native')
        ht=torch.tensor(h,device='cuda',dtype=torch.bfloat16).reshape(1,1,1024)
        if not np.array_equal(ht.double().cpu().numpy().reshape(1,1024),h):raise ValueError('Hidden not represented BF16')
        before=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=False
            with torch.inference_mode(),torch.autocast('cuda',enabled=False):
                # Native full-vocabulary same shape must reconstruct exactly.
                repeated=head(ht).double().cpu().numpy().reshape(-1)
                if not np.array_equal(repeated,native):raise ValueError('Stored hidden/native same-shape projection mismatch: '+str(float(np.max(abs(repeated-native)))))
                shadow=torch.nn.functional.linear(ht.float(),W,None)
                if not bool(torch.isfinite(shadow).all()):raise ValueError('Nonfinite full shadow projection')
                z=shadow.double().cpu().numpy().reshape(-1)
                roundtrip=shadow.to(torch.bfloat16).double().cpu().numpy().reshape(-1)
        finally:torch.backends.cuda.matmul.allow_tf32=before
        widened=native.astype(np.float32).astype(np.float64)
        if not np.array_equal(widened,native):raise ValueError('CAST_ONLY changed represented values')
        scores=score_full(z,[32,33,34,35],gold);native_score=score_full(native,[32,33,34,35],gold)
        f64=W64@h[0];f32=z[[32,33,34,35]];difference=f32-f64
        tol=protocol['readout']['fp32_fp64_validation'];ok=bool(np.allclose(f32,f64,atol=tol['atol'],rtol=tol['rtol']))
        controls={'C1_cast_only_exact':True,'native_projection_full_exact':True,'hidden_bf16_roundtrip_exact':True,
            'C2_full_equal':bool(np.array_equal(roundtrip,native)),'C2_full_mismatch_count':int(np.count_nonzero(roundtrip!=native)),
            'C2_full_max_abs':float(np.max(abs(roundtrip-native))),'C2_option_logits':roundtrip[[32,33,34,35]].tolist(),
            'C2_option_equal':bool(np.array_equal(roundtrip[[32,33,34,35]],native[[32,33,34,35]])),
            'C3_fp64_options':f64.tolist(),'C3_fp32_minus_fp64':difference.tolist(),'C3_max_abs':float(np.max(abs(difference))),
            'C3_gap_difference':float((np.sort(f32)[-1]-np.sort(f32)[-2])-(np.sort(f64)[-1]-np.sort(f64)[-2])),
            'C3_winner_disagreement':int(np.argmax(f32))!=int(np.argmax(f64)),'C3_tolerance_pass':ok,
            'full_native_finite':bool(np.isfinite(native).all()),'full_shadow_finite':bool(np.isfinite(z).all()),'full_hidden_finite':bool(np.isfinite(h).all()),'flags_restored':torch.backends.cuda.matmul.allow_tf32==before}
        if smoke:
            compensated=np.array([math.fsum(float(w)*float(v) for w,v in zip(row,h[0])) for row in W64])
            controls['C3_compensated_options']=compensated.tolist();controls['C3_compensated_max_abs']=float(max(abs(f64-compensated)))
        if not ok:raise ValueError('FP32/FP64 validation tolerance exceeded')
        return z,{'score':scores,'native_score':native_score,'controls':controls,'native_to_shadow_full_kl':kl_logits(native,z)}
    current='initialization'
    try:
        allsmoke={r['item_id']:r for r in load(case/'inputs/prompts.json')}; sg={r['base_id']:r for r in load(case/'inputs/gold.json')}
        baseline=None;priorid=None
        for cell in protocol['budget']['planned_smoke_cells']:
            current=cell;item_id,arm=cell.rsplit('--',1);item=allsmoke[item_id];gold='ABCD'.index(sg[item['base_id']]['gold'])
            if item_id!=priorid:baseline=None
            write_new(a.work_dir/'attempts'/('smoke-'+cell+'.json'),{'kind':'smoke_forward','cell':cell,'ordinal':len(list((a.work_dir/'attempts').glob('smoke-*')))+1})
            rec,priv=rt.diagnose(item,gold,arm,semantic,baseline)
            if not rec['validity_status']['valid']:raise ValueError('Smoke full validity failed')
            z,checks=project(priv['hidden']['final_norm_last'],priv['full_logits'],gold,True)
            write_new(a.work_dir/'smoke'/(cell+'.json'),{'cell':cell,'validity':rec['validity'],'readout':checks})
            if arm=='B':baseline=priv
            priorid=item_id
        native_rows=load(supp/'results/raw/native.json');native_map={(r['item_id'],r['arm']):r for r in native_rows}
        inputs=load(supp/'inputs/core.json');done=0;forwards=0
        for item in inputs:
            stage='standard-eval' if item['split']=='standard' else 'boundary-eval';original=a.original_run/stage
            baseline=restore_base(original/'private_vectors'/(item['item_id']+'--B.npz'))
            for arm in protocol['arms']:
                current=item['item_id']+'--'+arm;old=native_map[(item['item_id'],arm)];gold=old['gold_index']
                native=baseline['full_logits'] if arm=='B' else np.load(original/'private_vectors'/(current+'.npy'),allow_pickle=False)
                if arm=='B':priv=baseline;validity=old['validity'];origin='VERIFIED_STORED_HIDDEN'
                else:
                    if forwards>=protocol['budget']['ordinary_forwards_max']:raise RuntimeError('Forward budget exhausted')
                    write_new(a.work_dir/'attempts'/('core-'+current+'.json'),{'kind':'ordinary_prompt_forward','cell':current,'ordinal':forwards+1});forwards+=1
                    rec,priv=rt.diagnose(item,gold,arm,semantic,baseline);validity=rec['validity'];origin='BOUNDED_SAME_INPUT_REPLAY'
                    comparison={'full_native_exact':bool(np.array_equal(priv['full_logits'],native)),'options_exact':rec['option_logits']==old['option_logits'],'qkv_exact':rec['qkv_hashes']==old['qkv_hashes'],'full_validity':rec['validity_status'],'max_abs_native_difference':float(np.max(abs(priv['full_logits']-native)))}
                    write_new(a.work_dir/'private'/(current+'-integrity.json'),comparison)
                    if not comparison['full_native_exact'] or not comparison['options_exact'] or not comparison['qkv_exact'] or not rec['validity_status']['valid']:raise ValueError('FAILED_SAME_STATE_COMPARISON: '+json.dumps(comparison))
                z,checks=project(priv['hidden']['final_norm_last'],native,gold)
                with (a.work_dir/'private'/(current+'.npz')).open('xb') as stream:np.savez_compressed(stream,hidden=priv['hidden']['final_norm_last'],fp32_full_logits=z.astype(np.float32))
                public={k:old[k] for k in ['item_id','base_id','task','split','length','token_hash','arm','gold_index','label_ids']}
                public.update(readout='H_FP32',stage=stage,evidence_kind='posthoc_gpu_readout',origin=origin,validity=validity,readout_dtype='torch.float32',**checks)
                public.update(option_logits=checks['score']['option_logits'],full_lse=checks['score']['full_lse'],full_argmax=checks['score']['full_argmax'],private_vector_sha256=file_hash(a.work_dir/'private'/(current+'.npz')))
                write_new(a.work_dir/'cells'/(current+'.json'),public);done+=1
            print(json.dumps({'completed_cells':done,'ordinary_forwards':forwards,'item_id':item['item_id']}),flush=True)
        assert tensor_hash(head.weight)==w_hash and head.weight.dtype==torch.bfloat16
        write_new(a.work_dir/'completion.json',{'status':'COMPLETED','readout_cells':done,'ordinary_forwards':forwards,'smoke_forwards':len(protocol['budget']['planned_smoke_cells']),'stored_hidden_reused':done-forwards,'weight_unchanged':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    except Exception as exc:
        write_new(a.work_dir/('failure-'+str(time.time_ns())+'.json'),{'cell':current,'type':type(exc).__name__,'message':str(exc),'successful_verdict':False});raise

if __name__=='__main__':main()
