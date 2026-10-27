"""Same P4 dense/Marlin cores, projection-only. No model snapshot is opened."""
import argparse,importlib.util,sys,time
from pathlib import Path
import numpy as np
from common import read,write,identity,array_hash
from arithmetic_v2 import decode,numeric_equal,ulp_metrics
from prepare_v2 import load_artifact
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['B','C'],required=True);p.add_argument('--layer',type=int,choices=[0,25],required=True);p.add_argument('--block',type=int,choices=[0,1,2],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;loc=read(r/'references/locations.json');p4=Path(loc['p4_root']);case=Path(loc['original_case'])
    if a.phase=='C' and read(r/'summary/B_decision.json')['branch']!='D':raise RuntimeError('C_BRANCH_NOT_AUTHORIZED')
    data=read(r/'inputs_manifest/prepared.json');fixtures=[x for x in data['fixtures'] if x['phase']==a.phase and x['layer']==a.layer]
    q,s,artifact,prior=load_artifact(r,a.layer)
    oldworker=module('p5_p4_worker',p4/'source/gpu_worker.py');rt,layout=oldworker.original_runtime(case)
    import torch
    import importlib.metadata
    flags=rt.flags();assert flags==prior['flags'],'P4_FLAGS_DRIFT';assert torch.cuda.get_device_capability()==(12,0)
    a.output.mkdir(parents=True,exist_ok=False)
    with torch.inference_mode():
        torch.cuda.init();torch.cuda.synchronize();base=rt.memory();torch.cuda.reset_peak_memory_stats()
        linear=torch.nn.Linear(1024,2048,bias=False,dtype=torch.bfloat16,device='cuda');dense=rt.Dense(linear,q,'D1');packed=rt.Packed(linear,q);del linear
        assert dense.bias is None and packed.bias is None
        def bits(t):return t.detach().cpu().contiguous().view(torch.uint16).numpy().copy()
        def tensor(raw):return torch.from_numpy(np.asarray(raw).copy()).view(torch.bfloat16).cuda()
        def core(method,z):
            if method=='D1':return torch.nn.functional.linear(z,dense.weight,None)
            return packed.ops.marlin_gemm(a=z,c=None,b_q_weight=packed.codes,b_bias=None,b_scales=packed.scales,a_scales=None,global_scale=None,b_zeros=packed.empty,g_idx=packed.empty,perm=packed.empty,workspace=packed.workspace,b_q_type=packed.qtype,size_m=z.shape[0],size_n=2048,size_k=1024,is_k_full=True,use_atomic_add=False,use_fp32_reduce=True,is_zp_float=False)
        dw=bits(dense.weight);assert array_hash(dw)==prior['actual_D1_weight_bits_sha256'];del dw
        ps=bits(packed.scales);plain=np.empty_like(ps).reshape(-1);flat=ps.reshape(-1)
        for offset in range(64):
            row,col=divmod(offset,8);plain[offset::64]=flat[(col*8+row)::64]
        assert array_hash(plain.reshape(ps.shape))==prior['actual_unpermuted_scale_array_sha256']
        pack=layout.pack_marlin_input(q.codes).view(np.uint32)
        for n in range(8):assert np.array_equal(((pack>>(4*n))&15).astype(np.int16)-8,q.codes[:,n::8].T)
        controls=[];kernels={}
        if a.block==0:
            # Reuse the P4 stored pre-beta control, with its actual post-beta Z.
            source=next(x for x in read(p4/'inputs/projection_input_manifest.json')['inputs'] if x['kind']=='ACTUAL_DEV' and a.layer in x['layers'] and x.get('prebeta_fixture'))
            x=tensor(np.load(source['prebeta_fixture'],allow_pickle=False));z=tensor(np.load(source['path'],allow_pickle=False));assert torch.equal(z,(x.float()*dense.beta_inverse).bfloat16());assert not torch.equal(z,(z.float()*dense.beta_inverse).bfloat16())
            for method,mod in [('D1',dense),('P',packed)]:
                aa=mod(x);bb=core(method,z);torch.cuda.synchronize();assert torch.equal(aa,bb),'MODULE_CORE_BETA';controls.append({'method':method,'calls':2,'module_core_equal':True});del aa,bb
            from torch.profiler import profile,ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:y=core('P',z);torch.cuda.synchronize()
            del y;trace=a.output/'marlin_trace.json';prof.export_chrome_trace(str(trace));names=sorted({e.get('name','') for e in read(trace)['traceEvents'] if e.get('cat')=='kernel'});assert any('marlin' in x.lower() for x in names),'MARLIN_NOT_OBSERVED';kernels={'op':'torch.ops._C.marlin_gemm','kernel_names':names,'geometry':[512,1024,2048],'fallback':False};del x,z
        rows=[];out=a.output/'operator_outputs';out.mkdir()
        for f in fixtures:
            assert identity(f['path'])==f['identity'];zraw=np.load(f['path'],allow_pickle=False);assert array_hash(zraw)==f['array_sha256']
            if a.phase=='B':assert np.all(np.count_nonzero(decode(zraw),axis=1)<=1),'ONEHOT_CONTRACT'
            z=tensor(zraw);order=data['method_orders'][f"{a.phase}:{a.layer}:{a.block}:{f['fixture']}"]
            for method in order:
                torch.cuda.synchronize();start=time.perf_counter_ns();y=core(method,z);torch.cuda.synchronize();elapsed=time.perf_counter_ns()-start
                assert y.shape==(512,2048) and y.dtype==torch.bfloat16 and torch.isfinite(y).all();b=bits(y)
                row={'phase':a.phase,'layer':a.layer,'block':a.block,'fixture':f['fixture'],'method':method,'method_order':order,'input_sha256':array_hash(zraw),'output_sha256':array_hash(b),'output_dtype':'BF16_BITS_UINT16','shape':[512,2048],'valid_rows':f['valid_rows'],'host_diagnostic_ns':elapsed}
                if a.block==0:
                    dest=out/(f['fixture']+'_'+method+'.npy');np.save(dest,b,allow_pickle=False);row.update(output_path=str(dest),output_identity=identity(dest))
                else:
                    first=r/f"records/gpu-{a.phase}-l{a.layer}-b0/data/operator_outputs"/(f['fixture']+'_'+method+'.npy');old=np.load(first,allow_pickle=False);row.update(first_sha256=array_hash(old),bit_identical_to_first=np.array_equal(old,b),numeric_repeat=ulp_metrics(old,b))
                    if not np.array_equal(old,b):
                        ix=np.argwhere(old!=b)[:256];np.savez(out/(f['fixture']+'_'+method+'_mismatch.npz'),indices=ix,first=old[tuple(ix.T)],current=b[tuple(ix.T)])
                rows.append(row);del y,b
            assert array_hash(bits(z))==f['array_sha256'];del z
            assert rt.memory()['peak_allocated']<=2**31,'ALLOCATED_CAP'
        write(a.output/'receipt.json',{'phase':a.phase,'layer':a.layer,'block':a.block,'rows':rows,'main_core_calls':len(rows),'control_calls':sum(x['calls'] for x in controls),'profile_calls':1 if a.block==0 else 0,'controls':controls,'kernel':kernels,'flags':flags,'baseline':base,'memory':rt.memory(),'model_forwards':0,'artifact':artifact,'D1_weight_bits_sha256':prior['actual_D1_weight_bits_sha256'],'scale_array_sha256':prior['actual_unpermuted_scale_array_sha256'],'environment':{'python':sys.version,'executable':sys.executable,'versions':{k:importlib.metadata.version(k) for k in ['torch','numpy','transformers','vllm']},'torch_CUDA':torch.version.cuda,'device':torch.cuda.get_device_name(),'SM':torch.cuda.get_device_capability(),'torch_import_path':torch.__file__}})
        print({'phase':a.phase,'layer':a.layer,'block':a.block,'core_calls':len(rows),'peak':rt.memory()['peak_allocated']})
if __name__=='__main__':main()
