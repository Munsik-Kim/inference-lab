"""Projection-only GPU operations; imports Torch only after CLI/identity checks."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
from common import read,write,identity,array_hash

def original_runtime(case):
    # Explicit frozen modules; no ACTIVE_RUN and no writes to the original tree.
    saved={k:sys.modules.get(k) for k in ['common','layout']}
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
    try:
        original_common=load('case021_original_common',case/'source/common.py');sys.modules['common']=original_common
        layout=load('case021_original_layout',case/'source/layout.py');sys.modules['layout']=layout
        rt=load('case021_original_runtime',case/'source/runtime.py')
    finally:
        for key,value in saved.items():
            if value is None:sys.modules.pop(key,None)
            else:sys.modules[key]=value
    return rt,layout

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--job',choices=['inputs','operator'],required=True)
    parser.add_argument('--layer',type=int);parser.add_argument('--block',type=int);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    root=a.root;loc=read(root/'references/locations.json');case=Path(loc['original_case']);private=Path(loc['original_private'])
    expected=read(root/'references/external_inputs.json')
    for path,row in expected['existing_runtime_sources'].items():
        if identity(path)!=row:raise ValueError('RUNTIME_SOURCE_MISMATCH')
    for f in ['common.py','layout.py','runtime.py']:
        original=read(case/'PUBLIC_MANIFEST.json')['files']['source/'+f]
        if identity(case/'source'/f)!=original:raise ValueError('ORIGINAL_RUNTIME_CHANGED')
    import torch
    import importlib.metadata
    from numerics import weight_vector,bfdecode,bfbits32,bfround64
    from fixtures import inverse_scales
    rt,layout=original_runtime(case);flags=rt.flags()
    if torch.cuda.get_device_capability()!=(12,0):raise RuntimeError('GPU_SM_MISMATCH')
    original_locations=read(private/'locations.json');model=Path(original_locations['model']);config=read(model/'config.json');manifest=read(case/'inputs/manifest.json')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.cuda.init();torch.cuda.synchronize();baseline=rt.memory();torch.cuda.reset_peak_memory_stats()
    environment={'python':sys.version,'executable':sys.executable,'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','vllm','numpy']},'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),'flags':flags,'imports':{'torch':torch.__file__,'HF_qwen3':rt.hf.__file__},'baseline':baseline}
    def check_peak():
        current=rt.memory()
        if current['peak_allocated']>2**31:raise RuntimeError('PROJECTION_ALLOCATED_CAP')
        return current
    def tensor(bits):return torch.from_numpy(np.asarray(bits).copy()).view(torch.bfloat16).cuda()
    def bits(t):return t.detach().cpu().contiguous().view(torch.uint16).numpy().copy()
    with torch.inference_mode():
        if a.job=='inputs':
            target=root/'inputs/actual';target.mkdir(exist_ok=False);rows=[];norm_checks=[]
            for layer in [0,25,27]:
                artifact=next(r for r in expected['artifacts'] if r['projection']==f'model.layers.{layer}.self_attn.q_proj')
                if identity(artifact['path'])!={k:artifact[k] for k in ['bytes','sha256']}:raise ValueError('ARTIFACT_CHANGED')
                q=layout.load_rpi4(artifact['path']);key=f'model.layers.{layer}.input_layernorm.weight';w,whash=weight_vector(model/'model.safetensors',key,1024)
                if whash!=expected['used_norm_vectors'][key]['sha256']:raise ValueError('NORM_WEIGHT_CHANGED')
                norm=rt.hf.Qwen3RMSNorm(1024,eps=config['rms_norm_eps']).cuda().bfloat16();norm.weight.copy_(torch.from_numpy(w.astype(np.float32)).cuda().bfloat16())
                beta=torch.from_numpy(q.beta_inv).cuda().float()
                for doc in manifest['document_order']['DEV']:
                    source=private/'contexts/DEV'/doc/('input.npy' if layer==0 else f'layer{layer}_input.npy')
                    if identity(source)!=read(root/'references/input_identity.json')['files'][str(source)]:raise ValueError('SAVED_BLOCK_INPUT_CHANGED')
                    x=tensor(np.load(source,allow_pickle=False));xnorm=norm(x)
                    xf=x.float();explicit=(xf*torch.rsqrt(xf.pow(2).mean(-1,keepdim=True)+config['rms_norm_eps'])).bfloat16()*norm.weight
                    if not torch.equal(xnorm,explicit):raise ValueError('NATIVE_NORM_REPLAY')
                    z=(xnorm.float()*beta).bfloat16().reshape(512,1024).contiguous();zbits=bits(z)
                    dest=target/f'layer{layer}_{doc}_Z.npy';np.save(dest,zbits,allow_pickle=False)
                    prebeta=None
                    if doc==manifest['document_order']['DEV'][0]:
                        f=target/f'layer{layer}_{doc}_prebeta.npy';np.save(f,bits(xnorm.reshape(512,1024)),allow_pickle=False);prebeta=str(f)
                    rows.append({'layer':layer,'fixture':'ACTUAL_DEV_'+doc,'kind':'ACTUAL_DEV','document_id':doc,'path':str(dest),'identity':identity(dest),'input_source':str(source),'input_source_identity':identity(source),'native_input_norm_weight_sha256':whash,'beta_source':artifact['sha256'],'prebeta_fixture':prebeta,'shape':[512,1024],'encoding':'BF16_BITS_UINT16','array_sha256':array_hash(zbits),'valid_rows':512,'norm64':float(np.linalg.norm(bfdecode(zbits).astype(np.float64))),'min':float(bfdecode(zbits).min()),'max':float(bfdecode(zbits).max())})
                    norm_checks.append({'layer':layer,'document_id':doc,'native_vs_explicit_bit_identical':True})
                    del x,xnorm,xf,explicit,z
                del norm,beta;check_peak()
            write(a.output/'input_preparation.json',{'rows':rows,'native_norm_replay_checks':norm_checks,'full_model_forwards':0,'native_norm_calls':24,'explicit_norm_control_calls':24,'beta_transports':24,'GEMM_calls':0,'environment':environment,'memory':check_peak(),'original_checkpoint_current_full_hash':'NOT_REHASHED; selected norm vectors checked'})
            print(json.dumps({'status':'COMPLETE','Z_inputs':len(rows),'full_forwards':0}));return
        if a.layer not in [0,25,27] or a.block not in [0,1,2]:raise ValueError('PROJECTION_BLOCK')
        projection=next(r for r in expected['artifacts'] if r['projection']==f'model.layers.{a.layer}.self_attn.q_proj')
        if identity(projection['path'])!={k:projection[k] for k in ['bytes','sha256']}:raise ValueError('ARTIFACT_IDENTITY')
        q=layout.load_rpi4(projection['path']);O,I=q.shape
        dummy=torch.nn.Linear(I,O,bias=False,dtype=torch.bfloat16,device='cuda')
        dense=rt.Dense(dummy,q,'D1');packed=rt.Packed(dummy,q);del dummy
        if dense.bias is not None or packed.bias is not None:raise ValueError('BIAS_MISMATCH')
        def core(method,z):
            if method=='D1':return torch.nn.functional.linear(z,dense.weight,None)
            return packed.ops.marlin_gemm(a=z,c=None,b_q_weight=packed.codes,b_bias=None,b_scales=packed.scales,a_scales=None,global_scale=None,b_zeros=packed.empty,g_idx=packed.empty,perm=packed.empty,workspace=packed.workspace,b_q_type=packed.qtype,size_m=z.shape[0],size_n=O,size_k=I,is_k_full=True,use_atomic_add=False,use_fp32_reduce=True,is_zp_float=False)
        scale_bits=inverse_scales(bits(packed.scales))
        expected_scale=bfbits32((q.alpha_inv.astype(np.float32)*q.gamma.astype(np.float32)).T.copy())
        if not np.array_equal(scale_bits,expected_scale):raise ValueError('FORMAT_MISMATCH_SCALE')
        scale=bfdecode(scale_bits.T.copy()).astype(np.float64)
        exact=q.codes.astype(np.float64).reshape(O,I//64,64)*scale[...,None]
        if not np.array_equal(exact,exact.astype(np.float32).astype(np.float64)):raise ValueError('CODE_SCALE_PRODUCT_NOT_EXACT_FP32')
        expected_dense=bfbits32(bfround64(exact.reshape(O,I)));actual_dense=bits(dense.weight)
        if not np.array_equal(expected_dense,actual_dense):raise ValueError('FORMAT_MISMATCH_DENSE_BF16')
        # Independent bit extraction checks the input packing, not just a helper roundtrip.
        pack=layout.pack_marlin_input(q.codes).view(np.uint32)
        for nibble in range(8):
            if not np.array_equal(((pack>>(4*nibble))&15).astype(np.int16)-8,q.codes[:,nibble::8].T):raise ValueError('OFFSET_PACKING')
        contract={'projection':projection,'group':64,'bias':'NONE','actual_unpermuted_scale_array_sha256':array_hash(scale_bits),'actual_D1_weight_bits_sha256':array_hash(actual_dense),'codes_array_sha256':array_hash(q.codes),'exact_product_in_FP32':True,'independent_BF16_rounding_matches_actual_D1':True,'beta_application':'Only original module(x) applies beta; core consumes Z directly','shape':[O,I],'flags':flags}
        del exact,actual_dense,expected_dense,scale
        inputs=read(root/'inputs/projection_input_manifest.json');selected=[r for r in inputs['inputs'] if a.layer in r['layers']]
        if len(selected)!=15:raise ValueError('FIXTURE_COUNT')
        controls=[];kernel={};rows=[];outdir=a.output/'operator_outputs';outdir.mkdir()
        if a.block==0:
            actual=next(r for r in selected if r['kind']=='ACTUAL_DEV')
            x=tensor(np.load(actual['prebeta_fixture'],allow_pickle=False));z=tensor(np.load(actual['path'],allow_pickle=False))
            explicit_z=(x.float()*dense.beta_inverse).bfloat16()
            if not torch.equal(z,explicit_z) or torch.equal(z,(z.float()*dense.beta_inverse).bfloat16()):raise ValueError('BETA_TRANSPORT_FIXTURE')
            for method,module in [('D1',dense),('P',packed)]:
                module_y=module(x);core_y=core(method,z);torch.cuda.synchronize()
                if not torch.equal(module_y,core_y):raise ValueError('MODULE_CORE_MISMATCH')
                controls.append({'method':method,'module_core_bit_equal':True,'calls':2});del module_y,core_y
            from torch.profiler import profile,ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
                prof_y=core('P',z);torch.cuda.synchronize()
            del prof_y
            trace=a.output/'marlin_trace.json';prof.export_chrome_trace(str(trace));events=read(trace)['traceEvents']
            names=sorted({e.get('name','') for e in events if e.get('cat')=='kernel'})
            marlin=[name for name in names if 'marlin' in name.lower()]
            if not marlin:raise RuntimeError('ACTUAL_MARLIN_NOT_OBSERVED')
            kernel={'kernel_names':names,'marlin_kernels':marlin,'M':512,'O':O,'I':I,'profile_P_calls':1,'actual_op':'torch.ops._C.marlin_gemm','fallback':False}
            del x,z,explicit_z
        for index,entry in enumerate(selected):
            if identity(entry['path'])!=entry['identity']:raise ValueError('INPUT_HASH')
            raw=np.load(entry['path'],allow_pickle=False);z=tensor(raw).contiguous()
            if list(z.shape)!=[512,I] or array_hash(raw)!=entry['array_sha256']:raise ValueError('INPUT_SHAPE_HASH')
            order=inputs['method_orders'][f'{a.layer}:{a.block}:{entry["fixture"]}'];outputs={}
            for method in order:
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();t=time.perf_counter_ns();begin.record();y=core(method,z);end.record();end.synchronize();host=time.perf_counter_ns()-t;event=begin.elapsed_time(end)
                if y.dtype!=torch.bfloat16 or list(y.shape)!=[512,O] or not torch.isfinite(y).all():raise ValueError('OUTPUT_CONTRACT')
                ybits=bits(y);value=bfdecode(ybits).astype(np.float64)
                result={'layer':a.layer,'block':a.block,'fixture':entry['fixture'],'kind':entry['kind'],'method':method,'method_order':order,'M':512,'I':I,'O':O,'valid_rows':entry['valid_rows'],'input_sha256':entry['array_sha256'],'input_file_sha256':entry['identity']['sha256'],'output_array_sha256':array_hash(ybits),'output_dtype':'torch.bfloat16','host_call_ns':host,'event_elapsed_ms':event,'energy64':float(np.sum(value*value)),'min':float(value.min()),'max':float(value.max())}
                if a.block==0:
                    out=outdir/(entry['fixture']+'_'+method+'.npy');np.save(out,ybits,allow_pickle=False);result['output_path']=str(out);result['output_file']=identity(out)
                else:
                    first=root/'records'/f'gpu-q{a.layer}-b0'/'data/operator_outputs'/(entry['fixture']+'_'+method+'.npy')
                    reference=np.load(first,allow_pickle=False);delta=value-bfdecode(reference).astype(np.float64);different=ybits!=reference
                    result.update(first_block_output_sha256=array_hash(reference),bit_identical_to_first=bool(not different.any()),different_elements=int(different.sum()),repeat_SSE=float(np.sum(delta*delta)),repeat_max_abs=float(np.max(np.abs(delta))))
                    if different.any():
                        coordinates=np.argwhere(different)[:256];np.savez(outdir/(entry['fixture']+'_'+method+'_mismatch.npz'),coordinates=coordinates,first=reference[tuple(coordinates.T)],current=ybits[tuple(coordinates.T)])
                rows.append(result);del y,ybits,value
            if array_hash(bits(z))!=entry['array_sha256']:raise ValueError('INPUT_MUTATED')
            del z;check_peak()
        write(a.output/'operator_receipt.json',{'layer':a.layer,'block':a.block,'contract':contract,'rows':rows,'fixture_count':len(selected),'main_core_calls':len(rows),'control_core_or_module_calls':sum(r['calls'] for r in controls),'controls':controls,'kernel_evidence':kernel,'profile_calls':1 if a.block==0 else 0,'environment':environment,'memory':check_peak(),'full_model_forwards':0,'retention':'First block all BF16 bits, other blocks hashes and bounded mismatch coordinates'})
        print(json.dumps({'status':'COMPLETE','layer':a.layer,'block':a.block,'core_calls':len(rows),'peak_allocated':check_peak()['peak_allocated']}))

if __name__=='__main__':main()
