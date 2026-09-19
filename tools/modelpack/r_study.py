"""Fixed Case007 structures; native teacher targets and DEV-only ridge selection."""
from __future__ import annotations
import argparse
import copy
import gc
from pathlib import Path
import time
import numpy as np
from .common import CASE, digest, new_external, read, require, write
from .artifact import slice_mlp, tensor_hash, save_dense, load_dense
from .runtime import gpu_guard, diagnostic, replace
from .numerics import norm_record, score, fit_ridge, full_kl, paired


def config():return read(CASE/'configs/build_freeze.json')['R']


def retained(name: str) -> list[int]:
    removed=config()['removed'][name]
    return [i for i in range(3072) if i//192 not in removed]


def positions(length: int) -> list[int]:
    require(length>=32,'Insufficient positions')
    return [i*(length-1)//31 for i in range(32)]


def array(t):return t.detach().float().cpu().numpy().astype(np.float64)


def source_model(snapshot):
    from transformers import AutoModelForCausalLM
    import torch
    cfg=config();require(snapshot.name==cfg['revision'],'Wrong source revision')
    model=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
                 dtype=torch.bfloat16,attn_implementation='sdpa').eval().cuda()
    c=model.config
    require((c.num_hidden_layers,c.hidden_size,c.intermediate_size,c.hidden_act)==(28,1024,3072,'silu'),'Unexpected architecture')
    return model


def smoke(snapshot: Path, output: Path):
    import torch
    guard=gpu_guard();out=new_external(output);model=source_model(snapshot);original=model.model.layers[13].mlp
    rows=read(CASE/'inputs/R/smoke.json');records=[]
    for row in rows:
        z,capture,valid=diagnostic(model,row);x=capture['input'];y=capture['output']
        item={'id':row['id'],'validity':valid,'structures':{},'native_score':score(z,row['label_ids'],row['gold']),
              'native_full_logits_hash':tensor_hash(torch.tensor(z).to(torch.bfloat16).reshape(1,1,-1))}
        for name in config()['removed']:
            module=slice_mlp(original,retained(name))
            with torch.inference_mode():
                h=module.act_fn(module.gate_proj(x))*module.up_proj(x)
                sliced=module.down_proj(h)
                h_full=original.act_fn(original.gate_proj(x))*original.up_proj(x)
                mask=torch.zeros_like(h_full);mask[...,retained(name)]=h_full[...,retained(name)]
                masked=original.down_proj(mask)
            eq=norm_record(array(sliced),array(masked))
            require(eq['relative_error'] is not None and eq['relative_error']<=.01,'Sliced/masked smoke tolerance failed')
            with replace(model.model.layers[13],module): _,cc,v=diagnostic(model,row)
            require(cc['input_hash']==capture['input_hash'],'Different intervention input')
            require(torch.equal(cc['output'],sliced),'Integrated/same-shape local MLP mismatch')
            item['structures'][name]={'slice_mask':eq,'integrated_output_exact':True,
                    'activation_vs_full_slice':norm_record(array(h),array(h_full[...,retained(name)])),
                    'validity':v,'width':len(retained(name))}
            del module
        records.append(item)
    write(out/'smoke.json',{'status':'PASS','guard':guard,'records':records,
                          'tolerance_frozen_before_data':{'slice_mask_relative':.01,'integrated_exact':True}})
    print('R_SMOKE_PASS',len(records))


def calibrate(snapshot: Path, output: Path):
    import torch
    gpu_guard();out=new_external(output);model=source_model(snapshot);original=model.model.layers[13].mlp
    rows=read(CASE/'inputs/R/calibration.json');stats={name:[] for name in config()['removed']};ys=[];records=[]
    start=time.perf_counter()
    for row in rows:
        z,capture,valid=diagnostic(model,row);x=capture['input'];y=capture['output'];ps=positions(row['length'])
        ys.append(array(y[0,ps]));entry={'id':row['id'],'task':row['task'],'token_hash':row['token_hash'],
               'validity':valid,'baseline_score':score(z,row['label_ids'],row['gold']),'local':{}}
        for name in stats:
            module=slice_mlp(original,retained(name))
            with torch.inference_mode():
                h=module.act_fn(module.gate_proj(x))*module.up_proj(x);pred=module.down_proj(h)
            private=out/'activation_cells'/name;private.mkdir(parents=True,exist_ok=True)
            np.savez(private/(row['id']+'.npz'),H=h.float().cpu().numpy(),Y=y.float().cpu().numpy())
            stats[name].append(array(h[0,ps]))
            entry['local'][name]=norm_record(array(pred[0,ps]),array(y[0,ps]));del module
        records.append(entry)
        if len(records)%16==0:print('CALIBRATION',len(records),flush=True)
    y=np.concatenate(ys);np.save(out/'teacher.npy',y,allow_pickle=False)
    write(out/'calibration.json',records)
    original=original.cpu();del model;gc.collect();torch.cuda.empty_cache()
    fits={};grid=config()['eta_grid']
    for name,batches in stats.items():
        h=np.concatenate(batches);g=h.T@h/len(h);b=y.T@h/len(h)
        w0=array(original.down_proj.weight[:,retained(name)])
        np.savez(out/(name+'-statistics.npz'),G=g,B=b,W0=w0,H=h)
        fits[name]={}
        for eta in grid:
            fitted,info=fit_ridge(g,b,w0,eta)
            # Native evaluation uses represented BF16 W; FP64 solution is retained privately.
            rounded=torch.from_numpy(fitted).to(torch.bfloat16).float().numpy()
            np.save(out/f'{name}-eta-{eta}.npy',rounded,allow_pickle=False)
            np.save(out/f'{name}-eta-{eta}-fp64.npy',fitted,allow_pickle=False)
            fits[name][str(eta)]={**info,'rounding':norm_record(rounded,fitted),'sampled_vectors':len(h)}
            print('FIT',name,eta,flush=True)
    write(out/'fit.json',{'status':'PASS','fits':fits,'source_input_hash':digest(rows),
                          'native_capture_and_fit_seconds':time.perf_counter()-start})


def develop(snapshot: Path, calibration: Path, output: Path):
    import torch
    gpu_guard();out=new_external(output);require(read(calibration/'fit.json')['status']=='PASS','Incomplete calibration')
    model=source_model(snapshot);original=model.model.layers[13].mlp;rows=read(CASE/'inputs/R/development.json');records=[]
    for row in rows:
        z,capture,valid=diagnostic(model,row);x=capture['input'];ps=positions(row['length']);y=array(capture['output'][0,ps])
        entry={'id':row['id'],'task':row['task'],'validity':valid,'baseline_score':score(z,row['label_ids'],row['gold']),'local':{}}
        for name in config()['removed']:
            module=slice_mlp(original,retained(name));entry['local'][name]={}
            with torch.inference_mode():
                h=module.act_fn(module.gate_proj(x))*module.up_proj(x)
                entry['local'][name]['uncorrected']=norm_record(array(module.down_proj(h)[0,ps]),y)
                for eta in config()['eta_grid']:
                    w=np.load(calibration/f'{name}-eta-{eta}.npy',allow_pickle=False)
                    module.down_proj.weight.copy_(torch.from_numpy(w).to(device='cuda',dtype=torch.bfloat16))
                    pred=module.down_proj(h)
                    entry['local'][name][str(eta)]=norm_record(array(pred[0,ps]),y)
            del module
        records.append(entry)
        if len(records)%12==0:print('DEVELOPMENT',len(records),flush=True)
    means={}
    for eta in config()['eta_grid']:
        per_structure=[]
        for name in config()['removed']:
            per_structure.append(np.mean([r['local'][name][str(eta)]['squared_error']/r['local'][name][str(eta)]['reference_squared_norm'] for r in records]))
        means[str(eta)]=float(np.mean(per_structure))
    selected=min(config()['eta_grid'],key=lambda x:(means[str(x)],-x))
    write(out/'development.json',records)
    write(out/'selection.json',{'eta':selected,'rule':config()['eta_rule'],'mean_relative_squared_error':means,
                              'input_hash':digest(rows),'heldout_accessed':False})
    print('COMMON_ETA',selected,flush=True)


def export(snapshot: Path, calibration: Path, development: Path, output: Path):
    import torch
    gpu_guard();out=new_external(output);model=source_model(snapshot);original=model.model.layers[13].mlp
    cfg=config();eta=read(development/'selection.json')['eta'];reports={}
    source={'model':cfg['model'],'revision':cfg['revision']}
    # Keep only the one full model; standalone files are private.
    reports['R-B']=save_dense(model,out/'R-B',source,tokenizer_source=snapshot)
    for name in cfg['removed']:
        for repaired in [False,True]:
            arm=name+('-R' if repaired else '')
            module=slice_mlp(original,retained(name))
            if repaired:
                w=np.load(calibration/f'{name}-eta-{eta}.npy',allow_pickle=False)
                with torch.no_grad():module.down_proj.weight.copy_(torch.from_numpy(w).to(device='cuda',dtype=torch.bfloat16))
            with replace(model.model.layers[13],module):
                reports[arm]=save_dense(model,out/arm,source,retained(name),13,snapshot,
                            {'method':'fixed_structured_prune'+('_ridge_repair' if repaired else ''),
                             'removed_groups':cfg['removed'][name],'eta':eta if repaired else None,
                             'calibration_manifest':read(CASE/'inputs/R/manifest.json')['all_input_hash']})
            del module;print('EXPORTED',arm,flush=True)
    # Verify the only changed tensors between same-size arms are the down weights.
    for name in cfg['removed']:
        a,b=reports[name]['weights'],reports[name+'-R']['weights']
        require(set(a)==set(b),'Repair changed keys')
        changed=[k for k in a if a[k]!=b[k]]
        require(changed==['model.layers.13.mlp.down_proj.weight'],'Repair changed more than down projection')
    write(out/'export.json',{'status':'PASS','arms':{k:{'tensor_count':len(v['weights']),
                 'parameter_bytes':sum(x['bytes'] for x in v['weights'].values())} for k,v in reports.items()},'eta':eta})


def evaluate(artifacts: Path, arm: str, output: Path, baseline: Path | None, split='heldout'):
    import torch
    if split=='heldout':
        from .freeze import verify_eval
        verify_eval('R')
    gpu_guard();out=new_external(output);model=load_dense(artifacts/arm,'cuda');rows=read(CASE/'inputs/R'/(split+'.json'))
    require(arm=='R-B' or baseline is not None,'Candidate requires frozen baseline')
    records=[];(out/'vectors').mkdir();(out/'local').mkdir()
    for row in rows:
        z,capture,valid=diagnostic(model,row);ps=positions(row['length']);pred=array(capture['output'][0,ps])
        s=score(z,row['label_ids'],row['gold'])
        entry={'id':row['id'],'task':row['task'],'split':split,'arm':arm,'token_hash':row['token_hash'],
               'score':s,'validity':valid,'evidence_kind':'gpu_measurement','native_logits_dtype':'torch.bfloat16'}
        np.save(out/'vectors'/(row['id']+'.npy'),z,allow_pickle=False)
        np.save(out/'local'/(row['id']+'.npy'),pred,allow_pickle=False)
        if arm!='R-B':
            b=read(baseline/'cells'/(row['id']+'.json'))
            require(b['token_hash']==row['token_hash'] and b['validity']['input_hash']==valid['input_hash'],'Baseline pairing/input mismatch')
            bz=np.load(baseline/'vectors'/(row['id']+'.npy'),allow_pickle=False)
            by=np.load(baseline/'local'/(row['id']+'.npy'),allow_pickle=False)
            entry['sampled_positions']=ps
            entry['sampled_local']=[norm_record(a,b) for a,b in zip(pred,by)]
            entry.update(local=norm_record(pred,by),paired=paired(b['score'],s),full_kl_B_candidate=full_kl(bz,z))
        write(out/'cells'/(row['id']+'.json'),entry);records.append(entry)
        if len(records)%32==0:print('EVALUATE',arm,len(records),flush=True)
    write(out/'records.json',records);write(out/'status.json',{'status':'PASS','arm':arm,'count':len(records),'split':split,
                                                          'max_allocated':torch.cuda.max_memory_allocated(),'max_reserved':torch.cuda.max_memory_reserved()})



def calibration_readout(calibration: Path, development: Path, output: Path):
    """Reuse exact stored full-shape H; no additional transformer forward."""
    import torch
    gpu_guard();out=new_external(output);eta=read(development/'selection.json')['eta'];records=[]
    for name in config()['removed']:
        stats=np.load(calibration/(name+'-statistics.npz'),allow_pickle=False)
        weights={'uncorrected':stats['W0'], 'repaired':np.load(calibration/f'{name}-eta-{eta}.npy',allow_pickle=False)}
        for row in read(CASE/'inputs/R/calibration.json'):
            data=np.load(calibration/'activation_cells'/name/(row['id']+'.npz'),allow_pickle=False)
            h=torch.from_numpy(data['H']).to(device='cuda',dtype=torch.bfloat16);ps=positions(row['length'])
            record={'id':row['id'],'task':row['task'],'structure':name,'local':{}}
            for kind,w in weights.items():
                with torch.inference_mode():pred=torch.nn.functional.linear(h,torch.tensor(w,device='cuda',dtype=torch.bfloat16))
                record['local'][kind]=norm_record(array(pred[0,ps]),data['Y'][0,ps])
            records.append(record)
    write(out/'calibration_readout.json',records)
    write(out/'status.json',{'status':'PASS','rows':len(records),'transformer_forwards':0,'full_shape_hidden_reuse':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['smoke','calibrate','develop','export','evaluate','calibration-readout'])
    p.add_argument('--snapshot',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibration',type=Path);p.add_argument('--development',type=Path);p.add_argument('--artifacts',type=Path)
    p.add_argument('--arm');p.add_argument('--baseline',type=Path);p.add_argument('--split',default='heldout',choices=['smoke','heldout'])
    a=p.parse_args()
    if a.stage=='smoke':smoke(a.snapshot,a.output)
    elif a.stage=='calibrate':calibrate(a.snapshot,a.output)
    elif a.stage=='develop':develop(a.snapshot,a.calibration,a.output)
    elif a.stage=='export':export(a.snapshot,a.calibration,a.development,a.output)
    elif a.stage=='calibration-readout':calibration_readout(a.calibration,a.development,a.output)
    else:evaluate(a.artifacts,a.arm,a.output,a.baseline,a.split)
