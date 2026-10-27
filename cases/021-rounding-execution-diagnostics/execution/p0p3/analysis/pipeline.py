"""CPU-only recorded-evidence calculations. No Torch import or CUDA context."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
import numpy as np
from common import CASE,read,write,file_row,freeze_files,utc
from layout import bf16_float,metrics
from primitives import kl,log_prob,paired_bootstrap,vector_decomposition

CONDITIONS=['D0_R0','D1_R0','P_R0','D0_R1','D1_R1','P_R1']
PAIRS=list(dict.fromkeys([(f'{a}_{r}',f'{b}_{r}') for r in ['R0','R1'] for a,b in [('D0','D1'),('D1','P'),('D0','P')]]+[('D0_R0',b) for b in CONDITIONS if b!='D0_R0']))

def csvwrite(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:path.write_text('status\nNOT_RUN\n');return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def ingest(private,split):
    manifest=read(CASE/'inputs/manifest.json');scratch=private/'scratch'/split;out=private/'block_metrics'/split;out.mkdir(parents=True,exist_ok=True)
    for ref,cand in PAIRS:
        dest=out/f'{ref}__{cand}.json'
        if dest.exists() or not (scratch/ref).exists() or not (scratch/cand).exists():continue
        # A condition is complete only after its exact expected metadata count exists.
        expected=len(manifest['document_order'][split])*2
        if any(len(list((private/'main'/split/c).glob('*.json')))!=expected for c in [ref,cand]):continue
        rows=[]
        for doc in manifest['document_order'][split]:
            for n in [128,512]:
                stem=f'{doc}_n{n}_r0_blocks.npy';a=np.load(scratch/ref/stem,allow_pickle=False,mmap_mode='r');b=np.load(scratch/cand/stem,allow_pickle=False,mmap_mode='r')
                if a.shape!=b.shape or a.shape!=(28,n,1024):raise ValueError('BLOCK_SHAPE')
                pos=manifest['selected_positions'][str(n)]
                for layer in range(28):
                    row={'document_id':doc,'split':split,'length':n,'layer':layer,'reference':ref,'candidate':cand,'full_token_count':n,**metrics(bf16_float(a[layer]),bf16_float(b[layer]))}
                    selected=metrics(bf16_float(a[layer,pos]),bf16_float(b[layer,pos]))
                    row.update({'selected_'+k:v for k,v in selected.items()});rows.append(row)
        write(dest,{'rows':rows,'pair':[ref,cand],'scope':'Same tensor coordinates from separate full-path histories; not same local block input','input_manifest':file_row(CASE/'inputs/manifest.json')})
    # Only task-owned temporary full hidden arrays; selected raw and summaries remain.
    removed=[]
    for condition in CONDITIONS:
        d=scratch/condition
        if condition=='D0_R0' or not d.exists():continue
        needs=[(a,b) for a,b in PAIRS if condition in (a,b)]
        if all((out/f'{a}__{b}.json').exists() for a,b in needs):
            for f in d.glob('*_blocks.npy'):removed.append({'path':str(f),'bytes':f.stat().st_size});f.unlink()
            d.rmdir()
    if removed:
        from common import append
        append(private/'temporary_cleanup.jsonl',{'utc':utc(),'split':split,'removed':removed,'reason':'All fixed dependent full-block comparisons completed; raw selected-position arrays retained'})
    print(json.dumps({'split':split,'completed_pairs':len(list(out.glob('*.json'))),'expected_pairs':len(PAIRS),'temporary_files_removed':len(removed)}))

def load_logits(path):
    array=np.load(path,allow_pickle=False)
    if array.dtype==np.uint16:return bf16_float(array).astype(np.float64)
    if array.dtype not in (np.float32,np.float64):raise ValueError('LOGIT_ENCODING')
    return array.astype(np.float64)

def metric_records(private,split,repeat=False):
    manifest=read(CASE/'inputs/manifest.json');docs=manifest['repeat_documents'] if repeat else manifest['document_order'][split]
    base=private/('repeats' if repeat else 'main')/split;rows=[];positions=[];vectors=[]
    cycles=range(3) if repeat else range(1)
    for doc in docs:
        for n in ([512] if repeat else [128,512]):
            for cycle in cycles:
                logits={};lp={}
                for c in CONDITIONS:
                    stem=f'{doc}_n{n}_r{cycle}';meta=read(base/c/(stem+'.json'));path=base/c/(stem+'_logits.npy')
                    if file_row(path)!={k:meta['logits'][k] for k in ['bytes','sha256']}:raise ValueError('RAW_LOGIT_IDENTITY_MISMATCH')
                    if meta['input_sha256']!=file_row(CASE/'inputs'/(doc+'.npy'))['sha256'] or meta['condition']!=c or meta['full_logits_shape']!=[1,n,151936]:raise ValueError('FORWARD_CONTRACT_MISMATCH')
                    x=load_logits(path)
                    if x.shape!=(16,151936) or not np.isfinite(x).all():raise ValueError('LOGITS_INVALID')
                    logits[c]=x;lp[c]=log_prob(x)
                target=np.load(CASE/'inputs'/(doc+'.npy'),allow_pickle=False)[np.asarray(manifest['selected_positions'][str(n)])+1]
                for ref,cand in PAIRS:
                    p,q=lp[ref],lp[cand];k=np.sum(np.exp(p)*(p-q),axis=-1)
                    if np.min(k)<-1e-10:raise ValueError('MEANINGFUL_NEGATIVE_KL')
                    m=metrics(logits[ref],logits[cand]);refarg=logits[ref].argmax(-1);arg=logits[cand].argmax(-1)
                    nllref=-p[np.arange(16),target];nllcand=-q[np.arange(16),target]
                    top=np.partition(logits[ref],-2,axis=-1)[:,-2:];margin=top.max(-1)-top.min(-1)
                    row={'document_id':doc,'split':split,'length':n,'repeat':cycle,'reference':ref,'candidate':cand,'KL':float(k.mean()),'NLL_difference':float((nllcand-nllref).mean()),'argmax_flips':int((arg!=refarg).sum()),'positions':16,'reference_margin_mean':float(margin.mean()),**m};rows.append(row)
                    for j,pos in enumerate(manifest['selected_positions'][str(n)]):positions.append({'document_id':doc,'split':split,'length':n,'repeat':cycle,'position':pos,'target_position':pos+1,'target_id':int(target[j]),'reference':ref,'candidate':cand,'KL':float(k[j]),'NLL_reference':float(nllref[j]),'NLL_candidate':float(nllcand[j]),'argmax_flip':bool(arg[j]!=refarg[j]),'reference_margin':float(margin[j])})
                for r in ['R0','R1']:
                    vectors.append({'document_id':doc,'split':split,'length':n,'repeat':cycle,'rounding':r,'boundary':'selected16_full_vocabulary_logits',**vector_decomposition(logits['D0_'+r],logits['D1_'+r],logits['P_'+r])})
                del logits,lp
    return rows,positions,vectors

def analyze(private,split,out,repeat=False):
    out.mkdir(parents=True,exist_ok=False);start=time.process_time()
    rows,pos,vec=metric_records(private,split,repeat)
    csvwrite(out/'document_metrics.csv',rows);csvwrite(out/'position_metrics.csv',pos);csvwrite(out/'vector_decomposition.csv',vec)
    if not repeat:
        layerrows=[]
        for p in sorted((private/'block_metrics'/split).glob('*.json')):layerrows.extend(read(p)['rows'])
        csvwrite(out/'layer_metrics.csv',layerrows)
    primary=[]
    keys=sorted({(r['document_id'],r['length'],r['repeat']) for r in rows})
    for doc,n,cycle in keys:
        pair={r['reference']:r['KL'] for r in rows if r['document_id']==doc and r['length']==n and r['repeat']==cycle and (r['reference'],r['candidate']) in [('D1_R0','P_R0'),('D1_R1','P_R1')]}
        primary.append({'document_id':doc,'length':n,'repeat':cycle,'E0':pair['D1_R0'],'E1':pair['D1_R1'],'delta':pair['D1_R0']-pair['D1_R1']})
    csvwrite(out/'primary.csv',primary)
    summary={'split':split,'repeat':repeat,'document_metric_rows':len(rows),'position_rows':len(pos),'cpu_process_seconds':time.process_time()-start,'scope':'Stored full-vocabulary selected16 logits; no GPU execution','input_manifest':file_row(CASE/'inputs/manifest.json')}
    if repeat:
        diffs=[]
        for doc in sorted({r['document_id'] for r in primary}):
            d=[r['delta'] for r in primary if r['document_id']==doc];diffs.extend(abs(x-d[0]) for x in d)
        summary.update(r_repeat=max(diffs),tau_abs=max(1e-10,10*max(diffs)))
        samepath=[]
        manifest=read(CASE/'inputs/manifest.json')
        for doc in manifest['repeat_documents']:
            for condition in CONDITIONS:
                base=private/'repeats'/split/condition
                original=load_logits(base/f'{doc}_n512_r0_logits.npy')
                for cycle in [1,2]:
                    actual=load_logits(base/f'{doc}_n512_r{cycle}_logits.npy')
                    samepath.append({'document_id':doc,'condition':condition,'reference_repeat':0,'candidate_repeat':cycle,'KL':float(kl(original,actual).mean()),**metrics(original,actual),'exact_values_equal':bool(np.array_equal(original,actual))})
        csvwrite(out/'same_path_repeat.csv',samepath)
        summary['same_path_repeat_comparisons']=len(samepath)
        summary['same_path_max_relative_RMSE']=max(r['relative_RMSE'] for r in samepath)
        summary['same_path_max_KL']=max(r['KL'] for r in samepath)
    else:summary['primary512']=paired_bootstrap([r['delta'] for r in primary if r['length']==512])
    write(out/'summary.json',summary);print(json.dumps(summary))

def select(private,analysis,repeat,out,freeze,provisional=False):
    capture_path=private/'raw/p0-actual-capture-validation/capture_validation.json'
    capture=read(capture_path)
    if capture['full_forwards']!=18 or len(capture['rows'])!=3:raise ValueError('CAPTURE_VALIDATION_DENOMINATOR')
    for row in capture['rows']:
        for key in ['native_vs_R0_noop','R0_actual_CPU_copy_hook','R1_actual_CPU_copy_hook','R0_after_R1_restoration']:
            if row[key]['SSE']!=0 or row[key]['max_abs']!=0:raise ValueError('CAPTURE_VALIDATION_FAILED')
    layers=[]
    with (analysis/'layer_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    for layer in range(28):
        subset=[r for r in rows if r['reference']=='D1_R0' and r['candidate']=='P_R0' and int(r['layer'])==layer and int(r['length'])==512]
        if len(subset)!=8:raise ValueError('DEV_LAYER_DENOMINATOR')
        sse=math.fsum(float(r['SSE']) for r in subset);energy=math.fsum(float(r['reference_energy']) for r in subset)
        layers.append({'layer':layer,'SSE':sse,'energy':energy,'score':None if energy==0 else sse/energy})
    ranked=sorted([r for r in layers if r['score'] is not None],key=lambda r:(-r['score'],r['layer']))[:2];selected=[r['layer'] for r in ranked]
    manifest=read(CASE/'inputs/manifest.json');gate=read(private/'readiness.json')
    for doc in manifest['document_order']['DEV']:
        blocks=np.load(private/'scratch/DEV/D0_R0'/f'{doc}_n512_r0_blocks.npy',allow_pickle=False,mmap_mode='r');ctx=private/'contexts/DEV'/doc
        if not np.array_equal(np.load(ctx/'position_ids.npy',allow_pickle=False),np.arange(512,dtype=np.int64)[None,:]) or read(ctx/'metadata.json')['attention_mask'] is not None:raise ValueError('LOCAL_CONTEXT_CONTRACT')
        for idx in selected:
            value=np.load(ctx/'input.npy',allow_pickle=False) if idx==0 else blocks[idx-1:idx]
            np.save(ctx/f'layer{idx}_input.npy',value,allow_pickle=False)
    result={'utc':utc(),'dev_complete':True,'layers':selected,'layer_scores':layers,'tau_abs':read(repeat/'summary.json')['tau_abs'],'r_repeat':read(repeat/'summary.json')['r_repeat'],'intervention':{'status':'NOT_RUN','reason':'Optional boundary replacement omitted: same-input local replay and complete primary paired design prioritized; no replacement effect estimated'},'rounding_boundary_status':gate['rounding_boundary_status'],'no_op':gate['no_op'],'actual_capture_validation':file_row(capture_path),'gpu_source_bundle_sha256':read(freeze)['bundle_sha256'],'input_manifest':file_row(CASE/'inputs/manifest.json'),'analysis_freeze':freeze_files(list((CASE/'analysis').glob('*.py'))+[CASE/'protocol.json',CASE/'inputs/manifest.json']),'dev_analysis':{'path':str(analysis),'summary':file_row(analysis/'summary.json')},'repeat_analysis':{'path':str(repeat),'summary':file_row(repeat/'summary.json')},'eval_observed':False}
    if provisional:
        result['dev_complete']=False
        result['role']='DEV main/repeat completed; layers frozen for local replay; NOT an EVAL authorization'
        result['intervention']={'status':'PENDING_DEV_VALIDATION','reason':'Bounded optional replacement is being validated before final selection freeze'}
    write(out,result);print(json.dumps({'selected_layers':selected,'tau_abs':result['tau_abs'],'intervention':result['intervention']['status']}))

def retain_contexts(private,split,selection):
    manifest=read(CASE/'inputs/manifest.json');layers=read(selection)['layers']
    for doc in manifest['document_order'][split]:
        p=private/'scratch'/split/'D0_R0'/f'{doc}_n512_r0_blocks.npy';blocks=np.load(p,allow_pickle=False,mmap_mode='r');ctx=private/'contexts'/split/doc
        if not np.array_equal(np.load(ctx/'position_ids.npy',allow_pickle=False),np.arange(512,dtype=np.int64)[None,:]) or read(ctx/'metadata.json')['attention_mask'] is not None:raise ValueError('LOCAL_CONTEXT_CONTRACT')
        for idx in layers:
            dest=ctx/f'layer{idx}_input.npy'
            if dest.exists():continue
            value=np.load(ctx/'input.npy',allow_pickle=False) if idx==0 else blocks[idx-1:idx]
            np.save(dest,value,allow_pickle=False)
    print('same-input replay contexts retained')

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['ingest','analyze','repeat','select','contexts']);p.add_argument('--private',type=Path,required=True);p.add_argument('--split',default='DEV',choices=['DEV','EVAL']);p.add_argument('--output',type=Path);p.add_argument('--analysis',type=Path);p.add_argument('--repeat-analysis',type=Path);p.add_argument('--freeze',type=Path);p.add_argument('--selection',type=Path);p.add_argument('--provisional',action='store_true');a=p.parse_args()
    if a.command=='ingest':ingest(a.private,a.split)
    elif a.command in ['analyze','repeat']:analyze(a.private,a.split,a.output,a.command=='repeat')
    elif a.command=='select':select(a.private,a.analysis,a.repeat_analysis,a.output,a.freeze,a.provisional)
    elif a.command=='contexts':retain_contexts(a.private,a.split,a.selection)

if __name__=='__main__':main()
