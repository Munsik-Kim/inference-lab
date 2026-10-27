"""Second calculation path: recorded logits -> full-vocabulary FP64 audit.

Does not import benchmark functions, pipeline statistics, or saved PASS values.
Public mode checks only the disclosed small subset; local mode checks all main
selected logits. It never invokes a GPU or downloads an artifact.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

CASE=Path(__file__).resolve().parents[1]

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def decode(p):
    x=np.load(p,allow_pickle=False)
    if x.dtype==np.uint16:return (x.astype(np.uint32)*np.uint32(65536)).view(np.float32).astype(np.float64)
    if x.dtype not in (np.float32,np.float64):raise ValueError('UNSUPPORTED_ENCODING')
    return x.astype(np.float64)

def independent_kl(a,b):
    # A different normalization/reduction route from pipeline.log_prob.
    # Sequential logaddexp over151936 terms accumulated excessive roundoff in
    # the retained v1 failed audit. Max shifting plus compensated fsum keeps
    # the original normalization tolerance; no measurement tolerance changed.
    logza=np.asarray([float(row.max())+math.log(math.fsum(np.exp(row-row.max()))) for row in a])
    logzb=np.asarray([float(row.max())+math.log(math.fsum(np.exp(row-row.max()))) for row in b])
    logp=a-logza[:,None];logq=b-logzb[:,None]
    p=np.exp(logp)
    if not np.allclose(p.sum(-1),1,rtol=0,atol=2e-12):raise ValueError('NORMALIZATION')
    return np.einsum('ij,ij->i',p,logp-logq,dtype=np.float64,optimize=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--private',type=Path);parser.add_argument('--summary',type=Path);a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);started=time.process_time();manifest=json.loads((CASE/'inputs/manifest.json').read_text())
    source=json.loads((CASE/'provenance/freeze_v1.json').read_text())
    for name,row in source['files'].items():
        p=CASE/name
        if p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:raise ValueError('FROZEN_IDENTITY:'+name)
    for row in manifest['documents']:
        for ext,key in [('txt','text'),('npy','ids')]:
            p=CASE/'inputs'/(row['document_id']+'.'+ext)
            if digest(p)!=row[key]['sha256']:raise ValueError('INPUT_HASH')
    if a.private:
        if a.summary is None:raise ValueError('LOCAL_SUMMARY_REQUIRED')
        comparisons=list(csv.DictReader((a.summary/'position_metrics.csv').open()))
        base=a.private/'main';scope='All stored selected16 full-vocabulary logits for the requested split(s), not every token output'
        lookup={}
        for row in comparisons:
            key=(row['split'],row['document_id'],int(row['length']),row['reference'],row['candidate'])
            lookup.setdefault(key,[]).append(row)
        checked=0;maxdiff=0.;primary={}
        for (split,doc,n,ref,cand),expected in lookup.items():
            arrays=[]
            for condition in [ref,cand]:
                p=base/split/condition/f'{doc}_n{n}_r0_logits.npy';metadata=json.loads(p.with_name(f'{doc}_n{n}_r0.json').read_text())
                if digest(p)!=metadata['logits']['sha256']:raise ValueError('LOGIT_HASH')
                arrays.append(decode(p))
            actual=independent_kl(*arrays)
            ordered=sorted(expected,key=lambda r:int(r['position']))
            target=np.asarray([float(r['KL']) for r in ordered]);diff=float(np.max(np.abs(actual-target)));maxdiff=max(maxdiff,diff)
            if not np.allclose(actual,target,rtol=1e-9,atol=2e-11):raise ValueError('KL_DISAGREEMENT')
            if ref in ['D1_R0','D1_R1'] and cand.startswith('P_') and n==512:primary[(split,doc,ref)]=float(actual.mean())
            checked+=len(actual)
        deltas=[]
        for split,doc,ref in primary:
            if ref=='D1_R0':deltas.append({'split':split,'document_id':doc,'delta':primary[(split,doc,'D1_R0')]-primary[(split,doc,'D1_R1')]})
        layer_checked=0;layer_error=0.
        if (a.summary/'layer_metrics.csv').exists():
            layer_rows=list(csv.DictReader((a.summary/'layer_metrics.csv').open()))
            cache_key=None
            for row in layer_rows:
                key=(row['split'],row['document_id'],int(row['length']),row['reference'],row['candidate'])
                if key!=cache_key:
                    split,doc,n,ref,cand=key
                    tensors=[decode(base/split/c/f'{doc}_n{n}_r0_block_selected.npy') for c in [ref,cand]]
                    cache_key=key
                layer=int(row['layer']);aa,bb=[t[layer] for t in tensors];diff=bb-aa
                actual={'SSE':float(np.vdot(diff,diff)),'reference_energy':float(np.vdot(aa,aa)),'MAE_sum':float(np.abs(diff).sum()),'max_abs':float(np.abs(diff).max())}
                for name,value in actual.items():
                    expected=float(row['selected_'+name]);error=abs(value-expected);layer_error=max(layer_error,error)
                    if not math.isclose(value,expected,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('SELECTED_LAYER_STATISTIC')
                layer_checked+=1
        details={'positions_comparisons_checked':checked,'max_absolute_KL_difference':maxdiff,'selected_layer_rows_checked':layer_checked,'selected_layer_max_absolute_difference':layer_error,'primary_deltas':deltas}
    else:
        subset=json.loads((CASE/'fixtures/public_logits/manifest.json').read_text());checked=0;maxdiff=0.;seen=set()
        for row in subset['comparisons']:
            arrays=[]
            for key in ['reference_file','candidate_file']:
                p=CASE/'fixtures/public_logits'/row[key];entry=subset['files'][row[key]]
                if digest(p)!=entry['sha256'] or p.stat().st_size!=entry['bytes']:raise ValueError('PUBLIC_RAW_HASH')
                arrays.append(decode(p));seen.add(row[key])
            actual=independent_kl(*arrays);target=np.asarray(row['expected_KL']);diff=float(np.max(np.abs(actual-target)));maxdiff=max(maxdiff,diff)
            if not np.allclose(actual,target,rtol=1e-9,atol=2e-11):raise ValueError('PUBLIC_KL_DISAGREEMENT')
            checked+=len(actual)
        scope='Public DEV4 documents x2 fixed selected positions x6 conditions; no full local EVAL audit'
        details={'files_read':len(seen),'position_comparisons_checked':checked,'max_absolute_KL_difference':maxdiff}
    receipt={'scope':scope,**details,'cpu_process_seconds':time.process_time()-started,'python':sys.version,'numpy':np.__version__,'numpy_path':np.__file__,'torch_imported':'torch' in sys.modules,'cuda_initialization':'NOT_CALLED','verification':'RECALCULATED_FROM_LOGITS'}
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
