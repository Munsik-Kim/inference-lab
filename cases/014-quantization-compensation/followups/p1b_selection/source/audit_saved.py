"""Independent saved-scalar auditor. Does not import selector/analyzer/GPU code."""
import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,tol=1e-10):
    if a is None or b is None:assert a==b;return
    assert abs(float(a)-float(b))<=tol*max(1,abs(float(a)),abs(float(b))),(a,b)
def loadlines(p):return [json.loads(s) for s in p.read_text().splitlines() if s]

def audit(run):
    start=time.monotonic();summary=json.loads((run/'summary.json').read_text());frozen=json.loads((run/'selection_frozen.json').read_text())
    source=json.loads((run/'source_freeze.json').read_text());manifest=json.loads((run/'input_manifest.json').read_text())
    inputs={r['document_id_hash']:r for r in manifest['documents']};assert len(inputs)==112
    assets=json.loads((run/'reused_artifact_manifest.json').read_text())
    asset_hashes={r['label']:r for r in assets['artifacts']}
    assert frozen['input_manifest_sha256']==sha(run/'input_manifest.json')==source['input_manifest_sha256']
    assert frozen['FRESH_EVAL_forward_count_at_freeze']==0
    for rel,h in frozen['SELECT_cell_hashes'].items():assert sha(run/rel)==h
    raw=[];by={};groups=defaultdict(list)
    for p in sorted((run/'cells').glob('*/*.jsonl')):
        rr=loadlines(p);assert len(rr)==manifest['counts'][p.parent.name];raw+=rr
        for r in rr:
            k=r['target'],r['alpha'],r['split'],r['document_id_hash'];assert k not in by;by[k]=r;groups[k[:3]].append(r)
            assert r['status']=='COMPLETE' and r['input_identical_across_alpha'] and r['execution_source_sha256']==source['execution_source_sha256']
            assert r['token_hash']==inputs[r['document_id_hash']]['token_hash'] and r['split']==inputs[r['document_id_hash']]['split']
            asset=asset_hashes[f"P1_PRIVATE/artifacts/candidates/{r['target']}.alpha{r['alpha']:g}.int4"]
            assert r['artifact_sha256']==asset['sha256'] and r['artifact_bytes']==asset['bytes']
            close(r['T_inherited'],r['C_norm2']-2*r['dot_A_C']);close(r['T_cancellation'],2*r['dot_B0_C'])
            close(r['T_cont'],r['C_norm2']-2*r['dot_R0_C'])
            close(r['T_requant'],r['deltaB_norm2']-2*r['dot_R0minusC_deltaB'])
            close(r['Delta'],r['SSE_after']-r['SSE_before'])
            assert abs(r['Delta']-r['T_inherited']-r['T_cancellation']-r['T_requant'])<=1e-10*r['identity_scale']
            assert r['identity_relative_gap']<=1e-10 and r['valid_token_count']==255 and r['KL_position_count']==16
    assert len(raw)==3360 and len(groups)==90
    targets=sorted({r['target'] for r in raw});assert len(targets)==6
    choices={r['target']:r for r in frozen['choices']};old={r['target']:r for r in json.loads((run/'reused_artifact_manifest.json').read_text())['old_fit']}
    for t in targets:
        scores={}
        for a in [0,.25,.5,.75,1]:
            rr=groups[t,a,'SELECT-ID'];scores[a]=(math.fsum(r['SSE_after'] for r in rr),math.fsum(r['NLL_sum'] for r in rr)/sum(r['valid_token_count'] for r in rr))
        s=min(scores,key=lambda a:(scores[a][0],a));n=min(scores,key=lambda a:(scores[a][1],a))
        assert s==choices[t]['policy_alpha']['SELECT_SSE'] and n==choices[t]['policy_alpha']['SELECT_NLL']
        assert choices[t]['policy_alpha']['OLD_FIT']==old[t]['selected_alpha']
        for score in choices[t]['scores']:
            close(score['SSE'],scores[score['alpha']][0]);close(score['NLL'],scores[score['alpha']][1])
    for r in raw:
        assert r['K_sha256']==old[r['target']]['K_sha256'];close(r['lambda_value'],old[r['target']]['lambda'])
    # Recreate the nonexclusive A/B/C/D labels from individual scalar terms.
    oldcal={(r['block'],r['projection'],r['alpha']):r['T_inherited_sum'] for r in json.loads((run/'old_CAL_reference.json').read_text())['rows']}
    actual_labels={}
    for r in raw:
        if not r['alpha']:continue
        b=by[r['target'],0,r['split'],r['document_id_hash']];tol=1e-9*max(1,r['SSE_before'],r['SSE_after']);ls=[]
        if r['T_inherited'] < -tol and r['T_cont']>tol:ls.append('A_CANCELLATION_LOSS')
        if r['T_cont'] < -tol and r['Delta']>tol:ls.append('B_REQUANTIZATION_REVERSAL')
        if oldcal[r['block'],r['projection'],r['alpha']]<=tol and r['T_inherited']>tol:ls.append('C_CONTINUOUS_TRANSFER_LOSS')
        if r['Delta'] < -tol and (r['NLL_sum']-b['NLL_sum'])/255>1e-9:ls.append('D_LOCAL_NLL_DISAGREEMENT')
        actual_labels[r['target'],r['alpha'],r['split'],r['document_id_hash']]=ls
    classifications=json.loads((run/'classifications.json').read_text())
    for r in classifications['rows']:assert r['labels']==actual_labels[r['target'],r['alpha'],r['split'],r['document_id_hash']]
    assert len(classifications['rows'])==2688
    for split in manifest['counts']:
        counts=dict(Counter(l for key,v in actual_labels.items() if key[2]==split for l in v))
        assert counts==classifications['counts'][split]==summary['classification_counts'][split]
    alias=list(csv.DictReader((run/'policy_results.csv').open()));assert len(alias)==4704
    for r in alias:
        a=choices[r['target']]['policy_alpha'][r['policy']];assert float(r['alpha'])==a
        x=by[r['target'],a,r['split'],r['document_id_hash']]
        for k in ['SSE_after','NLL_sum','sampled_full_vocab_KL_sum','flip_count','valid_token_count','artifact_bytes']:close(float(r[k]),x[k])
        assert r['artifact_sha256']==x['artifact_sha256'] and r['token_hash']==x['token_hash']
    for filename in ['alpha_results.csv','policy_summary.csv']:
        for q in csv.DictReader((run/filename).open()):
            rr=groups[q['target'],float(q['alpha']),q['split']];n=sum(r['valid_token_count'] for r in rr);loss=math.fsum(r['NLL_sum'] for r in rr)
            close(q['NLL_sum'],loss);close(q['NLL_mean'],loss/n);close(q['PPL'],math.exp(loss/n))
            for k in ['SSE_after','SSE_before','reference_SSE','T_inherited','T_cancellation','T_cont','T_requant','BF16_output_residual_norm2']:close(q[k],math.fsum(r[k] for r in rr))
            close(q['mean_full_vocab_KL'],math.fsum(r['sampled_full_vocab_KL_sum'] for r in rr)/(16*len(rr)))
            close(q['argmax_flip_rate'],sum(r['flip_count'] for r in rr)/n)
    # Recompute every reported paired interval with an independently written index/reduction path.
    # No selector call occurs in this loop. A single shared matrix per split keeps target dependence.
    bootstrap_checks=0
    for split in ['SELECT-ID','FRESH-EVAL-ID','FRESH-SHIFT']:
        ids=sorted({r['document_id_hash'] for r in raw if r['split']==split});N=len(ids)
        ix=np.random.default_rng(6101519).integers(0,N,size=(10000,N))
        macro_records=[r for r in summary['macro_contrasts'] if r['split']==split]
        for m in macro_records:
            ratios=[];nlls=[];points=[];npoints=[]
            for t in targets:
                ca=choices[t]['policy_alpha'][m['candidate']];ba=choices[t]['policy_alpha'][m['baseline']]
                cr=[by[t,ca,split,d] for d in ids];br=[by[t,ba,split,d] for d in ids];z=[by[t,0,split,d] for d in ids]
                num=np.array([c['SSE_after']-b['SSE_after'] for c,b in zip(cr,br)])
                den=np.array([r['SSE_after'] for r in z]);nl=np.array([(c['NLL_sum']-b['NLL_sum'])/255 for c,b in zip(cr,br)])
                rat=np.sum(num[ix],axis=1)/np.sum(den[ix],axis=1);nn=np.sum(nl[ix],axis=1)/N
                ratios.append(rat);nlls.append(nn);points.append(float(num.sum()/den.sum()));npoints.append(float(nl.mean()))
                q=next(r for r in summary['target_contrasts'] if r['split']==split and r['target']==t and r['candidate']==m['candidate'] and r['baseline']==m['baseline'])
                close(q['SSE_difference_over_alpha0'],points[-1]);close(q['NLL_doc_mean_difference'],npoints[-1])
                for a,b in zip(q['SSE_pointwise_CI95'],np.percentile(rat,[2.5,97.5])):close(a,b)
                for a,b in zip(q['NLL_pointwise_CI95'],np.percentile(nn,[2.5,97.5])):close(a,b)
                bootstrap_checks+=2
            close(m['D_macro'],math.fsum(points)/6);close(m['NLL_macro_doc_mean_difference'],math.fsum(npoints)/6)
            for a,b in zip(m['D_macro_CI95'],np.percentile(np.array(ratios).mean(axis=0),[2.5,97.5])):close(a,b)
            for a,b in zip(m['NLL_macro_CI95'],np.percentile(np.array(nlls).mean(axis=0),[2.5,97.5])):close(a,b)
            bootstrap_checks+=2
    calls=loadlines(run/'forward_calls.jsonl');counts=dict(Counter(r['stage'] for r in calls));complete=json.loads((run/'completion.json').read_text())
    assert len(calls)==3813 and counts==complete['full_model_forward_counts'] and complete['fit_encode_calls']==0
    freeze_time=frozen['frozen_utc'];events=loadlines(run/'events.jsonl');first_eval=next(r for r in events if r['stage']=='CELL_COMPLETE' and r.get('split')=='FRESH-EVAL-ID')
    assert freeze_time<first_eval['utc'] and complete['selection_frozen_sha256']==sha(run/'selection_frozen.json')
    assert all(r['artifact_bytes']==(1179684 if r['target'].endswith('o_proj') else 1769508) for r in raw)
    return {'status':'PASS','audited_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'raw_rows':len(raw),'candidate_groups':90,'policy_alias_rows':len(alias),
            'selectors_independently_recomputed':12,'summary_rows_recomputed':216,'bootstrap_intervals_recomputed':bootstrap_checks,'bootstrap_resamples_each':10000,
            'source_file_sha256':sha(__file__),'full_model_forward_records':len(calls),'fit_encode_calls':0,'input_artifact_policy_pairing':'PASS','NLL_exp_PPL_KL_flip_SSE_decomposition':'PASS',
            'wall_seconds':time.monotonic()-start,'limits':'Saved scalar audit does not rerun model weights, full X/Z, GPU NLL or full-vocabulary logits. External artifacts identified, not verified by public bundle. Small private SMOKE arrays are checked separately.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=audit(Path(a.run))
    with Path(a.output).open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(r,indent=2))
