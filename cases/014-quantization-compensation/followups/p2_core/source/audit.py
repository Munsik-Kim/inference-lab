"""Independent stdlib/NumPy audit of raw P2 metrics and topological build receipts."""
import argparse,csv,hashlib,json,math,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(b):return hashlib.sha256(b).hexdigest()
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def close(a,b,tol=1e-10):
    assert abs(float(a)-float(b))<=tol*max(1,abs(float(a)),abs(float(b))),(a,b)
def ordered_groups(b):
    p=f'model.layers.{b}.'
    return [('qkv',[p+'self_attn.'+x+'_proj' for x in ['q','k','v']]),('o',[p+'self_attn.o_proj']),('gate_up',[p+'mlp.gate_proj',p+'mlp.up_proj']),('down',[p+'mlp.down_proj'])]

def audit(run):
    t=time.monotonic();run=Path(run);inp=json.loads((run/'input_manifest.json').read_text());frozen=json.loads((run/'artifact_frozen.json').read_text());source=json.loads((run/'source_freeze.json').read_text());model=json.loads((run/'model_manifest.json').read_text())
    assert frozen['input_manifest_sha256']==source['input_manifest_sha256']==sha(run/'input_manifest.json')
    assert frozen['spec_sha256']==source['spec_sha256'] and frozen['execution_source_sha256']==source['execution_source_sha256']
    assert sha(run/'artifacts.jsonl')==frozen['artifact_records_sha256'] and sha(run/'build_ledger.jsonl')==frozen['build_ledger_sha256']
    assert frozen['fresh_forward_count_at_freeze']==0 and frozen['RTN_K_fits']==0 and frozen['projection_K_fits']==392
    names=[n for b in range(28) for _,ns in ordered_groups(b) for n in ns];assert len(set(names))==196
    assert names==[r['name'] for r in model['modules']]
    shapes={r['name']:r['shape'] for r in model['modules']};artifacts=lines(run/'artifacts.jsonl');meta={}
    for r in artifacts:
        k=r['method'],r['name'];assert k not in meta;meta[k]=r;assert r['shape']==shapes[r['name']]
        o,i=r['shape'];p=((i+63)//64)*64;assert r['header_bytes']==36 and r['scale_bytes']==o*p//64*4 and r['code_bytes']==o*p//2
        assert r['artifact_bytes']==36+r['scale_bytes']+r['code_bytes']
        assert frozen['methods'][r['method']]['artifact_hashes'][r['name']]==r['sha256']
    assert len(meta)==588
    for m in ['RTN','COMP_QUARTER','COMP_HALF']:
        assert set(n for mm,n in meta if mm==m)==set(names)
        assert sum(meta[m,n]['artifact_bytes'] for n in names)==247733136==frozen['methods'][m]['core_bytes']
    ledger=lines(run/'build_ledger.jsonl');diag=lines(run/'CAL_diagnostics.jsonl');assert len(ledger)==224 and len(diag)==392
    dmap={(r['method'],r['name']):r for r in diag};assert len(dmap)==392
    cal=[d for d in inp['documents'] if d['split']=='CAL-FIT'];assert len(cal)==32
    cal_hashes=[d['token_hash'] for d in cal];factors=Counter()
    for method in ['COMP_QUARTER','COMP_HALF']:
        prefix=digest(b'P2_ORIGINAL_UNBUILT');seen=set();rr=[r for r in ledger if r['method']==method];assert len(rr)==112
        for b in range(28):
            before=prefix
            for g,ns in ordered_groups(b):
                r=rr[b*4+['qkv','o','gate_up','down'].index(g)]
                assert r['block']==b and r['group']==g and r['status']=='COMPLETE' and r['CAL_only'] and r['CAL_token_hashes']==cal_hashes
                assert r['source_sha256']==source['execution_source_sha256'] and r['input_manifest_sha256']==sha(run/'input_manifest.json')
                assert [x['name'] for x in r['group_members']]==ns and r['factorizations']==1 and r['shared_factor_projections']==len(ns)
                fd=r['factor_diagnostics'];assert fd['dimension']==shapes[ns[0]][1] and fd['tokens']==8192
                close(fd['lambda'],.001*fd['trace']/fd['dimension']);assert fd['min_eigenvalue']>0
                for member in r['group_members']:
                    q=dmap[method,member['name']];assert q['K_sha256']==member['K_sha256'] and member['artifact_sha256']==meta[method,member['name']]['sha256']
                    assert q['factor_id']==f'{method}/b{b}_{g}' and q['X_group_sha256']==r['X_group_sha256'] and q['Z_group_sha256']==r['Z_group_sha256']
                    close(q['lambda'],fd['lambda']);assert q['fit']['relative_solve_residual']<=1e-10
                    assert q['alpha']==(.25 if method=='COMP_QUARTER' else .5)
                    scale=max(1,q['SSE_before'],q['SSE_after'],abs(q['T_cont']),abs(q['T_requant']))
                    assert abs(q['Delta']-(q['SSE_after']-q['SSE_before']))<=1e-10*scale
                    assert abs(q['T_inherited']-(q['C_norm2']-2*q['dot_A_C']))<=1e-10*scale
                    assert abs(q['T_cancellation']-2*q['dot_B0_C'])<=1e-10*scale
                    assert abs(q['T_requant']-(q['deltaB_norm2']-2*q['dot_R0minusC_deltaB']))<=1e-10*scale
                    assert abs(q['Delta']-q['T_inherited']-q['T_cancellation']-q['T_requant'])<=1e-10*scale
                    if b==0 and g=='qkv':assert fd['X_equals_Z'] and q['fit']['K_norm2']==0 and meta[method,member['name']]['sha256']==meta['RTN',member['name']]['sha256']
                factors[method]+=1;seen.update(ns);prefix=digest(json.dumps([meta[method,n]['sha256'] for n in names if n in seen]).encode());assert prefix==r['prefix_after_sha256']
            checkpoint=json.loads((run/'checkpoints'/method/f'block{b:02d}.json').read_text());assert checkpoint['prefix_before_block_sha256']==before and checkpoint['prefix_after_block_sha256']==prefix and checkpoint['complete_projection_count']==7*(b+1)
    assert dict(factors)==frozen['fit_factorizations']
    inputs={d['document_id_hash']:d for d in inp['documents'] if d['split'] in ['EVAL-ID','SHIFT']};assert len(inputs)==80
    raw=[r for p in sorted((run/'metrics').glob('*.jsonl')) for r in lines(p)];by={}
    for r in raw:
        k=r['method'],r['split'],r['document_id_hash'];assert k not in by;by[k]=r
        assert r['status']=='COMPLETE' and r['valid_token_count']==255 and r['KL_position_count']==16 and r['token_hash']==inputs[r['document_id_hash']]['token_hash']
        assert r['artifact_frozen_sha256']==sha(run/'artifact_frozen.json') and r['execution_source_sha256']==source['execution_source_sha256']
        assert r['artifact_projection_count']==(0 if r['method']=='BF16' else 196)
        assert r['projection_core_bytes']==(model['original_projection_BF16_tensor_bytes'] if r['method']=='BF16' else frozen['methods'][r['method']]['core_bytes'])
    assert len(raw)==320
    summary=json.loads((run/'summary.json').read_text());method_summary={(r['method'],r['split']):r for r in summary['method_summaries']}
    csv_summary=list(csv.DictReader((run/'method_summary.csv').open()));assert len(csv_summary)==8
    interval_checks=0
    for split in ['EVAL-ID','SHIFT']:
        ids=sorted(d for d,x in inputs.items() if x['split']==split);N=len(ids);losses={}
        for method in ['BF16','RTN','COMP_QUARTER','COMP_HALF']:
            rr=[by[method,split,d] for d in ids];n=sum(r['valid_token_count'] for r in rr);S=math.fsum(r['NLL_sum'] for r in rr);losses[method]=np.array([r['NLL_sum']/255 for r in rr])
            q=method_summary[method,split];close(q['NLL_sum'],S);close(q['NLL_token_weighted'],S/n);close(q['PPL'],math.exp(S/n));assert q['valid_token_count']==n
            close(q['mean_full_vocab_KL'],math.fsum(r['sampled_full_vocab_KL_sum'] for r in rr)/(16*N));close(q['argmax_flip_rate'],sum(r['flip_count'] for r in rr)/n)
            c=next(x for x in csv_summary if x['method']==method and x['split']==split)
            for key in ['NLL_sum','NLL_token_weighted','PPL','mean_full_vocab_KL','argmax_flip_rate']:close(c[key],q[key])
        ix=np.random.default_rng(6101523).integers(0,N,size=(10000,N))
        for q in [r for r in summary['paired_comparisons'] if r['split']==split]:
            diff=losses[q['candidate']]-losses[q['baseline']];boot=np.sum(diff[ix],axis=1)/N;CI=np.percentile(boot,[2.5,97.5])
            close(q['mean_document_delta_NLL'],math.fsum(diff.tolist())/N)
            for x,y in zip(q['CI95'],CI):close(x,y)
            close(q['PPL_relative_change'],math.expm1(float(diff.mean())))
            for x,y in zip(q['PPL_relative_change_CI95'],[math.expm1(float(c)) for c in CI]):close(x,y)
            verdict='LOWER_NLL' if CI[1]<0 else 'HIGHER_NLL' if CI[0]>0 else 'INCONCLUSIVE';assert q['verdict']==verdict
            assert q['lower_documents']==int((diff<0).sum()) and q['higher_documents']==int((diff>0).sum()) and q['exact_ties']==int((diff==0).sum());close(q['median'],float(np.median(diff)));interval_checks+=2
    completion=json.loads((run/'completion.json').read_text());calls=lines(run/'forward_calls.jsonl');assert len(calls)==completion['total_full_model_forward_calls']
    assert sum(r['split'] in ['EVAL-ID','SHIFT'] for r in calls)==320
    event=lines(run/'events.jsonl');freeze_event=next(r for r in event if r['stage']=='ALL_ARTIFACTS_FROZEN');assert freeze_event['fresh_forwards']==0
    matrix=json.loads((run/'smoke_matrix_verification.json').read_text());assert matrix['status']=='PASS' and matrix['full_model_forwards']==4 and all(r['identity_relative_gap']<=1e-10 for r in matrix['rows'])
    reg=json.loads((run/'supplemental_smoke_registration.json').read_text());assert reg['fresh_forward_count_at_registration']==0 and reg['registered_utc']<frozen['frozen_utc']
    reload=json.loads((run/'fresh_process_reload.json').read_text());assert reload['status']=='PASS' and reload['independently_decoded_artifacts']==588 and reload['full_model_forwards']==4
    for r in reload['checks']:
        assert r['status']=='PASS' and r['full_logits_exact'] and r['NLL_exact']
        if r['method']!='BF16':assert r['full_logits_BF16_sha256']==json.loads((run/(r['method']+'.build_complete.json')).read_text())['smoke_full_logits_BF16_sha256']
    primary=next(r for r in summary['paired_comparisons'] if r['primary']);decision=json.loads((run/'decision.json').read_text());assert decision['status']=='COMPLETED_P2_CORE' and decision['primary']==primary
    return {'status':'PASS','raw_metrics_checked':320,'method_summaries_checked':8,'paired_comparisons_checked':10,'intervals_checked_including_PPL_transforms':interval_checks,'bootstrap_resamples':10000,
       'artifact_metadata_checked':588,'own_trajectory_groups_checked':224,'CAL_decompositions_checked':392,'prefix_checkpoints_checked':56,'fresh_process_reload_receipt':'PASS','fixed_SMOKE_matrix_receipt':'PASS',
       'worker_main_full_model_forward_records':len(calls),'wall_seconds':time.monotonic()-t,'audit_source_sha256':sha(__file__),
       'limits':'Independent scalar/provenance recomputation does not rerun private weights/GPU NLL/KL/full logits. Actual fresh GPU reload and independent decoding are separately recorded; external artifact hashes are identities in a public bundle.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=audit(Path(a.run))
    with Path(a.output).open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(r,indent=2))
