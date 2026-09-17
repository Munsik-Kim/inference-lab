"""Independent scalar path: math.fsum/log/exp, explicit first-max decisions.
Checks CPU arithmetic consistency, not independent GPU replication.
"""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

def scalar(r):
    x=r['option_logits'];assert len(x)==4 and all(math.isfinite(v) for v in x)
    m=max(x);logsum=m+math.log(math.fsum(math.exp(v-m) for v in x));q=[math.exp(v-logsum) for v in x];y=r['gold_index'];pred=next(i for i in range(4) if x[i]==m)
    return {'prediction':pred,'correct':pred==y,'top':[i for i in range(4) if x[i]==m],'nll':logsum-x[y],'brier':math.fsum((v-(i==y))**2 for i,v in enumerate(q)),'margin':x[y]-max(x[i] for i in range(4) if i!=y),'mass':math.exp(logsum-r['full_lse'])}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    def read(p):return json.loads(p.read_text())
    summary=read(a.summary);rows=read(a.case/'results/raw/native.json')+read(a.case/'results/raw/shadow.json'); protocol=read(a.case/'protocol.json')
    assert hashlib.sha256((a.case/'protocol.json').read_bytes()).hexdigest()==(a.case/'protocol.sha256').read_text().split()[0]
    for n,key in [('inputs/core.json','inputs_sha256'),('inputs/gold.json','gold_sha256'),('results/raw/native.json','native_sha256')]:assert hashlib.sha256((a.case/n).read_bytes()).hexdigest()==protocol[key]
    ix={};scores={};maxerr=0;norms=0
    for r in rows:
        key=(r['split'],r['base_id'],r['arm'],r['readout']);assert key not in ix;ix[key]=r;scores[key]=scalar(r)
        checks=r['validity'];assert len(checks)==11 and all(v is True for v in checks.values())
        if r['readout']=='H_FP32':
            for field,other in [('nll','choice_nll'),('brier','choice_brier'),('margin','gold_margin'),('mass','label_mass')]:
                error=abs(scores[key][field]-r['score'][other]);maxerr=max(maxerr,error);assert error<1e-11
            assert scores[key]['prediction']==r['score']['prediction'];assert all(r['controls'][k] is True for k in ['full_native_finite','full_hidden_finite','full_shadow_finite','native_projection_full_exact','C3_tolerance_pass'])
        else:
            metrics=[r['local_pooled'],r['local_last_query']]+list(r['hidden_differences'].values())+([r['native_pair_local']] if r['native_pair_local'] else [])
            for v in metrics:
                if v.get('relative_error') is not None:
                    rec=math.sqrt(v['error_squared_sum']/v['reference_squared_sum']);assert abs(rec-v['relative_error'])<1e-12;norms+=1
    # Independent control accounting from retained values, not success flags alone.
    control_counts={'cast_only_exact':0,'roundtrip_options_equal':0,'fp64_winner_disagreements':0};max_option=0.;max_gap=0.
    for r in rows:
        if r['readout']!='H_FP32':continue
        n=ix[(r['split'],r['base_id'],r['arm'],'H_NATIVE')];ctl=r['controls'];f32=np.asarray(r['option_logits'],dtype=np.float32)
        assert np.array_equal(np.asarray(n['option_logits'],dtype=np.float32).astype(np.float64),n['option_logits']);control_counts['cast_only_exact']+=1
        bits=f32.view(np.uint32);rounded=((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32).astype(np.float64).tolist()
        assert rounded==ctl['C2_option_logits'];control_counts['roundtrip_options_equal']+=int(rounded==n['option_logits'])
        fp64=ctl['C3_fp64_options'];diff=[r['option_logits'][j]-fp64[j] for j in range(4)]
        assert diff==ctl['C3_fp32_minus_fp64'];max_option=max(max_option,max(abs(v) for v in diff))
        gap32=sorted(r['option_logits'])[-1]-sorted(r['option_logits'])[-2];gap64=sorted(fp64)[-1]-sorted(fp64)[-2]
        assert gap32-gap64==ctl['C3_gap_difference'];max_gap=max(max_gap,abs(gap32-gap64))
        mismatch=r['option_logits'].index(max(r['option_logits']))!=fp64.index(max(fp64));assert mismatch==ctl['C3_winner_disagreement'];control_counts['fp64_winner_disagreements']+=mismatch
    for k,v in control_counts.items():assert v==summary['controls'][k]
    assert max_option==summary['controls']['max_fp32_fp64_option_abs'] and max_gap==summary['controls']['max_fp32_fp64_gap_abs']
    groups_checked=0;intervals_checked=0
    for key,report in summary['groups'].items():
        split,view,arm,task=key.split('/');items=sorted([r for r in rows if r['split']==split and r['readout']==view and r['arm']==arm and (task=='ALL' or r['task']==task)],key=lambda r:r['base_id']);pairs=[]
        for r in items:
            b=scores[(split,r['base_id'],'B',view)];c=scores[(split,r['base_id'],arm,view)];cell=('both_correct' if c['correct'] else 'regression') if b['correct'] else ('gain' if c['correct'] else 'both_wrong')
            pairs.append({'b':b,'c':c,'cell':cell,'task':r['task'],'flip':b['prediction']!=c['prediction'],'stratum':('unique' if len(b['top'])==1 else 'tied')+'/'+('unique' if len(c['top'])==1 else 'tied')})
        for stratum in [None,'unique/unique','tied/unique','unique/tied','tied/tied']:
            pp=[v for v in pairs if stratum is None or v['stratum']==stratum];g=report if stratum is None else report['tie_strata'][stratum]
            assert len(pp)==g['n'];assert sum(v['flip'] for v in pp)==g['flips']
            for cell in ['both_correct','regression','gain','both_wrong']:assert sum(v['cell']==cell for v in pp)==g[cell]
            assert sum(v['flip'] and v['cell']=='both_wrong' for v in pp)==g['wrong_to_wrong_changes']
        for m in ['nll','brier','margin']:
            delta=[v['c'][m]-v['b'][m] for v in pairs];mean=math.fsum(delta)/len(delta) if delta else None
            if delta:assert abs(mean-report['delta_'+m])<1e-11
        if split=='standard':
            rng=np.random.default_rng(617093);indices=[];present=[t for t in ['RETRIEVAL','COMPARISON','CODE'] if any(v['task']==t for v in pairs)]
            for t in present:
                ids=[i for i,v in enumerate(pairs) if v['task']==t];draw=rng.integers(0,len(ids),(5000,len(ids)));indices.append(np.asarray(ids)[draw])
            for m in ['nll','brier','margin']:
                delta=np.array([v['c'][m]-v['b'][m] for v in pairs]);rep=sum(delta[d].sum(1)/d.shape[1] for d in indices)/len(indices);ci=np.quantile(rep,[.025,.975],method='linear')
                assert max(abs(ci-np.array(report['posthoc_intervals']['delta_'+m]['ci95'])))<1e-11;intervals_checked+=1
        groups_checked+=1
    expected=[('A_PUBLIC',106,5,2),('V4',108,8,4)]
    for arm,correct,flips,gains in expected:
        g=summary['groups']['standard/H_NATIVE/'+arm+'/ALL'];assert g['B_description']['correct']==104 and g['candidate_description']['correct']==correct and g['flips']==flips and g['gain']==gains and g['regression']==0 and g['B_top_ties']==14
    for task,correct in [('RETRIEVAL',59),('COMPARISON',30),('CODE',15)]:assert summary['groups']['standard/H_NATIVE/A_PUBLIC/'+task]['B_description']['correct']==correct
    output={'status':'PASS','scalar_cells':len(rows),'distinct_existing_scenarios':len({r['base_id'] for r in rows}),'groups_checked':groups_checked,'bootstrap_intervals_checked':intervals_checked,'norm_summaries_checked':norms,'max_scalar_difference':maxerr,'original_totals_reconstructed':True,'scope':'Independent scalar and bootstrap calculations; no independent kernel/full-vocabulary replication'}
    with a.output.open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps(output))
if __name__=='__main__':main()
