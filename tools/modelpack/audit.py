"""Separate scalar implementation. No GPU replication or reconstruction of omitted vectors."""
from pathlib import Path
import math
import numpy as np
from .common import read,write,require


def close(a,b,name):require(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),name+' mismatch')


def scalar_score(record):
    s=record['score'];v=s['option_logits'];g=s['gold']
    require(len(v)==4 and all(math.isfinite(x) for x in v),'Invalid options')
    maximum=max(v);lc=maximum+math.log(math.fsum(math.exp(x-maximum) for x in v))
    q=[math.exp(x-lc) for x in v];pred=max(range(4),key=lambda i:v[i])
    require(pred==s['prediction'] and (pred==g)==s['correct'],'Prediction/gold mismatch')
    require([i for i in range(4) if v[i]==maximum]==s['top_ties'],'Tie mismatch')
    expected={'choice_nll':lc-v[g], 'full_gold_nll':s['full_log_normalizer']-v[g],
      'brier':math.fsum((q[i]-int(i==g))**2 for i in range(4)),
      'allowed_mass':math.exp(lc-s['full_log_normalizer']),
      'gold_margin':v[g]-max(v[i] for i in range(4) if i!=g),
      'winner_gap':sorted(v)[-1]-sorted(v)[-2]}
    for key,value in expected.items():close(value,s[key],key)
    for a,b in zip(q,s['choice_probabilities']):close(a,b,'probability')
    if 'local' in record:
        n=record['local'];close(math.sqrt(n['squared_error']/n['reference_squared_norm']),n['relative_error'],'norm ratio')
        close(math.sqrt(n['squared_error']/n['elements']),n['absolute_rms'],'absolute RMS')
        if 'sampled_local' in record:
            close(math.fsum(r['squared_error'] for r in record['sampled_local']),n['squared_error'],'sampled SSE sum')
            close(math.fsum(r['reference_squared_norm'] for r in record['sampled_local']),n['reference_squared_norm'],'sampled norm sum')
    return expected


def audit(case: Path,summary_path: Path,output: Path):
    summary=read(summary_path);checks={}
    for track,base in [('Q','Q-BF16'),('R','R-B')]:
        path=case/f'results/raw/{track}/model_records.json'
        if not path.exists():checks[track]={'status':'NOT_RUN'};continue
        rows=read(path);lookup={}
        for r in rows:
            key=(r['arm'],r['id']);require(key not in lookup,'Duplicate record');lookup[key]=r;scalar_score(r)
            require(r['validity'].get('full_model_outputs') is True and r['validity'].get('full_final_vocab') is True,'Missing full validity')
        ids=sorted(i for a,i in lookup if a==base);require(len(ids)==192,'Missing baseline')
        tasks=[lookup[base,i]['task'] for i in ids]
        rng=np.random.default_rng(808191);idx=np.concatenate([rng.choice([j for j,t in enumerate(tasks) if t==task],(5000,tasks.count(task))) for task in sorted(set(tasks))],axis=1)
        for arm,target in summary['tracks'][track]['arms'].items():
            pp=[(lookup[base,i]['score'],lookup[arm,i]['score']) for i in ids]
            require(sum(c['correct'] for b,c in pp)==target['correct'],'Correct count mismatch')
            counters={k:0 for k in ['both_correct','regression','gain','both_wrong','flip','wrong_to_different_wrong']}
            for b,c in pp:
                category=('both_correct' if c['correct'] else 'regression') if b['correct'] else ('gain' if c['correct'] else 'both_wrong')
                counters[category]+=1
                if b['prediction']!=c['prediction']:
                    counters['flip']+=1
                    if category=='both_wrong':counters['wrong_to_different_wrong']+=1
            require(counters==target['transitions'],'Transition reconstruction failure')
            for key in ['full_gold_nll','choice_nll','brier','gold_margin']:
                changes=np.array([c[key]-b[key] for b,c in pp]);close(math.fsum(changes)/192,target['delta_'+key]['mean'],'paired mean')
                ci=np.quantile(np.sum(changes[idx],axis=1)/192,[.025,.975])
                for a,b in zip(ci,target['delta_'+key]['ci95']):close(a,b,'paired interval')
            if arm.endswith('-R'):
                uncorrected=arm[:-2];raw=[lookup[uncorrected,i]['local']['squared_error'] for i in ids]
                fixed=[lookup[arm,i]['local']['squared_error'] for i in ids]
                observed=1-math.fsum(fixed)/math.fsum(raw)
                close(observed,summary['tracks'][track]['repair'][uncorrected]['pooled_recovery'],'pooled recovery')
                ci=np.quantile(1-np.sum(np.array(fixed)[idx],axis=1)/np.sum(np.array(raw)[idx],axis=1),[.025,.975])
                for a,b in zip(ci,summary['tracks'][track]['repair'][uncorrected]['ci95']):close(a,b,'recovery interval')
        checks[track]={'status':'PASS','records':len(rows),'scenarios':192}
    result={'status':'PASS','tracks':checks,'scope':'independent option/norm/paired calculations and same frozen bootstrap indices; full-vocabulary normalizers retained, full KL/vector/kernel checks excluded'}
    write(output,result);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(audit(a.case,a.summary,a.output)['status'])
