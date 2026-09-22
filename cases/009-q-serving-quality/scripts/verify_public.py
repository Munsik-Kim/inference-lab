"""Second, stdlib scalar path. No private tensors, requests or model execution."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics


def require(ok,message):
    if not ok:raise ValueError(message)


def same(a,b):
    require(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),f'numeric mismatch: {a} vs {b}')


def manifest(case):
    expected={}
    for line in (case/'SHA256SUMS').read_text().splitlines():
        h,name=line.split('  ',1);p=PurePosixPath(name)
        require(not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name and name not in expected,'unsafe/duplicate inventory')
        require(name!='SHA256SUMS','self hash')
        expected[name]=h
    actual={}
    for p in case.rglob('*'):
        require(not p.is_symlink(),'symlink')
        if p.is_file() and p.name!='SHA256SUMS':actual[p.relative_to(case).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    require(actual==expected,'inventory/hash mismatch')
    return len(actual)


def verify_serving(case):
    s=json.loads((case/'results/derived/serving_summary.json').read_text());groups=defaultdict(list)
    for line in (case/'results/raw/requests.jsonl').read_text().splitlines():
        r=json.loads(line);groups[(r['process_id'],r['condition'])].append(r)
    cells=0
    for c in s['cells']:
        if c['status']!='COMPLETE':continue
        rows=groups.pop((c['process_id'],c['condition']));require(len(rows)==c['n_requests'],'cell count')
        require(len({r['request_id'] for r in rows})==len(rows),'duplicate request')
        for r in rows:
            require(r['status']=='OK' and r['output_tokens']==256 and r['input_tokens']==c['input_tokens'] and r['warmup'] is False,'request validity')
            same(r['tpot_seconds'],(r['latency_seconds']-r['ttft_seconds'])/255)
        for m in ['ttft','tpot','latency']:
            v=[r[m+'_seconds']*1000 for r in rows]
            require(all(math.isfinite(x) and x>=0 for x in v),'invalid time')
            same(statistics.median(v),c[m+'_p50_ms']);same(statistics.quantiles(v,n=100,method='inclusive')[94],c[m+'_p95_ms'])
        same(sum(r['output_tokens'] for r in rows)/c['elapsed_seconds'],c['output_tokens_per_second']);cells+=1
    require(not groups,'unmatched raw cells')
    require(cells==s['complete_cells'],'complete count')
    return cells


def counts(pairs,normal=False):
    field='correct_norm' if normal else 'correct';answer='prediction_norm' if normal else 'prediction'
    b=[int(x[field]) for x,y in pairs];c=[int(y[field]) for x,y in pairs]
    diff=[x.get(answer,x.get('answer_hash'))!=y.get(answer,y.get('answer_hash')) for x,y in pairs]
    return {'n':len(b),'baseline_correct':sum(b),'candidate_correct':sum(c),'both_correct':sum(x and y for x,y in zip(b,c)),
      'both_wrong':sum(not x and not y for x,y in zip(b,c)), 'correct_to_wrong':sum(x and not y for x,y in zip(b,c)),
      'wrong_to_correct':sum(not x and y for x,y in zip(b,c)), 'all_answer_disagreement':sum(diff),
      'wrong_to_different_wrong':sum(not x and not y and z for x,y,z in zip(b,c,diff))}


def nll(row):
    values=row['scores'];m=max(values)
    return m+math.log(math.fsum(math.exp(x-m) for x in values))-values[row['gold']]


def verify_quality(case):
    s=json.loads((case/'results/derived/quality_summary.json').read_text());groups=defaultdict(lambda:defaultdict(dict))
    for line in (case/'results/raw/quality_scalars.jsonl').read_text().splitlines():
        r=json.loads(line);g=groups[r['task']+'/'+r['filter']][r['arm']]
        require(r['id'] not in g,'duplicate quality ID');g[r['id']]=r
        if 'scores' in r:
            require(all(math.isfinite(x) for x in r['scores']),'nonfinite likelihood')
            pred=max(range(len(r['scores'])),key=r['scores'].__getitem__)
            require(pred==r['prediction'] and r['correct']==(pred==r['gold']),'choice score/correctness mismatch')
    checked=0
    for task in s['tasks']:
        if 'groups' not in task:continue
        for name,result in task['groups'].items():
            g=groups.pop(name);b,c=g['BF16'],g['W4'];require(b.keys()==c.keys(),'missing quality pair')
            pairs=[(b[k],c[k]) for k in sorted(b)]
            for arm,data in [('BF16',b),('W4',c)]:
                exits={r['process_exit_code'] for r in data.values()}
                require(len(exits)==1,'mixed process exits')
                require((next(iter(exits))==0)==(task['process_status'][arm]=='COMPLETE'),'process failure status hidden')
            require(all(x['prompt_hash']==y['prompt_hash'] and x['doc_hash']==y['doc_hash'] and x.get('gold',x.get('gold_hash'))==y.get('gold',y.get('gold_hash')) for x,y in pairs),'quality pair provenance')
            if task['task']=='wikitext':
                for arm,data in [('BF16',b),('W4',c)]:
                    ll=math.fsum(r['loglikelihood'] for r in data.values());w=sum(r['words'] for r in data.values());by=sum(r['bytes'] for r in data.values())
                    require(w==result['arms'][arm]['words'] and by==result['arms'][arm]['bytes'],'rolling denominator')
                    same(math.exp(-ll/w),result['arms'][arm]['word_perplexity']);same(math.exp(-ll/by),result['arms'][arm]['byte_perplexity'])
            else:
                for key,value in counts(pairs).items():require(result[key]==value,'transition mismatch: '+key)
                if 'acc_norm' in result:
                    for key,value in counts(pairs,True).items():require(result['acc_norm'][key]==value,'normalized transition mismatch')
                if 'choice_score_diagnostic' in result:same(math.fsum(nll(y)-nll(x) for x,y in pairs)/len(pairs),result['choice_score_diagnostic']['mean_gold_NLL_delta_nats'])
            checked+=len(pairs)
    require(not groups,'unmatched quality group')
    return checked


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result={'manifest_files':manifest(a.case),'serving_cells':verify_serving(a.case),'quality_metric_pairs':verify_quality(a.case),
      'scope':'stdlib point metrics and declared file identities. GSM correctness/extraction fields are retained official results, not re-parsed private rationales. WikiText denominators are retained official fields, not recovered from excluded text. No independent GPU replay or bootstrap interval replication.'}
    result['status']='PASS'
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
