"""Export compact official-task scalars and recompute paired quality results.

No benchmark prose or generated rationale is exported. Official extraction and
metrics are retained; choice-score diagnostics are separately labelled.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def process_status(path):
    """A written result marker alone does not prove a successful process exit."""
    exit_path = path / 'process_exit.json'
    if exit_path.exists() and json.loads(exit_path.read_text())['returncode'] != 0:
        return 'FAILED'
    if (path / 'failure.json').exists():
        return 'FAILED'
    if (path / 'complete.json').exists() and exit_path.exists():
        return 'COMPLETE'
    return 'INCOMPLETE' if path.exists() else 'NOT_RUN'


def scalar_sample(task, sample):
    row = {'task': task, 'id': str(sample['doc_id']), 'filter': sample['filter'],
           'prompt_hash': digest(sample['arguments']), 'doc_hash': sample['doc_hash']}
    if task == 'wikitext':
        ll, words = sample['word_perplexity']
        ll2, nbytes = sample['byte_perplexity']
        require(ll == ll2 and math.isfinite(ll) and words > 0 and nbytes > 0, 'invalid rolling denominators')
        require(words == len(re.split(r'\s+', sample['doc']['page'])), 'word denominator differs from official original document')
        require(nbytes == len(sample['doc']['page'].encode('utf-8')), 'UTF8 denominator mismatch')
        row.update(loglikelihood=ll, words=words, bytes=nbytes)
    elif task == 'gsm8k':
        answer = sample['filtered_resps'][0]
        require(isinstance(answer, str) and sample['exact_match'] in (0, 1), 'invalid extracted answer')
        # Hash even extracted output: a failed parser can retain a long rationale.
        row.update(gold_hash=digest(str(sample['target'])), answer_hash=digest(answer),
                   correct=bool(sample['exact_match']), empty_extraction=not bool(answer))
    else:
        scores = [float(x[0]) for x in sample['filtered_resps']]
        require(len(scores) > 1 and all(math.isfinite(x) for x in scores), 'invalid choice likelihoods')
        target = sample['target']
        if not isinstance(target, int):
            # Official MMLU returns index; ARC returns an integer index as well.
            require(str(target).isdigit(), 'unsupported official target type')
        gold = int(target)
        require(0 <= gold < len(scores), 'gold index out of range')
        pred = max(range(len(scores)), key=scores.__getitem__)
        require(int(sample['acc']) == int(pred == gold), 'official acc/argmax mismatch')
        row.update(scores=scores, gold=gold, prediction=pred, correct=pred == gold)
        if 'acc_norm' in sample:
            choices = sample['doc']['choices']['text']
            lengths = [len(x) for x in choices]  # pinned Task uses character lengths
            require(len(lengths) == len(scores) and min(lengths) > 0, 'invalid normalization lengths')
            pn = max(range(len(scores)), key=lambda i: scores[i] / lengths[i])
            require(int(sample['acc_norm']) == int(pn == gold), 'official acc_norm mismatch')
            row.update(choice_character_lengths=lengths, prediction_norm=pn, correct_norm=pn == gold)
    return row


def wiki_aggregate(rows):
    require(bool(rows), 'empty WikiText group')
    ll = math.fsum(r['loglikelihood'] for r in rows)
    words = sum(r['words'] for r in rows)
    nbytes = sum(r['bytes'] for r in rows)
    require(words > 0 and nbytes > 0 and math.isfinite(ll), 'invalid PPL totals')
    return {'documents': len(rows), 'words': words, 'bytes': nbytes, 'loglikelihood': ll,
            'word_perplexity': math.exp(-ll / words), 'byte_perplexity': math.exp(-ll / nbytes),
            'bits_per_byte': -ll / nbytes / math.log(2)}


def pair_rows(baseline, candidate):
    def keyed(rows):
        out = {}
        for r in rows:
            key = (r['task'], r['filter'], r['id'])
            require(key not in out, 'duplicate sample key')
            out[key] = r
        return out
    b, c = keyed(baseline), keyed(candidate)
    require(b.keys() == c.keys(), 'missing/extra paired samples')
    pairs = []
    for key in sorted(b):
        x, y = b[key], c[key]
        for field in ('prompt_hash', 'doc_hash', 'gold', 'gold_hash', 'words', 'bytes', 'choice_character_lengths'):
            require(x.get(field) == y.get(field), 'paired mismatch: ' + field)
        pairs.append((x, y))
    return pairs


def validate_coverage(task, rows, freeze):
    expected_tasks = {name for name in freeze['tasks']
                      if name == task or task == 'mmlu' and name.startswith('mmlu_')}
    require({r['task'] for r in rows} == expected_tasks, 'missing/extra official task')
    filters = {'strict-match', 'flexible-extract'} if task == 'gsm8k' else {'none'}
    for name in expected_tasks:
        require({r['filter'] for r in rows if r['task'] == name} == filters,
                'missing/extra official filter')
        for fil in filters:
            members = [r for r in rows if r['task'] == name and r['filter'] == fil]
            require(len(members) == freeze['tasks'][name]['n_documents'], 'official split coverage mismatch')
            require(len({r['id'] for r in members}) == len(members), 'duplicate official document')


def generation_summary(audit, expected):
    outputs = [o for r in audit for o in r['outputs']]
    require(len(audit) == expected and len(outputs) == expected, 'generation coverage mismatch')
    require(all(type(o['tokens']) is int and 0 < o['tokens'] <= 1024
                and o['finish_reason'] in ('stop', 'length') for o in outputs),
            'invalid generation length/finish reason')
    return {'n': len(outputs), 'finish_reasons': dict(Counter(o['finish_reason'] for o in outputs)),
            'at_1024_token_cap': sum(o['tokens'] == 1024 for o in outputs),
            'total_generated_tokens': sum(o['tokens'] for o in outputs)}


def interval(values):
    return [float(x) for x in np.quantile(values, [.025, .975])]


def paired_mean_interval(values, rng, nboot=5000):
    a = np.asarray(values, dtype=float)
    require(a.size > 0 and np.isfinite(a).all(), 'invalid bootstrap values')
    # Bounded memory: no all-benchmarks giant index matrix.
    draws = []
    for start in range(0, nboot, 100):
        idx = rng.integers(0, len(a), size=(min(100, nboot-start), len(a)))
        draws.extend(a[idx].mean(axis=1).tolist())
    return interval(draws)


def choices(row):
    a = np.asarray(row['scores'], dtype=np.float64)
    logp = a - (np.max(a) + np.log(np.exp(a-np.max(a)).sum()))
    p = np.exp(logp); gold = row['gold']
    return logp, p, -float(logp[gold]), float(np.sum((p-np.eye(1, len(p), gold)[0])**2))


def transition_summary(pairs, rng, normalized=False):
    correct = 'correct_norm' if normalized else 'correct'
    prediction = 'prediction_norm' if normalized else 'prediction'
    b = [int(x[correct]) for x, _ in pairs]; c = [int(y[correct]) for _, y in pairs]
    d = [y-x for x, y in zip(b, c)]
    ans = [(x.get(prediction, x.get('answer_hash')), y.get(prediction, y.get('answer_hash'))) for x,y in pairs]
    result = {'n':len(pairs), 'baseline_correct':sum(b), 'candidate_correct':sum(c),
      'both_correct':sum(x and y for x,y in zip(b,c)), 'correct_to_wrong':sum(x and not y for x,y in zip(b,c)),
      'wrong_to_correct':sum(not x and y for x,y in zip(b,c)), 'both_wrong':sum(not x and not y for x,y in zip(b,c)),
      'all_answer_disagreement':sum(x!=y for x,y in ans),
      'wrong_to_different_wrong':sum(not x and not y and u!=v for x,y,(u,v) in zip(b,c,ans)),
      'accuracy_delta_candidate_minus_baseline':float(np.mean(d)), 'accuracy_delta_pointwise_95_interval':paired_mean_interval(d,rng),
      'conditional_denominators':{'baseline_correct':sum(b),'baseline_wrong':len(b)-sum(b)},
      'bootstrap_unit':'paired test item within task/filter'}
    result['correctness_flips_Dutta_definition'] = result['correct_to_wrong']+result['wrong_to_correct']
    if not normalized and 'scores' in pairs[0][0]:
        dn=[];db=[];kl=[]
        for x,y in pairs:
            lb,pb,nb,bb=choices(x);lc,pc,nc,bc=choices(y)
            dn.append(nc-nb);db.append(bc-bb);kl.append(float(np.sum(pb*(lb-lc))))
        result['choice_score_diagnostic']={'scope':'softmax of continuation choice likelihoods; not full vocabulary',
          'mean_gold_NLL_delta_nats':float(np.mean(dn)), 'NLL_delta_pointwise_95_interval':paired_mean_interval(dn,rng),
          'mean_Brier_delta':float(np.mean(db)), 'mean_KL_BF16_to_W4_nats':float(np.mean(kl))}
    return result


def analyze(quality, freeze):
    rows=[];tasks=[];rng=np.random.default_rng(909223)
    for task in ['arc_challenge','wikitext','gsm8k','mmlu']:
        paths={a:quality/f'full-{task}-{a}' for a in ['BF16','W4']}
        statuses = {a: process_status(p) for a, p in paths.items()}
        # Preserve complete calculations from a failed native finalization as
        # qualified evidence, never as a clean process or a replacement attempt.
        retained = all((p/'results.json').is_file() and (p/'complete.json').is_file()
                       and (p/'process_exit.json').is_file() and not (p/'failure.json').exists()
                       for p in paths.values())
        if not retained:
            tasks.append({'task':task,'status':'INCOMPLETE','arms':statuses})
            continue
        official={};by_arm={}
        for arm,p in paths.items():
            data=json.loads((p/'results.json').read_text());official[arm]=data['results'];ar=[]
            for name,samples in data['samples'].items():
                for sample in samples:ar.append(scalar_sample(name,sample)|{'arm':arm,'process_exit_code':json.loads((p/'process_exit.json').read_text())['returncode']})
            validate_coverage(task, ar, freeze)
            by_arm[arm]=ar;rows.extend(ar)
        pairs=pair_rows(by_arm['BF16'],by_arm['W4']);groups=defaultdict(list)
        for x,y in pairs:groups[(x['task'],x['filter'])].append((x,y))
        group_results={}
        for (name,fil),members in groups.items():
            label=name+'/'+fil
            if name=='wikitext':
                ag={a:wiki_aggregate([p[i] for p in members]) for i,a in enumerate(['BF16','W4'])}
                for arm in ag:
                    for metric in ['word_perplexity','byte_perplexity','bits_per_byte']:
                        require(math.isclose(ag[arm][metric],official[arm][name][metric+','+fil],rel_tol=1e-10,abs_tol=1e-10),'official PPL aggregate mismatch')
                boot=[]
                for _ in range(5000):
                    sel=rng.integers(0,len(members),len(members));n=sum(members[i][0]['words'] for i in sel)
                    boot.append(sum(members[i][0]['loglikelihood']-members[i][1]['loglikelihood'] for i in sel)/n)
                group_results[label]={'arms':ag,'word_NLL_delta_nats':-ag['W4']['loglikelihood']/ag['W4']['words']+ag['BF16']['loglikelihood']/ag['BF16']['words'],
                  'word_NLL_delta_pointwise_95_interval':interval(boot),'bootstrap_unit':'paired document, recomputed summed denominator'}
            else:
                s=transition_summary(members,rng)
                metric='exact_match' if name=='gsm8k' else 'acc'
                for arm,field in [('BF16','baseline_correct'),('W4','candidate_correct')]:
                    require(math.isclose(s[field]/s['n'],official[arm][name][metric+','+fil],abs_tol=1e-12),'official accuracy aggregate mismatch')
                if name=='arc_challenge':
                    s['acc_norm']=transition_summary(members,rng,True)
                    for arm,field in [('BF16','baseline_correct'),('W4','candidate_correct')]:
                        require(math.isclose(s['acc_norm'][field]/s['n'],official[arm][name]['acc_norm,'+fil],abs_tol=1e-12),'official normalized aggregate mismatch')
                group_results[label]=s
        entry={'task':task,'status':'COMPLETE' if all(v=='COMPLETE' for v in statuses.values()) else 'COMPUTED_WITH_PROCESS_FAILURE',
          'process_status':statuses,'official_results':official,'groups':group_results}
        if task=='mmlu':
            gs=list(group_results.values());require(len(gs)==57,'missing MMLU subject')
            d=np.array([g['candidate_correct']-g['baseline_correct'] for g in gs]);n=np.array([g['n'] for g in gs])
            idx=rng.integers(0,57,(5000,57));draws=d[idx].sum(axis=1)/n[idx].sum(axis=1)
            entry['overall_paired']={'subjects':57,'n':int(n.sum()),'accuracy_delta':float(d.sum()/n.sum()),'pointwise_95_interval':interval(draws),'bootstrap_unit':'paired subject cluster, item-weighted within resampled subjects',
              **{k:sum(g[k] for g in gs) for k in ['baseline_correct','candidate_correct','both_correct','correct_to_wrong','wrong_to_correct','both_wrong','all_answer_disagreement','wrong_to_different_wrong']}}
            for arm,key in [('BF16','baseline_correct'),('W4','candidate_correct')]:
                require(math.isclose(entry['overall_paired'][key]/entry['overall_paired']['n'],official[arm]['mmlu']['acc,none'],abs_tol=1e-12),'official weighted MMLU aggregate mismatch')
        if task=='gsm8k':
            entry['generation_budget']={}
            for arm,p in paths.items():
                entry['generation_budget'][arm]=generation_summary(
                    json.loads((p/'generation_audit.json').read_text()), freeze['tasks']['gsm8k']['n_documents'])
        tasks.append(entry)
    return {'tasks':tasks,'full_vocabulary_KL':'NOT_RUN','scope':'official full-split calculations; DEV excluded; no aggregate across benchmarks. COMPUTED_WITH_PROCESS_FAILURE retains auditable completed calculations but is not successful execution. No failed split is rerun.'},rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--quality',type=Path,required=True);p.add_argument('--freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result,rows=analyze(a.quality,json.loads(a.freeze.read_text()));a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'quality_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (a.output/'quality_scalars.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n' for r in rows))
    print(json.dumps({r['task']:r['status'] for r in result['tasks']}))


if __name__=='__main__':main()
