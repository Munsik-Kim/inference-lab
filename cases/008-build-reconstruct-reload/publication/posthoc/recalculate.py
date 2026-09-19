"""Post-hoc same-record diagnostics; no models, fits, new intervals or GPU calls."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

TASKS = ('retrieval', 'comparison', 'code')
ARMS = {'Q': ('Q-BF16', 'Q-W4'), 'R': ('R-B', 'I25', 'I25-R', 'P25', 'P25-R', 'S50', 'S50-R')}

def require(ok: bool, message: str) -> None:
    if not ok: raise ValueError(message)

def read(path: Path):
    def pairs(items):
        d = {}
        for k, v in items:
            require(k not in d, 'Duplicate JSON key: '+k); d[k] = v
        return d
    return json.loads(path.read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def mean(values):
    require(len(values) > 0, 'Empty mean')
    return math.fsum(values) / len(values)

def scores(record):
    v = record['score']; s = v['option_logits']; g = v['gold']
    require(len(s) == 4 and all(math.isfinite(x) for x in s), 'Nonfinite/options geometry')
    require(type(g) is int and 0 <= g < 4, 'Invalid gold')
    lv = v['full_log_normalizer']; require(math.isfinite(lv), 'Invalid normalizer')
    lc = max(s) + math.log(math.fsum(math.exp(x-max(s)) for x in s))
    result = {'full': lv-s[g], 'choice': lc-s[g], 'mass_nll': lv-lc,
              'mass': math.exp(lc-lv), 'prediction': s.index(max(s)), 'gold': g}
    result['correct'] = result['prediction'] == g
    for name, key in [('full','full_gold_nll'), ('choice','choice_nll'), ('mass','allowed_mass')]:
        require(math.isclose(result[name],v[key],rel_tol=1e-10,abs_tol=1e-10), 'Stored score mismatch: '+name)
    require(result['prediction'] == v['prediction'] and result['correct'] == v['correct'], 'Decision mismatch')
    require(math.isclose(result['full'],result['choice']+result['mass_nll'],abs_tol=1e-12), 'NLL identity')
    return result

def analyze(case: Path) -> dict:
    sources = {}; result = {'evidence_kind': 'POST_HOC_SAME_RECORD_DIAGNOSTIC',
        'new_gpu_runs': 0, 'tracks': {}, 'code_structure': {}, 'limits':
        'Full argmax membership and full KL are retained scalars, not reconstructed full vectors. No new uncertainty intervals. Shared synthetic templates; repeated arms are not independent samples.'}
    def load(name):
        sources[name] = sha(case/name); return read(case/name)
    for track, arms in ARMS.items():
        rows = load(f'results/raw/{track}/model_records.json'); lookup = {}
        inputs = load(f'inputs/{track}/heldout.json'); imap = {r['id']:r for r in inputs}
        require(len(imap) == len(inputs) == 192, 'Missing/duplicate heldout input')
        for r in rows:
            key = (r['arm'],r['id']); require(key not in lookup, 'Duplicate model cell')
            require(r['arm'] in arms and r['id'] in imap, 'Unknown arm/input')
            i = imap[r['id']]
            require(r['token_hash'] == i['token_hash'] and r['task'] == i['task'] and r['score']['gold'] == i['gold'], 'Input pairing mismatch')
            require(r['evidence_kind'] == 'gpu_measurement' and r['split'] == 'heldout', 'Wrong evidence kind')
            require(all(r.get('validity',{}).get(k) is True for k in ['full_model_outputs','full_final_vocab']), 'Missing full validity')
            lookup[key] = (r, scores(r))
        require(len(lookup) == 192*len(arms), 'Incomplete arm coverage')
        groups = {}
        for task in (*TASKS,'all'):
            ids = sorted(i for i,r in imap.items() if task == 'all' or r['task'] == task)
            require(len(ids) == (192 if task == 'all' else 64), 'Task denominator')
            group = {'n':len(ids),'arms':{}}
            for arm in arms:
                vv = [lookup[arm,i][1] for i in ids]; bb = [lookup[arms[0],i][1] for i in ids]
                group['arms'][arm] = {
                    'correct':sum(v['correct'] for v in vv),
                    'prediction_histogram':{x:sum(v['prediction']==j for v in vv) for j,x in enumerate('ABCD')},
                    'full_argmax_allowed_recorded':sum(lookup[arm,i][0]['score']['full_argmax_allowed'] for i in ids),
                    'mean_label_mass':mean([v['mass'] for v in vv]),
                    'mean_negative_log_mass':mean([v['mass_nll'] for v in vv]),
                    'negative_log_mean_mass':-math.log(mean([v['mass'] for v in vv])),
                    **{'delta_'+k:mean([v[k]-b[k] for b,v in zip(bb,vv)]) for k in ['full','choice','mass_nll']},
                    'regression':sum(b['correct'] and not v['correct'] for b,v in zip(bb,vv)),
                    'gain':sum(not b['correct'] and v['correct'] for b,v in zip(bb,vv))}
            groups[task] = group
        result['tracks'][track] = groups
        structure = {}
        for split in ('smoke','calibration','development','heldout'):
            all_rows = load(f'inputs/{track}/{split}.json')
            code = [r for r in all_rows if r['task']=='code']; mapping = Counter()
            for r in code:
                index = int(r['id'].rsplit('-',1)[1]); f = r['facts']
                require(f['count'] == 2+index%4 and r['gold'] == index%4, 'Count/label rule differs')
                expected = (f['start']+f['step']*f['count']*(f['count']-1)//2)%f['modulus']
                options = list(map(int,r['options']))
                require(int(r['gold_value'])==expected==options[r['gold']], 'Code oracle mismatch')
                require(sorted(options)==[expected,expected+1,expected+3,expected+7], 'Foil rule differs')
                mapping[str(f['count'])+'->'+'ABCD'[r['gold']]] += 1
            structure[split] = {'code_prompts':len(code),'count_to_gold_label':dict(sorted(mapping.items())),
                'gold_is_smallest_option':sum(int(r['gold_value'])==min(map(int,r['options'])) for r in code)}
        result['code_structure'][track] = structure
    sources['../../tools/modelpack/data.py'] = sha(case.parents[1]/'tools/modelpack/data.py')
    result['sources'] = sources
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();case=a.case.resolve();out=a.output.resolve()
    require(not out.exists() and not out.is_relative_to(case), 'Use a new external output')
    result=analyze(case);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
    print('PASS: same-record scores, task/arm coverage, generator structure')

if __name__ == '__main__': main()
