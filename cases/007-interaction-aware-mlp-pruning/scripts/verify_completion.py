"""Additional independent CPU validation of completion, model and timing intervals.

Added for the resumed run; does not change frozen measurement/analysis code.
Checks retained scalars, not independent GPU replication or full-vocabulary KL.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

TASKS = ['RETRIEVAL', 'COMPARISON', 'CODE']
SEED = 707180

def read(path):
    return json.loads(path.read_text())

def close(actual, expected):
    if not np.allclose(actual, expected, rtol=1e-10, atol=1e-11):
        raise ValueError('Independent interval or scalar mismatch')

def draws(tasks):
    rng = np.random.default_rng(SEED)
    return np.concatenate([rng.choice(np.flatnonzero(np.array(tasks) == task),
                            (5000, tasks.count(task)), replace=True)
                           for task in TASKS], axis=1)

def score(row):
    s = row['score']['option_logits']; gold = row['score']['gold']
    z = max(s); logsum = z + math.log(math.fsum(math.exp(x-z) for x in s))
    return [logsum-s[gold],
            math.fsum((math.exp(x-logsum)-(j == gold))**2 for j, x in enumerate(s)),
            s[gold]-max(x for j, x in enumerate(s) if j != gold)]

def require_validity(row, selection_hash):
    if not (row.get('valid') is True and row.get('full_blocks_checked') == 28
            and row.get('full_final_norm_checked') is True
            and row.get('full_logits_finite') is True
            and row.get('selection_sha256') == selection_hash):
        raise ValueError('Missing full-output validity')
    if row['method'] != 'B':
        e = row['integrated_vs_standalone']['relative_error']
        if e is None or not math.isfinite(e) or e > .01:
            raise ValueError('Invalid integrated/standalone match')

def verify(case, summary):
    raw = case/'results/raw'; selection = read(raw/'selection.json')
    methods = ['B', *selection['selections']]
    selhash = hashlib.sha256((raw/'selection.json').read_bytes()).hexdigest()
    rows = [read(p) for p in sorted((raw/'heldout').glob('c007-*.json'))]
    lookup = {(r['id'], r['method']): r for r in rows}
    ids = sorted(r['id'] for r in read(case/'inputs/HELD_OUT.json'))
    if len(lookup) != len(rows) or set(lookup) != {(i,m) for i in ids for m in methods}:
        raise ValueError('Incomplete/duplicate held-out cells')
    for r in rows:
        require_validity(r, selhash)
    tasks = [lookup[i,'B']['task'] for i in ids]; ix = draws(tasks)
    for m in methods:
        values = np.array([score(lookup[i,m]) for i in ids])
        baseline = np.array([score(lookup[i,'B']) for i in ids])
        diff = values-baseline
        for k, name in enumerate(['nll','brier','margin']):
            target = summary['model'][m]['delta_'+name]
            close(diff[:,k].mean(), target['mean'])
            close(np.quantile(diff[ix,k].mean(axis=1), [.025,.975]), target['ci95'])
        for task in TASKS:
            actual=sum(lookup[i,m]['score']['prediction']==lookup[i,m]['score']['gold']
                       for i in ids if lookup[i,m]['task']==task)
            close(actual,summary['model'][m]['per_task_correct'][task])
    notes=read(case/'results/derived/posthoc_completion_notes.json')
    for budget in [4,8]:
        per=notes['direct_model_contrasts'][str(budget)]
        iv=np.array([score(lookup[i,f'INDEPENDENT_{budget}']) for i in ids])
        pv=np.array([score(lookup[i,f'PAIRWISE_{budget}']) for i in ids])
        for k,name in enumerate(['nll','brier','margin']):
            delta=pv[:,k]-iv[:,k]
            close(delta.mean(),per[name]['mean_pairwise_minus_independent'])
            close(np.quantile(delta[ix].mean(axis=1),[.025,.975]),per[name]['ci95'])
        correctness=lambda r:int(r['score']['prediction']==r['score']['gold'])
        delta=np.array([correctness(lookup[i,f'PAIRWISE_{budget}'])-
                        correctness(lookup[i,f'INDEPENDENT_{budget}']) for i in ids])
        close(delta.mean(),per['correct']['mean_pairwise_minus_independent'])
        close(np.quantile(delta[ix].mean(axis=1),[.025,.975]),per['correct']['ci95'])
    timing = []
    for rr in range(3):
        stage = read(raw/f'timing{rr}/summary.json')
        if stage.get('status') != 'PASS' or stage.get('round') != rr:
            raise ValueError('Timing stage incomplete')
        timing.extend(read(raw/f'timing{rr}/blocks.json'))
    rng = np.random.default_rng(SEED)
    for boundary in ['MLP','MODEL_PREFILL']:
        cell = [r for r in timing if r['boundary'] == boundary]
        lut = {(r['id'], r['round'], r['block'], r['method']):r for r in cell}
        tids = sorted({r['id'] for r in cell})
        expected = {(i,rr,b,m) for i in tids for rr in range(3) for b in range(5) for m in methods}
        if len(tids) != 6 or len(lut) != len(cell) or set(lut) != expected:
            raise ValueError('Timing pairing incomplete')
        ti = draws([lut[i,0,0,'B']['task'] for i in tids])
        rounds = rng.integers(0,3,(5000,3))
        blocks = rng.integers(0,5,(5000,6,3,5))
        # Assemble independent sample arrays by explicit cell lookup. No analysis import.
        for m in methods:
            samples = np.empty((5000,6,3,5))
            for d in range(6):
                for rr in range(3):
                    for b in range(5):
                        values = np.array([lut[i,r,k,'B']['wall_ms']/lut[i,r,k,m]['wall_ms']
                                           for i in tids for r in range(3) for k in range(5)])
                        keys = ti[:,d]*15+rounds[:,rr]*5+blocks[:,d,rr,b]
                        samples[:,d,rr,b] = values[keys]
            ci = np.quantile(np.median(samples.reshape(5000,-1), axis=1), [.025,.975])
            close(ci, summary['timing'][boundary][m]['ci95'])
    return {'status':'PASS','heldout_cells':len(rows),'model_interval_checks':15,
            'posthoc_direct_model_contrast_checks':8,
            'timing_interval_checks':10,'timing_records':len(timing),
            'full_output_validity':'PASS','audit_scope':'Independent scalar/interval consistency; not GPU replication; full-vocabulary KL not reconstructable from public options'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.case.resolve()):
        raise ValueError('New external output required')
    result=verify(a.case,read(a.summary))
    a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))

if __name__ == '__main__':
    main()
