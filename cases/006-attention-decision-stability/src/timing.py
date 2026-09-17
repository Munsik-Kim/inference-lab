"""Timing primitives; CPU mocks are tests, never GPU evidence."""
from __future__ import annotations
import time
import numpy as np


def wall_block(call, synchronize, calls: int = 5, clock=time.perf_counter) -> float:
    if calls <= 0:
        raise ValueError('Positive call count required')
    synchronize()
    start = clock()
    for _ in range(calls):
        call()
    synchronize()
    elapsed = clock()-start
    if elapsed <= 0:
        raise ValueError('Invalid clock result')
    return elapsed/calls


def paired_ratios(rows: list[dict], arm: str) -> list[dict]:
    index={}
    for r in rows:
        if r.get('evidence_kind')!='gpu_timing' or r.get('mock'):
            raise ValueError('Mock/non-GPU timing cannot enter published timing')
        if not np.isfinite(r['seconds_per_call']) or r['seconds_per_call']<=0:
            raise ValueError('Invalid time')
        key=(r['base_id'],r['process'],r['block'],r['arm'])
        if key in index:
            raise ValueError('Duplicate timing cell')
        index[key]=r
    output=[]
    for key,r in sorted(index.items()):
        if key[-1]!='B':
            continue
        candidate=index.get((*key[:-1],arm))
        if candidate is None or candidate['token_hash']!=r['token_hash']:
            raise ValueError('Missing or mismatched timing pair')
        output.append({k:r[k] for k in ('base_id','process','block')} |
                      {'ratio':r['seconds_per_call']/candidate['seconds_per_call']})
    if len(output)!=sum(r['arm']==arm for r in rows):
        raise ValueError('Unpaired candidate timing')
    return output


def timing_interval(pairs: list[dict], seed: int = 606902, repetitions: int = 5000) -> dict:
    docs=sorted({r['base_id'] for r in pairs}); processes=sorted({r['process'] for r in pairs})
    if not docs or not processes:
        raise ValueError('No paired timing')
    cells={(d,p):np.array([r['ratio'] for r in pairs if r['base_id']==d and r['process']==p])
           for d in docs for p in processes}
    if any(not len(x) for x in cells.values()) or len({len(x) for x in cells.values()})!=1:
        raise ValueError('Missing/unbalanced document/process cells')
    rng=np.random.default_rng(seed); boot=[]
    for _ in range(repetitions):
        ds=rng.integers(len(docs),size=len(docs))
        ps=rng.integers(len(processes),size=len(processes)) # Shared process draw across documents.
        values=[]
        for d in ds:
            for p in ps:
                cell=cells[docs[d],processes[p]]
                values.extend(cell[rng.integers(len(cell),size=len(cell))])
        boot.append(float(np.median(values)))
    return {'paired_ratio_median':float(np.median([r['ratio'] for r in pairs])),
            'ci95':np.quantile(boot,[.025,.975],method='linear').tolist(),
            'independent_base_scenarios':len(docs),'local_process_rounds':len(processes),
            'scope':'Pointwise document/global-process/paired-block resampling; same hardware',
            'latency_scope':'Block-mean per-call; not service-request p95'}
