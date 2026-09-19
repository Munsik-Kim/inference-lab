"""Independent paired cost means and frozen hierarchical interval calculation."""
from pathlib import Path
import argparse,json,math
import numpy as np


def verify(case,summary,output):
    s=json.loads(summary.read_text());checks={}
    for track,baseline in [('Q','Q-BF16'),('R','R-B')]:
        records=json.loads((case/f'results/raw/{track}/timing.json').read_text());table={}
        for r in records:
            key=(r['arm'],r['boundary'],r['round'],r['id'],r['block'])
            if key in table or r['calls']!=3 or not math.isfinite(r['wall_seconds_per_call']) or r['wall_seconds_per_call']<=0:raise ValueError('Invalid/duplicate timing cell')
            table[key]=r['wall_seconds_per_call']
        ids=sorted({r['id'] for r in records})
        expected=json.loads((case/f'inputs/{track}/manifest.json').read_text())['timing_ids']
        if len(ids)!=12 or set(ids)!=set(expected):raise ValueError('Timing input manifest mismatch')
        rng=np.random.default_rng(808192)
        draws=rng.integers(0,3,(5000,3,1,1));units=np.concatenate([rng.integers(j*4,(j+1)*4,(5000,1,4,1)) for j in range(3)],axis=2)
        blocks=rng.integers(0,5,(5000,3,12,5))
        for boundary,arms in s['timing'][track].items():
            arrays={a:np.array([[[table[a,boundary,r,i,k] for k in range(5)] for i in ids] for r in [1,2,3]]) for a in arms}
            for arm,row in arms.items():
                mean=math.fsum(arrays[arm].ravel())/180
                if not math.isclose(mean*1000,row['wall_ms'],rel_tol=1e-12):raise ValueError('Timing mean mismatch')
                ratios=arrays[baseline][draws,units,blocks].sum(axis=(1,2,3))/arrays[arm][draws,units,blocks].sum(axis=(1,2,3))
                np.testing.assert_allclose(np.percentile(ratios,[2.5,97.5]),row['ci95'],rtol=1e-12,atol=1e-12)
        checks[track]={'records':len(records),'scenarios':12,'rounds':3,'status':'PASS'}
    if output.exists():raise ValueError('New output file required')
    output.write_text(json.dumps({'status':'PASS','tracks':checks,'scope':'paired raw wall-clock block means and same frozen hierarchical resampling; no new timing'},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.case,a.summary,a.output)
