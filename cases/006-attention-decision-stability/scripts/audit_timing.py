"""Independent scalar timing audit; no import of the production timing analyzer."""
import argparse, json, math, statistics, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.storage import load, write_new


def check(cells, summary):
    records = [r for cell in cells for r in cell['rows']]
    index = {}
    for r in records:
        key = (r['base_id'], r['process'], r['block'], r['arm'])
        if key in index or r['evidence_kind'] != 'gpu_timing' or r.get('mock'):
            raise ValueError('Invalid timing identity or evidence kind')
        if not math.isfinite(r['seconds_per_call']) or r['seconds_per_call'] <= 0:
            raise ValueError('Nonpositive/nonfinite time')
        index[key] = r
    docs = sorted({r['base_id'] for r in records})
    processes = sorted({r['process'] for r in records})
    if len(docs) != 12 or processes != [0, 1, 2]:
        raise ValueError('Incomplete fixed timing design')
    output = {}
    for arm in ('A_PUBLIC', 'V4'):
        ratios = {}
        for d in docs:
            for p in processes:
                cell = []
                for block in range(5):
                    b, c = index[d,p,block,'B'], index[d,p,block,arm]
                    if b['token_hash'] != c['token_hash']:
                        raise ValueError('Mismatched paired inputs')
                    cell.append(b['seconds_per_call']/c['seconds_per_call'])
                ratios[d,p] = cell
        rng = np.random.default_rng(606902)
        boot = []
        for _ in range(5000):
            ds = rng.integers(len(docs), size=len(docs))
            ps = rng.integers(len(processes), size=len(processes))
            draw = []
            for di in ds:
                for pi in ps:
                    values = ratios[docs[di],processes[pi]]
                    draw.extend(values[i] for i in rng.integers(5,size=5))
            boot.append(statistics.median(draw))
        # Independent linear order-statistic interpolation.
        ordered = sorted(boot)
        def quantile(q):
            pos=q*(len(ordered)-1); lo=math.floor(pos); frac=pos-lo
            return ordered[lo]*(1-frac)+ordered[min(lo+1,len(ordered)-1)]*frac
        actual = {'median': statistics.median([v for vs in ratios.values() for v in vs]),
                  'ci95': [quantile(.025), quantile(.975)]}
        target=summary['model_timing'][arm]
        if not math.isclose(actual['median'],target['paired_ratio_median'],abs_tol=1e-12):
            raise ValueError('Timing point estimate differs')
        if any(not math.isclose(x,y,abs_tol=1e-12) for x,y in zip(actual['ci95'],target['ci95'])):
            raise ValueError('Timing interval differs')
        output[arm]=actual
    return {'status':'PASS','raw_timing_rows':len(records),'documents':len(docs),
            'shared_local_processes':len(processes),'recomputed':output,
            'scope':'Independent scalar aggregation and resampling; not new GPU replication'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timing',type=Path,required=True);p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cells=[load(f)['payload'] for f in sorted(a.timing.glob('*.json'))] if a.timing.is_dir() else load(a.timing)
    result=check(cells,load(a.summary));write_new(a.output,result);print(json.dumps(result))


if __name__=='__main__':main()
