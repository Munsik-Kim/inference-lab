"""Post-hoc descriptive M2-minus-M1 model contrasts and timing variability.

Uses completed frozen measurements only; does not alter primary study status.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import bootstrap_indices,interval

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    read=lambda p:json.loads(p.read_text())
    raw=a.case/'results/raw';rows=[read(p) for p in sorted((raw/'heldout').glob('c007-*.json'))]
    lut={(r['id'],r['method']):r for r in rows};ids=sorted({r['id'] for r in rows})
    if len(lut)!=960 or len(ids)!=192:raise ValueError('Incomplete evidence')
    ix=bootstrap_indices([lut[i,'B']['task'] for i in ids])
    result={'evidence_kind':'POST_HOC_DESCRIPTIVE_COMPLETION_NOTES','selection_changed':False,
            'direct_model_contrasts':{},'per_task':{},'timing_by_process':{}}
    for budget in [4,8]:
        per={}
        for metric in ['nll','brier','margin','correct']:
            diff=[float(lut[i,f'PAIRWISE_{budget}']['score'][metric])-float(lut[i,f'INDEPENDENT_{budget}']['score'][metric]) for i in ids]
            per[metric]={'mean_pairwise_minus_independent':float(np.mean(diff)),
                         'ci95':interval(diff,ix),'n':192}
        result['direct_model_contrasts'][str(budget)]=per
    for task in ['RETRIEVAL','COMPARISON','CODE']:
        result['per_task'][task]={}
        for m in ['B','INDEPENDENT_4','PAIRWISE_4','INDEPENDENT_8','PAIRWISE_8']:
            rs=[r for r in rows if r['task']==task and r['method']==m]
            result['per_task'][task][m]={'n':len(rs),'correct':sum(r['score']['correct'] for r in rs),
                'mean_nll':float(np.mean([r['score']['nll'] for r in rs]))}
    for rr in range(3):
        tr=read(raw/f'timing{rr}/blocks.json');per={}
        for boundary in ['MLP','MODEL_PREFILL']:
            per[boundary]={}
            for m in sorted({r['method'] for r in tr}):
                vals=[r['wall_ms'] for r in tr if r['boundary']==boundary and r['method']==m]
                per[boundary][m]={'n':len(vals),'median_ms':float(np.median(vals)),
                                 'p95_block_mean_ms':float(np.quantile(vals,.95))}
        result['timing_by_process'][str(rr)]=per
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['direct_model_contrasts']))

if __name__=='__main__':main()
