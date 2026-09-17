"""Measured-only summary builder. Empty results stay empty."""
from __future__ import annotations
from .metrics import outcomes, paired, score_options
from .statistics import balanced_interval, paired_join, scenario_draws
from .storage import load
from .validity import assess, study_status


def summarize(case):
    records=load(case/'results/records.json')
    status=study_status(records,576,load(case/'results/semantic_status.json'))
    manifest=load(case/'inputs/manifest.json')
    result={'status':status,'prepared_inputs':manifest['counts'],
            'measured_record_count':len(records),'arms':{a:{'status':'NOT_RUN','metrics':None} for a in ('B','A_PUBLIC','V4')},
            'measurement_note':'No test/mock records may enter this report',
            'deployment_verdict':'NOT_ASSESSED'}
    if not records:
        return result, []
    joined=paired_join(records)
    pairs=[]
    for item,group in joined.items():
        for r in group.values():
            if not assess(r.get('validity',{}))['valid']:
                raise ValueError('Invalid/missing validity evidence; stop successful summaries')
            r['score']=score_options(r['option_logits'],r['gold_index'],r['full_lse'],r['full_argmax'],r['label_ids'])
        for arm in ('A_PUBLIC','V4'):
            b,c=group['B'],group[arm]
            pairs.append({k:b[k] for k in ('item_id','base_id','task','split','length')} |
                         {'arm':arm,**paired(b['score'],c['score'])})
    primary=[r for r in pairs if r['split']=='standard' and r['length']==4096]
    if primary:
        draws=scenario_draws(primary)
        for arm in ('A_PUBLIC','V4'):
            rows=[r for r in primary if r['arm']==arm]
            result['arms'][arm]={'status':'MEASURED','outcomes':outcomes(rows),
                               'metrics':{m:balanced_interval(rows,m,draws) for m in ('delta_choice_nll','delta_brier','delta_gold_margin')}}
        result['arms']['B']={'status':'MEASURED_CONTROL','metrics':None}
        result['bootstrap_draws']=draws
    return result,pairs
