"""Gold-grounded measured summaries. Standard, selected stress and lengths separate."""
from __future__ import annotations
import numpy as np
from .metrics import paired,score_options,outcomes
from .statistics import paired_join,scenario_draws,balanced_interval,gap_bin,zero_event_upper
from .storage import digest
from .validity import assess
from .tasks import TASKS


def distribution(values):
    x=np.asarray([v for v in values if v is not None],dtype=np.float64)
    if not x.size:return {'n':0,'mean':None,'median':None,'p95':None,'min':None,'max':None}
    if not np.isfinite(x).all():raise ValueError('Nonfinite summary input')
    return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),
            'p95':float(np.quantile(x,.95,method='linear')),'min':float(x.min()),'max':float(x.max())}


def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(center-half),float(center+half)]


def analyze_records(records):
    groups=paired_join(records);pairs=[]
    for item,group in groups.items():
        for row in group.values():
            if not assess(row.get('validity',{}))['valid']:raise ValueError('Invalid full-output evidence')
            row['recalculated_score']=score_options(row['option_logits'],row['gold_index'],row['full_lse'],row['full_argmax'],row['label_ids'])
        b=group['B']
        for arm in ('A_PUBLIC','V4'):
            c=group[arm];pair=paired(b['recalculated_score'],c['recalculated_score'])
            pair.update({k:b[k] for k in ('item_id','base_id','task','split','length','token_hash')})
            pair.update({'arm':arm,'B_score':b['recalculated_score'],'candidate_score':c['recalculated_score'],
                         'delta_accuracy':float(c['recalculated_score']['correct'])-float(b['recalculated_score']['correct']),
                         'full_vocab_kl':c['full_vocab_kl_B_to_candidate'],'local_reference_error':c['local_pooled']['relative_error'],
                         'local_last_query_error':c['local_last_query']['relative_error'],
                         'native_pair_local':c['native_pair_local'],'hidden_differences':c['hidden_differences'],
                         'near_tie':b['recalculated_score']['winner_gap']<.1,
                         'noise_sensitive':b['recalculated_score']['winner_gap']==0,
                         'validity':c['validity'],'gap_bin':gap_bin(b['recalculated_score']['winner_gap'])})
            pairs.append(pair)
    summary={'scope':'One layer (13) prompt prefill, fixed BF16 Qwen3-0.6B weights; no downstream deployment verdict',
             'deployment_verdict':'NOT_ASSESSED','groups':{},'paired_items':len(pairs),'measured_arm_records':len(records)}
    for split,length in sorted({(r['split'],r['length']) for r in pairs}):
        subset=[r for r in pairs if r['split']==split and r['length']==length]
        key=f'{split}/L{length}';entry={'independent_scenarios':len({r['base_id'] for r in subset}),'arms':{},
             'evidence_kind':'BASELINE_CONDITIONED_STRESS' if split=='boundary_pool' else 'STANDARD_EVAL' if split=='standard' and length==4096 else 'SECONDARY_LENGTH' if split=='standard' else 'DEVELOPMENT'}
        tasks_present={r['task'] for r in subset}
        draws=scenario_draws(subset) if tasks_present==set(TASKS) else None
        entry['bootstrap_indices_sha256']=digest(draws['indices']) if draws else None
        for arm in ('A_PUBLIC','V4'):
            rows=[r for r in subset if r['arm']==arm]
            stats={'outcomes':outcomes(rows),'B_accuracy':float(np.mean([r['B_score']['correct'] for r in rows])),
                   'candidate_accuracy':float(np.mean([r['candidate_score']['correct'] for r in rows])),
                   'task_balanced':{m:balanced_interval(rows,m,draws) for m in ('delta_choice_nll','delta_brier','delta_gold_margin','delta_accuracy')} if draws else None,
                   'B_choice_nll':distribution([r['B_score']['choice_nll'] for r in rows]),
                   'candidate_choice_nll':distribution([r['candidate_score']['choice_nll'] for r in rows]),
                   'B_brier':distribution([r['B_score']['choice_brier'] for r in rows]),
                   'candidate_brier':distribution([r['candidate_score']['choice_brier'] for r in rows]),
                   'B_label_mass':distribution([r['B_score']['label_mass'] for r in rows]),
                   'candidate_label_mass':distribution([r['candidate_score']['label_mass'] for r in rows]),
                   'B_full_gold_nll':distribution([r['B_score']['full_gold_nll'] for r in rows]),
                   'candidate_full_gold_nll':distribution([r['candidate_score']['full_gold_nll'] for r in rows]),
                   'B_full_argmax_allowed':sum(r['B_score']['full_argmax_allowed'] for r in rows),
                   'candidate_full_argmax_allowed':sum(r['candidate_score']['full_argmax_allowed'] for r in rows),
                   'B_gap':distribution([r['base_gap'] for r in rows]),'full_vocab_kl':distribution([r['full_vocab_kl'] for r in rows]),
                   'local_reference_error':distribution([r['local_reference_error'] for r in rows]),
                   'near_zero_local_count':sum(r['local_reference_error'] is None for r in rows),'R_undefined_count':sum(r['R'] is None for r in rows),
                   'per_task':{},'bins':[],'predictive_AUC':'NOT_ESTIMABLE_NO_PREDICTIVE_PROTOCOL'}
            for task in TASKS:
                t=[r for r in rows if r['task']==task]
                stats['per_task'][task]={'n':len(t),'outcomes':outcomes(t),
                    'B_correct':sum(r['B_score']['correct'] for r in t),'candidate_correct':sum(r['candidate_score']['correct'] for r in t),
                    'mean_delta_nll':float(np.mean([r['delta_choice_nll'] for r in t])) if t else None}
                if draws and t:
                    values=np.array([next(r['delta_choice_nll'] for r in t if r['base_id']==sid) for sid in draws['ids'][task]])
                    boot=values[np.asarray(draws['indices'][task])].mean(axis=1)
                    stats['per_task'][task]['delta_nll_ci95']=np.quantile(boot,[.025,.975],method='linear').tolist()
                for label in ('EXACT_TIE','[0.0,0.1)','[0.1,0.5)','[0.5,1.0)','[1.0,inf)'):
                    bin_rows=[r for r in t if r['gap_bin']==label]
                    counts=outcomes(bin_rows)
                    stats['bins'].append({'task':task,'gap_bin':label,**counts,
                         'flip_wilson95':wilson(counts['flips']['numerator'],counts['flips']['denominator']),
                         'regression_wilson95':wilson(counts['regression_given_B_correct']['numerator'],counts['regression_given_B_correct']['denominator']),
                         'interval_scope':'Pointwise binomial-style descriptive interval within this bin; stress selection is not ordinary-workload sampling'})
            stats['zero_flip_sensitivity_limit']=zero_event_upper(len(rows)) if not stats['outcomes']['flips']['numerator'] and split=='standard' and length==4096 else None
            entry['arms'][arm]=stats
        summary['groups'][key]=entry
    summary['example_selection']={}
    primary=[r for r in pairs if r['split']=='standard' and r['length']==4096]
    for arm in ('A_PUBLIC','V4'):
        rows=sorted([r for r in primary if r['arm']==arm],key=lambda r:(abs(r['delta_choice_nll']),r['item_id']))
        summary['example_selection'][arm]={'rule':'Post-hoc descriptive examples; no representative rate claim','median_absolute_nll_change':rows[len(rows)//2]['item_id'] if rows else None,'largest_absolute_nll_change':rows[-1]['item_id'] if rows else None,
             'first_regression_id':min((r['item_id'] for r in rows if r['cell']=='regression'),default=None),
             'first_gain_id':min((r['item_id'] for r in rows if r['cell']=='gain'),default=None)}
    return summary,pairs
