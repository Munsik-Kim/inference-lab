"""Rebuild tables and four figures from saved unit/row measurement files."""
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare import CASE, write_json
from audit import clean

LABELS = {'nearest_ceil': 'Nearest, headroom scale', 'nearest_efq_scale': 'Nearest, EFQ scale',
          'efq_mmlu': 'EFQ-MMLU', 'efq_mean': 'EFQ-Mean', 'efq_balance': 'EFQ-Balance',
          'efq_lut': 'EFQ-LUT', 'efq_calibrated': 'EFQ calibrated',
          'online_fp32': 'FP32 online', 'dense_fp32': 'FP32 dense'}


def read_units(stage):
    return [json.loads(x) for x in (CASE/'results'/f'{stage}_units.jsonl').read_text().splitlines()]


def stats(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return {'n': len(values), 'median': float(np.median(values)) if values else None,
            'p95': float(np.quantile(values, .95)) if values else None,
            'worst': float(max(values)) if values else None}


def grouped(units, keys):
    buckets = defaultdict(list)
    for u in units:
        buckets[tuple(u[k] for k in keys)].append(u)
    out = []
    for key, rows in sorted(buckets.items()):
        out.append({**dict(zip(keys, key)), **stats([r['relative_error'] for r in rows]),
                    'absolute_frobenius': stats([r['absolute_error_frobenius'] for r in rows]),
                    'near_zero_reference_units': sum(r['near_zero_reference'] for r in rows),
                    'failed_valid_rows': sum(r['failed_valid_rows'] for r in rows)})
    return out


def screen(units, plan):
    primary = [u for u in units if u['block'] == plan['primary_block']]
    baseline = plan['screen']['baseline']
    floor = plan['screen']['baseline_relative_error_floor']
    checks = []
    for length in plan['lengths']:
        groups = {method: stats([u['relative_error'] for u in primary if u['length'] == length and u['method'] == method])
                  for method in [baseline, 'efq_calibrated']}
        for metric, limit in [('median', 1.10), ('p95', 1.25)]:
            b, e = groups[baseline][metric], groups['efq_calibrated'][metric]
            ratio = e/b if b is not None and b > floor and e is not None else None
            checks.append({'length': length, 'scope': metric, 'baseline': b, 'calibrated': e,
                           'ratio': ratio, 'limit': limit, 'pass': None if ratio is None else ratio <= limit})
        for layer in plan['layers']:
            values = {m: stats([u['relative_error'] for u in primary if u['length'] == length and u['layer'] == layer and u['method'] == m])['median'] for m in [baseline, 'efq_calibrated']}
            b, e = values[baseline], values['efq_calibrated']
            ratio = e/b if b is not None and b > floor and e is not None else None
            checks.append({'length': length, 'layer': layer, 'scope': 'layer_length_median', 'baseline': b,
                           'calibrated': e, 'ratio': ratio, 'limit': 2., 'pass': None if ratio is None else ratio <= 2.})
    bad = sum(u['failed_valid_rows'] for u in primary if u['method'] == 'efq_calibrated')
    near = sum(u['near_zero_reference'] for u in primary if u['method'] == 'efq_calibrated')
    passed = all(c['pass'] is True for c in checks) and bad == 0 and near == 0
    return {'baseline': baseline, 'passed': passed, 'decision': 'CONSIDER_LIMITED_KERNEL_FEASIBILITY' if passed else 'HOLD_KERNEL_WORK',
            'checks': checks, 'failed_valid_rows': bad, 'near_zero_reference_units': near,
            'ratio_holds': sum(c['pass'] is None for c in checks)}


def rankdata(x):
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), dtype=float)
    sorted_x = np.asarray(x)[order]
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]: end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def main():
    spec = json.loads((CASE/'configs/experiment_spec.json').read_text())
    plan = spec['plan']
    units = read_units('eval')
    dev = read_units('dev')
    synth = read_units('synthetic-eval')
    primary = [u for u in units if u['block'] == 32]
    summary = {'length': grouped(primary, ['length', 'method']),
               'layer_length': grouped(primary, ['length', 'layer', 'method']),
               'head_length': grouped(primary, ['length', 'head', 'method']),
               'layer': grouped(primary, ['layer', 'method']),
               'language': grouped(primary, ['language', 'method']),
               'sensitivity': grouped(units, ['block', 'length', 'method']),
               'synthetic': grouped(synth, ['family', 'length', 'block', 'method']),
               'screen': screen(units, plan)}
    # Compare calibration objective on the exact same subset definition,
    # changing only the document split. This is not a refit on evaluation.
    transfer = []
    c = plan['calibration']
    for stage, data in [('dev', dev), ('eval', units)]:
        for method in plan['methods']:
            subset = [u for u in data if u['block'] == 32 and u['length'] == c['length'] and u['layer'] == c['layer'] and u['head'] in c['heads'] and u['method'] == method]
            value = math.sqrt(sum(u['error_squared'] for u in subset)/sum(u['reference_squared'] for u in subset))
            transfer.append({'split': stage, 'method': method, 'head_units': len(subset), 'relative_frobenius': value})
    summary['calibration_transfer'] = transfer
    chosen = json.loads((CASE/'results/calibration.json').read_text())
    measured = next(x['relative_frobenius'] for x in transfer if x['split']=='dev' and x['method']=='efq_calibrated')
    assert abs(measured - chosen['selected_objective']) < 2e-6, 'Calibration/evaluator objective mismatch.'
    rng = np.random.default_rng(plan['metrics']['bootstrap']['seed'])
    bootstrap = []
    docs = sorted({u['document_id'] for u in primary})
    assert len(docs) == 16
    for length in plan['lengths']:
        values = []
        for doc in docs:
            mean = {m: np.mean([u['relative_error'] for u in primary if u['document_id']==doc and u['length']==length and u['method']==m]) for m in ['efq_calibrated', 'nearest_efq_scale']}
            values.append(mean['efq_calibrated']-mean['nearest_efq_scale'])
        draws = rng.integers(0, len(docs), size=(2000, len(docs)))
        samples = np.array(values)[draws].mean(1)
        bootstrap.append({'length': length, 'mean_document_error_difference': float(np.mean(values)), 'ci95': np.quantile(samples,[.025,.975]).tolist(), 'documents':16})
    summary['paired_document_bootstrap'] = bootstrap
    row_data = []
    with gzip.open(CASE/'results/eval_rows.csv.gz', 'rt') as f:
        for row in csv.DictReader(f):
            if row['block']=='32': row_data.append(row)
    associations = []
    for method in ['efq_mean', 'efq_calibrated']:
        rows = [r for r in row_data if r['method']==method]
        for field in ['score_spread','entropy_nats','top_logit_gap','removed_reference_mass','max_row_max_jump']:
            vals = [(float(r[field]),float(r['relative_error'])) for r in rows if r[field] and r['relative_error']]
            x,y=np.array(vals).T
            good=np.isfinite(x)&np.isfinite(y); x,y=x[good],y[good]
            corr=float(np.corrcoef(rankdata(x),rankdata(y))[0,1]) if np.std(x)>0 and np.std(y)>0 else None
            associations.append({'method':method,'feature':field,'row_pairs':len(x),'spearman':corr,
                                 'scope':'Exploratory pooled dependent rows; no p-value or causal claim.'})
    summary['exploratory_associations'] = associations
    summary['row_diagnostics'] = []
    for method in plan['methods']:
        rows = [r for r in row_data if r['method']==method]
        summary['row_diagnostics'].append({'method':method, **{field:stats([float(r[field]) for r in rows if r[field]]) for field in ['cosine','js_nats','top8_overlap','effective_probability_tie_fraction','e2m1_zero_fraction','removed_reference_mass','abs_error']}})
    # Representative units: worst EFQ-Mean, worst calibrated, and largest
    # calibrated/nearest ratio above floor. Full unit list remains available.
    keys = ['document_id','length','layer','head']
    base_index = {tuple(u[k] for k in keys):u for u in primary if u['method']=='nearest_efq_scale'}
    representatives=[]
    for method in ['efq_mean','efq_calibrated']:
        worst=max((u for u in primary if u['method']==method and u['relative_error'] is not None),key=lambda u:u['relative_error'])
        representatives.append({'selection':'largest_absolute_relative_output_error',**worst,'same_scale_nearest':base_index[tuple(worst[k] for k in keys)]['relative_error']})
    candidates=[u for u in primary if u['method']=='efq_calibrated' and base_index[tuple(u[k] for k in keys)]['relative_error'] > 1e-5]
    worst=max(candidates,key=lambda u:u['relative_error']/base_index[tuple(u[k] for k in keys)]['relative_error'])
    representatives.append({'selection':'largest_calibrated_to_same_scale_nearest_ratio',**worst,'same_scale_nearest':base_index[tuple(worst[k] for k in keys)]['relative_error']})
    summary['representative_units']=representatives
    write_json(CASE/'results/aggregate.json',clean(summary))
    flat=[]
    for x in summary['length']:
        flat.append({k:x[k] for k in ['length','method','n','median','p95','worst','near_zero_reference_units','failed_valid_rows']})
    with (CASE/'results/length_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    plt.rcParams.update({'font.size':10,'figure.dpi':130,'axes.spines.top':False,'axes.spines.right':False})
    methods=['nearest_ceil','nearest_efq_scale','efq_mmlu','efq_mean','efq_balance','efq_lut','efq_calibrated']
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),sharey=True)
    for ax,n in zip(axes,plan['lengths']):
        rows={x['method']:x for x in summary['length'] if x['length']==n}
        med=[rows[m]['median'] for m in methods];hi=[rows[m]['p95'] for m in methods]
        ax.bar(np.arange(len(methods)),med,color=['#78828d','#435b70','#b8a085','#b67b4b','#cfad6d','#74998b','#287b75'])
        ax.scatter(np.arange(len(methods)),hi,marker='_',s=90,color='black',label='p95')
        ax.set_xticks(np.arange(len(methods)),[LABELS[m] for m in methods],rotation=60,ha='right');ax.set_title(f'{n} tokens; 192 head units')
    axes[0].set_ylabel('Relative output Frobenius error');axes[-1].legend();fig.suptitle('Real Qwen attention: median bars and p95 marks, block 32');fig.tight_layout();fig.savefig(CASE/'results/length_error.png');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,layer in zip(axes,plan['layers']):
        mm=['nearest_efq_scale','efq_mean','efq_calibrated']
        arrays=[[u['relative_error'] for u in primary if u['layer']==layer and u['method']==m and u['relative_error'] is not None] for m in mm]
        ax.boxplot(arrays,tick_labels=[LABELS[m] for m in mm],showfliers=True);ax.tick_params(axis='x',labelrotation=30);ax.set_title(f'Layer {layer}; all three lengths')
    axes[0].set_ylabel('Relative output error');fig.tight_layout();fig.savefig(CASE/'results/layer_error.png');plt.close(fig)
    selected_rows=[r for r in row_data if r['method']=='efq_calibrated' and r['relative_error']]
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,field in zip(axes,['score_spread','entropy_nats','removed_reference_mass']):
        rows=[r for r in selected_rows if r[field]]
        ax.scatter([float(r[field]) for r in rows],[float(r['relative_error']) for r in rows],s=4,alpha=.2,rasterized=True)
        ax.set_xlabel(field.replace('_',' '));ax.set_ylim(bottom=0)
    axes[0].set_ylabel('Row output relative error');fig.suptitle('Calibrated EFQ: exploratory associations, dependent sampled rows');fig.tight_layout();fig.savefig(CASE/'results/error_associations.png');plt.close(fig)
    rep=representatives[1]
    mm=['nearest_ceil','nearest_efq_scale','efq_mmlu','efq_mean','efq_balance','efq_lut','efq_calibrated']
    rows=[r for r in row_data if all(str(r[k])==str(rep[k]) for k in keys)]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for m in mm:
        rr=sorted([r for r in rows if r['method']==m],key=lambda r:int(r['query_position']))
        axes[0].plot([int(r['query_position']) for r in rr],[float(r['relative_error']) if r['relative_error'] else np.nan for r in rr],label=LABELS[m],marker='.',linewidth=1)
    cr=[r for r in rows if r['method']=='efq_calibrated']
    axes[1].bar([int(r['query_position']) for r in cr],[float(r['removed_reference_mass']) for r in cr],width=rep['length']/40,color='#287b75')
    axes[0].set_ylabel('Row output relative error');axes[1].set_ylabel('Reference mass removed');axes[0].legend(fontsize=7)
    for ax in axes:ax.set_xlabel('Query position')
    fig.suptitle(f"Largest calibrated unit error: {rep['document_id']}, N={rep['length']}, layer={rep['layer']}, head={rep['head']}")
    fig.tight_layout();fig.savefig(CASE/'results/representative_failure.png');plt.close(fig)
    print(json.dumps({'screen':summary['screen'],'transfer':transfer,'representatives':representatives},indent=2))


if __name__=='__main__':
    main()
