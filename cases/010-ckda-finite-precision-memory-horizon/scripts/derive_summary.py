"""Build a lightweight presentation view from immutable v1/v2 recorded scalars."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from verify_unified import CASE, sha, verify_snapshots


def derive(case=CASE):
    case = Path(case)
    verify_snapshots(case)
    sources = {}
    def read(path):
        sources[path] = {'path': path, 'sha256': sha(case/path), 'bytes': (case/path).stat().st_size}
        return json.loads((case/path).read_text())
    def csvrows(path):
        sources[path] = {'path': path, 'sha256': sha(case/path), 'bytes': (case/path).stat().st_size}
        with (case/path).open() as f: return list(csv.DictReader(f))
    p1 = read('versions/v1/configs/protocol.json')
    p2 = read('versions/v2/protocol_v2.json')
    combined = read('versions/v2/results/fresh-summary/combined.json')
    ledger = read('versions/v2/results/runtime_byte_ledger.json')
    historical = read('versions/v2/results/historical-summary/summary.json')
    historical_v1 = read('versions/v2/results/historical-reanalysis/summary.json')
    rows = []
    for seed in p2['fresh']['model_seeds']:
        timing = read(f'versions/v2/results/timing/seed{seed}/timing.json')
        for arm in p2['fresh']['arms']:
            raw = read(f'versions/v2/results/fresh/seed{seed}/{arm}/summary.json')
            if raw['N'] != p2['fresh']['N'] or len(raw['tau']) != raw['N'] or raw['family_size'] != p2['fresh']['simultaneous_family']:
                raise ValueError('fresh denominator/family mismatch')
            rmst = statistics.mean(raw['T'] if t is None else min(t-1,raw['T']) for t in raw['tau'])
            if rmst != raw['RMST0']:
                raise ValueError('RMST0 does not match first-failure records')
            empirical = max([0]+[t for t in range(1,raw['T']+1) if sum(x is not None and x <= t for x in raw['tau'])/raw['N'] <= .05])
            if empirical != raw['empirical_T05_all_tokens']:
                raise ValueError('empirical horizon does not match first-failure records')
            chosen = {k:raw[k] for k in ['model_seed','arm','N','T','RMST0','empirical_T05_all_tokens','supported_T05_grid','token_accuracy','final_quarter_accuracy','terminal_count','grid_rows','checkpoint_sha256','tokens_sha256','sample_ids_sha256']}
            trials = [r for r in timing['rows'] if r['arm'] == arm]
            if len(trials) != 3 or any(not math.isclose(r['ms_per_group_token'],r['seconds']*1000/(timing['N']*timing['T']),rel_tol=1e-14) for r in trials):
                raise ValueError('timing repeat/denominator mismatch')
            chosen['cpu_ms_per_group_token'] = statistics.median(r['ms_per_group_token'] for r in trials)
            chosen['cpu_timing_scope'] = timing['scope']
            chosen['source'] = f'versions/v2/results/fresh/seed{seed}/{arm}/summary.json'
            rows.append(chosen)
    keyed = {(r['model_seed'],r['arm']):r for r in rows}
    if len({(r['tokens_sha256'],r['sample_ids_sha256']) for r in rows}) != 1:
        raise ValueError('fresh inputs not paired')
    for pair in combined['paired_comparisons']:
        delta = keyed[pair['model_seed'],pair['candidate']]['RMST0'] - keyed[pair['model_seed'],pair['baseline']]['RMST0']
        if delta != pair['RMST0_delta']:
            raise ValueError('paired mean mismatch')
    for row in ledger['rows']:
        data = row['v2']
        for n,total in data['total_bytes'].items():
            if total != data['shared_bytes']+int(n)*data['per_stream_persistent_bytes']:
                raise ValueError('byte budget mismatch')
    base = {(r['model_seed'],r['arm']):r['v2'] for r in ledger['rows']}
    native, int8 = base[0,'NATIVE_FP32']['per_stream_persistent_bytes'],base[0,'UNIFORM_8']['per_stream_persistent_bytes']
    precision_pairs = csvrows('versions/v2/results/diagnostic-summary/precision_pairs.csv')
    if any(int(r['scored_prediction_disagreements']) or int(r['tau_equal_sequences']) != int(r['N']) for r in precision_pairs):
        raise ValueError('precision identity observation changed')
    precision_endpoints = csvrows('versions/v2/results/diagnostic-summary/precision_endpoints.csv')
    trace = csvrows('versions/v2/results/diagnostic-summary/trace_extrema.csv')
    promoted = []
    for seed in p2['fresh']['model_seeds']:
        source = f'versions/v2/inputs/codecs/seed{seed}/selection.json'
        selection = read(source)
        bits = selection['bits']
        # The source stores one width per flattened head/key channel.
        flattened = [x for row in bits for x in row] if bits and isinstance(bits[0], list) else bits
        if len(set(selection['accepted_channels'])) != len(selection['accepted_channels']):
            raise ValueError('duplicate mixed channel selection')
        if {i for i, width in enumerate(flattened) if width == 6} != set(selection['accepted_channels']):
            raise ValueError('mixed promoted-channel map differs from bit widths')
        if any(width not in (5, 6) for width in flattened):
            raise ValueError('unexpected mixed comparator precision')
        promoted.append({'model_seed':seed, 'count':len(selection['accepted_channels']),
                         'total_channels':len(flattened), 'accepted_channels':selection['accepted_channels'],
                         'rule':selection['rule'], 'source':source})
    return {
        'schema':'diova-case010-unified-summary-v1',
        'derivation_scope':'Existing recorded scalars; no new inference, training, calibration, timing or TEST evaluation.',
        'model': {'name':p2['model'], 'checkpoint_sha256':p2['checkpoint_sha256'], 'upstream_commit':p2['upstream_commit'], 'architecture':p1['architecture'], 'training':p1['training']},
        'methods': {'mixed_promoted_channels_by_seed':promoted},
        'stages': {
            'v1':{'phase':'A: storage exploration and stronger budget baselines','N':p1['splits']['TEST']['sequences'],'T':p1['splits']['TEST']['length'],'grid':p1['evaluation']['horizons'],'families':[546,630],'original_arms':26,'supplemented_arms':30,'comparisons':historical_v1['historical_comparisons']},
            'v2':{'phase':'B: failure-aware restart and fresh-input comparison','N':p2['fresh']['N'],'T':p2['fresh']['T'],'grid':p2['fresh']['confidence_grid'],'family':p2['fresh']['simultaneous_family'],'arms':p2['fresh']['arms'],'generator_seed':p2['fresh']['generator_seed'],'reuses_same_three_checkpoints':True}},
        'headline': {'native_stream_bytes':native,'int8_stream_bytes':int8,'reduction_pct':100*(native-int8)/native,'unit':'serialized recurrent-state bytes per stream',
                     'int8_minus_native_RMST0_by_seed':[{'model_seed':s,'delta_tokens':keyed[s,'UNIFORM_8']['RMST0']-keyed[s,'NATIVE_FP32']['RMST0']} for s in (0,1,2)]},
        'fresh': {'rows':rows,'paired_comparisons':combined['paired_comparisons'],'interval_scope':combined['scope'],'pooled_checkpoint_estimate':False,'RMST0_definition':'Mean consecutive correct group tokens before first failure, excluding the failing token; censored streams contribute Tmax.'},
        'storage': {'rows':ledger['rows'],'scope':ledger['scope'],'comparison_stream_count':128,'evaluation_sequence_count':1024,'timing_N':16,'timing_T':128,'timing_repeats':3,'timing_threads':2},
        'restart':{'counts':historical['counts'],'interpretation':historical['interpretation'],'failure_cases':[{k:r[k] for k in ['model_seed','arm','original_failure_write','actual_terminal_write','v1_full_split_predictions_equal','v2_full_split_predictions_equal','fresh_resume_immediately_after_failure']} for r in historical['failure_cases']],'scope':'Historical diagnostics retained from v2, not new model executions in this integration.'},
        'precision':{'N':32,'T':2048,'pairs':precision_pairs,'endpoints':precision_endpoints,'scope':p2['precision']['coefficients']},
        'trace':{'rank2_state_norm_extrema':[r for r in trace if r['arm']=='LOWRANK_4_8_R2' and r['metric']=='state_norm'],'scope':'Separate 32-input diagnostic cohort; extrema are not causal explanations.'},
        'sources':[sources[k] for k in sorted(sources)]}


def figure(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'NATIVE_FP32':'Native FP32','UNIFORM_8':'INT8','UNIFORM_5':'INT5','LOWRANK_4_8_R2':'Rank2','MIXED_5_6_BUDGET':'Mixed5/6'}
    colors=['#253749','#537f92','#77836a','#ad6940','#9b7bb2']
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True,layout='constrained')
    for seed,ax in enumerate(axes):
        for arm,color in zip(data['stages']['v2']['arms'],colors):
            row=next(r for r in data['fresh']['rows'] if r['model_seed']==seed and r['arm']==arm)
            budget=next(r['v2']['total_bytes']['128'] for r in data['storage']['rows'] if r['model_seed']==seed and r['arm']==arm)
            ax.scatter(budget/1024,row['empirical_T05_all_tokens'],color=color,marker='o',label=labels[arm])
            ax.scatter(budget/1024,row['supported_T05_grid'],color=color,marker='x')
        ax.set_xscale('log');ax.set_title(f'Fixed checkpoint seed {seed}');ax.set_xlabel('Serialized total KiB, N=128 streams');ax.grid(alpha=.2)
    axes[0].set_ylabel('Correct-readout horizon (group tokens)')
    axes[2].legend(fontsize=8)
    fig.suptitle('Stage B / v2: 1,024 fresh sequences per fixed checkpoint\nCircle: empirical T0.05; cross: 195-family supported grid lower bound',fontsize=11)
    fig.savefig(output,dpi=130,metadata={'Software':'DIOVA recorded-scalar presentation'})
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,default=CASE)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figure',type=Path)
    a=p.parse_args()
    if a.output.exists() or (a.figure and a.figure.exists()): raise ValueError('use new output paths')
    data=derive(a.case)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    if a.figure:
        a.figure.parent.mkdir(parents=True,exist_ok=True);figure(data,a.figure)
    print(json.dumps({'status':'PASS','fresh_cells':len(data['fresh']['rows']),'source_files':len(data['sources']),'model_runs':0}))
if __name__=='__main__':main()
