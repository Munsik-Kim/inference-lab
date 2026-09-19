"""CPU-only paired summaries from retained native scores and norm statistics."""
from pathlib import Path
import numpy as np
from .common import read,write,require,digest
from .numerics import bootstrap_indices,recovery

SEED=808191;REPS=5000


def interval(values):return [float(x) for x in np.quantile(values,[.025,.975])]


def estimate(x,indices):
    a=np.asarray(x,dtype=np.float64);require(np.isfinite(a).all(),'Invalid summary values')
    return {'mean':float(a.mean()),'ci95':interval(a[indices].mean(axis=1))}


def join(records,arm,base):
    maps={}
    for name in [base,arm]:
        subset=[r for r in records if r['arm']==name];lookup={r['id']:r for r in subset}
        require(len(lookup)==len(subset)>0,'Missing/duplicate arm records');maps[name]=lookup
    require(set(maps[base])==set(maps[arm]),'Unpaired scenario sets')
    pairs=[]
    for ident in sorted(maps[base]):
        b,c=maps[base][ident],maps[arm][ident]
        require(b['token_hash']==c['token_hash'] and b['task']==c['task'] and b['score']['gold']==c['score']['gold'],'Pair metadata mismatch')
        pairs.append((b,c))
    return pairs


def model_summary(records,baseline):
    arms=sorted({r['arm'] for r in records});result={}
    for arm in arms:
        pairs=join(records,arm,baseline);require(len(pairs)==192,'Heldout count differs from freeze')
        tasks=[b['task'] for b,c in pairs];indices=bootstrap_indices(tasks,REPS,SEED)
        items=[c['score'] for b,c in pairs];bs=[b['score'] for b,c in pairs]
        cells={'both_correct':0,'regression':0,'gain':0,'both_wrong':0,'flip':0,'wrong_to_different_wrong':0}
        for b,c in zip(bs,items):
            bc,cc=b['correct'],c['correct'];flip=b['prediction']!=c['prediction']
            cells['both_correct']+=int(bc and cc);cells['regression']+=int(bc and not cc)
            cells['gain']+=int(not bc and cc);cells['both_wrong']+=int(not bc and not cc)
            cells['flip']+=int(flip);cells['wrong_to_different_wrong']+=int(not bc and not cc and flip)
        bcorrect=sum(b['correct'] for b in bs);cweight=sum(c['correct'] for c in items)
        row={'n':192,'correct':cweight,'baseline_correct':bcorrect,'baseline_wrong':192-bcorrect,
             'transitions':cells,'regression_denominator':bcorrect,'gain_denominator':192-bcorrect,
             'conditional_regression':cells['regression']/bcorrect if bcorrect else None,
             'conditional_gain':cells['gain']/(192-bcorrect) if bcorrect<192 else None,
             'top_ties':sum(len(c['top_ties'])>1 for c in items),
             'allowed_mass_mean':float(np.mean([c['allowed_mass'] for c in items])),
             'full_argmax_allowed':sum(c['full_argmax_allowed'] for c in items),
             'winner_gap_quantiles':np.quantile([c['winner_gap'] for c in items],[0,.25,.5,.75,1]).tolist(),
             'per_task':{}}
        row['accuracy_delta']=estimate([int(c['correct'])-int(b['correct']) for b,c in zip(bs,items)],indices)
        for key in ['full_gold_nll','choice_nll','brier','gold_margin']:
            row['delta_'+key]=estimate([c[key]-b[key] for b,c in zip(bs,items)],indices)
        row['full_kl_B_candidate_mean']=0. if arm==baseline else float(np.mean([c['full_kl_B_candidate'] for b,c in pairs]))
        for task in sorted(set(tasks)):
            pp=[(b,c) for b,c in pairs if b['task']==task]
            ii=bootstrap_indices([task]*len(pp),REPS,SEED)
            row['per_task'][task]={'n':len(pp),'correct':sum(c['score']['correct'] for b,c in pp),
               'delta_full_gold_nll':estimate([c['score']['full_gold_nll']-b['score']['full_gold_nll'] for b,c in pp],ii),
               'flip':sum(b['score']['prediction']!=c['score']['prediction'] for b,c in pp),
               'regression':sum(b['score']['correct'] and not c['score']['correct'] for b,c in pp),
               'gain':sum(not b['score']['correct'] and c['score']['correct'] for b,c in pp)}
        if arm!=baseline and 'local' in pairs[0][1]:
            rr=np.array([c['local']['relative_error'] for b,c in pairs]);absolute=[c['local']['absolute_rms'] for b,c in pairs]
            row['local']={'mean_relative':float(rr.mean()),'median_relative':float(np.median(rr)),
               'p95_relative':float(np.quantile(rr,.95)),'absolute_rms_mean':float(np.mean(absolute)),
               'worst_prompt':pairs[int(rr.argmax())][1]['id'],'max_relative':float(rr.max())}
        result[arm]=row
    return {'baseline':baseline,'arms':result,'bootstrap':{'replicates':REPS,'seed':SEED,
       'unit':'paired base scenario, resampled within task; 64 per task, equal task weights',
       'indices_sha256':digest(indices.tolist()),'coverage':'pointwise descriptive, not equivalence'},
       'zero_event_note':'Zero observed regressions do not imply zero population risk. Shared synthetic templates limit generalization.'}


def repair_summary(records):
    result={}
    for name in ['I25','P25','S50']:
        pairs=join(records,name+'-R',name);idx=bootstrap_indices([a['task'] for a,b in pairs],REPS,SEED)
        a=np.array([a['local']['squared_error'] for a,b in pairs]);b=np.array([b['local']['squared_error'] for a,b in pairs])
        values=1-b[idx].sum(axis=1)/a[idx].sum(axis=1)
        result[name]={'pooled_recovery':recovery(a,b),'ci95':interval(values),
            'error_energy_uncorrected':float(a.sum()),'error_energy_repaired':float(b.sum()),
            'improved_prompts':int((b<a).sum()),'worse_prompts':int((b>a).sum()),'n':len(pairs),
            'delta_full_gold_nll_repair_minus_uncorrected':estimate([b['score']['full_gold_nll']-a['score']['full_gold_nll'] for a,b in pairs],idx),
            'delta_choice_nll_repair_minus_uncorrected':estimate([b['score']['choice_nll']-a['score']['choice_nll'] for a,b in pairs],idx)}
    return result


def timing_summary(records,baseline):
    result={};arms=sorted({r['arm'] for r in records});boundaries=sorted({r['boundary'] for r in records})
    ids=sorted({r['id'] for r in records});rounds=sorted({r['round'] for r in records})
    require(len(ids)==12 and rounds==[1,2,3],'Incomplete timing rounds')
    lookup={ (r['arm'],r['boundary'],r['round'],r['id'],r['block']):r['wall_seconds_per_call'] for r in records}
    require(len(lookup)==len(records),'Duplicate timing cells')
    rng=np.random.default_rng(SEED+1)
    process=rng.integers(0,3,(REPS,3,1,1));scenario=np.concatenate([rng.integers(j*4,(j+1)*4,(REPS,1,4,1)) for j in range(3)],axis=2);block=rng.integers(0,5,(REPS,3,12,5))
    for boundary in boundaries:
        base=np.array([[[lookup[baseline,boundary,r,i,k] for k in range(5)] for i in ids] for r in rounds])
        result[boundary]={}
        for arm in arms:
            arr=np.array([[[lookup[arm,boundary,r,i,k] for k in range(5)] for i in ids] for r in rounds])
            values=base[process,scenario,block].mean(axis=(1,2,3))/arr[process,scenario,block].mean(axis=(1,2,3))
            result[boundary][arm]={'wall_ms':float(arr.mean()*1000),'speedup':float(base.mean()/arr.mean()),
              'ci95':interval(values),'round_wall_ms':(arr.mean(axis=(1,2))*1000).tolist(),
              'n_scenarios':12,'process_rounds':3,'blocks_per_scenario_round':5,'calls_per_block':3}
    return result


def analyze(case: Path,output: Path):
    result={'schema':1,'deployment':'NOT_ASSESSED','tracks':{},'timing':{}}
    for track,base in [('Q','Q-BF16'),('R','R-B')]:
        file=case/f'results/raw/{track}/model_records.json'
        if not file.exists():result['tracks'][track]={'status':'MEASUREMENT_NOT_RUN'};continue
        records=read(file)
        for r in records:
            require(r['evidence_kind']=='gpu_measurement' and r['split']=='heldout','Fixture/non-heldout entered results')
            v=r.get('validity',{});require(v.get('full_final_vocab') is True and v.get('full_model_outputs') is True,'Incomplete full validity')
        out=model_summary(records,base)
        if track=='R':out['repair']=repair_summary(records)
        result['tracks'][track]=out
        timing=case/f'results/raw/{track}/timing.json'
        if timing.exists():result['timing'][track]=timing_summary(read(timing),base)
    write(output/'summary.json',result);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();from .common import new_external
    analyze(a.case,new_external(a.output))
