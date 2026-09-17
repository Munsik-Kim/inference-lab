"""CPU reanalysis of exact retained option scores; no GPU or model dependency."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

ARMS=('B','A_PUBLIC','V4');TASKS=('RETRIEVAL','COMPARISON','CODE')
STRATA=('unique/unique','tied/unique','unique/tied','tied/tied')

def load(p):return json.loads(Path(p).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def score(r):
    s=np.asarray(r['option_logits'],dtype=np.float64)
    if s.shape!=(4,) or not np.isfinite(s).all():raise ValueError('Invalid options')
    y=r['gold_index']; assert type(y) is int and 0<=y<4
    lse=float(s.max()+np.log(np.exp(s-s.max()).sum()));q=np.exp(s-lse);pred=int(np.argmax(s))
    top=np.flatnonzero(s==s.max()).tolist();target=np.eye(4)[y];fl=r['full_lse']
    if not math.isfinite(fl) or fl<lse-1e-10:raise ValueError('Invalid full normalizer')
    return {'s':s.tolist(),'q':q.tolist(),'log_q':(s-lse).tolist(),'prediction':pred,'gold':y,'correct':pred==y,'top':top,'unique':len(top)==1,'gold_in_top':y in top,'gap':float(np.sort(s)[-1]-np.sort(s)[-2]),'nll':lse-float(s[y]),'brier':float(np.square(q-target).sum()),'margin':float(s[y]-max(s[j] for j in range(4) if j!=y)),'label_mass':math.exp(lse-fl),'full_nll':fl-float(s[y]),'full_argmax_allowed':r['full_argmax'] in r['label_ids']}

def valid(r):
    required=('full_attention_outputs_finite','full_block_outputs_finite','full_final_hidden_finite','full_logits_finite','inputs_unchanged','same_layer13_qkv','routing_valid','dtype_layout_valid','answer_interface_valid','native_B_validated','effective_backend_verified')
    if any(r.get('validity',{}).get(k) is not True for k in required):raise ValueError('Absent or failed full validity')
    if r['readout']=='H_FP32':
        for k in ('full_native_finite','full_shadow_finite','full_hidden_finite','native_projection_full_exact','hidden_bf16_roundtrip_exact','C3_tolerance_pass','flags_restored'):
            if r.get('controls',{}).get(k) is not True:raise ValueError('Missing/failed readout validity '+k)

def join(rows,require_shadow=True):
    result={}
    for r in rows:
        if r.get('evidence_kind') not in ('recorded_gpu_measurement','posthoc_gpu_readout'):raise ValueError('Mock or unknown evidence')
        valid(r);key=(r['split'],r['base_id'],r['length'],r['arm'],r['readout'])
        if key in result:raise ValueError('Duplicate cell')
        result[key]=r
    scenarios=sorted({k[:3] for k in result})
    for key in scenarios:
        cells=[result.get((*key,a,v)) for a in ARMS for v in (('H_NATIVE','H_FP32') if require_shadow else ('H_NATIVE',))]
        if any(r is None for r in cells):raise ValueError('Missing pair cell')
        for field in ('token_hash','gold_index','label_ids','task'):
            if any(r[field]!=cells[0][field] for r in cells):raise ValueError('Mismatched '+field)
    return result,scenarios

def pair(b,c):
    if b['gold']!=c['gold']:raise ValueError('Gold mismatch')
    cell='both_correct' if b['correct'] and c['correct'] else 'regression' if b['correct'] else 'gain' if c['correct'] else 'both_wrong'
    flip=b['prediction']!=c['prediction'];d=np.array(c['s'])-b['s'];osc=float(np.ptp(d))
    return {'cell':cell,'flip':flip,'wrong_to_wrong':flip and cell=='both_wrong','stratum':('unique' if b['unique'] else 'tied')+'/'+('unique' if c['unique'] else 'tied'),'B_gold_in_top':b['gold_in_top'],'candidate_gold_in_top':c['gold_in_top'],'top_disjoint':not bool(set(b['top'])&set(c['top'])),'delta_nll':c['nll']-b['nll'],'delta_brier':c['brier']-b['brier'],'delta_margin':c['margin']-b['margin'],'option_kl':float(np.sum(np.array(b['q'])*(np.array(b['log_q'])-c['log_q']))),'B_gap':b['gap'],'R':osc/b['gap'] if b['gap']>0 else None,'B':b,'candidate':c}

def outcomes(ps):
    n=len(ps);cells={k:sum(p['cell']==k for p in ps) for k in ('both_correct','regression','gain','both_wrong')};bc=cells['both_correct']+cells['regression'];bw=n-bc
    return {'n':n,**cells,'flips':sum(p['flip'] for p in ps),'wrong_to_wrong_changes':sum(p['wrong_to_wrong'] for p in ps),'B_gold_in_top':sum(p['B_gold_in_top'] for p in ps),'candidate_gold_in_top':sum(p['candidate_gold_in_top'] for p in ps),'top_sets_disjoint':sum(p['top_disjoint'] for p in ps),'top_sets_intersect':sum(not p['top_disjoint'] for p in ps),'conditional_regression':{'numerator':cells['regression'],'denominator':bc,'rate':cells['regression']/bc if bc else None},'conditional_gain':{'numerator':cells['gain'],'denominator':bw,'rate':cells['gain']/bw if bw else None},'unconditional_regression':cells['regression']/n if n else None,'unconditional_gain':cells['gain']/n if n else None,'accuracy_delta':(cells['gain']-cells['regression'])/n if n else None}

def describe(ps):
    result=outcomes(ps);result['tie_strata']={s:outcomes([p for p in ps if p['stratum']==s]) for s in STRATA}
    result['B_top_ties']=sum(not p['B']['unique'] for p in ps);result['candidate_top_ties']=sum(not p['candidate']['unique'] for p in ps)
    for metric in ('delta_nll','delta_brier','delta_margin','option_kl'):
        result[metric]=float(np.mean([p[metric] for p in ps])) if ps else None
    for arm in ('B','candidate'):
        vals=[p[arm] for p in ps]
        result[arm+'_description']={'correct':sum(v['correct'] for v in vals),'top_tie_gold_label_positions':[sum(not v['unique'] and v['gold']==j for v in vals) for j in range(4)],'predicted_label_positions':[sum(v['prediction']==j for v in vals) for j in range(4)],'top_set_membership':[sum(j in v['top'] for v in vals) for j in range(4)],'gap_quantiles':np.quantile([v['gap'] for v in vals],[0,.25,.5,.75,.95,1]).tolist() if vals else None,'mean_label_mass':float(np.mean([v['label_mass'] for v in vals])) if vals else None,'mean_full_nll':float(np.mean([v['full_nll'] for v in vals])) if vals else None,'full_argmax_allowed':sum(v['full_argmax_allowed'] for v in vals)}
    return result

def bootstrap_indices(tasks,seed=617093,reps=5000):
    rng=np.random.default_rng(seed);groups=[np.flatnonzero(np.array(tasks)==t) for t in TASKS if t in tasks]
    return [g[rng.integers(0,len(g),size=(reps,len(g)))] for g in groups]

def intervals(ps):
    draws=bootstrap_indices([p['task'] for p in ps]);res={}
    for field in ('delta_nll','delta_brier','delta_margin'):
        x=np.array([p[field] for p in ps]);means=np.mean([x[idx].mean(1) for idx in draws],axis=0)
        res[field]={'task_balanced_mean':float(np.mean([np.mean([p[field] for p in ps if p['task']==t]) for t in TASKS if any(p['task']==t for p in ps)])),'ci95':np.quantile(means,[.025,.975]).tolist()}
    for name,cell,denom in [('regression','regression',('both_correct','regression')),('gain','gain',('gain','both_wrong'))]:
        numerator=np.array([p['cell']==cell for p in ps]); denominator=np.array([p['cell'] in denom for p in ps]);num=sum(numerator[i].sum(1) for i in draws);den=sum(denominator[i].sum(1) for i in draws);good=den>0
        res['conditional_'+name]={'undefined_draws':int((~good).sum()),'ci95':np.quantile(num[good]/den[good],[.025,.975]).tolist() if good.any() else None,'warning':'Descriptive conditional count ratio; zero events do not imply zero population risk'}
    return res

def gap_bin(g):return 'exact_tie' if g==0 else '(0,0.1)' if g<.1 else '[0.1,0.5)' if g<.5 else '[0.5,1)' if g<1 else '[1,infinity)'

def rank(x):
    x=np.asarray(x);return np.array([np.count_nonzero(x<v)+(np.count_nonzero(x==v)+1)/2 for v in x])
def spearman(x,y):
    good=[(a,b) for a,b in zip(x,y) if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    if len(good)<3:return {'n':len(good),'rho':None,'status':'INSUFFICIENT_DATA'}
    a,b=map(rank,zip(*good))
    if np.ptp(a)==0 or np.ptp(b)==0:return {'n':len(good),'rho':None,'status':'CONSTANT_FEATURE'}
    return {'n':len(good),'rho':float(np.corrcoef(a,b)[0,1]),'status':'DESCRIPTIVE_ONLY_NO_P_VALUE'}

def analyze(rows):
    has_shadow=any(r['readout']=='H_FP32' for r in rows);idx,scenarios=join(rows,has_shadow);ps=[]
    for key in scenarios:
        for view in (('H_NATIVE','H_FP32') if has_shadow else ('H_NATIVE',)):
            b=score(idx[(*key,'B',view)])
            for arm in ARMS[1:]:
                c=score(idx[(*key,arm,view)]);p=pair(b,c);r=idx[(*key,arm,view)]
                p.update(split=key[0],base_id=key[1],length=key[2],arm=arm,readout=view,task=r['task'],item_id=r['item_id'],original_B_tied=not score(idx[(*key,'B','H_NATIVE')])['unique']);ps.append(p)
    groups={};transitions={};associations={}
    for split in sorted({p['split'] for p in ps}):
        for arm in ARMS[1:]:
            for view in (('H_NATIVE','H_FP32') if has_shadow else ('H_NATIVE',)):
                for task in ('ALL',*TASKS):
                    sel=[p for p in ps if p['split']==split and p['arm']==arm and p['readout']==view and (task=='ALL' or p['task']==task)]
                    key='/'.join([split,view,arm,task]);groups[key]=describe(sel)
                    groups[key]['original_B_tie_groups']={str(t):describe([p for p in sel if p['original_B_tied']==t]) for t in (False,True)}
                    groups[key]['gap_bins']={b:outcomes([p for p in sel if gap_bin(p['B_gap'])==b]) for b in ('exact_tie','(0,0.1)','[0.1,0.5)','[0.5,1)','[1,infinity)')}
                    if split=='standard' and sel:groups[key]['posthoc_intervals']=intervals(sel)
                    if view=='H_NATIVE' and sel:
                        local=[idx[(p['split'],p['base_id'],p['length'],arm,'H_NATIVE')]['native_pair_local']['relative_error'] for p in sel]
                        associations[key]={'candidate_vs_B_local_vs_abs_margin':spearman(local,[abs(p['delta_margin']) for p in sel]),'candidate_vs_B_local_vs_abs_nll':spearman(local,[abs(p['delta_nll']) for p in sel]),'predictor_comparison':'INSUFFICIENT_EVENTS; no classifier fitted','unit':'base scenario; pooled32query x16head output per item, same original evidence'}
            if has_shadow:
                a={p['base_id']:p for p in ps if p['split']==split and p['arm']==arm and p['readout']=='H_NATIVE'};b={p['base_id']:p for p in ps if p['split']==split and p['arm']==arm and p['readout']=='H_FP32'}
                transitions[split+'/'+arm]={'native_flip_persists':sum(a[k]['flip'] and b[k]['flip'] for k in a),'native_flip_vanishes':sum(a[k]['flip'] and not b[k]['flip'] for k in a),'new_shadow_flip':sum(not a[k]['flip'] and b[k]['flip'] for k in a),'persistent_flip_changed_correctness_category':sum(a[k]['flip'] and b[k]['flip'] and a[k]['cell']!=b[k]['cell'] for k in a),'nll_readout_interaction':float(np.mean([np.mean([b[k]['delta_nll']-a[k]['delta_nll'] for k in a if a[k]['task']==t]) for t in TASKS if any(a[k]['task']==t for k in a)]))}
    controls={}
    if has_shadow:
        sh=[r for r in rows if r['readout']=='H_FP32'];cs=[r['controls'] for r in sh]
        controls={'cells':len(sh),'cast_only_exact':sum(c['C1_cast_only_exact'] for c in cs),'roundtrip_full_equal':sum(c['C2_full_equal'] for c in cs),'roundtrip_options_equal':sum(c['C2_option_equal'] for c in cs),'fp64_winner_disagreements':sum(c['C3_winner_disagreement'] for c in cs),'max_fp32_fp64_option_abs':max(c['C3_max_abs'] for c in cs),'max_fp32_fp64_gap_abs':max(abs(c['C3_gap_difference']) for c in cs),'max_roundtrip_full_abs':max(c['C2_full_max_abs'] for c in cs),'all_full_validity':all(c['full_shadow_finite'] and c['full_hidden_finite'] for c in cs)}
    return {'evidence_kind':'POST_HOC_SAME_INPUT_READOUT_DIAGNOSTIC','scenarios':len(scenarios),'groups':groups,'flip_readout_transitions':transitions,'controls':controls,'local_associations':associations,'original_study':'COMPLETED_CONTROLLED_STUDY','supplemental_readout':'COMPLETED' if has_shadow and len(scenarios)==238 else 'NOT_RUN','deployment':'NOT_ASSESSED','publication':'NOT_PUBLISHED_BY_THIS_TASK'},ps

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.case.resolve()):p.error('New external output directory required')
    rows=load(a.case/'results/raw/native.json');shadow=a.case/'results/raw/shadow.json'
    if shadow.exists():rows+=load(shadow)
    summary,pairs=analyze(rows);save(a.output/'summary.json',summary);save(a.output/'pairs.json',pairs)
    print(json.dumps({'scenarios':summary['scenarios'],'readout':summary['supplemental_readout']}))
if __name__=='__main__':main()
