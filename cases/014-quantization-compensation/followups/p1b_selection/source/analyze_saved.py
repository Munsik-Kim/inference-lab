"""Analyze immutable P1b scalar records with shared-document bootstrap."""
import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from p1b_common import ALPHAS, TARGETS, save, sha, utc, jsonl
from policies import map_policies, POLICIES
from bootstrap import shared_indices, contrast_samples

CONTRASTS=[('SELECT_SSE',b) for b in ['OLD_FIT','BASE_ZERO','FIXED_QUARTER','FIXED_HALF']]+[('SELECT_NLL',b) for b in ['BASE_ZERO','SELECT_SSE','FIXED_HALF']]+[(p,'BASE_ZERO') for p in ['OLD_FIT','FIXED_QUARTER','FIXED_HALF','FIXED_FULL']]
TERMS=['SSE_after','SSE_before','reference_SSE','A_norm2','C_norm2','B_norm2','B0_norm2','deltaB_norm2','dot_A_B0','dot_A_C','dot_B0_C','T_inherited','T_cancellation','T_cont','T_requant','fixedB_SSE','continuous_SSE','arithmetic_residual_norm2','cast_residual_norm2','BF16_output_residual_norm2']

def csvwrite(path, rows):
    if Path(path).exists():raise FileExistsError(path)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'x',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def build(run,case):
    start=time.monotonic();rows=[r for p in sorted((run/'cells').glob('*/*.jsonl')) for r in jsonl(p)]
    frozen=json.loads((run/'selection_frozen.json').read_text());choice={r['target']:r['policy_alpha'] for r in frozen['choices']}
    grouped=defaultdict(list);lookup={}
    for r in rows:
        key=r['target'],r['alpha'],r['split'],r['document_id_hash']
        if key in lookup:raise ValueError('duplicate raw measurement')
        if r['status']!='COMPLETE':raise ValueError('incomplete measurement')
        lookup[key]=r;grouped[key[:3]].append(r)
    assert len(rows)==3360
    mapped=map_policies(rows,frozen['choices']);assert len(mapped)==4704
    splits=['SELECT-ID','FRESH-EVAL-ID','FRESH-SHIFT'];alphas=[];policyrows=[];contrasts=[];macro=[];mechanisms=[];classifications=[];signs=[]
    oldcal=defaultdict(list)
    if (run/'old_CAL_reference.json').exists():
        for r in json.loads((run/'old_CAL_reference.json').read_text())['rows']:
            oldcal[r['block'],r['projection'],r['alpha']]=[r['T_inherited_sum']]
    else:
        sources={}
        for p in sorted((case/'results/p1_r2/cells').glob('*.jsonl')):
            sources[p.name]=sha(p)
            for r in jsonl(p):
                if r['split']=='CAL-FIT':oldcal[r['block'],r['projection'],r['alpha']].append(r['T_inherited'])
        if len(oldcal)!=30:raise ValueError('missing historical CAL scalar reference')
        save(run/'old_CAL_reference.json',{'rows':[{'block':b,'projection':p,'alpha':a,'T_inherited_sum':math.fsum(v)} for (b,p,a),v in sorted(oldcal.items())],
             'historical_source_cell_hashes':sources,'purpose':'C classification only; never input to selectors; old EVAL scores not used'})
    for r in rows:
        if not r['alpha']:continue
        b=lookup[r['target'],0,r['split'],r['document_id_hash']];tol=1e-9*max(1,r['SSE_after'],r['SSE_before']);labels=[]
        if r['T_inherited'] < -tol and r['T_cont']>tol:labels.append('A_CANCELLATION_LOSS')
        if r['T_cont'] < -tol and r['Delta']>tol:labels.append('B_REQUANTIZATION_REVERSAL')
        if math.fsum(oldcal[r['block'],r['projection'],r['alpha']])<=tol and r['T_inherited']>tol:labels.append('C_CONTINUOUS_TRANSFER_LOSS')
        if r['Delta'] < -tol and (r['NLL_sum']-b['NLL_sum'])/255>1e-9:labels.append('D_LOCAL_NLL_DISAGREEMENT')
        classifications.append(dict(target=r['target'],alpha=r['alpha'],split=r['split'],document_id_hash=r['document_id_hash'],labels=labels))
        bound=2*math.sqrt(r['SSE_after']*r['arithmetic_residual_norm2'])+r['arithmetic_residual_norm2']+2*math.sqrt(b['SSE_after']*b['arithmetic_residual_norm2'])+b['arithmetic_residual_norm2']
        actual=r['BF16_output_residual_norm2']-b['BF16_output_residual_norm2']
        signs.append(dict(target=r['target'],alpha=r['alpha'],split=r['split'],document_id_hash=r['document_id_hash'],algebra_Delta=r['Delta'],actual_BF16_module_Delta=actual,
                          arithmetic_Delta_absolute_bound=bound,algebra_sign_certified=abs(r['Delta'])>bound,
                          measured_direction_agrees=(actual>0)==(r['Delta']>0) if actual and r['Delta'] else None))
    def aggregate(t,a,split):
        rr=sorted(grouped[t,a,split],key=lambda r:r['document_id_hash']);base=[lookup[t,0,split,r['document_id_hash']] for r in rr]
        sums={k:math.fsum(r[k] for r in rr) for k in TERMS};n=sum(r['valid_token_count'] for r in rr);nll=math.fsum(r['NLL_sum'] for r in rr)
        sse0=sums['SSE_before'];diff=[r['SSE_after']-b['SSE_after'] for r,b in zip(rr,base)]
        return dict(target=t,split=split,alpha=a,documents=len(rr),valid_token_count=n,NLL_sum=nll,NLL_mean=nll/n,PPL=math.exp(nll/n),
             SSE_vs_alpha0_ratio=sums['SSE_after']/sse0-1 if sse0 else None,
             NLL_vs_alpha0_doc_mean=math.fsum((r['NLL_sum']-b['NLL_sum'])/r['valid_token_count'] for r,b in zip(rr,base))/len(rr),
             mean_full_vocab_KL=math.fsum(r['sampled_full_vocab_KL_sum'] for r in rr)/sum(r['KL_position_count'] for r in rr),
             argmax_flip_rate=sum(r['flip_count'] for r in rr)/n,artifact_bytes=rr[0]['artifact_bytes'],artifact_sha256=rr[0]['artifact_sha256'],
             documents_lower_SSE_vs_alpha0=sum(v<0 for v in diff),documents_higher_SSE_vs_alpha0=sum(v>0 for v in diff),documents_exact_tie_SSE_vs_alpha0=sum(v==0 for v in diff),
             median_document_relative_SSE_vs_alpha0=float(np.median([r['SSE_after']/b['SSE_after']-1 for r,b in zip(rr,base)])),
             max_identity_relative_gap=max(r['identity_relative_gap'] for r in rr),**sums)
    for split in splits:
        ids=sorted({r['document_id_hash'] for r in rows if r['split']==split});n=len(ids)
        ix=shared_indices(n)
        for t in TARGETS:
            for a in ALPHAS:alphas.append(aggregate(t,a,split))
            base=grouped[t,0,split];an=math.fsum(r['A_norm2'] for r in base);un=math.fsum(r['U_norm2'] for r in base);au=math.fsum(r['dot_A_U'] for r in base)
            star=au/un if un else None
            mechanisms.append({'target':t,'split':split,'A_norm2':an,'U_norm2':un,'dot_A_U':au,'alpha_cont_star_raw':star,'alpha_cont_star_clipped':min(1,max(0,star)) if star is not None else None,
                'pooled_cosine_A_U':au/math.sqrt(an*un) if an*un else None,'norm_A_over_norm_U':math.sqrt(an/un) if un else None,'use':'explanatory continuous inherited-error diagnostic, not a selector'})
            for policy,a in choice[t].items():policyrows.append(dict(aggregate(t,a,split),policy=policy))
        def array(policy, field):
            return np.array([[lookup[t,choice[t][policy],split,d][field] for d in ids] for t in TARGETS],dtype='f8')
        denominator=array('BASE_ZERO','SSE_after')
        for cp,bp in CONTRASTS:
            c=array(cp,'SSE_after');b=array(bp,'SSE_after');lc=array(cp,'NLL_sum')/255;lb=array(bp,'NLL_sum')/255
            sb,sm=contrast_samples(c,b,denominator,ix);nb,nm=contrast_samples(lc,lb,np.ones_like(lc),ix)
            points=(c-b).sum(axis=-1)/denominator.sum(axis=-1);npoints=(lc-lb).mean(axis=-1)
            for j,t in enumerate(TARGETS):
                sdiff=c[j]-b[j];ndiff=lc[j]-lb[j]
                contrasts.append({'target':t,'split':split,'candidate':cp,'baseline':bp,'candidate_alpha':choice[t][cp],'baseline_alpha':choice[t][bp],
                    'SSE_difference_over_alpha0':float(points[j]),'SSE_pointwise_CI95':np.quantile(sb[j],[.025,.975]).tolist(),
                    'NLL_doc_mean_difference':float(npoints[j]),'NLL_pointwise_CI95':np.quantile(nb[j],[.025,.975]).tolist(),
                    'SSE_lower_documents':int((sdiff<0).sum()),'SSE_higher_documents':int((sdiff>0).sum()),'SSE_exact_tie_documents':int((sdiff==0).sum()),
                    'SSE_doc_relative_median':float(np.median(sdiff/denominator[j])),
                    'NLL_lower_documents':int((ndiff<0).sum()),'NLL_higher_documents':int((ndiff>0).sum()),'NLL_exact_tie_documents':int((ndiff==0).sum()),'documents':n})
            macro.append({'split':split,'candidate':cp,'baseline':bp,'fixed_targets':6,'documents':n,
                'D_macro':float(points.mean()),'D_macro_CI95':np.quantile(sm,[.025,.975]).tolist(),
                'NLL_macro_doc_mean_difference':float(npoints.mean()),'NLL_macro_CI95':np.quantile(nm,[.025,.975]).tolist(),
                'primary':split=='FRESH-EVAL-ID' and cp=='SELECT_SSE' and bp=='OLD_FIT',
                'interpretation':'equal-weight fixed interventions; not a jointly compensated full model'})
    csvwrite(run/'alpha_results.csv',alphas);csvwrite(run/'policy_summary.csv',policyrows);csvwrite(run/'paired_contrasts.csv',contrasts)
    fields=['run_id','target','block','projection','split','document_id_hash','token_hash','alpha','policy','artifact_sha256','artifact_bytes','SSE_after','SSE_before','NLL_sum','valid_token_count','sampled_full_vocab_KL_sum','KL_position_count','flip_count','T_inherited','T_cancellation','T_requant','BF16_output_residual_norm2','arithmetic_residual_norm2','status']
    csvwrite(run/'policy_results.csv',[{k:r[k] for k in fields} for r in mapped])
    cls={s:dict(Counter(l for r in classifications if r['split']==s for l in r['labels'])) for s in splits}
    signsummary={s:{'comparisons':sum(r['split']==s for r in signs),'measured_direction_agrees':sum(r['split']==s and r['measured_direction_agrees'] is True for r in signs),
                    'bound_certified':sum(r['split']==s and r['algebra_sign_certified'] for r in signs),
                    'measured_agreement_but_not_bound_certified':sum(r['split']==s and r['measured_direction_agrees'] is True and not r['algebra_sign_certified'] for r in signs)} for s in splits}
    save(run/'classifications.json',{'rows':classifications,'counts':cls,'comparisons':{s:sum(r['split']==s for r in classifications) for s in splits},'overlap_allowed':True})
    save(run/'arithmetic_sign_robustness.json',{'definition':'sum across two arms of 2sqrt(SSE*arithmetic_residual_norm2)+arithmetic_residual_norm2','rows':signs,'summary':signsummary})
    result={'status':'ANALYZED','raw_candidate_rows':len(rows),'policy_alias_rows':len(mapped),'independent_documents':{'SELECT-ID':32,'FRESH-EVAL-ID':64,'FRESH-SHIFT':16},
            'alpha_summary_rows':len(alphas),'policy_summary_rows':len(policyrows),'choices':frozen['choices'],'macro_contrasts':macro,'target_contrasts':contrasts,
            'continuous_diagnostics':mechanisms,'classification_counts':cls,'arithmetic_sign_summary':signsummary,
            'confidence_scope':'10,000 paired document resamples seed6101519; all targets/policies share indices; ratios re-evaluated, selectors frozen; conditional on old K and this SELECT; only EVAL SELECT_SSE vs OLD_FIT D_macro primary',
            'GPU_metrics_not_recomputed':True,'analysis_seconds':time.monotonic()-start,'analysis_source_sha256':sha(__file__),'created_utc':utc()}
    save(run/'summary.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--run',required=True);a=p.parse_args()
    r=build(Path(a.run),Path(a.case));print(json.dumps({'raw_rows':r['raw_candidate_rows'],'primary':next(m for m in r['macro_contrasts'] if m['primary'])},indent=2))
