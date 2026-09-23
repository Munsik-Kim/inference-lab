"""Sequence-level endpoints; diagnostic cohorts never enlarge fresh sample size."""
from pathlib import Path
import sys
import numpy as np
from .common import GRID,write_json,file_sha
V1=Path(__file__).resolve().parent/'v1_reference'
if str(V1) not in sys.path:sys.path.insert(0,str(V1))
from codec.survival import clopper_pearson_upper

def metrics(predictions,gold,*,family=195):
    pred=np.asarray(predictions);gold=np.asarray(gold)
    if pred.shape!=(len(gold),gold.shape[1]+1) or pred.dtype.kind not in 'iu' or gold.dtype.kind not in 'iu':
        raise ValueError('Expected BOS-inclusive predictions and group-only gold')
    if np.any((pred<-1)|(pred>5)) or np.any((gold<0)|(gold>5)):raise ValueError('Invalid class')
    n,t=gold.shape;correct=pred[:,1:]==gold
    fail=~correct;has=fail.any(1);first=fail.argmax(1)+1
    tau=[int(x) if seen else None for x,seen in zip(first,has)]
    times=np.where(has,first,t+1)
    survival=np.array([(times>h).mean() for h in range(1,t+1)])
    eligible=lambda values,eps:None if not np.any(values<=eps) else int(np.flatnonzero(values<=eps)[-1]+1)
    rows=[]
    grid=[x for x in GRID if x<=t]
    if t not in grid:grid.append(t)
    for h in grid:
        k=int((times<=h).sum());upper=None if family is None else clopper_pearson_upper(k,n,alpha=.05/family)
        rows.append({'horizon':h,'failures':k,'F':k/n,'survival':1-k/n,'p_upper':upper})
    supported=lambda eps:max([x['horizon'] for x in rows if x['p_upper'] is not None and x['p_upper']<=eps],default=None)
    return {'N':n,'T':t,'tau':tau,'RMST0':float(np.minimum(times-1,t).mean()),
        'empirical_T05_all_tokens':eligible(1-survival,.05),'empirical_T01_all_tokens':eligible(1-survival,.01),
        'supported_T05_grid':supported(.05),'supported_T01_grid':supported(.01),
        'grid_rows':rows,'family_size':family,'alpha':.05,'epsilon_primary':.05,
        'token_accuracy':float(correct.mean()),'final_quarter_accuracy':float(correct[:,-max(t//4,1):].mean()),
        'bos_accuracy':float((pred[:,0]==0).mean()),'bos_unscored_in_tau':True,
        'first_failure_censored':int((~has).sum()),'invalid_prediction_tokens':int((pred[:,1:]<0).sum()),
        'RMST0_definition':'mean(min(tau-1,Tmax)); censored contributes Tmax; first wrong token excluded',
        'confidence_scope':'diagnostic, no confidence family claimed' if family is None else 'grid only; pointwise curves outside grid have no simultaneous interval'},correct,survival

def save_result(folder,*,predictions,gold,identity,execution,ledger,first_terminal,terminal_codes,extra=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for name in ['predictions.npz','summary.json']:
        if (folder/name).exists():raise ValueError('Completed result file already exists')
    summary,correct,survival=metrics(predictions,gold,family=identity.get('family_size',195))
    np.savez_compressed(folder/'predictions.npz',predictions=np.asarray(predictions,dtype='i1'),
        correctness=np.packbits(correct,axis=1,bitorder='little'),gold=np.asarray(gold,dtype='u1'))
    summary.update(schema='case010-failure-aware-v2-cell-v1',**identity,execution=execution,ledger=ledger,
        first_terminal_write=first_terminal,terminal_codes=terminal_codes,
        terminal_count=sum(x!=0 for x in terminal_codes),artifact_sha256=file_sha(folder/'predictions.npz'))
    if extra:summary.update(extra)
    write_json(folder/'summary.json',summary)
    return summary

def paired_summary(a,b,*,bootstrap_seed,repeats=5000):
    for k in ['N','T','model_seed','cohort','input_manifest_sha256','sample_ids_sha256']:
        if a[k]!=b[k]:raise ValueError('Paired identity mismatch: '+k)
    av=np.array([a['T'] if t is None else t-1 for t in a['tau']],dtype=np.float64)
    bv=np.array([b['T'] if t is None else t-1 for t in b['tau']],dtype=np.float64)
    delta=av-bv;rng=np.random.default_rng(bootstrap_seed)
    boot=np.empty(repeats)
    for start in range(0,repeats,100):
        size=min(100,repeats-start);inds=rng.integers(0,len(delta),size=(size,len(delta)))
        boot[start:start+size]=delta[inds].mean(1)
    return {'candidate':a['arm'],'baseline':b['arm'],'model_seed':a['model_seed'],'N':a['N'],
        'RMST0_delta':float(delta.mean()),'pointwise_95_CI':np.quantile(boot,[.025,.975]).tolist(),
        'bootstrap_seed':bootstrap_seed,'bootstrap_repeats':repeats,'unit':'independent input sequence within one fixed checkpoint',
        'positive_difference_sequences':int((delta>0).sum()),'negative_difference_sequences':int((delta<0).sum()),
        'grid_risk_difference':[{'horizon':x['horizon'],'candidate_minus_baseline_F':x['F']-y['F']}
             for x,y in zip(a['grid_rows'],b['grid_rows'])],
        'interval_scope':'pointwise paired RMST bootstrap; not a multiple-comparison confirmatory claim',
        'checkpoint_pooling':False}
