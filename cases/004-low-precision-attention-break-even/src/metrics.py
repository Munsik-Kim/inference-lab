"""Metrics and cluster-preserving statistical summaries; NumPy only."""
import numpy as np


def unit_error(actual,reference,floor=1e-6):
    a=np.asarray(actual,dtype=np.float64);r=np.asarray(reference,dtype=np.float64)
    if a.shape!=r.shape:raise ValueError('Shape mismatch')
    bad=~np.isfinite(a).all(axis=-1);bad_ref=~np.isfinite(r).all(axis=-1)
    if bad.any() or bad_ref.any():return dict(relative_output_error=None,absolute_rms_error=None,cosine=None,invalid_rows=int((bad|bad_ref).sum()),near_zero=False,status='NUMERICAL_REJECT')
    norm=float(np.linalg.norm(r));rms=float(np.sqrt(np.mean(r*r)));diff=float(np.linalg.norm(a-r))
    an=float(np.linalg.norm(a));near=rms<=floor
    return dict(relative_output_error=None if near else diff/norm,absolute_rms_error=float(np.sqrt(np.mean((a-r)**2))),cosine=None if near or an==0 else float(np.sum(a*r)/(an*norm)),invalid_rows=0,near_zero=near,status='NUMERICAL_REVIEW' if near else 'OK',error_squared_sum=float(np.sum((a-r)**2)),reference_squared_sum=float(np.sum(r*r)),actual_squared_sum=float(np.sum(a*a)),dot_sum=float(np.sum(a*r)))


def numerical_screen(units):
    if not units:return {'status':'NOT_RUN','median':None,'p95':None,'invalid_rows':None,'near_zero_units':None}
    invalid=sum(u['invalid_rows'] for u in units);near=sum(u['near_zero'] for u in units)
    vals=[u['relative_output_error'] for u in units if u['relative_output_error'] is not None]
    median=float(np.median(vals)) if vals else None;p95=float(np.quantile(vals,.95)) if vals else None
    status='NUMERICAL_REJECT' if invalid else 'NUMERICAL_REVIEW' if near or len(vals)!=len(units) else 'PASS' if median<=.01 and p95<=.03 else 'NUMERICAL_REJECT'
    return dict(status=status,median=median,p95=p95,invalid_rows=invalid,near_zero_units=near,units=len(units))


def paired_timing_bootstrap(round_pairs,seed=470004,repetitions=2000):
    """Outer process round, inner paired block within each retained document."""
    if len(round_pairs)<2:raise ValueError('Need multiple process rounds')
    arrays=[(np.atleast_2d(np.asarray(a,float)),np.atleast_2d(np.asarray(b,float))) for a,b in round_pairs]
    for a,b in arrays:
        if a.shape!=b.shape or a.size==0 or not np.isfinite(a).all() or not np.isfinite(b).all() or np.min(a)<=0 or np.min(b)<=0:raise ValueError('Invalid pairs')
    aa=np.concatenate([a.ravel() for a,b in arrays]);bb=np.concatenate([b.ravel() for a,b in arrays])
    point=float(np.median(aa/bb));ratio=float(np.median(aa)/np.median(bb))
    rng=np.random.default_rng(seed);boot=[]
    for _ in range(repetitions):
        pairs=[]
        for ri in rng.integers(0,len(arrays),len(arrays)):
            a,b=arrays[ri]
            for ad,bd in zip(a,b):
                ids=rng.integers(0,len(ad),len(ad));pairs.extend(ad[ids]/bd[ids])
        boot.append(np.median(pairs))
    return dict(speedup=point,ratio_of_median_costs=ratio,ci95=[float(v) for v in np.quantile(boot,[.025,.975])],process_rounds=len(arrays),paired_blocks=sum(a.size for a,b in arrays),bootstrap_replicates=repetitions)


def document_bootstrap(grouped,seed=480004,repetitions=2000):
    ids=sorted(grouped);arrays=[np.asarray(grouped[d],float) for d in ids]
    rng=np.random.default_rng(seed);med=[];tail=[]
    for _ in range(repetitions):
        values=np.concatenate([arrays[i] for i in rng.integers(0,len(ids),len(ids))]);med.append(np.median(values));tail.append(np.quantile(values,.95))
    return dict(documents=len(ids),median_ci95=np.quantile(med,[.025,.975]).tolist(),p95_ci95=np.quantile(tail,[.025,.975]).tolist())


def decision(speed,error,real_supported,interface=True):
    reasons=[]
    if not interface:return 'UNSUPPORTED',['INTERFACE_NOT_VERIFIED']
    if not real_supported:reasons.append('NO_MATCHING_REAL_QWEN_FIDELITY')
    if error and error['status'] in ['NUMERICAL_REJECT','NUMERICAL_REVIEW']:reasons.append(error['status'])
    if not speed:reasons.append('TIMING_NOT_RUN')
    elif speed['speedup']<1.10:reasons.append('SPEEDUP_BELOW_1_10')
    elif speed['ci95'][0]<=1.0:reasons.append('SPEEDUP_INTERVAL_INCLUDES_ONE')
    if not real_supported:return 'SYNTHETIC_ONLY',reasons
    if not error or error['status']=='NOT_RUN':return 'NOT_RUN',reasons+['FIDELITY_NOT_RUN']
    if error['status']!='PASS':return error['status'],reasons
    if speed is None:return 'NOT_RUN',reasons
    if speed['speedup']<1.10:return 'KEEP_BF16',reasons
    if speed['ci95'][0]<=1:return 'INCONCLUSIVE',reasons
    return 'LOCAL_CANDIDATE',[]
