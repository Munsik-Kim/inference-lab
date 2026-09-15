"""Crossed document/global-process clusters, with paired blocks within cells."""
import numpy as np

def resample_indices(rng,documents,processes,blocks):
 # One global process draw is shared across all sampled documents.
 return (rng.integers(documents,size=documents),rng.integers(processes,size=processes),rng.integers(blocks,size=(documents,processes,blocks)))

def timing(base,other,seed=505901,repetitions=5000):
 a=np.asarray(base,float);b=np.asarray(other,float)
 if a.shape!=b.shape or a.ndim!=3 or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a<=0) or np.any(b<=0):raise ValueError('Invalid paired timing cube [document,global process,block]')
 ratios=a/b;rng=np.random.default_rng(seed);boot=[]
 for _ in range(repetitions):
  ds,ps,bs=resample_indices(rng,*a.shape)
  boot.append(np.median(ratios[ds[:,None,None],ps[None,:,None],bs]))
 return dict(speedup=float(np.median(ratios)),ci95=np.quantile(boot,[.025,.975],method='linear').tolist(),bootstrap_seed=seed,repetitions=repetitions,documents=a.shape[0],processes=a.shape[1],blocks_per_cell=a.shape[2],ratio_of_medians=float(np.median(a)/np.median(b)))

def error_ci(values,seed=505902,repetitions=5000):
 a=np.asarray(values,float)
 if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('Expected document x head values')
 rng=np.random.default_rng(seed);med=[];tail=[]
 for _ in range(repetitions):
  sampled=a[rng.integers(len(a),size=len(a))].ravel();med.append(np.median(sampled));tail.append(np.quantile(sampled,.95,method='linear'))
 return dict(median_ci95=np.quantile(med,[.025,.975]).tolist(),p95_ci95=np.quantile(tail,[.025,.975]).tolist(),documents=len(a),seed=seed,repetitions=repetitions)
