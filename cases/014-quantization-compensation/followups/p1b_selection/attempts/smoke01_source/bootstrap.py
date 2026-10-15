"""Fixed-policy shared document bootstrap; never performs selection."""
import numpy as np

def shared_indices(documents, resamples=10000, seed=6101519):
    if documents<1:raise ValueError('no documents')
    return np.random.default_rng(seed).integers(0,documents,size=(resamples,documents))

def contrast_samples(candidate, baseline, denominator, indices):
    c,b,d=(np.asarray(x,dtype=np.float64) for x in [candidate,baseline,denominator])
    if c.shape!=b.shape or c.shape!=d.shape or c.ndim!=2:
        raise ValueError('expected aligned target × document arrays')
    if not all(np.isfinite(x).all() for x in [c,b,d]) or np.any(d<0):
        raise ValueError('invalid data/denominator')
    numer=(c-b)[:,indices].sum(axis=-1)
    denom=d[:,indices].sum(axis=-1)
    if np.any(denom<=0):raise ValueError('undefined denominator')
    ratios=numer/denom
    return ratios,ratios.mean(axis=0)
