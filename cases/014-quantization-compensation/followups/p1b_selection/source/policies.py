"""SELECT-only exact argmin. No EVAL, fitting or model import."""
import math
from collections import defaultdict
from p1b_common import ALPHAS, TARGETS

POLICIES = ['BASE_ZERO','OLD_FIT','FIXED_QUARTER','FIXED_HALF','FIXED_FULL','SELECT_SSE','SELECT_NLL']

def select(rows, old_fit, targets=TARGETS):
    groups = defaultdict(list)
    seen = set()
    prompts = {}
    for r in rows:
        if r['split'] != 'SELECT-ID':
            raise ValueError('selector accepts SELECT-ID only')
        key = r['target'], r['alpha'], r['document_id_hash']
        if key in seen or r['target'] not in targets or r['alpha'] not in ALPHAS:
            raise ValueError('duplicate/unknown target or alpha')
        if r['status'] != 'COMPLETE' or not all(math.isfinite(r[k]) for k in ['SSE_after','NLL_sum']):
            raise ValueError('invalid measurement')
        doc = r['document_id_hash']
        if doc in prompts and prompts[doc] != r['token_hash']:
            raise ValueError('same document has different model input')
        prompts[doc] = r['token_hash']
        seen.add(key)
        groups[r['target'], r['alpha']].append(r)
    expected = None
    result = []
    for t in targets:
        scores = []
        for a in ALPHAS:
            rr = sorted(groups[t,a], key=lambda r:r['document_id_hash'])
            ids = tuple(r['document_id_hash'] for r in rr)
            if not ids or (expected is not None and ids != expected):
                raise ValueError('missing/different documents or cohort')
            expected = ids
            if any(r['valid_token_count'] != 255 for r in rr):
                raise ValueError('token count mismatch')
            hashes = {r['artifact_sha256'] for r in rr}
            if len(hashes) != 1:
                raise ValueError('artifact mismatch')
            scores.append({'alpha':a,'SSE':math.fsum(r['SSE_after'] for r in rr),
                           'NLL':math.fsum(r['NLL_sum'] for r in rr)/sum(r['valid_token_count'] for r in rr),
                           'artifact_sha256':next(iter(hashes))})
        sse = min(scores, key=lambda s:(s['SSE'],s['alpha']))['alpha']
        nll = min(scores, key=lambda s:(s['NLL'],s['alpha']))['alpha']
        old = old_fit[t]
        if old not in ALPHAS:
            raise ValueError('unregistered OLD_FIT')
        result.append({'target':t,'policy_alpha':dict(zip(POLICIES,[0,old,.25,.5,1,sse,nll])),
                       'scores':scores,'SELECT_documents':len(expected)})
    return result

def map_policies(rows, frozen):
    """Alias existing observations; never run inference for a policy."""
    lookup = {}
    cohorts = defaultdict(set)
    for r in rows:
        key = r['target'],r['alpha'],r['split'],r['document_id_hash']
        if key in lookup:
            raise ValueError('duplicate row')
        lookup[key] = r
        cohorts[r['split']].add(r['document_id_hash'])
    for split, ids in cohorts.items():
        for f in frozen:
            for a in ALPHAS:
                for doc in ids:
                    if (f['target'],a,split,doc) not in lookup:
                        raise ValueError('incomplete candidate grid')
    mapped = []
    for f in frozen:
        for split, ids in sorted(cohorts.items()):
            for policy, alpha in f['policy_alpha'].items():
                for doc in sorted(ids):
                    key = f['target'],alpha,split,doc
                    if key not in lookup:
                        raise ValueError('missing policy measurement')
                    r = lookup[key]
                    score = next(s for s in f['scores'] if s['alpha'] == alpha)
                    if r['artifact_sha256'] != score['artifact_sha256']:
                        raise ValueError('policy artifact changed')
                    mapped.append(dict(r, policy=policy))
    return mapped
