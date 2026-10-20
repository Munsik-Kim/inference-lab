from collections import Counter
from .common import load,save
import pathlib

def summarize(result_root,planned):
    root=pathlib.Path(result_root);episodes=[load(p) for p in sorted((root/'episodes').glob('*.json'))]
    by_id={e['episode_id']:e for e in episodes}
    if len(by_id)!=len(episodes):raise ValueError('duplicate cells')
    expected={r['episode_id'] for r in planned}
    if set(by_id)-expected:raise ValueError('unplanned cell')
    groups=[]
    for kv in dict.fromkeys(p['kv'] for p in planned):
        for menu in (8,32):
            for cap in (256,1024):
                rows=[e for e in episodes if (e['kv'],e['menu'],e['cap'])==(kv,menu,cap)]
                observed=[e for e in rows if e['availability']=='OBSERVED']
                groups.append({'kv':kv,'menu':menu,'cap':cap,'planned':12,'completed':len(rows),'observed':len(observed),
                  'verified':sum(e['VERIFIED_WORKFLOW_SUCCESS']is True for e in observed),'state':sum(e['STATE_CORRECT'] for e in observed),
                  'final':sum(e['FINAL_ANSWER_CORRECT'] for e in observed),'false_done':sum(e['CLAIMED_DONE_WITH_WRONG_STATE'] for e in observed),
                  'terminals':dict(Counter(e['terminal_reason'] for e in rows)),
                  'first_unmet':dict(Counter(e['first_unmet_obligation'] for e in observed)),
                  'tokens':sum(e['output_tokens'] for e in rows),'model_calls':sum(e['model_calls'] for e in rows),'tool_attempts':sum(e['tool_attempts'] for e in rows),
                  'wall_episode_sum_seconds':sum(e['wall_seconds'] for e in rows)})
    pairs=[]
    lookup={(e['task_id'],e['kv'],e['menu'],e['cap']):e for e in episodes if e['availability']=='OBSERVED'}
    def pair(keys_a,keys_b,label):
        counts=Counter();n=0
        for a,b in zip(keys_a,keys_b):
            if a not in lookup or b not in lookup:continue
            x=lookup[a]['VERIFIED_WORKFLOW_SUCCESS'];y=lookup[b]['VERIFIED_WORKFLOW_SUCCESS'];n+=1
            counts['both_success' if x and y else 'loss' if x else 'recovery' if y else 'both_fail']+=1
        pairs.append({'comparison':label,'observed_pairs':n,**{k:counts[k] for k in ('both_success','loss','recovery','both_fail')}})
    tids=sorted({r['task_id'] for r in planned});kvs=list(dict.fromkeys(r['kv'] for r in planned))
    for kv in kvs:
        for menu in (8,32):pair([(t,kv,menu,256)for t in tids],[(t,kv,menu,1024)for t in tids],f'{kv}/menu{menu}: cap256 to1024')
    if len(kvs)==2:
        for menu in (8,32):
            for cap in (256,1024):pair([(t,kvs[0],menu,cap)for t in tids],[(t,kvs[1],menu,cap)for t in tids],f'menu{menu}/cap{cap}: {kvs[0]} to{kvs[1]}')
    base=[e for e in episodes if (e['kv'],e['menu'],e['cap'],e['availability'])==('B16',8,1024,'OBSERVED')]
    family={f:sum(e['VERIFIED_WORKFLOW_SUCCESS']is True for e in base if e['family']==f) for f in 'ABCD'}
    usable=None if len(base)!=12 else sum(family.values())>=9 and min(family.values())>=2
    return {'status':'COMPLETED_P1' if len(episodes)==len(planned) and all(e['availability']=='OBSERVED' for e in episodes) else 'PARTIAL_P1',
      'planned':len(planned),'completed':len(episodes),'missing':sorted(expected-set(by_id)),'groups':groups,'paired':pairs,
      'baseline_usable':usable,'baseline_family_success':family,'all_tokens':sum(e['output_tokens'] for e in episodes),
      'all_model_calls':sum(e['model_calls'] for e in episodes),'all_tool_attempts':sum(e['tool_attempts'] for e in episodes)}
