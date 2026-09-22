"""Bounded integer-program fixtures; no exec/eval or model inference.

Facts and labels use independently seeded streams. Foils bracket the answer, so
minimum/maximum rules score zero by construction; middle-rank membership remains
an explicit design cue, not evidence of cue-free natural tasks.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import random


def oracle(f):
    x=f['start']
    for i in range(f['count']):x+=f['step']*i
    return x%f['modulus']


def generate():
    rows=[]
    for split,n in [('calibration',96),('development',48),('heldout',192)]:
        facts_rng=random.Random('diova-data-v2-facts-'+split)
        labels_rng=random.Random('diova-data-v2-labels-'+split)
        ids_rng=random.Random('diova-data-v2-order-'+split)
        labels=list(range(4))*(n//4);labels_rng.shuffle(labels)
        for i in range(n):
            # This domain constraint precedes label assignment and any model run.
            # Bracketing requires an interior answer; reject only endpoint facts.
            while True:
                f={'start':facts_rng.randrange(100,10000),'step':facts_rng.randrange(1,13),'count':facts_rng.randrange(2,13),'modulus':facts_rng.randrange(17,80)}
                y=oracle(f)
                if 0<y<f['modulus']-1:break
            if y!=(f['start']+f['step']*f['count']*(f['count']-1)//2)%f['modulus']:raise ValueError('oracle mismatch')
            foils=[facts_rng.randrange(y),facts_rng.randrange(y+1,f['modulus'])]
            foils.append(facts_rng.choice([v for v in range(f['modulus']) if v not in [y,*foils]]))
            labels_rng.shuffle(foils);options=foils[:];options.insert(labels[i],y)
            ident=hashlib.sha256(json.dumps([split,f,options],sort_keys=True).encode()).hexdigest()[:16]
            rows.append({'id':'data-v2-'+ident,'version':'2.1-domain-valid','split':split,'facts':f,'options':options,'gold':labels[i], 'answer':y})
        segment=rows[-n:];ids_rng.shuffle(segment);rows[-n:]=segment
    return rows


def audit(rows):
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('duplicate IDs')
    if Counter(r['split'] for r in rows)!=Counter(calibration=96,development=48,heldout=192):raise ValueError('missing/extra split fixtures')
    if len({tuple(sorted(r['facts'].items())) for r in rows})!=len(rows):raise ValueError('repeated program facts')
    result={}
    for split in ('calibration','development','heldout'):
        group=[r for r in rows if r['split']==split];counts=defaultdict(Counter)
        for r in group:
            o=r['options'];a=oracle(r['facts'])
            if len(o)!=4 or len(set(o))!=4 or o[r['gold']]!=a or not min(o)<a<max(o) or not all(0<=v<r['facts']['modulus'] for v in o):raise ValueError('invalid gold/foils/domain')
            counts[r['facts']['count']][r['gold']]+=1
        result[split]={'n':len(group),'label_counts':dict(Counter(r['gold'] for r in group)),
            'count_by_gold':{str(k):dict(v) for k,v in sorted(counts.items())},
            'constant_label_correct':max(Counter(r['gold'] for r in group).values()),
            'minimum_option_correct':sum(r['options'].index(min(r['options']))==r['gold'] for r in group),
            'maximum_option_correct':sum(r['options'].index(max(r['options']))==r['gold'] for r in group),
            'index_mod_4_correct':sum(i%4==r['gold'] for i,r in enumerate(group)),
            'count_to_label_in_sample_best_correct':sum(max(v.values()) for v in counts.values()),
            'deterministic_count_to_label':all(len(v)==1 for v in counts.values())}
        if result[split]['deterministic_count_to_label']:raise ValueError('deterministic count-label coupling')
    train=[r for r in rows if r['split']=='calibration'];test=[r for r in rows if r['split']=='heldout']
    votes=defaultdict(Counter)
    for r in train:votes[r['facts']['count']][r['gold']]+=1
    mapping={k:max(range(4),key=lambda label:v[label]) for k,v in votes.items()}
    result['heldout']['calibration_count_rule_correct']=sum(mapping[r['facts']['count']]==r['gold'] for r in test)
    result['scope']='Input/oracle checks only; no data-v2 model evaluation. Gold is always interior by construction. Best-count rule above is in-sample descriptive; separate calibration-fitted rule is also reported.'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);rows=generate()
    (a.output/'inputs.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    (a.output/'audit.json').write_text(json.dumps(audit(rows),indent=2)+'\n')
