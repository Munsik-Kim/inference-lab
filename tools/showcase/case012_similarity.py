"""Post-hoc visual grouping for review convenience, never image/label equivalence."""
import argparse
import hashlib
import json
from pathlib import Path
from common import read, require, sha

PATH = 'presentation/content/case012-similarity.json'
THRESHOLDS = {'phash63': 16, 'dhash64': 18, 'rgb_mae': 0.085}


def complete_link(ids, distances):
    """Every pair in a group must meet the cap; no chaining via a middle image."""
    groups = [[i] for i in sorted(ids)]
    while True:
        choices = []
        for a in range(len(groups)):
            for b in range(a+1, len(groups)):
                score = max(distances[frozenset((x,y))] for x in groups[a] for y in groups[b])
                if score <= 1: choices.append((score, groups[a], groups[b], a, b))
        if not choices: return groups
        *_, a, b = min(choices)
        groups[a] = sorted(groups[a]+groups[b]); groups.pop(b)


def group_id(members):
    return 'similar-'+hashlib.sha256('\n'.join(sorted(members)).encode()).hexdigest()[:16]


def load_groups(root, data):
    result = read(root/PATH)
    require(result['schema']=='case012-visual-groups-v1' and result['thresholds']==THRESHOLDS,
            'Case012 similarity configuration mismatch')
    require(result['generator_sha256']==sha(Path(__file__)), 'Case012 similarity generator changed')
    rows = {r['image_id']: r for r in data['pairs']}
    require(result['source_images']=={i:r['image_sha256'] for i,r in rows.items()},
            'Case012 similarity image identity mismatch')
    seen = []
    for g in result['groups']:
        members = g['members']; seen.extend(members)
        require(members==sorted(set(members)) and all(i in rows for i in members), 'Invalid visual members')
        require(g['group_id']==group_id(members) and g['representative'] in members, 'Invalid visual representative')
        require(all(rows[i]['prompt_id']==g['prompt_id'] for i in members), 'Cross-prompt visual group')
        require(0<=g['max_normalized_distance']<=1, 'Visual group exceeds complete-link cap')
    require(len(seen)==192 and set(seen)==set(rows), 'Visual group coverage mismatch')
    return result


def generate(root):
    # Optional tooling only. Pages build and checks consume and verify the manifest
    # with stdlib; no vision model, original environment change or quality labels.
    import numpy as np
    from PIL import Image
    from case012_pages import load_case012
    data=load_case012(root); rows={r['image_id']:r for r in data['pairs']}
    d=np.cos(np.pi*(2*np.arange(32)+1)[None,:]*np.arange(8)[:,None]/64)
    features={}
    for i,row in rows.items():
        with Image.open(root/data['images'][row['asset']]) as im:
            rgb=np.asarray(im.convert('RGB').resize((32,32),Image.Resampling.LANCZOS),dtype=float)/255
            gray=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
            z=(d@gray@d.T).reshape(-1)[1:]
            ph=sum(int(x>np.median(z))<<n for n,x in enumerate(z))
            gray9=np.asarray(im.convert('L').resize((9,8),Image.Resampling.LANCZOS),dtype=float)
            dh=sum(int(x)<<n for n,x in enumerate((gray9[:,1:]>gray9[:,:-1]).reshape(-1)))
            features[i]=(ph,dh,rgb)
    groups=[]
    for pid in sorted({r['prompt_id'] for r in rows.values()}):
        ids=sorted(i for i,r in rows.items() if r['prompt_id']==pid); distances={}
        for n,a in enumerate(ids):
            for b in ids[n+1:]:
                x,y=features[a],features[b]
                distances[frozenset((a,b))]=max((x[0]^y[0]).bit_count()/16,
                    (x[1]^y[1]).bit_count()/18,float(np.abs(x[2]-y[2]).mean())/.085)
        for members in complete_link(ids,distances):
            rep=min(members,key=lambda a:(sum(distances[frozenset((a,b))] for b in members if a!=b),a))
            pairs=[distances[frozenset((a,b))] for n,a in enumerate(members) for b in members[n+1:]]
            groups.append({'group_id':group_id(members),'prompt_id':pid,'members':members,
                           'representative':rep,'max_normalized_distance':max(pairs,default=0)})
    return {'schema':'case012-visual-groups-v1','purpose':'POST_HOC_REVIEW_CONVENIENCE; not semantic equivalence or new model evaluation',
            'method':'same request; complete-link pHash63/dHash64/RGB32 mean absolute difference; medoid representative',
            'thresholds':THRESHOLDS,'generator_sha256':sha(Path(__file__)),
            'source_images':{i:r['image_sha256'] for i,r in sorted(rows.items())},'groups':groups,
            'uses_AI_labels':False,'new_model_inference':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'Use a new grouping manifest output')
    result=generate(a.repo.resolve());a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps({'groups':len(result['groups']),'images':len(result['source_images'])}))
