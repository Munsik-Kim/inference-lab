"""Frozen S3 inputs; labels are an independent integer evaluator, never features."""
from pathlib import Path
from itertools import permutations
import argparse
import datetime
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPECS = {'smoke': (61601, 8, 32, 202), 'fit': (61101, 512, 256, 101),
         'dev': (61201, 128, 256, 202), 'fresh': (61301, 1024, 2048, 303)}
BANDS = [(1,32),(33,64),(65,128),(129,256)]

def sha(data): return hashlib.sha256(data).hexdigest()
def file_sha(path): return sha(Path(path).read_bytes())
def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def gold_trace(tokens):
    """Lexicographic S3 IDs; product x_t ... x_1, identity is label 0."""
    values=np.asarray(tokens)
    if values.ndim!=2 or values.dtype.kind not in 'iu' or np.any(values<0) or np.any(values>5):
        raise ValueError('Expected integer S3 tokens [sequences,time]')
    elements=list(permutations(range(3))); index={x:i for i,x in enumerate(elements)}
    state=np.tile(np.arange(3),(len(values),1)); gold=np.empty(values.shape,dtype=np.uint8)
    elements=np.asarray(elements)
    for t in range(values.shape[1]):
        current=elements[values[:,t]]
        state=np.take_along_axis(current,state,axis=1)
        gold[:,t]=[index[tuple(x)] for x in state]
    return gold

def load(role):
    m=json.loads((ROOT/'inputs/manifest.json').read_text())['cohorts'][role]
    arrays=[]
    for key in ['tokens','gold']:
        path=ROOT/m[key+'_path']
        if file_sha(path)!=m[key+'_sha256']: raise ValueError('Changed input '+role+'/'+key)
        arrays.append(np.load(path,allow_pickle=False))
    tokens,gold=arrays
    if tokens.shape!=(m['sequences'],m['group_tokens']) or gold.shape!=tokens.shape or tokens.dtype!=np.uint8 or gold.dtype!=np.uint8:
        raise ValueError('Input shape/dtype differs from frozen design')
    ids_path=ROOT/m['sample_ids_path']
    if file_sha(ids_path)!=m['sample_ids_sha256']:raise ValueError('Changed sample IDs')
    ids=json.loads(ids_path.read_text())
    if len(ids)!=len(tokens) or len(set(ids))!=len(ids):raise ValueError('Duplicate/missing sample IDs')
    if [sha(row.tobytes()) for row in tokens]!=m['per_sequence_token_sha256']:raise ValueError('Sequence bytes changed')
    if not np.array_equal(gold_trace(tokens),gold):raise ValueError('Independent S3 gold mismatch')
    return tokens,gold,m

def freeze(case010):
    target=ROOT/'inputs/manifest.json'
    if target.exists():raise ValueError('Input freeze already exists')
    case010=Path(case010); matches=[]
    seeds={v[0] for v in SPECS.values()}|{61401,61501}
    def walk(v,path,trail=''):
        if isinstance(v,dict):
            for k,x in v.items():
                if 'seed' in k.lower() and type(x)==int and x in seeds:
                    matches.append({'path':path,'key':trail+'/'+k,'seed':x})
                walk(x,path,trail+'/'+k)
        elif isinstance(v,list):
            for i,x in enumerate(v):walk(x,path,trail+'/'+str(i))
    for p in case010.rglob('*.json'):
        try:walk(json.loads(p.read_text()),str(p.relative_to(case010)))
        except (UnicodeDecodeError,json.JSONDecodeError):continue
    if matches:raise ValueError('Previously used generator seed: '+str(matches))
    historical=[(str(p.relative_to(case010)),np.load(p,allow_pickle=False))
                for p in (case010/'versions/v2/inputs').rglob('tokens.npy')]
    manifest={'schema':'case011-inputs-v1','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'task':'S3 all six elements, independent uniform; left cumulative product',
              'generator':'PCG64 SeedSequence([role_seed,Case010_split_tag,sequence_index]); int64 draws then uint8',
              'prior_seed_scan_matches':matches,'seed_scan_scope':'all Case010 JSON integer seed fields',
              'cohorts':{},'complete_sequence_duplicate_checks':[],'fitting_positions_seed':61501}
    generated=[]
    for role,(seed,n,t,tag) in SPECS.items():
        tokens=np.stack([np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,tag,i]))).integers(6,size=t,dtype=np.int64) for i in range(n)]).astype(np.uint8)
        gold=gold_trace(tokens); folder=ROOT/'inputs'/role;folder.mkdir(exist_ok=False)
        for kind,array in [('tokens',tokens),('gold',gold)]:np.save(folder/(kind+'.npy'),array,allow_pickle=False)
        ids=[f'case011:{role}:{seed}:{i:06d}' for i in range(n)];write_json(folder/'sample_ids.json',ids)
        for other_name,other in historical+generated:
            # Equal complete lengths only; naturally shared short prefixes are not duplicate base sequences.
            count=len(set(map(bytes,tokens))&set(map(bytes,other))) if t==other.shape[1] else 0
            manifest['complete_sequence_duplicate_checks'].append({'cohort':role,'other':other_name,'equal_length':t==other.shape[1],'duplicates':count})
            if count:raise ValueError('Complete input sequence duplicate')
        if len(set(map(bytes,tokens)))!=n:raise ValueError('Within-cohort duplicate')
        generated.append((role,tokens))
        manifest['cohorts'][role]={'seed':seed,'split_tag':tag,'sequences':n,'group_tokens':t,
          **{kind+'_path':str((folder/(kind+'.npy')).relative_to(ROOT)) for kind in ['tokens','gold']},
          **{kind+'_sha256':file_sha(folder/(kind+'.npy')) for kind in ['tokens','gold']},
          'sample_ids_path':str((folder/'sample_ids.json').relative_to(ROOT)),
          'sample_ids_sha256':file_sha(folder/'sample_ids.json'),
          'per_sequence_token_sha256':[sha(row.tobytes()) for row in tokens]}
    positions=np.empty((512,32),dtype=np.int16)
    for i in range(512):
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([61501,i])))
        positions[i]=np.concatenate([np.sort(rng.choice(np.arange(a,b+1),8,replace=False)) for a,b in BANDS])
    np.save(ROOT/'inputs/mixed_positions.npy',positions,allow_pickle=False)
    manifest['mixed_positions_sha256']=file_sha(ROOT/'inputs/mixed_positions.npy')
    manifest['short_rows']=manifest['mixed_rows']=16384
    write_json(target,manifest)
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case010',required=True);a=p.parse_args()
    d=freeze(a.case010);print(json.dumps({'status':'FROZEN','cohorts':list(d['cohorts']),'manifest_sha256':file_sha(ROOT/'inputs/manifest.json')}))
