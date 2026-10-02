"""Model-free contracts. No CUDA or model imports."""
import hashlib, json, math, os
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON literal: '+x)))

def dump(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def validate_setting(s):
    if type(s.get('loops')) is not int or s['loops'] not in (1,2,4): raise ValueError('loops must be 1, 2 or 4')
    if type(s.get('steps')) is not int or not 1<=s['steps']<=125: raise ValueError('steps outside [1,125]')
    return s

def counts(s,cfg=6):
    validate_setting(s); calls=s['steps']*(1 if cfg==1 else 2)
    return {'model':calls,'pre':calls*6,'core':calls*5*s['loops'],'post':calls*6,'text_preamble':calls*2,'decode':calls,'joint_blocks':calls*(12+5*s['loops'])}

def verify_freeze(root):
    root=Path(root); freeze=load(root/'configs/input_freeze.json')
    for name,h in freeze['files'].items():
        if sha(root/name)!=h: raise ValueError('Frozen input modified: '+name)
    return freeze

def unique_records(rows):
    ids=[r['job_id'] for r in rows]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate job_id')
    for r in rows:
        if r['status']=='SUCCESS':
            for k in ('complete_seconds','sampler_gpu_ms'):
                if not math.isfinite(r[k]) or r[k]<0: raise ValueError('Invalid time: '+r['job_id'])
            if r['image_shape']!=[512,512,3] or not r['finite']: raise ValueError('Invalid successful image')
    return rows

def verify_attempt(path):
    path=Path(path); inv=load(path/'inventory.json')
    expected=set(inv)|{'inventory.json'}
    actual={str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
    if actual!=expected: raise ValueError('Attempt inventory mismatch')
    for name,x in inv.items():
        if (path/name).stat().st_size!=x['bytes'] or sha(path/name)!=x['sha256']: raise ValueError('Attempt bytes mismatch: '+name)
    return load(path/'record.json')

def commit_attempt(partial, final, record):
    partial,final=Path(partial),Path(final)
    if final.exists(): raise ValueError('Refuse overwrite completed attempt')
    dump(partial/'record.json',record)
    inventory={str(p.relative_to(partial)): {'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(partial.rglob('*')) if p.is_file()}
    dump(partial/'inventory.json',inventory)
    for name,x in inventory.items():
        if sha(partial/name)!=x['sha256']: raise ValueError('Attempt inventory failure')
    partial.rename(final)

def annotation_validate(data, items, rubric_hash):
    if data.get('schema')!='case012-annotations-v1' or data.get('rubric_sha256')!=rubric_hash: raise ValueError('Annotation schema/rubric mismatch')
    if data.get('evaluator',{}).get('type') not in ('human','model-assisted'): raise ValueError('Evaluator identity required')
    if not data.get('evaluator',{}).get('id'): raise ValueError('Evaluator id required')
    expected={i['image_id']:i for i in items}; seen=set()
    for row in data.get('rows',[]):
        key=row['image_id']
        if key in seen or key not in expected: raise ValueError('Duplicate/unknown annotation image')
        seen.add(key); item=expected[key]
        if row['image_sha256']!=item['image_sha256']: raise ValueError('Annotation image hash mismatch')
        if set(row['values'])!={c['id'] for c in item['constraints']}: raise ValueError('Constraint coverage mismatch')
        if any(v not in ('satisfied','not_satisfied','uncertain') for v in row['values'].values()): raise ValueError('Unknown annotation value')
    return {'complete':len(seen)==len(expected),'annotated':len(seen),'expected':len(expected),'missing':sorted(set(expected)-seen)}
