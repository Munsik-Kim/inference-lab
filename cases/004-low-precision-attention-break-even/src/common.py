"""Portable record/spec helpers. No GPU imports."""
import hashlib
import math
import json
from pathlib import Path
from datetime import datetime, timezone

CASE = Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n')

def utc():return datetime.now(timezone.utc).isoformat()

def shape_id(s):
    return f"{s['family']}-b{s['batch']}-hq{s['hq']}-hk{s['hkv']}-n{s['length']}-d{s['dim']}-c{int(s['causal'])}"

def shapes():
    out=[dict(family='synthetic',batch=1,hq=8,hkv=8,length=n,dim=d,causal=c) for n in [512,1024,2048,4096,8192,16384] for d in [64,128] for c in [True,False]]
    out += [dict(family='qwen',batch=1,hq=16,hkv=8,length=n,dim=128,causal=True) for n in [512,2048,4096]]
    return [{**s,'id':shape_id(s),'scale':s['dim']**-0.5,'input_dtype':'bfloat16','output_dtype':'bfloat16','layout':'BHND','layer':13 if s['family']=='qwen' else None} for s in out]

def verify_spec(case=CASE):
    case=Path(case);p=case/'configs/experiment_spec.json';s=json.loads(p.read_text())
    expected=(case/'configs/experiment_spec.sha256').read_text().split()[0]
    assert sha(p)==expected,'Specification hash mismatch'
    for n,h in s['frozen_files'].items():assert sha(case/n)==h,n
    return s,expected

def allow_confirmation(split,spec):
    if split!='confirmation':raise ValueError('Confirmation requires held-out IDs')
    assert spec['protocol_version']==1

def ensure_real(record):
    if record.get('evidence_kind')!='gpu_measurement' or record.get('mock',False):
        raise ValueError('Mock/synthetic CPU fixture is not GPU timing evidence')
    if record.get('status')=='OK':
        for k in ['wall_ms','event_ms']:
            if not record[k] or not all(math.isfinite(x) and x>0 for x in record[k]):raise ValueError('Invalid measured latency')
