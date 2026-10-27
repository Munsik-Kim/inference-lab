"""File identity and schema helpers only. No model or CUDA imports."""
import csv
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

def utc(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def identity(p):
    p=Path(p); h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024), b''): h.update(b)
    return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def append(p,d):
    with Path(p).open('a') as f:f.write(json.dumps(d,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def csvwrite(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows or [{'status':'NOT_RUN'}])
def freeze(root,paths,external=None):
    files={str(Path(p).relative_to(root)):identity(p) for p in sorted(paths)}
    body={'files':files,'external':external or {}}
    return {**body,'bundle_sha256':hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'utc':utc()}
def verify(root,record):
    body={k:record[k] for k in ['files','external']}
    assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==record['bundle_sha256'],'FREEZE_DIGEST'
    for name,row in record['files'].items():
        p=(Path(root)/name).resolve()
        if not p.is_relative_to(Path(root).resolve()) or identity(p)!=row:raise ValueError('SOURCE_MISMATCH:'+name)
    for name,row in record['external'].items():
        if identity(name)!=row:raise ValueError('EXTERNAL_IDENTITY_MISMATCH:'+name)
def array_hash(a): return hashlib.sha256(a.tobytes(order='C')).hexdigest()
