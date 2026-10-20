import hashlib
import json
import pathlib
import datetime

def dumps(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(x):
    return hashlib.sha256(dumps(x).encode()).hexdigest()

def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save(path,x):
    p=pathlib.Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)

def load(path):
    return json.loads(pathlib.Path(path).read_text())
