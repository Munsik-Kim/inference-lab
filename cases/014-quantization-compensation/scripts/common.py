"""Case014 local I/O: immutable completed outputs and explicit hashes."""
import hashlib,json,os
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def digest(b):return hashlib.sha256(b).hexdigest()
def save(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(p)
 t=p.with_name(p.name+'.partial')
 with t.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
 os.replace(t,p)
def append(path,obj):
 with open(path,'a') as f:f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
