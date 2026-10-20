import datetime,hashlib,json,pathlib,struct

def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(canonical(x).encode()).hexdigest()
def sha_file(path):
 h=hashlib.sha256()
 with open(path,'rb')as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def token_hash(ids):return hashlib.sha256(b''.join(struct.pack('<I',i)for i in ids)).hexdigest()
def load(p):return json.loads(pathlib.Path(p).read_text())
def save(p,x,replace=False):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists()and not replace:raise FileExistsError(p)
 q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');json.loads(q.read_text());q.replace(p)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
