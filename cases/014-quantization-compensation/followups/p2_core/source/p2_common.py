"""P2 immutable I/O, identities and native projection-group order."""
import hashlib,json,os
from pathlib import Path
from datetime import datetime,timezone

REVISION='c1899de289a04d12100db370d81485cdf75e47ca'
METHODS=['BF16','RTN','COMP_QUARTER','COMP_HALF']
ALPHA={'RTN':0.,'COMP_QUARTER':.25,'COMP_HALF':.5}
POSITIONS=[0,16,33,50,67,84,101,118,135,152,169,186,203,220,237,254]
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def digest(b):return hashlib.sha256(b).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def save(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial')
    if p.exists() or tmp.exists():raise FileExistsError(p)
    with tmp.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
    os.replace(tmp,p)
def append(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
def jsonl(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def groups(block):
    base=f'model.layers.{block}.'
    return [('qkv',[base+'self_attn.'+p+'_proj' for p in ['q','k','v']]),
            ('o',[base+'self_attn.o_proj']),('gate_up',[base+'mlp.gate_proj',base+'mlp.up_proj']),('down',[base+'mlp.down_proj'])]
def all_names(layers=28):return [n for b in range(layers) for _,names in groups(b) for n in names]

