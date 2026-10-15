"""Immutable local P1b I/O and registered identities. Standard library only."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ALPHAS = [0, .25, .5, .75, 1]
BLOCKS = [4, 12, 20]
PROJECTIONS = ['o_proj', 'down_proj']
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
POSITIONS = [0,16,33,50,67,84,101,118,135,152,169,186,203,220,237,254]

def target(block, projection):
    return f'model.layers.{block}.' + ('self_attn.' if projection == 'o_proj' else 'mlp.') + projection

TARGETS = [target(b, p) for b in BLOCKS for p in PROJECTIONS]

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def digest(b):
    return hashlib.sha256(b).hexdigest()

def utc():
    return datetime.now(timezone.utc).isoformat()

def save(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() or p.with_name(p.name + '.partial').exists():
        raise FileExistsError(p)
    tmp = p.with_name(p.name + '.partial')
    with tmp.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, p)

def append(path, obj):
    with Path(path).open('a') as f:
        f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()

def jsonl(path):
    with open(path) as f:
        return [json.loads(s) for s in f if s.strip()]

def forbidden(*args, **kwargs):
    raise RuntimeError('P1B_READ_ONLY: fitting/encoding/requantization forbidden')
