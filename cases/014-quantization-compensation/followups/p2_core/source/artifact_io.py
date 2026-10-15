"""Original codec, validated temporary file, then atomic final-name publication."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
import codec
from independent_codec import read

def write_new(path,w):
    p=Path(path);tmp=p.with_name(p.name+'.partial')
    if p.exists() or tmp.exists():raise FileExistsError(p)
    b,stats=codec.encode(w);p.parent.mkdir(parents=True,exist_ok=True)
    with tmp.open('xb') as f:f.write(b)
    a,q,s=codec.decode(tmp.read_bytes());ia,iq,iss=read(tmp)
    assert np.array_equal(a,ia) and np.array_equal(q,iq) and np.array_equal(s,iss)
    assert np.isfinite(a).all() and tmp.stat().st_size==len(b)
    tmp.rename(p)
    return a,q,s,dict(stats,artifact_bytes=len(b),header_bytes=codec.HEADER.size,scale_bytes=s.nbytes,code_bytes=len(b)-codec.HEADER.size-s.nbytes)
