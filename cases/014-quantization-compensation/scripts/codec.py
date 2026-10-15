"""C014INT4 v1: little-endian 36-byte header, FP32 scales, low nibble first.
No method identity stored in runtime payload; row-major input-axis group64.
"""
import struct
from pathlib import Path
import numpy as np
MAGIC=b'C014INT4';VERSION=1
HEADER=struct.Struct('<8s7I')
def encode(w,group=64):
 w=np.asarray(w,dtype=np.float32)
 if w.ndim!=2 or min(w.shape)<1 or group!=64 or not np.isfinite(w).all():raise ValueError('invalid matrix/group')
 o,i=w.shape;p=((i+group-1)//group)*group
 v=np.pad(w,((0,0),(0,p-i))).reshape(o,-1,group)
 mx=np.max(np.abs(v),axis=-1)
 s=np.divide(mx,np.float32(7),dtype=np.float32);s=np.where(mx==0,np.float32(1),s).astype('<f4')
 if not np.isfinite(s).all() or (s<=0).any():raise ValueError('scale underflow/nonpositive')
 pre=v/s[...,None]; rounded=np.rint(pre);codes=np.clip(rounded,-8,7).astype(np.int8)
 valid=np.zeros((o,p),bool);valid[:,:i]=True;valid=valid.reshape(v.shape)
 stats={'endpoint_rate':float(np.mean(np.abs(codes[valid])==7)),'minus8_rate':float(np.mean(codes[valid]==-8)),
 'pre_range_outside_rate':float(np.mean(((pre < -8)|(pre>7))[valid])), 'clipping_rate':float(np.mean(((rounded < -8)|(rounded>7))[valid]))}
 flat=(codes.reshape(-1).astype(np.int16)&15).astype(np.uint8)
 packed=flat[::2]|(flat[1::2]<<4)
 b=HEADER.pack(MAGIC,VERSION,o,i,group,p,s.size,packed.size)+s.tobytes()+packed.tobytes()
 return b,stats

def decode(b):
 if len(b)<HEADER.size:raise ValueError('short header')
 magic,version,o,i,g,p,ns,nc=HEADER.unpack_from(b)
 if magic!=MAGIC or version!=VERSION or g!=64 or not(0<i<=p and o>0) or p!=((i+63)//64)*64 or ns!=o*p//g or nc!=o*p//2:raise ValueError('header/shape/version')
 if len(b)!=HEADER.size+4*ns+nc:raise ValueError('payload length')
 s=np.frombuffer(b,dtype='<f4',count=ns,offset=HEADER.size).copy().reshape(o,p//g)
 if not np.isfinite(s).all() or (s<=0).any():raise ValueError('invalid scales')
 q=np.frombuffer(b,dtype=np.uint8,offset=HEADER.size+4*ns)
 n=np.empty(nc*2,dtype=np.int8);n[::2]=q&15;n[1::2]=q>>4;n[n>=8]-=16;n=n.reshape(o,p)
 if i<p and np.any(n[:,i:]!=0):raise ValueError('nonzero padding')
 w=(n.reshape(o,p//g,g).astype(np.float32)*s[...,None]).reshape(o,p)[:,:i].copy()
 return w,n[:,:i].copy(),s

def write_reload(path,w):
 p=Path(path)
 if p.exists():raise FileExistsError(p)
 b,stats=encode(w);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix('.partial');tmp.write_bytes(b);tmp.replace(p)
 a,q,s=decode(p.read_bytes())
 return a,q,s,dict(stats,artifact_bytes=len(b),header_bytes=HEADER.size,scale_bytes=s.nbytes,code_bytes=len(b)-HEADER.size-s.nbytes)
if __name__=='__main__':
 import argparse,json
 a=argparse.ArgumentParser();a.add_argument('artifact');a.add_argument('--output',required=True);x=a.parse_args()
 w,q,s=decode(Path(x.artifact).read_bytes());np.save(x.output,w,allow_pickle=False)
