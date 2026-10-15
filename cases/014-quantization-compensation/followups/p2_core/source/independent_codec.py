"""Independent read-only header/nibble decoder for P2 reload verification."""
import struct
import numpy as np

def read(path):
    b=path.read_bytes()
    if len(b)<36:raise ValueError('short header')
    magic,v,o,i,g,p,ns,nc=struct.unpack('<8s7I',b[:36])
    if magic!=b'C014INT4' or v!=1 or g!=64 or p!=((i+63)//64)*64 or i<1 or ns!=o*p//64 or nc!=o*p//2 or len(b)!=36+4*ns+nc:raise ValueError('header')
    s=np.frombuffer(b,dtype='<f4',count=ns,offset=36).copy().reshape(o,p//64)
    if not np.isfinite(s).all() or (s<=0).any():raise ValueError('scale')
    packed=np.frombuffer(b,dtype='u1',offset=36+4*ns)
    q=np.stack([packed&15,packed>>4],axis=1).reshape(o,p).astype('i2');q=np.where(q>=8,q-16,q).astype('i1')
    if np.any(q[:,i:]):raise ValueError('padding')
    w=(q.reshape(o,p//64,64).astype('f4')*s[...,None]).reshape(o,p)[:,:i].copy()
    return w,q[:,:i],s
