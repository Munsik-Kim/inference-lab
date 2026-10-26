"""Fixture/wrapper utilities; upstream codec and GEMM are unchanged."""
import hashlib, json, pathlib, sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
UPSTREAM='344c712aeb6620e39eb78930fb922447f0e123f3'
METHODS=['A','B','C']
SHAPES={'SMALL':dict(V=8192,L=8,H=128,Z=32,G=8,D=4),'LARGE':dict(V=32768,L=16,H=256,Z=32,G=8,D=4)}
def sha(b):return hashlib.sha256(b).hexdigest()
def file_info(p):
 p=pathlib.Path(p); b=p.read_bytes();return {'bytes':len(b),'sha256':sha(b)}
def dump(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):return json.loads(pathlib.Path(p).read_text())
def upstream():
 sys.path.insert(0,str(ROOT/'vendor/optic/src'))
 from ple_eval.compact_adapter import load_compact_ple,compact_resident_bytes
 return load_compact_ple,compact_resident_bytes
def fixture(s,seed):
 r=np.random.default_rng(seed);V,L,H,Z,G,D=[s[k] for k in ['V','L','H','Z','G','D']]
 meta={'compressed_ple_format_id':'optic_npz','ple_shape':[V,L*H],'latent_mode':'discrete','d_ple':H,'latent_dim':Z,'n_groups':G,'group_dim':D,'codebook_size':256,'code_bits':8}
 a={'token_codes_packed_uint8':r.integers(256,size=V*G,dtype=np.uint8),'codebooks_fp16':r.standard_normal((G,256,D)).astype(np.float16),'decoder_weight_fp16':(r.standard_normal((L,H,Z))*.15+np.arange(L)[:,None,None]*.01).astype(np.float16),'decoder_bias_fp16':(r.standard_normal((L,H))*.2+np.arange(L)[:,None]*.1+.07).astype(np.float16)}
 return meta,a
def save_fixture(path,meta,a):np.savez(path,metadata=np.asarray(json.dumps(meta)),**a)
def read_fixture(path):
 with np.load(path,allow_pickle=False) as z:return json.loads(z['metadata'].item()),{k:z[k].copy() for k in z.files if k!='metadata'}
def array_inventory(a):return {k:{'shape':list(v.shape),'dtype':str(v.dtype),'bytes':v.nbytes,'sha256':sha(v.tobytes())} for k,v in a.items()}
def validate_ids(ids,V):
 ids=np.asarray(ids)
 if ids.dtype!=np.int64:raise TypeError('common adapter requires CPU-generated int64 IDs')
 if ids.size and (ids.min()<0 or ids.max()>=V):raise IndexError('token ID outside [0,V)')
 return ids
def reference(meta,a,ids,loop=False):
 """Independent FP64 affine formula over STORED FP16 values; no Torch imports."""
 ids=np.asarray(ids);V=meta['ple_shape'][0];G=meta['n_groups'];D=meta['group_dim'];L,H,Z=a['decoder_weight_fp16'].shape
 codes=a['token_codes_packed_uint8'].reshape(V,G); cb=a['codebooks_fp16'].astype(np.float64);w=a['decoder_weight_fp16'].astype(np.float64);b=a['decoder_bias_fp16'].astype(np.float64)
 if loop:
  out=np.empty((ids.size,L,H),dtype=np.float64)
  for i,t in enumerate(ids.reshape(-1)):
   z=np.empty(Z,dtype=np.float64)
   for g in range(G):
    for d in range(D):z[g*D+d]=cb[g,int(codes[t,g]),d]
   for l in range(L):
    for h in range(H):
     value=float(b[l,h])
     for j in range(Z):value+=float(z[j])*float(w[l,h,j])
     out[i,l,h]=value
 else:
  latent=cb[np.arange(G)[None,:],codes[ids.reshape(-1)]].reshape(ids.size,Z)
  out=np.einsum('mz,lhz->mlh',latent,w,optimize=False)+b
 return out.reshape(*ids.shape,L,H)
def metrics(out,ref):
 x=np.asarray(out,dtype=np.float64);r=np.asarray(ref,dtype=np.float64)
 if x.shape!=r.shape:return {'pass':False,'shape':list(x.shape),'expected':list(r.shape)}
 def m(x,r):
  d=x-r;norm=float(np.linalg.norm(r));ma=float(np.max(np.abs(d),initial=0));rel=float(np.linalg.norm(d)/max(norm,1e-12));finite=bool(np.isfinite(x).all())
  return {'relative_L2':rel,'max_abs':ma,'rmse':float(np.sqrt(np.mean(d*d))) if d.size else 0.,'zero_reference':norm==0,'finite':finite,'pass':finite and (ma<=1e-6 if norm==0 else rel<=1e-2)}
 all_=m(x,r);all_['layers']=[m(x[...,l,:],r[...,l,:]) for l in range(r.shape[-2])];all_['pass']=all_['pass'] and all(z['pass'] for z in all_['layers']);return all_
def functions(torch,decoder,table=None):
 def A(ids):return table.index_select(0,ids.reshape(-1)).reshape(*ids.shape,decoder.layers*decoder.d_ple)
 def B(ids):return decoder(ids)
 def C(ids):
  u,inv=torch.unique(ids.reshape(-1),sorted=True,return_inverse=True)
  return decoder(u).index_select(0,inv).reshape(*ids.shape,decoder.layers*decoder.d_ple)
 return {'A':A,'B':B,'C':C}
def tensor_inventory(named):
 seen=set();rows=[]
 for name,t,persistent in named:
  st=t.untyped_storage();key=(str(t.device),st.data_ptr(),st.nbytes())
  rows.append({'name':name,'dtype':str(t.dtype),'shape':list(t.shape),'numel':t.numel(),'element_size':t.element_size(),'payload_bytes':t.numel()*t.element_size(),'storage_bytes':st.nbytes(),'storage_key':list(key),'unique_storage':key not in seen,'device':str(t.device),'persistent':persistent});seen.add(key)
 return rows
def module_tensors(m):
 return [(n,t,True) for n,t in m.named_parameters()]+[(n,t,n not in m._non_persistent_buffers_set) for n,t in m.named_buffers()]
