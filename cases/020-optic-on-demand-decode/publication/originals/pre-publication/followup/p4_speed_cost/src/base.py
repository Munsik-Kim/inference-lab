"""P4 paths, unchanged-format fixture/reference and thin method wrappers. Torch is lazy."""
import pathlib,json,hashlib,sys,itertools,statistics,math
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
CASE=ROOT.parents[1]
PRIVATE=pathlib.Path('/home/mskim/Codex_main/case020-p4-private-20261026-1451')
UPSTREAM='344c712aeb6620e39eb78930fb922447f0e123f3'
def read(p):return json.loads(pathlib.Path(p).read_text())
def dump(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(b):return hashlib.sha256(b).hexdigest()
def info(p):b=pathlib.Path(p).read_bytes();return {'bytes':len(b),'sha256':sha(b)}
def stats(a):
 x=np.array(a,dtype='float64');return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'p95':float(np.quantile(x,.95,method='linear')),'min':float(x.min()),'q25':float(np.quantile(x,.25,method='linear')),'q75':float(np.quantile(x,.75,method='linear')),'IQR':float(np.quantile(x,.75,method='linear')-np.quantile(x,.25,method='linear'))}
def upstream():
 sys.path.insert(0,str(CASE/'vendor/optic/src'))
 from ple_eval.compact_adapter import load_compact_ple
 return load_compact_ple
def artifact(w):return CASE/'inputs/LARGE.npz' if w==32 else ROOT/f'fixtures/Z{w}.npz'
def arrays(path):
 with np.load(path,allow_pickle=False) as z:return json.loads(z['metadata'].item()),{k:z[k].copy() for k in z.files if k!='metadata'}
def fixture(w):
 G=w//4;r=np.random.default_rng(np.random.SeedSequence([61026020,4,w]));L,H,V=16,256,32768
 meta={'compressed_ple_format_id':'optic_npz','ple_shape':[V,L*H],'latent_mode':'discrete','d_ple':H,'latent_dim':w,'n_groups':G,'group_dim':4,'codebook_size':256,'code_bits':8}
 aa={'token_codes_packed_uint8':r.integers(256,size=V*G,dtype=np.uint8),'codebooks_fp16':r.standard_normal((G,256,4)).astype(np.float16),'decoder_weight_fp16':(r.standard_normal((L,H,w))*.15+np.arange(L)[:,None,None]*.01).astype(np.float16),'decoder_bias_fp16':(r.standard_normal((L,H))*.2+np.arange(L)[:,None]*.1+.07).astype(np.float16)}
 return meta,aa
def ref_chunks(meta,a,ids,chunk=128):
 """FP64 on stored arrays. No [M,L,H,Z] broadcast and no Torch/upstream calls."""
 G=meta['n_groups'];Z=meta['latent_dim'];codes=a['token_codes_packed_uint8'].reshape(meta['ple_shape'][0],G);cb=a['codebooks_fp16'].astype(np.float64);W=a['decoder_weight_fp16'].reshape(-1,Z).astype(np.float64);bias=a['decoder_bias_fp16'].astype(np.float64)
 for start in range(0,len(ids),chunk):
  part=ids[start:start+chunk];latent=np.concatenate([cb[g,codes[part,g]] for g in range(G)],axis=1);yield start,(latent@W.T).reshape(len(part),16,256)+bias
def numerical_stats(x,ref):
 d=np.asarray(x,dtype=np.float64)-ref;r=np.asarray(ref,dtype=np.float64);dims=(0,2)
 es=(d*d).sum(axis=dims);rs=(r*r).sum(axis=dims);ma=np.max(np.abs(d),axis=dims);finite=np.isfinite(x).all(axis=dims)
 return {'error_sumsq':es.tolist(),'reference_sumsq':rs.tolist(),'max_abs':ma.tolist(),'finite':finite.tolist(),'count_per_layer':int(x.shape[0]*x.shape[2])}
def merge_errors(parts):
 es=np.sum([p['error_sumsq'] for p in parts],axis=0);rs=np.sum([p['reference_sumsq'] for p in parts],axis=0);ma=np.max([p['max_abs'] for p in parts],axis=0);fi=np.all([p['finite'] for p in parts],axis=0);count=sum(p['count_per_layer'] for p in parts);rel=np.sqrt(es)/np.maximum(np.sqrt(rs),1e-12);overall=float(np.sqrt(es.sum())/max(np.sqrt(rs.sum()),1e-12));passed=bool(fi.all() and all(ma[i]<=1e-6 if rs[i]==0 else rel[i]<=1e-2 for i in range(16)) and (ma.max()<=1e-6 if rs.sum()==0 else overall<=1e-2))
 return {'error_sumsq':es.tolist(),'reference_sumsq':rs.tolist(),'max_abs':ma.tolist(),'finite':fi.tolist(),'count_per_layer':count,'relative_L2_layers':rel.tolist(),'relative_L2':overall,'rmse_layers':np.sqrt(es/count).tolist(),'pass':passed}
def methods(torch,decoder,table,ids,names):
 def A():return table.index_select(0,ids.reshape(-1))
 def B():return decoder(ids)
 def C():
  u,inv=torch.unique(ids.reshape(-1),sorted=True,return_inverse=True);return decoder(u).index_select(0,inv)
 def S():
  sorted_ids,permutation=torch.sort(ids.reshape(-1));inverse=torch.argsort(permutation);return decoder(sorted_ids).index_select(0,inverse)
 return {k:{'A':A,'B':B,'C':C,'S':S}[k] for k in names}
def inventory(named):
 rows=[];seen=set()
 for n,t in named:
  st=t.untyped_storage();key=(str(t.device),st.data_ptr(),st.nbytes());rows.append({'name':n,'shape':list(t.shape),'dtype':str(t.dtype),'numel':t.numel(),'element_size':t.element_size(),'payload_bytes':t.numel()*t.element_size(),'storage_bytes':st.nbytes(),'key':list(key),'unique':key not in seen});seen.add(key)
 return rows
def module_inventory(m):return inventory(list(m.named_parameters())+list(m.named_buffers()))
def validate_ids(ids):
 if ids.dtype!=np.int64:raise TypeError('int64 IDs required')
 if ids.size and (ids.min()<0 or ids.max()>=32768):raise IndexError('IDs outside [0,V)')
def balanced_orders(names,n,seed):
 r=np.random.default_rng(np.random.SeedSequence(seed));perms=list(itertools.permutations(names));r.shuffle(perms);offset=int(r.integers(len(perms)));perms=perms[offset:]+perms[:offset];return [list(perms[i%len(perms)]) for i in range(n)]
def source_identity():
 f=read(ROOT/'provenance/freeze.json')
 for rel,v in f['files'].items():
  if info(ROOT/rel)!=v:raise ValueError('SOURCE_FREEZE_MISMATCH: '+rel)
 return f['bundle_sha256']
