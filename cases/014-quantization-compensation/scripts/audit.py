"""Independent stdlib auditor. Rebuild scalar terms and aggregates, never trust summaries.
Does not validate hidden model matrices/logits. P0 exact is checked with Fraction.
"""
import argparse,json,math,hashlib
from pathlib import Path
from fractions import Fraction as F

def close(a,b,scale=1,tol=1e-10):
 if abs(a-b)>tol*max(scale,abs(a),abs(b),1):raise AssertionError((a,b,scale))
def audit(root):
 root=Path(root)
 fixture=Path(__file__).resolve().parents[1]/'tests/fixtures/matrix_identity.json'
 if fixture.exists():
  f=json.loads(fixture.read_text());W,X,Z,Q,Q0=[f[k] for k in ['W','X','Z','What','What0']]
  def mul(a,b):return [[math.fsum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
  def sub(a,b):return [[x-y for x,y in zip(ra,rb)] for ra,rb in zip(a,b)]
  def ss(a):return math.fsum(x*x for row in a for x in row)
  teacher=mul(W,X);SSE=ss(sub(teacher,mul(Q,Z)));SSE0=ss(sub(teacher,mul(Q0,Z)))
  close(SSE,f['scalars']['SSE_after']);close(SSE0,f['scalars']['SSE_before']);close(SSE-SSE0,f['scalars']['Delta'])
  # Independent two's-complement decode of the saved fixture; no codec import.
  import struct
  b=bytes.fromhex(f['codec_hex']);magic,version,o,i,g,padded,ns,nc=struct.unpack_from('<8s7I',b);assert magic==b'C014INT4' and version==1
  scales=struct.unpack_from('<'+'f'*ns,b,36);raw=b[36+4*ns:];codes=[((v>>(4*k))&15) for v in raw for k in [0,1]];codes=[q-16 if q>=8 else q for q in codes]
  decoded=[[codes[row*padded+col]*scales[row*(padded//64)+col//64] for col in range(i)] for row in range(o)]
  for a,brow in zip(decoded,Q):
   for a,bv in zip(a,brow):close(a,bv,tol=1e-7)
 p0=json.loads((root/('p0_full.json' if (root/'p0_full.json').exists() else 'p0.json')).read_text());f=p0['exact']['finite_codebook'];assert F(f['teacher'])==F(377,125);assert F(f['BASE_abs_error'])==F(2,125);assert F(f['COMP_abs_error'])==F(123,125)
 for r in p0['exact']['fixed_errors']:
  eps=F(r['epsilon']);assert F(r['COMP_abs_error'])==eps+eps**2
 assert len(p0['dither']['rows'])==18
 for r in p0['dither']['rows']:
  assert r['n_seeds']==2000;assert not r['failures'];assert r['status']=='VALID'
  if 'seed_SSE' in r:
   vals=r['seed_SSE'];close(math.fsum(vals)/len(vals),r['mean_SSE'])
   base=next(q['seed_SSE'] for q in p0['dither']['rows'] if q['depth']==r['depth'] and q['step']==r['step'] and q['alpha']==0)
   ds=[a-b for a,b in zip(vals,base)];mean=math.fsum(ds)/len(ds);se=(math.fsum((d-mean)**2 for d in ds)/(len(ds)-1)/len(ds))**.5;close(mean,r['paired_delta']);close(se,r['SE'])
 p1=root/'p1';files=sorted((p1/'cells').glob('*.jsonl'));groups={};seen=set();paired={}
 for file in files:
  rows=[json.loads(l) for l in file.read_text().splitlines()];assert len(rows)==96
  for r in rows:
   key=(r['block'],r['projection'],r['alpha'],r['split'],r['document_id_hash']);assert key not in seen;seen.add(key)
   inherited=r['C_norm2']-2*r['dot_A_C'];cancellation=2*r['dot_B0_C'];cont=r['C_norm2']-2*r['dot_R0_C'];req=r['deltaB_norm2']-2*r['dot_R0minusC_deltaB']
   delta=r['SSE_after']-r['SSE_before'];scale=max(1,abs(r['SSE_after']),abs(r['SSE_before']))
   for name,v in [('T_inherited',inherited),('T_cancellation',cancellation),('T_cont',cont),('T_requant',req),('Delta',delta)]:close(r[name],v,scale)
   close(delta,inherited+cancellation+req,scale);close(cont,inherited+cancellation,scale);assert r['valid_token_count']==255;assert r['KL_position_count']==16;assert r['input_identical_across_alpha']
   assert 0<=r['flip_count']<=255 and r['sampled_full_vocab_KL_sum']>=-1e-3
   g=groups.setdefault((r['block'],r['projection'],r['split'],r['alpha']),[]);g.append(r)
   pk=(r['block'],r['projection'],r['split'],r['document_id_hash']);paired.setdefault(pk,{})[r['alpha']]=r
 if files:
  assert len(files)==30
  for rs in paired.values():
   assert sorted(rs)==[0,.25,.5,.75,1];assert len({r['token_hash'] for r in rs.values()})==1;assert len({r['artifact_bytes'] for r in rs.values()})==1
  assert all(len(v)=={'CAL-FIT':32,'DEV-SELECT':16,'EVAL-ID':32,'SHIFT':16}[k[2]] for k,v in groups.items())
  selected=json.loads((p1/'fit_frozen.json').read_text())['selected']
  for sel in selected:
   block=int(sel['target'].split('.')[2]);proj=sel['target'].split('.')[-1];totals={a:math.fsum(r['SSE_after'] for r in groups[(block,proj,'CAL-FIT',a)]) for a in [0,.25,.5,.75,1]}
   assert min(totals,key=lambda a:(totals[a],a))==sel['selected_alpha']
 return {'P0':'PASS','P1':'PASS' if files else 'NOT_RUN','grid_rows':len(seen),'grid_cells':len(files),'paired_document_target_groups':len(paired),'checked':'raw scalar identities, CAL selector, pairing, counts, units, artifact byte equality; no weights/X/Z/logit re-evaluation','independent_auditor':'stdlib math.fsum/Fraction; does not import fit/decompose/runner','groups':{str(k):{'documents':len(v),'NLL_sum':math.fsum(r['NLL_sum'] for r in v),'tokens':sum(r['valid_token_count'] for r in v),'PPL':math.exp(math.fsum(r['NLL_sum'] for r in v)/sum(r['valid_token_count'] for r in v)),'SSE':math.fsum(r['SSE_after'] for r in v)} for k,v in groups.items()}}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);a=p.parse_args();v=audit(a.results);Path(a.output).write_text(json.dumps(v,indent=2));print({k:v[k] for k in v if k!='groups'})
