"""Read existing Arrow caches only. No download; book IDs or content-group IDs.
Requires the existing dataset environment (pyarrow) and tokenizer, not GPU.
"""
import argparse,json,hashlib,re
from pathlib import Path
import numpy as np
import pyarrow as pa
from transformers import AutoTokenizer
from common import save,sha,digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--pg19',required=True);p.add_argument('--pile',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(exist_ok=True,parents=True)
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 pg=pa.ipc.open_stream(pa.memory_map(a.pg19,'r')).read_all().to_pylist();pile=pa.ipc.open_stream(pa.memory_map(a.pile,'r')).read_all().to_pylist()
 entries=[];seen=set();private=[]
 def candidates(rows,corpus):
  cs=[]
  for i,row in enumerate(rows):
   if corpus=='pile' and row['meta']['pile_set_name']!='ArXiv':continue
   text=row['text'];h=digest(text.encode());ident=row['url'] if corpus=='pg19' else h
   cs.append((digest(('6101515:'+corpus+':'+ident).encode()),i,h,ident,row))
  return sorted(cs)
 selected=[]
 for corpus,rows,counts in [('pg19',pg,[('CAL-FIT',32),('DEV-SELECT',16),('EVAL-ID',32),('SMOKE',1)]),('pile',pile,[('SHIFT',16)])]:
  cs=candidates(rows,corpus);at=0
  for split,count in counts:
   got=0
   while got<count and at<len(cs):
    order,index,h,ident,row=cs[at];at+=1
    if ident in seen or h in seen:continue
    # Common deterministic tokenization: no special tokens, no chat, no loss filter.
    ids=tok(row['text'],add_special_tokens=False).input_ids
    if len(ids)<512:continue
    offset=256+int(order[:8],16)%(len(ids)-511);window=ids[offset:offset+256]
    # Exact and normalized first-256-token duplicate checks; no text exported.
    nh=digest(re.sub(r'\W+',' ',tok.decode(window).lower()).encode())
    if nh in seen:continue
    seen.update([ident,h,nh]);got+=1
    item={'split':split,'document_id_hash':digest((corpus+':'+ident).encode()),'source':'PG-19 book' if corpus=='pg19' else 'Pile ArXiv text record','source_revision':'c021754c8e01c5b1cc83a1f549c1f97fbbb756b8' if corpus=='pg19' else '127bfedcd5047750df5ccf3a12979a47bfa0bafa','source_split':'test' if corpus=='pg19' else 'train','row':index,'offset':offset,'full_document_tokens':len(ids),'content_sha256':h,'token_hash':digest(np.array(window,dtype='<i8').tobytes()),'token_count':256,'group_basis':'book URL; one book per split' if corpus=='pg19' else 'full record content hash; original article ID unavailable','prior_use':'No PG19/Pile reference found in bounded current cases/configs audit; usage outside that audit UNKNOWN','license':'PG19 Apache-2.0 dataset card; original book rights not re-assigned' if corpus=='pg19' else 'mirror card bigscience-bloom-rail-1.0; source article terms vary; text not redistributed'}
    entries.append(item);private.append(dict(item,input_ids=window,source_id=ident))
   if got!=count:raise ValueError(f'insufficient {corpus} {split}: {got}/{count}')
 # Limited near-duplicate check of selected window token 5-gram sets.
 grams=[set(tuple(r['input_ids'][k:k+5]) for k in range(252)) for r in private];near=[]
 for i in range(len(grams)):
  for j in range(i):
   sim=len(grams[i]&grams[j])/max(1,len(grams[i]|grams[j]))
   if sim>=.8:near.append([i,j,sim])
 if near:raise ValueError('near duplicate selected windows: '+str(near))
 save(out/'documents.private.json',private)
 save(out/'input_manifest.json',{'documents':entries,'counts':{s:sum(x['split']==s for x in entries) for s in set(x['split'] for x in entries)},'selection':'seed6101515 hash of corpus and stable ID; length and duplicate filtering only','boundary_limit':'PG19 URLs identify books. Pile rows are content-grouped extracted records, not verified article-disjointness. Pretraining exposure UNKNOWN.','near_duplicate':'selected256-token 5gram Jaccard>=0.8 rejected; full-source near-duplicates not exhaustively checked','source_files':{Path(q).name:{'bytes':Path(q).stat().st_size,'sha256':sha(q)} for q in [a.pg19,a.pile]},'private_token_file_sha256':sha(out/'documents.private.json')})
 print(json.dumps({'counts':{s:sum(x['split']==s for x in entries) for s in set(x['split'] for x in entries)},'tokenizer':a.model,'status':'INPUTS_FROZEN'}))
if __name__=='__main__':main()
