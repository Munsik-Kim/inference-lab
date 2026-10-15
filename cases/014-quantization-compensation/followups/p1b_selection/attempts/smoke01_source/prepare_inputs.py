"""Bounded streaming of existing Arrow caches. No download or model forward."""
import argparse
import json
import re
import time
from pathlib import Path
import numpy as np
import pyarrow as pa
from transformers import AutoTokenizer
from p1b_common import save, sha, digest, utc

def canonical(url):
    m = re.search(r'/ebooks/(\d+)', url)
    if not m:
        raise ValueError('PG19 canonical ebook ID missing')
    return 'gutenberg-ebook-' + str(int(m.group(1)))

def batches(path):
    with pa.memory_map(str(path), 'r') as f:
        yield from pa.ipc.open_stream(f)

def main():
    ap=argparse.ArgumentParser()
    for k in ['model','pg19','pile','old-inputs','private','output']:
        ap.add_argument('--'+k, required=True)
    a=ap.parse_args(); start=time.monotonic()
    private=Path(a.private); out=Path(a.output)
    old=json.loads(Path(a.old_inputs).read_text())
    tok=AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    oldids={canonical(d['source_id']) for d in old if d['source'].startswith('PG')}
    contents={d['content_sha256'] for d in old}; tokens={d['token_hash'] for d in old}
    def norm(ids):return re.sub(r'\W+', ' ', tok.decode(ids).lower()).strip()
    normal={digest(norm(d['input_ids']).encode()) for d in old}
    gramsets=[set(tuple(d['input_ids'][i:i+5]) for i in range(252)) for d in old]
    metadata=[]; exclusions=[]; near_max=0.; comparisons=0
    # First shard only is the registered finite candidate pool, not the entire train corpus.
    idx=0
    for b in batches(a.pg19):
        for url in b.column('url').to_pylist():
            cid=canonical(url)
            if cid not in oldids:
                order=digest(f'case014-p1b-v1|pg19|{cid}|6101517'.encode())
                metadata.append((order,idx,cid))
            else:exclusions.append({'corpus':'pg19','row':idx,'reason':'previous canonical book ID'})
            idx+=1
    metadata.sort(); ranks={row:(order,cid,rank) for rank,(order,row,cid) in enumerate(metadata)}
    # Read each selected text at most once from mmap. Hash order is applied before tokenization.
    # Metadata-only first pass determines the first 160 candidates; no model-based filtering.
    candidates=set(row for _,row,_ in metadata[:160]); text_rows={}; idx=0
    for b in batches(a.pg19):
        for j in range(b.num_rows):
            if idx in candidates:text_rows[idx]=b.column('text')[j].as_py()
            idx+=1
    accepted=[]
    def window(text, corpus, cid, row, order, source_split):
        nonlocal near_max, comparisons
        h=digest(text.encode())
        if h in contents:
            exclusions.append({'corpus':corpus,'row':row,'reason':'full content duplicate'});return None
        ids=tok(text,add_special_tokens=False).input_ids
        if len(ids)<512:
            exclusions.append({'corpus':corpus,'row':row,'reason':'under512 tokens'});return None
        offset=256+int(order[:8],16)%(len(ids)-511); w=ids[offset:offset+256]
        th=digest(np.array(w,dtype='<i8').tobytes());nh=digest(norm(w).encode())
        g=set(tuple(w[j:j+5]) for j in range(252))
        sims=[len(g&s)/max(1,len(g|s)) for s in gramsets]
        comparisons+=len(sims);mx=max(sims,default=0);near_max=max(near_max,mx)
        if th in tokens or nh in normal or mx>=.8:
            exclusions.append({'corpus':corpus,'row':row,'reason':'window exact/normalized/5gram duplicate','maximum_jaccard':mx});return None
        contents.add(h);tokens.add(th);normal.add(nh);gramsets.append(g)
        return {'document_id_hash':digest((corpus+':'+cid).encode()),'source':'PG19 book' if corpus=='pg19' else 'Pile ArXiv content group',
                'source_revision':'c021754c8e01c5b1cc83a1f549c1f97fbbb756b8' if corpus=='pg19' else '127bfedcd5047750df5ccf3a12979a47bfa0bafa',
                'source_split':source_split,'row':row,'canonical_id_hash':digest(cid.encode()),'content_sha256':h,'offset':offset,
                'full_document_tokens':len(ids),'token_count':256,'token_hash':th,'max_window_5gram_jaccard_with_prior_accepted':mx,
                'prior_use':'Excluded union of confirmed Case014 P0/P1 inputs; bounded repo cases/research/configs scan found no other PG19/Pile use outside Case014. Other work and pretraining exposure UNKNOWN.',
                'license':'PG19 dataset card Apache-2.0; original book rights remain' if corpus=='pg19' else 'Pile mirror bigscience-bloom-rail-1.0; original article terms vary; no raw text redistribution',
                'group_basis':'canonical Gutenberg ebook ID and full content hash' if corpus=='pg19' else 'full content hash; article ID unavailable',
                'input_ids':w,'source_id':cid}
    for order,row,cid in metadata[:160]:
        r=window(text_rows.pop(row), 'pg19',cid,row,order,'train/shard00000-of00019')
        if r:accepted.append(r)
        if len(accepted)==96:break
    text_rows.clear()
    if len(accepted)!=96:raise ValueError(f'PG19 insufficient eligible candidates: {len(accepted)}/96; no model outputs opened')
    perm=np.random.default_rng(6101517).permutation(96)
    for i,j in enumerate(perm):accepted[j]['split']='SELECT-ID' if i<32 else 'FRESH-EVAL-ID'
    pcs=[];idx=0
    for b in batches(a.pile):
        meta=b.column('meta');text=b.column('text')
        for j in range(b.num_rows):
            if meta[j].as_py()['pile_set_name']=='ArXiv':
                t=text[j].as_py();cid=digest(t.encode())
                if cid not in contents:pcs.append((digest(f'case014-p1b-v1|pile|{cid}|6101517'.encode()),idx,cid,t))
            idx+=1
    for order,row,cid,text in sorted(pcs):
        r=window(text,'pile',cid,row,order,'train')
        if r:r['split']='FRESH-SHIFT';accepted.append(r)
        if sum(d['split']=='FRESH-SHIFT' for d in accepted)==16:break
    assert len(accepted)==112,'fresh SHIFT insufficient; register source amendment before GPU'
    accepted.sort(key=lambda r:(r['split'],r['document_id_hash']))
    smoke=next(d for d in old if d['split']=='SMOKE').copy();smoke['split']='REUSED_SMOKE'
    save(private/'inputs/documents.private.json',accepted+[smoke])
    public=[{k:v for k,v in r.items() if k not in ['input_ids','source_id']} for r in accepted]
    manifest={'frozen_utc':utc(),'documents':public,'counts':{s:sum(r['split']==s for r in accepted) for s in ['SELECT-ID','FRESH-EVAL-ID','FRESH-SHIFT']},
              'smoke':{k:v for k,v in smoke.items() if k not in ['input_ids','source_id']},'old_input_file_sha256':sha(a.old_inputs),
              'source_files':{Path(p).name:{'bytes':Path(p).stat().st_size,'sha256':sha(p)} for p in [a.pg19,a.pile]},
              'selection_algorithm':'first cached PG19 train shard metadata hash sort: sha256(case014-p1b-v1|pg19|canonical_id|6101517), first96 eligible of first160 ranked candidates; seed6101517 permutation assigns32 SELECT /64 EVAL from same pool; Pile same namespace/order, first16 eligible unused ArXiv groups',
              'PG19_candidate_metadata_count':len(metadata),'PG19_pool_total_rows':len(metadata)+sum(e['corpus']=='pg19' for e in exclusions if e['reason']=='previous canonical book ID'),
              'old_excluded_documents':len(old),'old_excluded_PG19_books':len(oldids),'exclusions':exclusions,
              'near_duplicate_check':{'rule':'exact canonical ID/full content/window token hash/normalized decoded window and 5gram token Jaccard >=0.8; old97 vs new112 and among new112', 'comparisons':comparisons,'maximum_observed_jaccard':near_max,'full_source_near_duplicate_search':False},
              'private_tokens_sha256':sha(private/'inputs/documents.private.json'),'new_transfer_bytes':0,'data_prepare_seconds':time.monotonic()-start,
              'limits':'PG19 train vs old test source split; no strict identical-distribution proof. Pile content groups, not verified articles. SELECT is development data, not final performance sample.'}
    save(out/'input_manifest.json',manifest)
    print(json.dumps({'status':'INPUTS_FROZEN','counts':manifest['counts'],'private_tokens_sha256':manifest['private_tokens_sha256'],'seconds':manifest['data_prepare_seconds']}),flush=True)

if __name__=='__main__':main()
