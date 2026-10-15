"""Fresh P2 books/content groups; reused original CAL32. No model forward."""
import argparse,json,re,time
from pathlib import Path
import numpy as np
import pyarrow as pa
from transformers import AutoTokenizer
from p2_common import save,sha,digest,utc

def canonical(d):
    v=d['source_id']
    if not d['source'].startswith('PG'):return v
    m=re.search(r'/ebooks/(\d+)',v)
    return 'gutenberg-ebook-'+str(int(m.group(1))) if m else v
def batches(path):
    with pa.memory_map(str(path),'r') as f:yield from pa.ipc.open_stream(f)
def main():
    ap=argparse.ArgumentParser()
    for k in ['case','private','pg19','pile']:ap.add_argument('--'+k,required=True)
    a=ap.parse_args();start=time.monotonic();case=Path(a.case);pr=Path(a.private);out=case/'results/p2_r1'
    spec_path=case/'followups/p2_core/specs/p2.json';spec=json.loads(spec_path.read_text());identity=json.loads((pr/'identity.private.json').read_text())
    old0=json.loads(Path(identity['CAL_inputs']).read_text());old1=json.loads(Path(identity['P1b_inputs']).read_text());unique={}
    for d in old0+old1:unique[(d['source'].startswith('PG'),canonical(d))]=d
    prior=list(unique.values());oldids={canonical(d) for d in prior};contents={d['content_sha256'] for d in prior};tokens={d['token_hash'] for d in prior}
    tok=AutoTokenizer.from_pretrained(identity['snapshot']['model_path'],local_files_only=True)
    norm=lambda ids:re.sub(r'\W+',' ',tok.decode(ids).lower()).strip()
    normal={digest(norm(d['input_ids']).encode()) for d in prior}
    grams=[set(tuple(d['input_ids'][j:j+5]) for j in range(252)) for d in prior];comparisons=0;maximum=0.;excluded=[]
    def make(text,cid,corpus,row,order):
        nonlocal comparisons,maximum
        h=digest(text.encode())
        if cid in oldids or h in contents:return None
        ids=tok(text,add_special_tokens=False).input_ids
        if len(ids)<512:excluded.append({'corpus':corpus,'row':row,'reason':'under512 tokens'});return None
        offset=256+int(order[:8],16)%(len(ids)-511);w=ids[offset:offset+256];th=digest(np.array(w,dtype='<i8').tobytes());nh=digest(norm(w).encode())
        g=set(tuple(w[j:j+5]) for j in range(252));sim=[len(g&s)/max(1,len(g|s)) for s in grams];mx=max(sim,default=0);maximum=max(maximum,mx);comparisons+=len(sim)
        if th in tokens or nh in normal or mx>=.8:excluded.append({'corpus':corpus,'row':row,'reason':'duplicate window','max_Jaccard':mx});return None
        oldids.add(cid);contents.add(h);tokens.add(th);normal.add(nh);grams.append(g)
        return {'split':'EVAL-ID' if corpus=='pg19' else 'SHIFT','document_id_hash':digest(('p2|'+corpus+'|'+cid).encode()),'source':'PG19 book' if corpus=='pg19' else 'Pile ArXiv content group',
                'source_revision':'c021754c8e01c5b1cc83a1f549c1f97fbbb756b8' if corpus=='pg19' else '127bfedcd5047750df5ccf3a12979a47bfa0bafa','source_split':'train/shard00000-of00019' if corpus=='pg19' else 'train',
                'row':row,'source_id':cid,'canonical_id_hash':digest(cid.encode()),'content_sha256':h,'offset':offset,'full_document_tokens':len(ids),'token_hash':th,'token_count':256,'input_ids':w,
                'group_basis':'canonical ebook ID + content hash' if corpus=='pg19' else 'full content hash; original article ID unavailable','max_window_5gram_Jaccard':mx,
                'license':'PG19 dataset card Apache-2.0; original book rights remain' if corpus=='pg19' else 'Pile mirror bigscience-bloom-rail-1.0; article terms vary; no raw export',
                'prior_use':'Excluded all confirmed P0/P1/P1b209 unique inputs. Other project history/pretraining exposure UNKNOWN.'}
    meta=[];idx=0
    for b in batches(a.pg19):
        for url in b.column('url').to_pylist():
            cid='gutenberg-ebook-'+str(int(re.search(r'/ebooks/(\d+)',url).group(1)))
            if cid not in oldids:meta.append((digest(f'case014-p2-v1|pg19|{cid}|6101521'.encode()),idx,cid))
            idx+=1
    meta.sort();needed={row for _,row,_ in meta[:128]};texts={};idx=0
    for b in batches(a.pg19):
        for j in range(b.num_rows):
            if idx in needed:texts[idx]=b.column('text')[j].as_py()
            idx+=1
    fresh=[]
    for order,row,cid in meta[:128]:
        d=make(texts.pop(row),cid,'pg19',row,order)
        if d:fresh.append(d)
        if len(fresh)==64:break
    texts.clear()
    if len(fresh)<32:raise ValueError(f'Insufficient new PG19: {len(fresh)}/32; no P2 output opened')
    if len(fresh)<64:fresh=fresh[:32]
    pile=[];idx=0;arxiv_count=0
    for b in batches(a.pile):
        for j in range(b.num_rows):
            if b.column('meta')[j].as_py()['pile_set_name']=='ArXiv':
                arxiv_count+=1;text=b.column('text')[j].as_py();cid=digest(text.encode())
                if cid not in contents:pile.append((digest(f'case014-p2-v1|pile|{cid}|6101521'.encode()),idx,cid,text))
            idx+=1
    shift=[]
    for order,row,cid,text in sorted(pile):
        d=make(text,cid,'pile',row,order)
        if d:shift.append(d)
        if len(shift)==16:break
    if len(shift)<16:shift=[]
    fresh+=shift
    cal=[d.copy() for d in old0 if d['split']=='CAL-FIT'];smoke=next(d.copy() for d in old0 if d['split']=='SMOKE');assert len(cal)==32
    fresh.sort(key=lambda d:(d['split'],d['document_id_hash']));save(pr/'inputs/documents.private.json',cal+[smoke]+fresh)
    public=lambda d:{k:v for k,v in d.items() if k not in ['input_ids','source_id']}
    counts=dict(EVAL_ID=sum(d['split']=='EVAL-ID' for d in fresh),SHIFT=len(shift))
    save(out/'input_manifest.json',{'frozen_utc':utc(),'documents':[public(d) for d in cal+[smoke]+fresh],'counts':{'CAL-FIT':32,'SMOKE':1,'EVAL-ID':counts['EVAL_ID'],'SHIFT':counts['SHIFT']},
         'original_CAL_token_file_sha256':sha(identity['CAL_inputs']),'P1b_token_file_sha256':sha(identity['P1b_inputs']),'private_tokens_sha256':sha(pr/'inputs/documents.private.json'),
         'prior_unique_inputs_excluded':len(prior),'prior_PG19_books':sum(d['source'].startswith('PG') for d in prior),'prior_ArXiv_groups':sum(not d['source'].startswith('PG') for d in prior),
         'selection':'registered seed6101521 namespace hash order; first64 eligible PG19 of first128 ranked within cached train shard0; first16 eligible unused ArXiv groups; no model score filter',
         'sources':{Path(p).name:{'bytes':Path(p).stat().st_size,'sha256':sha(p)} for p in [a.pg19,a.pile]},'PG19_metadata_pool':len(meta),'ArXiv_cached_rows':arxiv_count,
         'near_duplicate_check':{'comparisons':comparisons,'maximum_Jaccard':maximum,'threshold':.8,'scope':'prior209 vs new80 and among new80; token256 windows; full-source near duplicates unverified'},
         'excluded':excluded,'new_transfer_bytes':0,'pretraining_exposure':'UNKNOWN','article_disjointness':'PG19 canonical books; Pile content-group only','data_prepare_seconds':time.monotonic()-start})
    if counts!={'EVAL_ID':64,'SHIFT':16}:save(out/'availability_amendment.json',{'before_any_P2_model_output':True,'registered_counts':{'EVAL-ID':counts['EVAL_ID'],'SHIFT':len(shift)},'reason':'source/length/duplicate availability only','SHIFT_status':'NOT_RUN_SOURCE' if not shift else 'REGISTERED'})
    print(json.dumps({'status':'INPUTS_FROZEN','fresh_counts':counts,'excluded_prior':len(prior),'cached_ArXiv_rows':arxiv_count,'data_prepare_seconds':time.monotonic()-start}),flush=True)
if __name__=='__main__':main()
