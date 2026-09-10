"""Dataset/evidence/strict scorer checks; never executes a model."""
import argparse,datetime,hashlib,json,pathlib,re
from data import FIELDS,LABELS,messages,read_jsonl,dump
from score import self_test

def parse_value(field,text):
    if text=='미기재':return None
    if field in ['owner','task']:return text
    numbers=[int(n.replace(',','')) for n in re.findall(r'\d[\d,]*',text)]
    if field=='due_date':
        assert len(numbers)==3
        return datetime.date(*numbers).isoformat()
    if '만' in text:return numbers[0]*10000+(numbers[1] if len(numbers)>1 else 0)
    assert len(numbers)==1
    return numbers[0]

def verify(root,tokenizers=None):
    stats={'scorer':self_test(),'documents_checked':0,'rendered_gold_checks':0,'input_allowlist_checks':0,'tokenizer_pair_checks':0,'representative_documents_read_by':'Codex; no human review claimed'}
    facts={x['id']:x for x in read_jsonl(root/'data/facts.jsonl')};ids=set();scenarios=set();texts=set()
    manifest=json.loads((root/'data/manifest.json').read_text())
    for name,h in manifest['files_sha256'].items():assert hashlib.sha256((root/'data'/name).read_bytes()).hexdigest()==h,name
    rev={v:k for k,v in LABELS.items()}
    for split,n in [('dev',20),('eval',100)]:
        docs=read_jsonl(root/f'data/{split}_inputs.jsonl');golds={x['id']:x for x in read_jsonl(root/f'data/{split}_gold.jsonl')}
        assert len(docs)==n and len(golds)==n
        assert sum(d['bucket']=='short' for d in docs)==n//2
        for d in docs:
            assert d['id'] not in ids and facts[d['id']]['scenario_id'] not in scenarios and d['document'] not in texts
            ids.add(d['id']);scenarios.add(facts[d['id']]['scenario_id']);texts.add(d['document'])
            assert facts[d['id']]['target_active'] is True
            target_lines=[s for s in d['document'].splitlines() if s.startswith(d['target_id']+' [')]
            events=[]
            for line in target_lines:
                m=re.fullmatch(re.escape(d['target_id'])+r' \[(.+) v(\d+)\] (.+)',line);assert m,line
                status,version,fields=m.groups();patch={}
                for item in fields.split(' / '):
                    k,v=item.split(': ',1);field=rev[k];patch[field]=parse_value(field,v)
                events.append((int(version),status,patch,line))
            reconstructed={};quotes={}
            for version,status,patch,line in sorted(events):
                if status in ['접수·확정','확정 변경']:
                    reconstructed.update(patch);quotes.update({f:line for f in patch})
            assert reconstructed==golds[d['id']]['values'],d['id']
            assert quotes==golds[d['id']]['evidence'],d['id']
            assert set(reconstructed)==set(FIELDS)
            assert all(q in d['document'] for q in quotes.values())
            assert len(d['document'].splitlines())==len(set(d['document'].splitlines()))
            model_input=messages(d)
            injected={**d,'gold':'EVAL_GOLD_CANARY','evidence':'EVIDENCE_CANARY','scenario_metadata':'META_CANARY'}
            assert model_input==messages(injected)
            assert all(marker not in json.dumps(model_input) for marker in ['EVAL_GOLD_CANARY','EVIDENCE_CANARY','META_CANARY'])
            target=512 if d['bucket']=='short' else 4096
            assert target*.9<=d['input_tokens']<=target*1.1
            if tokenizers:
                encoded=[tok.apply_chat_template(model_input,add_generation_prompt=True,tokenize=True,return_dict=False) for tok in tokenizers]
                assert encoded[0]==encoded[1] and len(encoded[0])==d['input_tokens'],d['id']
                stats['tokenizer_pair_checks']+=1
            stats['documents_checked']+=1;stats['rendered_gold_checks']+=4;stats['input_allowlist_checks']+=1
    stats['passed']=True
    return stats

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]);p.add_argument('--model-paths',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path);a=p.parse_args()
    ts=None
    if a.model_paths:
        from transformers import AutoTokenizer
        paths=json.loads(a.model_paths.read_text());ts=[AutoTokenizer.from_pretrained(paths[k],local_files_only=True,trust_remote_code=False) for k in ['bf16','fp8']]
    result=verify(a.root,ts)
    if a.output:dump(a.output,result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
