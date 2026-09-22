"""Freeze official task contexts and few-shot identities without model inference.

Uses a pinned installed lm-evaluation-harness and its task samplers. Dataset text
stays in the private cache; public manifests retain IDs and hashes only.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tokenizer',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from lm_eval.tasks import TaskManager, get_task_dict
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
    tasks=get_task_dict(['arc_challenge','wikitext','gsm8k','mmlu'],TaskManager())
    def walk(d):
        for k,v in d.items():
            if isinstance(v,dict):yield from walk(v)
            else:yield k,v
    def chat(history,add_generation_prompt=True):
        return tok.apply_chat_template(history,tokenize=False,add_generation_prompt=add_generation_prompt,continue_final_message=not add_generation_prompt,enable_thinking=None)
    output=[];summaries={}
    for name,t in sorted(walk(tasks)):
        if not hasattr(t,'dataset'):continue
        shot=5 if name=='gsm8k' or name.startswith('mmlu_') else 0
        t.set_config('num_fewshot',shot);t.set_fewshot_seed(909222)
        if name=='gsm8k':
            g=dict(t.config.generation_kwargs);g['max_gen_toks']=1024;t.set_config('generation_kwargs',g)
        fewshots=[]
        if shot:
            original=t.sampler.sample
            def sample(*args,_original=original,**kwargs):
                docs=_original(*args,**kwargs);fewshots.append([digest(d) for d in docs]);return docs
            t.sampler.sample=sample
        t.build_all_requests(apply_chat_template=name!='wikitext',fewshot_as_multiturn=False,chat_template=chat if name!='wikitext' else None,tokenizer_name='pinned-qwen4b')
        lengths=[]
        for inst in t.instances:
            args=inst.arguments;prompt=args[0]
            ids=tok.encode(prompt,add_special_tokens=False);lengths.append(len(ids))
            output.append({'task':name,'task_version':str(t.config.metadata.get('version')),'doc_id':inst.doc_id,'request_index':inst.idx,'request_type':inst.request_type,'document_hash':digest(inst.doc),'arguments_hash':digest(args),'context_token_hash':digest(ids),'context_tokens':len(ids)})
        summaries[name]={'num_fewshot':shot,'n_documents':len({x.doc_id for x in t.instances}),'n_requests':len(t.instances),'max_context_tokens':max(lengths),'config':t.config.to_dict(),'fewshot_document_hashes_by_draw':fewshots,'dataset_fingerprints':{k:dict(n=len(v),fingerprint=v._fingerprint) for k,v in t.dataset.items()}}
        print(name,summaries[name]['n_documents'],max(lengths),flush=True)
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'requests.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in output))
    (a.output/'tasks.json').write_text(json.dumps(summaries,indent=2,default=str)+'\n')
    (a.output/'hashes.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.output.iterdir() if f.is_file()},indent=2)+'\n')

if __name__=='__main__':main()
