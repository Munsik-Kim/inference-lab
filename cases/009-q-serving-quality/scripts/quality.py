"""Pinned official lm-eval tasks with pre-inference request-manifest checks.

Benchmark samples/private text stay outside Git. The public exporter retains
scalar measurements and request hashes. No quantization/build code is invoked.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--tokenizer',required=True)
    p.add_argument('--task',choices=['arc_challenge','wikitext','gsm8k','mmlu'],required=True)
    p.add_argument('--arm',choices=['BF16','W4'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--dev',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from lm_eval import evaluator
    from lm_eval.models.vllm_causallms import VLLM
    from lm_eval.tasks import TaskManager
    expected=Counter()
    for line in (a.freeze/'requests.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['task']==a.task or a.task=='mmlu' and r['task'].startswith('mmlu_'):
            expected[(r['task'],r['doc_id'],r['request_index'],r['arguments_hash'])]+=1
    observed=Counter();audit=[]
    class CheckedVLLM(VLLM):
        def check_requests(self,requests):
            for r in requests:
                key=(r.task_name,r.doc_id,r.idx,digest(r.arguments))
                if not a.dev and (expected[key]!=1 or observed[key]):
                    raise ValueError('Request differs from pre-evaluation freeze or is duplicated: '+str(key[:3]))
                observed[key]+=1
        def loglikelihood(self,requests,**kwargs):
            self.check_requests(requests);return super().loglikelihood(requests,**kwargs)
        def loglikelihood_rolling(self,requests,**kwargs):
            self.check_requests(requests);return super().loglikelihood_rolling(requests,**kwargs)
        def generate_until(self,requests,**kwargs):
            self.check_requests(requests);return super().generate_until(requests,**kwargs)
        def _model_generate(self,requests,generate=False,**kwargs):
            out=super()._model_generate(requests,generate=generate,**kwargs)
            if generate:
                for r in out:
                    audit.append({'input_token_hash':digest(r.prompt_token_ids),'input_tokens':len(r.prompt_token_ids),
                        'outputs':[{'tokens':len(x.token_ids),'finish_reason':x.finish_reason,'stop_reason':x.stop_reason} for x in r.outputs]})
            return out
    try:
        model=CheckedVLLM(pretrained=a.model,tokenizer=a.tokenizer,dtype='bfloat16',
            max_model_len=8192,max_gen_toks=1024,tensor_parallel_size=1,batch_size='auto',
            add_bos_token=False,seed=909222,enable_thinking=None,
            enforce_eager=False,max_num_seqs=16,max_num_batched_tokens=2048,
            gpu_memory_utilization=0.87,
            enable_prefix_caching=False,enable_chunked_prefill=True,async_scheduling=False,
            kv_cache_memory_bytes=4*1024**3,kv_cache_dtype='bfloat16',
            attention_config={'backend':'FLASH_ATTN'},generation_config='vllm')
        manager=TaskManager()
        tasks=[a.task]
        if a.dev:
            loaded=manager.load(['arc_challenge','wikitext'])['tasks']
            tasks=[]
            for name,t in loaded.items():
                t.set_config('test_split','validation');tasks.append(t)
            # DEV is parser/runtime smoke only; use ARC alone here, WikiText separate.
            tasks=[t for t in tasks if t.config.task==a.task]
        result=evaluator.simple_evaluate(model=model,tasks=tasks,task_manager=manager,
            num_fewshot=5 if a.task in ('gsm8k','mmlu') else 0,
            apply_chat_template=a.task!='wikitext',fewshot_as_multiturn=False,
            gen_kwargs={'max_gen_toks':1024} if a.task=='gsm8k' else None,
            random_seed=909222,numpy_random_seed=909222,torch_random_seed=909222,fewshot_random_seed=909222,
            limit=2 if a.dev else None,log_samples=True,bootstrap_iters=5000)
        if not a.dev and observed!=expected:raise ValueError('Missing/extra official requests')
        (a.output/'results.json').write_text(json.dumps(result,indent=2,default=str,allow_nan=False)+'\n')
        (a.output/'generation_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        # Stop owned engine workers while Python modules are still alive.
        # The outer process return code remains a separate validity requirement.
        model.model.llm_engine.engine_core.shutdown(timeout=30)
        (a.output/'complete.json').write_text(json.dumps({'status':'PASS','arm':a.arm,'task':a.task,'dev':a.dev,'requests':sum(observed.values()),'request_freeze_match':not a.dev})+'\n')
    except Exception as e:
        (a.output/'failure.json').write_text(json.dumps({'status':'FAILED','error':type(e).__name__+': '+str(e),'requests_checked':sum(observed.values())})+'\n')
        raise

if __name__=='__main__':main()
