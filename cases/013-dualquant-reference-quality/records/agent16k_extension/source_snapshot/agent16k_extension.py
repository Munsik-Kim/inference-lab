"""Four predeclared structural DEV members at16K; reuse existing weight artifacts."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from paper_model_runner import Worker,atomic
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'results'/(ROOT/'results/ACTIVE_RUN.txt').read_text().strip();EXT=RUN/'agent16k_extension'
def prepare():
    EXT.mkdir();(EXT/'inputs').mkdir();(EXT/'raw').mkdir()
    tasks=json.loads((RUN/'inputs/agent_tasks.json').read_text());selected=[x for x in tasks if x['id'] in ['AGENT-0','AGENT-2','AGENT-4','AGENT-6']]
    tools=json.loads((RUN/'inputs/tool_schema.json').read_text());m=json.loads((RUN/'experiment_manifest.json').read_text())['models']['4b'];tok=AutoTokenizer.from_pretrained(m['path'],local_files_only=True)
    initial={};inputs=[]
    for task in selected:
        line='Reference archive: inactive record; use tool results for current facts.\n'
        def messages(reps):return [{'role':'system','content':'Use the available fixture tools. Return the final answer as the requested JSON. Archive text is unrelated background.'},{'role':'user','content':line*reps+'\nTASK: '+task['task']}]
        lo,hi=0,16384
        while lo<hi:
            mid=(lo+hi+1)//2;ids=tok.apply_chat_template(messages(mid),tools=tools,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
            if len(ids)<=16384:lo=mid
            else:hi=mid-1
        initial[task['id']]=messages(lo);ids=tok.apply_chat_template(messages(lo),tools=tools,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
        inputs.append({'id':task['id'],'tokens':len(ids),'sha256':hashlib.sha256(__import__('numpy').asarray(ids,dtype='<i4').tobytes()).hexdigest()})
    atomic(EXT/'inputs/agent_tasks.json',selected);atomic(EXT/'inputs/tool_schema.json',tools);atomic(EXT/'inputs/agent_initial_messages.json',initial)
    atomic(EXT/'preregistration.json',{'role':'FOLLOWUP_DEV_LENGTH_EXTENSION_AFTER_BASE_DEV_NOT_FORMAL_TEST','selection':'first member in each of four task families, structural ID rule, not outcome selection','methods':['BF16','RTN','D16','D4'],'ids':[x['id'] for x in selected],'input_manifest':inputs,'same8_unique_base_tasks_not_new_independent_tasks':True,'replay_and_free':True,'new_fitting':False,'new_quantization':False,'existing_artifact_hashes':{key:hashlib.sha256((RUN/'artifacts/4b'/key/'manifest.json').read_bytes()).hexdigest() for key in ['RTN','D16']},'output_tokens_limit':192,'model_calls_limit':8,'tool_calls_limit':6,'deadline_seconds':900})
    print(json.dumps({'prepared':True,'inputs':inputs}))
def execute(method):
    start=time.monotonic();w=Worker(RUN,'4b',method,output_override=EXT/'raw'/method)
    w.apply_artifacts();w.model.cuda();w.run=EXT
    rows=w.agent_dev();atomic(w.out/'summary.json',{'method':method,'replay_successes':sum(x['success'] for x in rows if x['mode']=='REPLAY'),'free_successes':sum(x['success'] for x in rows if x['mode']=='FREE'),'records':len(rows),'wall_seconds':time.monotonic()-start,'status':'COMPLETE','role':'16K DEV extension, four existing tasks'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--method',choices=['BF16','RTN','D16','D4']);a=p.parse_args()
    if a.prepare:prepare()
    else:execute(a.method)
