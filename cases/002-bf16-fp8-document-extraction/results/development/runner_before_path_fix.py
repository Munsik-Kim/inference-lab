"""One local vLLM server at a time; sequential HTTP/SSE measurement, no retries."""
import argparse,hashlib,http.client,json,os,pathlib,random,signal,subprocess,sys,threading,time,traceback,urllib.request
from data import messages,read_jsonl,dump

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

class MemorySampler:
    """Global device memory, not torch allocator or process-private memory."""
    def __init__(self,path):
        import pynvml
        self.n=pynvml;self.n.nvmlInit();self.h=self.n.nvmlDeviceGetHandleByIndex(0)
        self.path=path;self.phase='idle_before_start';self.stop=threading.Event();self.error=None
        self.thread=threading.Thread(target=self.sample,daemon=True)
    def sample(self):
        with self.path.open('w') as f:
            while not self.stop.is_set():
                now=time.time();t=time.perf_counter()
                try:
                    m=self.n.nvmlDeviceGetMemoryInfo(self.h);u=self.n.nvmlDeviceGetUtilizationRates(self.h)
                    row={'epoch_s':now,'monotonic_s':t,'phase':self.phase,'device_used_bytes':m.used,'device_free_bytes':m.free,'device_total_bytes':m.total,'gpu_util_pct':u.gpu}
                except Exception as e:row={'epoch_s':now,'phase':self.phase,'error':str(e)};self.error=str(e)
                f.write(json.dumps(row)+'\n');f.flush();self.stop.wait(.1)
    def start(self):self.thread.start()
    def close(self):self.stop.set();self.thread.join();self.n.nvmlShutdown()

def request(doc,port,model,settings,meta):
    payload={'model':model,'messages':messages(doc),'stream':True,'stream_options':{'include_usage':True},**settings}
    # Assert that only these user-visible inputs appear in the chat message.
    assert set(payload['messages'][0])=={'role','content'}
    encoded=json.dumps(payload,ensure_ascii=False).encode()
    row={**meta,'case_id':doc['id'],'split':doc['split'],'bucket':doc['bucket'],'input_ref':f'data/{doc["split"]}_inputs.jsonl#{doc["id"]}',
         'request_sha256':hashlib.sha256(encoded).hexdigest(),'expected_input_tokens':doc['input_tokens'],'raw_output':'','chunks':[],
         'input_tokens':None,'output_tokens':None,'finish_reason':None,'error':None,'ttft_s':None,'total_s':None,'done_received':False}
    start=time.perf_counter();row['started_epoch_s']=time.time()
    conn=http.client.HTTPConnection('127.0.0.1',port,timeout=180)
    try:
        conn.request('POST','/v1/chat/completions',body=encoded,headers={'Content-Type':'application/json'})
        response=conn.getresponse();row['http_status']=response.status
        if response.status!=200:raise RuntimeError(f'HTTP {response.status}: '+response.read().decode())
        for line in response:
            # vLLM emits each SSE JSON event on one data line.
            if not line.startswith(b'data:'):continue
            elapsed=time.perf_counter()-start;text=line[5:].strip()
            if text==b'[DONE]':row['done_received']=True;break
            item=json.loads(text)
            if 'error' in item:raise RuntimeError(json.dumps(item['error']))
            if item.get('usage'):
                row['usage']=item['usage'];row['input_tokens']=item['usage']['prompt_tokens'];row['output_tokens']=item['usage']['completion_tokens']
            for choice in item.get('choices',[]):
                delta=choice.get('delta',{});content=delta.get('content') or ''
                if content:
                    if row['ttft_s'] is None:row['ttft_s']=elapsed
                    row['raw_output']+=content;row['chunks'].append({'arrival_s':elapsed,'text':content})
                if delta.get('reasoning_content') or delta.get('reasoning'):
                    row.setdefault('unexpected_reasoning',[]).append(delta)
                if choice.get('finish_reason') is not None:row['finish_reason']=choice['finish_reason']
        row['total_s']=time.perf_counter()-start
        if not row['done_received']:raise RuntimeError('SSE ended before [DONE]')
        if row['input_tokens']!=doc['input_tokens']:raise RuntimeError(f'Input-token mismatch: {row["input_tokens"]} vs {doc["input_tokens"]}')
        if not row['raw_output']:raise RuntimeError('No generated content')
        if row['output_tokens'] is None:raise RuntimeError('Missing engine usage')
    except Exception as e:
        row['total_s']=time.perf_counter()-start;row['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
    finally:conn.close()
    # Approximation includes final transport/completion overhead; chunks are NOT tokens.
    if row['output_tokens'] and row['output_tokens']>1 and row['ttft_s'] is not None:
        row['approx_post_first_s_per_token']=(row['total_s']-row['ttft_s'])/(row['output_tokens']-1)
    return row

def run_server(a,label,round_no,position,config,spec_hash):
    paths=json.loads(a.model_paths.read_text());models=json.loads((a.root/'provenance/models.json').read_text())
    modelpath=pathlib.Path(paths[label]);run_id=f'{a.run_id}-{a.stage}-r{round_no}-{position}-{label}'
    work=a.work/run_id;work.mkdir(parents=True,exist_ok=False)
    cfg_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    env=os.environ.copy()
    # Remove inherited experimental overrides, then set only recorded common overrides.
    for key in list(env):
        if key.startswith('VLLM_'):env.pop(key)
    env.update(config.get('environment',{}))
    env.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','PYTHONNOUSERSITE':'1',
                'CUDA_HOME':str(pathlib.Path(sys.prefix)/'lib/python3.12/site-packages/nvidia/cu13'),
                'TRITON_CACHE_DIR':str(a.cache/'triton'),'VLLM_CACHE_ROOT':str(a.cache/'vllm'),
                'TORCHINDUCTOR_CACHE_DIR':str(a.cache/'torchinductor'),'FLASHINFER_WORKSPACE_BASE':str(a.cache/'flashinfer')})
    cmd=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',str(modelpath),'--tokenizer',str(modelpath),
         '--served-model-name',label,'--host','127.0.0.1','--port',str(a.port),*config['engine_args']]
    metadata={'run_id':run_id,'stage':a.stage,'model':label,'model_id':models[label]['repo_id'],'revision':models[label]['revision'],
              'round':round_no,'model_order_position':position,'common_config_sha256':cfg_hash,'experiment_spec_sha256':spec_hash,
              'runner_sha256':sha(pathlib.Path(__file__)),'sampling':config['sampling'],'command':cmd,
              'runtime_environment_overrides':{k:env[k] for k in config.get('environment',{})},'started_epoch_s':time.time(),
              'memory_measurement':'NVML entire device, sampled every 100 ms; includes Windows/background usage. Not torch allocator or per-process attribution.'}
    dump(work/'server.json',metadata)
    sampler=MemorySampler(work/'memory.jsonl');sampler.start();time.sleep(2)
    process=None;log=(work/'server.log').open('w')
    try:
        sampler.phase='startup';t0=time.perf_counter();process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while True:
            if process.poll() is not None:raise RuntimeError(f'Server exited during startup: {process.returncode}; see server.log')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health',timeout=1) as res:
                    if res.status==200:break
            except Exception:pass
            if time.perf_counter()-t0>900:raise TimeoutError('Server health timeout after 900 seconds')
            time.sleep(1)
        metadata['startup_to_health_s']=time.perf_counter()-t0;sampler.phase='loaded';time.sleep(2)
        dev=read_jsonl(a.root/'data/dev_inputs.jsonl')
        warm=[next(d for d in dev if d['bucket']==b) for b in ['short','long']]
        if a.stage=='eval':
            docs=read_jsonl(a.root/'data/eval_inputs.jsonl')
            random.Random(20260910+round_no).shuffle(docs)
        else:docs=dev
        with (work/'requests.jsonl').open('w') as out:
            for phase,requests in [('warmup',warm),('measured',docs)]:
                sampler.phase=phase
                for idx,doc in enumerate(requests):
                    row=request(doc,a.port,label,config['sampling'],{k:metadata[k] for k in ['run_id','model','model_id','revision','round','model_order_position','common_config_sha256','experiment_spec_sha256']})
                    row['phase']=phase;row['request_order']=idx
                    out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
                    print(json.dumps({'run_id':run_id,'phase':phase,'index':idx,'case_id':doc['id'],'seconds':round(row['total_s'],3),'tokens':row['output_tokens'],'error':row['error']}),flush=True)
                    if phase=='warmup' and row['error']:raise RuntimeError('Warmup transport/execution failure; details in requests.jsonl')
        metadata['measurement_complete']=True
    except Exception as e:
        metadata['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
        print(json.dumps({'run_id':run_id,'error':metadata['error']}),flush=True)
    finally:
        sampler.phase='shutdown'
        if process is not None:
            metadata['server_exit_before_shutdown']=process.poll()
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=45)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait();metadata['forced_own_server_shutdown']=True
            metadata['server_exit_code']=process.returncode
        log.close();time.sleep(2);sampler.close();metadata['ended_epoch_s']=time.time();metadata['memory_sampler_error']=sampler.error
        dump(work/'server.json',metadata)
    return not metadata.get('error')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1]);p.add_argument('--model-paths',type=pathlib.Path,required=True)
    p.add_argument('--work',type=pathlib.Path,required=True);p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--config',type=pathlib.Path,required=True)
    p.add_argument('--stage',choices=['dev','eval'],required=True);p.add_argument('--model',choices=['bf16','fp8']);p.add_argument('--run-id',required=True);p.add_argument('--port',type=int,default=8765)
    a=p.parse_args();a.root=a.root.resolve();cfg=json.loads(a.config.read_text());spec_hash=None
    if a.stage=='eval':
        spec=json.loads((a.root/'configs/experiment_spec.json').read_text());spec_hash=sha(a.root/'configs/experiment_spec.json')
        for relative,expected in spec['frozen_files_sha256'].items():assert sha(a.root/relative)==expected,relative
        assert cfg==spec['common_config'],'Evaluation configuration differs from frozen spec'
        assert a.model is None,'Evaluation runs the complete paired schedule'
        schedule=[(1,['bf16','fp8']),(2,['fp8','bf16']),(3,['bf16','fp8'])]
    else:schedule=[(0,[a.model] if a.model else ['bf16','fp8'])]
    for rnd,labels in schedule:
        for pos,label in enumerate(labels,1):
            if not run_server(a,label,rnd,pos,cfg,spec_hash):sys.exit(1)
    print('All scheduled server runs completed.',flush=True)

if __name__=='__main__':main()
