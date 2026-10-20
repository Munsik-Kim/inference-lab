"""Offline CPU tokenizer/template only. No torch/transformers/model import."""
import argparse,copy,json,pathlib,random
from .common import save,sha_file,token_hash,digest,utc

def prepare(root,snapshot):
 import tokenizers
 from jinja2.sandbox import ImmutableSandboxedEnvironment
 cfg=json.loads((snapshot/'tokenizer_config.json').read_text());tok=tokenizers.Tokenizer.from_file(str(snapshot/'tokenizer.json'));env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,extensions=['jinja2.ext.loopcontrols']);env.filters['tojson']=lambda x,**kw:json.dumps(x,ensure_ascii=False,**kw)
 env.globals['raise_exception']=lambda msg:(_ for _ in()).throw(ValueError(msg));template=env.from_string(cfg['chat_template'])
 def make(pid,length,prefix,question,kind,expected=None):
  padding='';padding_count=0
  def messages():return[{'role':'system','content':'Read the synthetic text and respond in English. No external service or tool is available.'},{'role':'user','content':prefix+'\n'+padding+'\n'+question}]
  def render():return template.render(messages=messages(),add_generation_prompt=True,enable_thinking=False,tools=None,bos_token=cfg.get('bos_token'),eos_token=cfg.get('eos_token'))
  for _ in range(10):
   text=render();ids=tok.encode(text,add_special_tokens=False).ids;gap=length-len(ids)
   if gap==0:break
   # Adjust only synthetic filler before registration; never cut facts/question/template.
   padding_count+=gap
   if padding_count<0:raise ValueError('prompt facts alone exceed length')
   padding=' archive'*padding_count
  text=render();ids=tok.encode(text,add_special_tokens=False).ids
  if len(ids)!=length:raise ValueError('exact template-inclusive length unavailable')
  return {'prompt_id':pid,'kind':kind,'input_length':length,'token_ids':ids,'input_token_sha256':token_hash(ids),'rendered_prompt_sha256':__import__('hashlib').sha256(text.encode()).hexdigest(),'messages':messages(),'length_definition':'Complete model-input IDs, including Qwen chat template/system/user/assistant generation prefix. Sent to /v1/completions as IDs with add_special_tokens=False.','expected_answer':expected,'no_truncation':True}
 rows=[]
 for length in (8192,16384):
  for i in range(8):
   rows.append(make(f'B-L{length}-{i:02}',length,f'SYNTHETIC LOG {i}.\nStation ALPHA recorded {17+i} units; station BETA recorded {29+i} units. These are invented records.','Continue the synthetic station log with a plain-English description of the sequence. Do not call tools.','benchmark'))
  for i in range(8):
   if i<4:
    value=['CEDAR_Q7','ORBIT_M4','COPPER_V8','HARBOR_K2'][i];prefix=f'SYNTHETIC REFERENCE NOTE\nThe verification label for reference NOTE-{i} is {value}. All other text is unrelated archive filler.';question=f'Copy the verification label for NOTE-{i}. Return only the label, without punctuation or explanation.';kind='copy_canary';expected=value
   else:
    prefix=f'SYNTHETIC NOTE {i}. A library lends books, receives returns, and sorts them by shelf. This note contains no real account data.';question=['Describe why clear labels help a small library.','Explain a simple way to organize a reading list.','Write one short sentence about keeping records consistent.','Describe the difference between storing information and checking it.'][i-4];kind='open_text';expected=None
   rows.append(make(f'S-L{length}-{i:02}',length,prefix,question,kind,expected))
 rows.append(make('P-SHORT',512,'This is an isolated serving probe.','Write a short neutral description of an archive.','probe'))
 save(root/'data/prompts.json',rows)
 manifest=[{k:v for k,v in r.items()if k not in ('messages','token_ids')}for r in rows]
 save(root/'data/input_manifest.json',{'created_utc':utc(),'rows':manifest,'tokenizer_files':[{'name':f.name,'bytes':f.stat().st_size,'sha256':sha_file(f)}for f in (snapshot/'tokenizer.json',snapshot/'tokenizer_config.json')],'tokenizers_version':tokenizers.__version__,'torch_or_model_loaded':False})
 conditions=[(l,c)for l in (8192,16384)for c in (1,2,4)];blocks=[];orders=[]
 for rnd,kvs in [(1,['B16','F4']),(2,['F4','B16']),(3,['B16','F4'])]:
  cc=conditions.copy();random.Random(6102002+rnd).shuffle(cc);orders.append({'round':rnd,'kv_order':kvs,'conditions':[list(x)for x in cc]})
  for kv in kvs:
   for l,c in cc:blocks.append({'block_id':f'R{rnd}_{kv}_L{l}_C{c}','round':rnd,'kv':kv,'length':l,'concurrency':c,'prompt_ids':[f'B-L{l}-{i:02}'for i in range(8)]})
 protocol={'schema':'case016-nvfp4-technical-v1','created_utc':utc(),'registered_before_new_gpu_requests':True,'model':'Qwen/Qwen3-4B-Instruct-2507','revision':'cdbee75f17c01a7cc42f958dc650907174af0554','engine':'vllm==0.32.0','weights':'bfloat16','kv':{'B16':'auto','F4':'nvfp4'},'pool_bytes':4294967296,'max_model_len':28672,'max_num_seqs':4,'max_num_batched_tokens':2048,'chunked_prefill':True,'prefix_cache':False,'async_scheduling':False,'attention_backend':'FLASHINFER','tp':1,'temperature':0,'thinking':False,'endpoint':'/v1/completions','add_special_tokens':False,'timed_ignore_eos':True,'timed_output_tokens':128,'sanity_ignore_eos':False,'sanity_max_tokens':64,'raw_token_recording':True,'stream_interval':1,'tools':None,'request_seconds_including_client_queue':180,'block_drain_seconds':30,'worker_seconds_including_p0_warmup_failures':7200,'max_infra_retries':1,'orders':orders,'planned_timed_blocks':36,'planned_timed_requests':288,'warmup':'One benchmark prompt of each length per owned server/round,12 total, excluded from timed results.','sanity':'16 separate inputs per KV once,32 natural-end requests;8 copy canary inputs and8 open-text inputs per KV. No full KL/PPL or agent endpoint.','optional_logprob':'NOT_RUN_SCOPE; no extra teacher-forced requests','graph_selection_before_timed':'Reuse validated eager mode for P0 finite checks. Probe default CUDA graph mode in separate owned servers with4 short512/16token requests per KV. Select graph only if both startup<=480s,cache/support/request checks pass. Graph errors lead to a common eager amendment before timed results, never based on speed/quality.','p0_probes':'2 eager32-token actual-attention finite/profile probes, up to8 graph16-token probes, one reused original tiny2-layout kernel check. No agent sweep.','numeric_tolerance':{'same_compressed_dequant_atol':0.02,'same_compressed_dequant_rtol':0.02,'finite_attention_required':True,'bf16_vs_quantization_error':'record only, no lossless threshold'},'primary_conditions':[[16384,2],[16384,4]],'round_uncertainty':'All3 paired ratios plus median/range. No per-request bootstrap for makespan.','timing_definitions':{'makespan':'first8-request queue launch to last terminal HTTP close; includes failures and client-slot wait','client_ttft':'first nonempty returned token-ID SSE event minus actual HTTP dispatch; role/empty/usage chunks excluded','client_decode_mean':'last token-ID event minus first token-ID event, divided by output_token_count-1; client-observed average, not per-token ITL','engine_queue_decode':'dedicated-server Prometheus interval sum/count deltas; no invented per-request attribution','successful_request':'HTTP completed,usage and original prompt IDs match,128 observed outputIDs and finish length; quality not scored in timed request success'},'seeds':{'inputs':6102001,'order':6102002,'analysis':6102003},'new_agent_tasks':False,'agent_P2_P3':False,'github_publication':'APPROVED_NORMAL_PR_AND_MERGE','pages_dispatch':'NOT_APPROVED'}
 save(root/'specs/protocol.json',protocol);save(root/'data/planned_blocks.json',blocks);print(json.dumps({'prompts':len(rows),'blocks':len(blocks),'timed_requests':288,'model_loaded':False}))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--snapshot',type=pathlib.Path,required=True);a=p.parse_args();prepare(a.root,a.snapshot)
if __name__=='__main__':main()
