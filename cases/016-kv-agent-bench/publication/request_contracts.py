"""Independent request-body/counter-window checks from retained raw files."""
import hashlib,json,struct,pathlib

def check(case):
 t=case/'followups/nvfp4-technical-r1';prompts={r['prompt_id']:r for r in json.loads((t/'data/prompts.json').read_text())}
 n=0
 for section in ['blocks','p0','graph','warmup','sanity']:
  for p in (t/'results'/section).glob('*/request_*.json'):
   r=json.loads(p.read_text());ids=prompts[r['prompt_id']]['token_ids']
   assert hashlib.sha256(b''.join(struct.pack('<I',x)for x in ids)).hexdigest()==r['input_token_sha256']
   body={'model':'case016-tech-qwen3-4b','prompt':ids,'add_special_tokens':False,'temperature':0,'max_tokens':r['output_limit'],'ignore_eos':r['ignore_eos'],'stop_token_ids':[],'stream':True,'stream_interval':1,'stream_options':{'include_usage':True},'return_token_ids':True}
   h=hashlib.sha256(json.dumps(body,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
   assert h==r['request_sha256'],p
   n+=1
 assert n==342
 windows=[]
 for folder in sorted((t/'results/blocks').iterdir()):
  if not folder.is_dir():continue
  def scalar(path,name):
   values=[]
   for line in path.read_text().splitlines():
    if not line.startswith('#')and line.split('{')[0].split(' ')[0]==name:values.append(float(line.split()[-1]))
   return sum(values)if values else None
  counts={}
  for name in ['request_queue_time_seconds','request_decode_time_seconds','request_prefill_time_seconds','time_to_first_token_seconds']:
   a=scalar(folder/'metrics_before.txt','vllm:'+name+'_count');b=scalar(folder/'metrics_after.txt','vllm:'+name+'_count')
   counts[name]=None if a is None or b is None else b-a
   if counts[name]is not None:assert counts[name]==8,(folder,name,counts[name])
  windows.append({'block':folder.name,'histogram_count_deltas':counts})
 commands=[]
 for p in sorted((t/'results/servers').glob('R*/launch.json')):
  cmd=json.loads(p.read_text())['command'];idx=cmd.index('--kv-cache-dtype');cmd[idx+1]='<KV_DTYPE>';commands.append(cmd)
 assert len(commands)==6 and all(c==commands[0]for c in commands)
 return {'status':'PASS','independently_rebuilt_request_body_hashes':n,'actual_input_ID_hashes':n,'metric_windows':windows,'timed_common_launch_commands':6,'allowed_command_difference':'KV dtype only;private paths replaced by shared identity labels','fresh_GPU_requests':0}
