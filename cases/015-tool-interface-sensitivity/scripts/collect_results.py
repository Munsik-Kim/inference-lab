"""Publish local review copies with explicit path transforms; retain private originals."""
import argparse,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from interface_sensitivity.common import atomic,read,sha,digest,require,rows

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--private',type=Path,required=True);a=p.parse_args();case=Path(__file__).resolve().parents[1];result=case/'results/p1_r1'
 transforms={str(case):'<CASE_ROOT>',str(a.private):'<PRIVATE_RESULT_ROOT>'}
 runtime=read(a.private/'runtime.json')
 transforms.update({runtime['model']['path']:'<EXISTING_WSL_QWEN3_4B_SNAPSHOT>',runtime['original_result_root']:'<ORIGINAL_CASE013_R2>',runtime['research_source']:'<ORIGINAL_CASE013_SOURCE>',runtime['artifact_root']:'<ORIGINAL_CASE013_4B_ARTIFACTS>',
  str(Path(runtime['research_source']).parent/'.venv-reference/bin/python'):'<EXISTING_REFERENCE_PYTHON>',
  str(next(p for p in Path(runtime['model']['path']).parents if p.name=='inference-lab')):'<ORIGINAL_LIBRARY_ROOT>'})
 transforms=dict(sorted(transforms.items(),key=lambda x:-len(x[0])))
 def clean(value):
  if isinstance(value,str):
   for src,target in transforms.items():value=value.replace(src,target)
   return value
  if isinstance(value,list):return [clean(x) for x in value]
  if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
  return value
 export=[];allrows=[];smoke=[];receipts=[]
 for m in ['BF16','RTN','D16']:
  d=a.private/'run/workers'/m/'attempt01'
  for f in sorted(d.rglob('*.json')):
   if 'fixtures' in f.parts or 'smoke_fixtures' in f.parts:continue
   original=read(f);sanitized=clean(original);rel=f.relative_to(a.private/'run');dest=result/rel
   if f.parent.name=='cells':
    require(sanitized==original,'model-response private path requires dedicated token/history-safe export amendment')
    allrows.append(original)
   if f.parent.name=='smoke':smoke.append(original)
   if f.name=='receipt.json':receipts.append(original)
   atomic(dest,sanitized,True)
   export.append({'source_label':'PRIVATE_RUN/'+str(rel),'source_sha256':sha(f),'export':str(dest.relative_to(case)),
     'export_sha256':sha(dest),'transform':'path labels in metadata only' if original!=sanitized else 'byte representation JSON formatted; object unchanged','source_object_sha256':digest(original),'export_object_sha256':digest(sanitized)})
  log=a.private/'run'/(m+'-attempt01.log')
  if log.exists():
   target=result/'attempts'/log.name;target.parent.mkdir(exist_ok=True);text=clean(log.read_text());target.write_text(text)
   export.append({'source_label':'PRIVATE_RUN/'+log.name,'source_sha256':sha(log),'export':str(target.relative_to(case)),'export_sha256':sha(target),'transform':'known absolute path labels only'})
 allrows.sort(key=lambda r:r['episode_id'])
 require(len({r['episode_id'] for r in allrows})==len(allrows),'duplicate cells')
 target=result/'episodes.jsonl';require(not target.exists(),'preserve previous collected records')
 target.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in allrows))
 (result/'tool_events.jsonl').write_text(''.join(json.dumps({'episode_id':r['episode_id'],**e},ensure_ascii=False,allow_nan=False)+'\n' for r in allrows for e in r['tool_events']))
 commands=read(a.private/'run/gpu_commands.json');atomic(result/'gpu_commands.json',clean(commands),True)
 for name in ['p0_tests_attempt01.log','p0_tests_attempt02.log']:
  f=a.private/name;dest=result/name;dest.write_text(clean(f.read_text()))
  export.append({'source_label':'PRIVATE/'+name,'source_sha256':sha(f),'export':str(dest.relative_to(case)),'export_sha256':sha(dest),'transform':'known absolute path labels only'})
 atomic(case/'provenance/export_map.json',{'transform_rules':[{'private_prefix_label':target,'applied_to':'metadata/log text, never generated response/action/value'} for target in transforms.values()],
  'private_originals_preserved':True,'files':export,'original_response_objects_unchanged':True},True)
 peaks=[r for r in allrows+smoke]
 atomic(result/'resource_summary.json',{'primary_episodes':len(allrows),'smoke_episodes':len(smoke),
  'primary_model_calls':sum(r['model_calls'] for r in allrows),'smoke_model_calls':sum(r['model_calls'] for r in smoke),
  'primary_generated_tokens':sum(r['generated_tokens'] for r in allrows),'smoke_generated_tokens':sum(r['generated_tokens'] for r in smoke),
  'primary_tool_attempts':sum(r['tool_attempts'] for r in allrows),'smoke_tool_attempts':sum(r['tool_attempts'] for r in smoke),
  'worker_wall_seconds':sum(r['wall_seconds'] for r in commands),'worker_wall_includes':'load/decode/smoke/generation/CPU/I-O/process lifetime; not GPU-busy or serving latency',
  'peak_torch_allocated_bytes':max(r['peak_allocated_bytes'] for r in peaks),'peak_torch_reserved_bytes':max(r['peak_reserved_bytes'] for r in peaks),
  'device_free_total_samples':[{'episode_id':r['episode_id'],'request':s['model_turn'],'free_total':s['device_free_total_after_request']} for r in peaks for s in r['responses']],
  'device_scope':'CUDA device free/total samples, not a continuous system-wide Windows peak','continuous_NVML_peak':None,
  'model_load_order':[r['method'] for r in commands],'actual_process_exits':[r['exit_code'] for r in commands],
  'retries':0,'new_model_download':0,'new_quantization_or_solver':0,'package_changes':0,'serving_benefit':'NOT_ESTABLISHED'},True)
 print(json.dumps({'primary_records':len(allrows),'smoke_records':len(smoke),'exports':len(export)}))
if __name__=='__main__':main()
