"""Export each interface separately; preserve diova-compare exact-input pairing."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from interface_sensitivity.common import atomic,canonical,digest,read,rows,sha,require

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--records',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 require(not a.output.exists(),'output directory must be new');a.output.mkdir(parents=True)
 records=rows(a.records);case=Path(__file__).resolve().parents[1];tasks={t['id']:t for t in read(case/'data/agent_tasks.json')};written=[]
 for interface in ['I0_ORIGINAL','I1_ORDER_ONLY','I2_STYLE_RENAME','I3_OPAQUE_RENAME']:
  for method in ['BF16','RTN','D16']:
   rr=sorted([r for r in records if r['method']==method and r['interface_id']==interface],key=lambda x:x['task_id'])
   require(len(rr)==8 and len({r['task_id'] for r in rr})==8,'exactly eight distinct observed task rows required')
   exported=[];side=[]
   for r in rr:
    require(type(r['functional_success']) is bool and r['input_token_hash'],'missing observation cannot become a failed export')
    exported.append({'schema_version':1,'sample_id':r['task_id'],'task':'case015-free-functional','task_version':'v1|'+interface,
     'prompt_hash':r['input_token_hash'],'gold':[canonical(tasks[r['task_id']]['gold'])],
     'gold_definition':'case015-functional-v1-required-actions-final-state-and-legacy-json-subset',
     'model_id':r['model_id']+'@'+r['model_revision'],'artifact_id':r['artifact_hash'],
     'output_type':'extracted_answer','answer':canonical(r['final_answer']) if r['final_answer'] is not None else '__PARSE_FAILURE__',
     'correct':r['functional_success'],'evidence_kind':'measurement'})
    side.append({'episode_id':r['episode_id'],'source_record_sha256':digest(r),'source_file_sha256':sha(a.records),
     'legacy_success':r['legacy_success'],'final_fixture':r['final_fixture'],'tool_events':r['tool_events'],
     'answer_contract':'extracted JSON; correct additionally requires real actions/state; same answer can have correct=false'})
   d=a.output/interface;d.mkdir(exist_ok=True);f=d/(method+'.jsonl');f.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in exported));(d/(method+'.sidecar.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in side))
   written.append({'path':str(f.relative_to(a.output)),'rows':len(exported),'sha256':sha(f)})
 atomic(a.output/'manifest.json',{'source_file_sha256':sha(a.records),'converter_sha256':sha(Path(__file__)),'files':written,
 'scope':'same interface only; cross-interface prompts deliberately differ, use canonical-task/mapping analyzer instead','process_exit_code':'not exported; original worker receipts available separately'})
 print(json.dumps({'status':'PASS','method_interface_files':len(written),'records':sum(r['rows'] for r in written)}))
if __name__=='__main__':main()
