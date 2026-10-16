"""Comparable common event prefixes only; never align arbitrary free turns."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from interface_sensitivity.common import read,rows,atomic,require
case=Path(__file__).resolve().parents[1]
records=rows(case/'results/p1_r1/episodes.jsonl');look={(r['task_id'],r['method'],r['interface_id']):r for r in records};out=[]
def key(e):
 return {'action':e['semantic_tool_id'] if e['semantic_tool_id'] else {'UNRESOLVED_EXACT_RAW_NAME':e['raw_name']},'args':e['raw_arguments'],'state_before':e['state_before'],'state_after':e['state_after'],'result':e['tool_result']}
for r in records:
 if r['interface_id']=='I0_ORIGINAL':continue
 a=look[r['task_id'],r['method'],'I0_ORIGINAL'];b=r
 require(a['canonical_task_hash']==b['canonical_task_hash'],'different canonical tasks')
 n=0
 while n<min(len(a['tool_events']),len(b['tool_events'])) and key(a['tool_events'][n])==key(b['tool_events'][n]):n+=1
 ea=a['tool_events'][n] if n<len(a['tool_events']) else None;eb=b['tool_events'][n] if n<len(b['tool_events']) else None
 out.append({'task_id':r['task_id'],'method':r['method'],'changed_interface':r['interface_id'],'baseline_episode':a['episode_id'],'changed_episode':b['episode_id'],
  'common_event_prefix_length':n,'next_actions_comparable_same_state':ea is not None and eb is not None and ea['state_before']==eb['state_before'],
  'baseline_next_event':key(ea) if ea else None,'changed_next_event':key(eb) if eb else None,
  'meaning':'observed execution-prefix comparison; no internal semantic or first-token causal claim'})
atomic(case/'results/p1_r1/common_event_prefixes.json',out,True)
print(json.dumps({'comparisons':len(out)}))
