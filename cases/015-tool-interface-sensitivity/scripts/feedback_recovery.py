"""Post-hoc refine recovery denominators to actual feedback and later model turns."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from interface_sensitivity.common import rows,atomic
case=Path(__file__).resolve().parents[1];records=rows(case/'results/p1_r1/episodes.jsonl');out=[]
for r in records:
 errors=[e for e in r['tool_events'] if e['execution_status']=='ERROR' or e['tool_result'].get('valid') is False]
 opportunities=[e for e in errors if any(s['model_turn']>e['model_turn'] for s in r['responses'])]
 later=[e for e in r['tool_events'] if any(e['model_turn']>o['model_turn'] for o in opportunities)]
 if opportunities:
  out.append({'episode_id':r['episode_id'],'method':r['method'],'interface_id':r['interface_id'],'feedback_events':len(errors),'feedback_events_with_later_model_turn':len(opportunities),
   'later_registered_name':any(e['semantic_tool_id'] is not None for e in later),
   'later_correct_required_action':any(e['semantic_choice_correct'] is True and e['schema_result']=='VALID' and e['argument_value_correct'] is True and e['execution_status']=='SUCCESS' and e['tool_result'].get('valid') is not False for e in later),
   'final_functional_success':r['functional_success']})
atomic(case/'results/p1_r1/feedback_recovery.posthoc.json',{'phase':'POST_HOC_DENOMINATOR_REFINEMENT','definition':'actual error result or negative validation, with a later model turn; same-turn calls are not error-feedback recovery. Sidecar-only WRONG_TOOL is excluded.',
 'original_registered_broad_continuation_preserved':'summary.json call_metrics/recovery; that field includes sidecar labels and later tool events',
 'episodes':out,'episodes_evaluable':len(out),'later_registered_name_episodes':sum(r['later_registered_name'] for r in out),'later_correct_required_action_episodes':sum(r['later_correct_required_action'] for r in out),'final_functional_successes':sum(r['final_functional_success'] for r in out)},True)
print(json.dumps({'feedback_opportunity_episodes':len(out),'later_registered_name_episodes':sum(r['later_registered_name'] for r in out),'final_success':sum(r['final_functional_success'] for r in out)}))
