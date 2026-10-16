"""Post-run annotations; frozen raw verdicts and execution policy are unchanged."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from order_regression.common import rows,read,atomic,require


def diagnostics(root=ROOT):
    root=Path(root);records=rows(root/'results/a1_r1/episodes.jsonl');spec=read(root/'specs/protocol.json');out=[]
    for r in records:
        parse=[e for e in r['errors'] if e['type']=='PARSE_ERROR']
        wrong=[e['event_id'] for e in r['tool_events'] if e['semantic_tool_id'] in ('order.read','order.write','order.validate') and e['schema_result']=='UNKNOWN_ORDER' and e['execution_status']=='ERROR']
        terminal=r['terminal_reason'];annotation=None
        if terminal=='BUDGET_EXHAUSTED' and parse and r['model_calls']<spec['generation']['max_model_calls'] and r['attempted_tool_calls']<spec['generation']['max_tool_calls']:
            terminal='PARSE_ERROR';annotation='Frozen runner retained its initial terminal label on malformed-only response. No model/tool budget was exhausted. Episode still lacks write, validation and final answer; primary failure is unchanged.'
        out.append({'episode_id':r['episode_id'],'raw_terminal_reason':r['terminal_reason'],'derived_terminal_reason':terminal,
            'terminal_annotation':annotation,'parse_error_markers':len(parse),'bad_argument_value_event_ids':wrong,
            'ARGUMENT_VALUE_WRONG_including_rejected_unknown_order_id':bool(wrong or r['ARGUMENT_VALUE_WRONG']),
            'raw_ARGUMENT_VALUE_WRONG':r['ARGUMENT_VALUE_WRONG'],'verified_workflow_success_unchanged':r['VERIFIED_WORKFLOW_SUCCESS']})
    return {'stage':'A1_POSTHOC_DIAGNOSTIC_ANNOTATION','raw_source_and_scoring_unchanged':True,'primary_outcomes_affected':0,
        'annotation_scope':'Correct two diagnostic metadata limitations; no rerun, feedback, parser or endpoint change',
        'malformed_early_stop_terminal_annotations':sum(r['terminal_annotation'] is not None for r in out),
        'bad_argument_value_episodes':sum(r['ARGUMENT_VALUE_WRONG_including_rejected_unknown_order_id'] for r in out),'episodes':out}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=diagnostics();atomic(a.output,v,True);print(json.dumps({k:v[k] for k in v if k!='episodes'}))
