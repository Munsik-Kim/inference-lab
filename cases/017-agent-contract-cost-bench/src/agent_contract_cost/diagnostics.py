"""Read-only explanatory ledgers; no score, parser, or policy revision."""
import argparse
import json
from collections import Counter
from pathlib import Path

from .common import CASE, read, write_once


def build(stage, output):
    root=CASE/'results'/f'{stage.lower()}_r1'
    episodes=[read(p) for p in sorted((root/'cells').glob('*.json'))]
    lookup={(e['task_id'],e['kv'],e['policy']):e for e in episodes}
    groups=[]; pairs=[]; failures=[]
    for kv in ('K0','K1'):
        for policy in ('C0','G','N','E'):
            rows=[e for e in episodes if e['kv']==kv and e['policy']==policy]
            requests=[r for e in rows for r in e['requests']]
            valid=[r for r in requests if r.get('parse') and r['parse']['accepted']]
            rejected=[r for r in requests if r.get('parse') and not r['parse']['accepted']]
            groups.append({'kv':kv,'policy':policy,'episodes':len(rows),'requests':len(requests),
                'parse_accepted_actions':len(valid),'contract_rejected_requests':len(rejected),
                'public_contract_error_codes':dict(Counter(r['parse']['error']['code'] for r in rejected)),
                'first_unmet_obligation':dict(Counter(str(e['reported_endpoint']['first_unmet_obligation']) for e in rows)),
                'workflow_success':sum(e['reported_endpoint']['VERIFIED_WORKFLOW_SUCCESS'] is True for e in rows),
                'state_correct':sum(e['reported_endpoint']['STATE_CORRECT'] is True for e in rows),
                'final_correct':sum(e['reported_endpoint']['FINAL_ANSWER_CORRECT'] is True for e in rows),
                'unsupported_success_claim':sum(e['reported_endpoint']['UNSUPPORTED_SUCCESS_CLAIM'] is True for e in rows),
                'claimed_done_with_wrong_state':sum(e['reported_endpoint']['CLAIMED_DONE_WITH_WRONG_STATE'] is True for e in rows),
                'server_finish_reasons':dict(Counter(str(r['response'].get('finish_reason')) for r in requests)),
                'max_output_tokens':max((r['response']['output_tokens'] for r in requests),default=None),
                'request_cap_reached':sum(r['response']['output_tokens']==r['request_plan']['max_tokens'] for r in requests),
                'sixth_nonfinal_censoring':sum(e['budget']['final_opportunity_blocked_by_registered_attempt_budget'] for e in rows),
                'counter_note':'generated_raw_objects counts a strict complete single JSON dictionary, not every JSON-looking fragment.'})
    for task_id in sorted({e['task_id'] for e in episodes}):
        for kv in ('K0','K1'):
            for a,b in (('N','C0'),('E','C0'),('E','N')):
                left,right=lookup.get((task_id,kv,a)),lookup.get((task_id,kv,b))
                if not left or not right:continue
                l,r=left['requests'][0],right['requests'][0]
                pairs.append({'task_id':task_id,'kv':kv,'pair':a+'-'+b,
                    'same_initial_messages':l['messages']==r['messages'],
                    'same_initial_token_ids':l['input_token_ids']==r['input_token_ids'],
                    'same_first_output_ids':l['response']['output_token_ids']==r['response']['output_token_ids'],
                    'same_first_raw':l['response']['raw_text']==r['response']['raw_text'],
                    'interpretation':'Observed equality is not a shared-failure intervention; these were independent free runs.'})
    for e in episodes:
        if e['reported_endpoint']['VERIFIED_WORKFLOW_SUCCESS'] is True:continue
        failures.append({'task_id':e['task_id'],'kv':e['kv'],'policy':e['policy'],'terminal':e['terminal_reason'],
            'endpoint':e['reported_endpoint'],
            'trajectory':[{'generation_turn':r['generation_turn'],'raw_text':r['response']['raw_text'],
                'available_result_event_ids':r['available_result_event_ids'],'parse':r['parse'],
                'event_id':r['event']['event_id'] if r.get('event') else None,
                'status':r['event']['status'] if r.get('event') else None,
                'result':r['event']['result'] if r.get('event') else None,
                'version':r['event']['version_after'] if r.get('event') else r['state_version']}
                for r in e['requests']]})
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_once(output/'groups.json',groups);write_once(output/'initial_free_pairs.json',pairs)
    write_once(output/'failed_trajectories.json',failures)
    return {'stage':stage,'groups':groups,'initial_pair_count':len(pairs),
        'initial_input_mismatches':sum(not p['same_initial_token_ids'] or not p['same_initial_messages'] for p in pairs),
        'initial_output_mismatches':sum(not p['same_first_output_ids'] for p in pairs),
        'failed_workflows':len(failures),'new_model_requests':0,'scores_changed':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,choices=('P1','P2'))
    p.add_argument('--output',required=True);a=p.parse_args(argv)
    print(json.dumps(build(a.stage,a.output),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
