"""Endpoint audit without trusting the runner's saved terminal label.

The frozen scorer's extra terminal_budget_or_infrastructure label is generic:
it is also appended for semantic INCORRECT_STATE/INCORRECT_FINAL and contract
failures. It must not be interpreted as evidence of an exhausted budget. We
preserve it and check that dropping this eligibility gate changes no endpoint.
"""
import argparse
import json
from pathlib import Path

from agent_contract_cost.common import CASE,read,write_once,sha_file
from agent_contract_cost.audit import independent_endpoint


def check(output):
    rows=[]
    for stage,split in [('P1','dev'),('P2','holdout')]:
        tasks={t['task_id']:t for t in read(CASE/f'data/{split}_tasks.json')}
        for p in sorted((CASE/f'results/{stage.lower()}_r1/cells').glob('*.json')):
            e=read(p)
            r=independent_endpoint(tasks[e['task_id']],e['events'],e['final_action'],e['final_state'],
                                   e['state_version'],e['final_available_result_event_ids'],terminal=None)
            rows.append({'stage':stage,'task_id':e['task_id'],'kv':e['kv'],'policy':e['policy'],
                'reported':e['reported_endpoint']['VERIFIED_WORKFLOW_SUCCESS'],
                'without_stored_terminal_eligibility':r['VERIFIED_WORKFLOW_SUCCESS'],
                'extra_generic_terminal_label':'terminal_budget_or_infrastructure' in e['reported_endpoint']['unmet_obligations'],
                'actual_terminal_reason':e['terminal_reason'],'source_file':p.relative_to(CASE).as_posix(),
                'source_sha256':sha_file(p)})
    bad=[r for r in rows if r['reported']!=r['without_stored_terminal_eligibility']]
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    result={'status':'PASS' if not bad else 'MISMATCH','cells':len(rows),'endpoint_changes':len(bad),
        'extra_generic_terminal_labels':sum(r['extra_generic_terminal_label'] for r in rows),
        'interpretation':'The extra unmet-obligation terminal label is not an exhaustion diagnosis; read actual terminal_reason and counters. No original scores or labels changed.',
        'dependency':'Actual events/state already independently replayed in the corresponding stage audit.',
        'new_model_requests':0,'scores_changed':False,'source_sha256':sha_file(Path(__file__))}
    write_once(out/'rows.json',rows);write_once(out/'receipt.json',result)
    if bad:raise ValueError('Endpoint differs without saved terminal gate')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(check(a.output),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
