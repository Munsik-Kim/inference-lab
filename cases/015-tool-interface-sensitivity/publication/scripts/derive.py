"""Read measured raw episodes, independently checked contracts and paired summaries."""
import json
import sys
import hashlib
import importlib.util
from pathlib import Path
CASE=Path(__file__).resolve().parents[2]
A1REL='followups/order-api-regression-v1'


def derive(case=CASE):
    case=Path(case);a1=case/A1REL
    sys.path.insert(0,str(case/'src'));sys.path.insert(0,str(a1/'src'))
    from interface_sensitivity.common import read,rows,require,sha
    from interface_sensitivity.analysis import summarize as p1_summary
    from order_regression.analysis import summarize as a1_summary
    old=rows(case/'results/p1_r1/episodes.jsonl');new=rows(a1/'results/a1_r1/episodes.jsonl')
    spec=read(case/'specs/protocol.json')
    p1=p1_summary(old,read(case/'results/p1_r1/planned_cells.json'),spec['bootstrap_replicates'],spec['bootstrap_seed'])
    a1data=a1_summary(new,read(a1/'results/a1_r1/planned_cells.json'))
    require(p1==read(case/'results/p1_r1/summary.json'),'old P1 raw/summary mismatch')
    require(a1data==read(a1/'results/a1_r1/summary.json'),'A1 raw/summary mismatch')
    diagnostic_spec=importlib.util.spec_from_file_location('case015_a1_diagnostics',a1/'postprocess/diagnostics.py')
    diagnostic_module=importlib.util.module_from_spec(diagnostic_spec);diagnostic_spec.loader.exec_module(diagnostic_module)
    annotations=diagnostic_module.diagnostics(a1)
    require(annotations==read(a1/'results/a1_r1/diagnostic_annotations.json'),'A1 diagnostic annotations differ from raw')
    sources=['results/p1_r1/episodes.jsonl','results/p1_r1/planned_cells.json','provenance/source_freeze.json',
        A1REL+'/results/a1_r1/episodes.jsonl',A1REL+'/results/a1_r1/planned_cells.json',A1REL+'/provenance/source_freeze.json',
        A1REL+'/provenance/input_manifest.json',A1REL+'/results/a1_r1/runtime_manifest.json']
    provenance={name:{'sha256':sha(case/name),'bytes':(case/name).stat().st_size} for name in sources}
    branching=[]
    by={(r['scenario_id'],r['method'],r['api_version']):r for r in new}
    for m in ['BF16','D16','RTN']:
        for sid in sorted({r['scenario_id'] for r in new}):
            if (sid,m,'API_V1') not in by or (sid,m,'API_V2') not in by:continue
            x,y=by[sid,m,'API_V1'],by[sid,m,'API_V2'];pairs=list(zip(x['tool_events'],y['tool_events']));first=None
            for i,(a,b) in enumerate(pairs):
                if (a['semantic_tool_id'],a['raw_arguments'],a['generated_in_turn'],a['generated_after_event_ids'])!=(b['semantic_tool_id'],b['raw_arguments'],b['generated_in_turn'],b['generated_after_event_ids']):first=i;break
            if first is None and len(x['tool_events'])!=len(y['tool_events']):first=len(pairs)
            branching.append({'method':m,'scenario_id':sid,'first_semantic_or_generation_group_difference_event_index':first,
                'both_verified':x['VERIFIED_WORKFLOW_SUCCESS'] is True and y['VERIFIED_WORKFLOW_SUCCESS'] is True,
                'same_initial_fixture':True,'same_model_visible_policy_input':False,
                'note':'API names and input token hashes differ. Matching semantic event prefix is not identical model policy input.'})
    return {'schema_version':1,'project':'Case015','model_id':read(case/'specs/methods.json')['model_id'],
        'model_revision':read(case/'specs/methods.json')['revision'],'evidence_kind':'historical_P1_plus_new_measured_A1_recalculated_from_raw',
        'execution_source_sha256':{'P1':read(case/'provenance/source_freeze.json')['execution_source_sha256'],
            'A1':read(a1/'provenance/source_freeze.json')['execution_source_sha256']},
        'P1':{'run_id':'p1_r1','summary':p1,'denominator':'8 previously observed tasks x3 methods x4 interfaces=96;4 families;4K/8K',
            'endpoint':'FUNCTIONAL_SUCCESS (historical scorer case015-strict-functional-v1)',
            'observation_boundary':'same pre-execution fixture is distinct from information received before generation; P1 did not separately enforce A1 receipt-order contract'},
        'A1':{'run_id':'a1_r1','summary':a1data,'denominator':'16 new synthetic scenarios nested in4 templates of one order workflow x3 methods x2 API names=96',
            'endpoint':'VERIFIED_WORKFLOW_SUCCESS (case015-a1-verified-workflow-v1)',
            'runtime':read(a1/'results/a1_r1/runtime_manifest.json'),'branching':branching,'diagnostic_annotations':annotations},
        'source_identity':provenance,'execution_boundary':'read-only saved INT4 -> dense BF16 reference; no ENUM/feedback/alias; no packed serving',
        'not_run':['P2 recovery diagnostics','fresh64/P3','CIPR training','other models','external business systems']}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x') as f:json.dump(derive(),f,ensure_ascii=False,indent=2);f.write('\n')
