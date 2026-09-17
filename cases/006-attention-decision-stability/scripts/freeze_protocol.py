"""Create immutable local design/evaluation freezes; no external preregistration."""
import argparse,json,sys
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import load,write_new,file_hash,digest
from src.statistics import select_boundary

CODE_FILES=['src/tasks.py','src/storage.py','src/metrics.py','src/validity.py','src/statistics.py','src/reference.py','src/intervention.py',
            'src/model_runtime.py','src/trajectory.py','src/timing.py','src/study_analysis.py','scripts/measure_scores.py',
            'scripts/run_secondary.py','scripts/measure_model_cost.py','scripts/prepare_evaluation.py','scripts/analyze_study.py',
            'scripts/audit_study.py','scripts/verify_results.py','scripts/freeze_protocol.py']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['design','eval'],required=True)
    p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dev-report',type=Path);p.add_argument('--design',type=Path);p.add_argument('--inventory',type=Path);p.add_argument('--pool-cells',type=Path);p.add_argument('--selection-output',type=Path)
    a=p.parse_args();hashes={name:file_hash(a.case/name) for name in CODE_FILES}
    if a.phase=='design':
        if a.dev_report is None:raise ValueError('Completed DEV report required')
        dev=load(a.dev_report)
        if dev['groups']['dev/L4096']['independent_scenarios']!=96:raise ValueError('Incomplete DEV')
        spec=load(a.case/'configs/design_draft.json')
        spec.update({'version':'1.0','status':'DESIGN_FREEZE','design_freeze':True,'evaluation_manifest_freeze':False})
        spec['resource_policy']['existing_compute_process']='Registration alone is not activity; five samples, >=3 utilization >10% blocks; free memory >=13312 MiB'
        spec['stage_readiness']={'semantic_smoke':'PASS','DEV_B_only':'COMPLETE','DEV_paired':'COMPLETE','design_freeze':'COMPLETE','evaluation':'NOT_ACCESSED'}
        spec['planned_generation']['implemented']='Own-cache bounded runner implemented; no evaluation outputs accessed'
        spec['decision_rules']={'primary_arms':['A_PUBLIC','V4'],'no_winner_selection':True,'deployment_verdict':'NOT_ASSESSED',
              'presentation_revisions_after_B_DEV':0,'weak_baseline_notes':['DEV comparison 13/32','DEV code 10/32'],'near_tie':'B gap < 0.1 raw logit units','noise_sensitive':'Exact ties only; B repeat floor bitwise zero',
              'native_head_policy':'All arms native logits_to_keep=1. Same hidden and same-shape LM head bitwise match required; full-position BF16 head shape may differ by rounding',
              'examples':'Fixed hash-selected IDs plus post-hoc median/largest absolute NLL change and first regression/gain by ID; label post-hoc',
              'no_new_input_after_results':True}
        frozen={'kind':'DESIGN_FREEZE','utc':datetime.now(timezone.utc).isoformat(),'spec':spec,'code_hashes':hashes,'dev_report_sha256':file_hash(a.dev_report),
                'semantic_report_sha256':file_hash(a.case/'provenance/integration_pass.json'),'input_hashes':{},'note':'Local pre-evaluation freeze, not external preregistration'}
    else:
        if any(x is None for x in (a.design,a.inventory,a.pool_cells,a.selection_output)):raise ValueError('Design, inventory, B-only pool and new selection path required')
        design=load(a.design)
        if design['kind']!='DESIGN_FREEZE' or design['code_hashes']!=hashes:raise ValueError('Design/source drift')
        rows=[load(f)['payload'] for f in sorted(a.pool_cells.glob('*.json'))]
        if len(rows)!=192 or any(r['arm']!='B' or r['split']!='boundary_pool' for r in rows):raise ValueError('Incomplete B-only pool')
        if any(not r['validity_status']['valid'] for r in rows):raise ValueError('Invalid pool record')
        selection=select_boundary(rows);selection['B_pool_score_hash']=digest(rows)
        inventory=load(a.inventory);selected=set(selection['selected_ids'])
        allowed=[item for item in inventory['input_hashes'] if '-standard-' in item or any(item.startswith(sid+'-L') for sid in selected)]
        frozen={'kind':'EVAL_MANIFEST_FREEZE','utc':datetime.now(timezone.utc).isoformat(),'design_sha256':file_hash(a.design),'code_hashes':hashes,
                'input_hashes':inventory['input_hashes'],'candidate_item_ids':sorted(allowed),'selection':selection,
                'boundary_pool_B_only':True,'candidate_evaluation_outputs_accessed':False,'inventory_sha256':file_hash(a.inventory)}
        write_new(a.selection_output,selection)
    write_new(a.output,frozen);print(json.dumps({'phase':a.phase,'sha256':file_hash(a.output),'code_files':len(hashes),'kind':frozen['kind']}))


if __name__=='__main__':main()
