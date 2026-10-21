"""One model-free replay of every present registered stage in a new directory."""
import argparse
import json
import sys
from pathlib import Path

from .common import CASE, read, write_once


def verify(output):
    # Heavy model modules are neither needed nor accepted by this entry point.
    class NoModelImports:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'torch','transformers','vllm','xgrammar'}:
                raise RuntimeError('Model-free verification forbids runtime import: '+fullname)
    sys.meta_path.insert(0,NoModelImports())
    from .audit import audit_run
    from .analysis import analyze
    from .cli import validate, oracle_check
    from .prepare import check_freeze
    from .review import load_review
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    validation=validate(output/'validate');gold=oracle_check(output/'oracle-check')
    stages=[]
    for stage,split in [('P1','dev'),('P2','holdout')]:
        root=CASE/f'results/{stage.lower()}_r1'
        if not (root/'completion.json').exists():continue
        check_freeze(read(CASE/f'provenance/execution_freeze_{stage.lower()}.json'))
        tasks=read(CASE/f'data/{split}_tasks.json')
        audited=audit_run(tasks,root,output/(stage+'-audit'))
        if not audited['receipt']['audit_pass']:raise ValueError('Stored raw replay failed: '+stage)
        if audited['rows']!=read(root/'audit/audited_rows.json'):raise ValueError('Retained audit differs: '+stage)
        summary=analyze(audited['rows'],output/(stage+'-analysis'),tasks=tasks,stage=stage)
        if summary!=read(root/'analysis/summary.json'):raise ValueError('Retained aggregate differs: '+stage)
        stages.append({'stage':stage,'cells':len(audited['rows']),'status':summary['status'],
                       'saved_raw_replayed_in_new_fixture':True,'aggregate_recomputed_from_fresh_audit':True})
    display=load_review(CASE)
    receipt={'status':'PASS','evidence_kind':'STORED_TRACE_REPLAY_AND_SYNTHETIC_CONTRACT_TEST',
        'stages':stages,'validated_tasks':validation['tasks'],'scripted_gold_tasks':gold['tasks'],
        'display_episodes':len(display['episodes']),'new_model_requests':0,
        'retokenization':'NOT_RUN; saved IDs/hash compared, actual tokenizer used in original prepare/run',
        'private_model_or_environment_required':False,'upstream_import_source_hashes_checked':True}
    write_once(output/'verification.json',receipt)
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args(argv)
    print(json.dumps(verify(a.output),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
