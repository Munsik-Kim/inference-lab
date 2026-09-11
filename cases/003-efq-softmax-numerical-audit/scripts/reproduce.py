"""Create a fresh output copy; never clear the supplied case or its results."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from huggingface_hub import snapshot_download
from prepare import CASE


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['full','fixed-evaluation'],default='full')
    p.add_argument('--model-cache',type=Path,required=True)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    out=args.output_dir.resolve()
    assert not out.exists(), 'Choose a new output directory.'
    assert not out.is_relative_to(CASE), 'Output must be outside the supplied case.'
    ignore=shutil.ignore_patterns('results','__pycache__','experiment_spec.json','experiment_spec.sha256','eval_traces.json') if args.mode=='full' else shutil.ignore_patterns('results','__pycache__')
    shutil.copytree(CASE,out,ignore=ignore)
    (out/'results').mkdir()
    if args.mode=='fixed-evaluation':
        # Original development evidence is an explicit reference, not a new run.
        for name in ['calibration.json','dev_run.json','dev_units.jsonl','dev_rows.csv.gz','synthetic-dev_run.json']:
            shutil.copyfile(CASE/'results'/name,out/'results'/name)
    for split in ['dev','eval']:(args.work_dir/'traces'/split).mkdir(parents=True,exist_ok=True)
    assert not list((args.work_dir/'traces').glob('*/*.pt')), 'Choose an empty trace directory.'
    model=json.loads((CASE/'provenance/model.json').read_text())
    snapshot_download(model['model_id'],revision=model['revision'],cache_dir=args.model_cache,
                      allow_patterns=[f['name'] for f in model['files']],token=False)
    def run(script,*extra):
        subprocess.run([sys.executable,str(out/'scripts'/script),*map(str,extra)],cwd=out,check=True)
    if args.mode=='full':
        run('prepare.py','--model-cache',args.model_cache)
        run('extract_traces.py','--stage','validate','--model-cache',args.model_cache,'--work-dir',args.work_dir)
        run('record_tests.py')
        run('extract_traces.py','--stage','dev','--model-cache',args.model_cache,'--work-dir',args.work_dir)
        for stage in ['calibrate','dev','synthetic-dev','freeze']:
            run('audit.py','--stage',stage,'--work-dir',args.work_dir)
    run('extract_traces.py','--stage','eval','--model-cache',args.model_cache,'--work-dir',args.work_dir)
    for stage in ['eval','synthetic-eval']:run('audit.py','--stage',stage,'--work-dir',args.work_dir)
    run('analyze.py')
    print('Reproduction output:',out)


if __name__=='__main__':main()
