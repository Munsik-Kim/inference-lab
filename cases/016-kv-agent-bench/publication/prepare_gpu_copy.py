"""Prepare a fresh source/input-only case copy; never run a model or touch the old run."""
import argparse,pathlib,shutil,json
CASE=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
if a.output.exists():raise ValueError('fresh external directory required')
if a.output.resolve().is_relative_to(CASE):raise ValueError('copy outside original case')
a.output.mkdir(parents=True)
paths=['src','provenance/model_identity.json','results/p0_diagnostics_r2/F8_final_config_cache_before.json','followups/nvfp4-technical-r1/src','followups/nvfp4-technical-r1/tests','followups/nvfp4-technical-r1/data','followups/nvfp4-technical-r1/specs/protocol.json']
for name in paths:
 s=CASE/name;d=a.output/name;d.parent.mkdir(parents=True,exist_ok=True)
 if s.is_dir():shutil.copytree(s,d,ignore=shutil.ignore_patterns('__pycache__'))
 else:shutil.copyfile(s,d)
print(json.dumps({'status':'PREPARED_INPUT_SOURCE_ONLY','new_gpu_requests':0,'existing_run_modified':False,'next':'Create private runtime JSON for this fresh copy; run GPU command only after local support/resource gates.'}))
