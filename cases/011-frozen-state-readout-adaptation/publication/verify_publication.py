"""Verify fixed originals and the separate exact public inventory."""
import argparse,json
from pathlib import Path
from integrity import CASE,verify_original,verify_support,verify_protected,verify_manifest

def verify(require_full=False):
    return {'status':'PASS','original':verify_original(),'reference_files':verify_support(),
            'publication':verify_manifest(),'protected':verify_protected(required=require_full),
            'new_model_runs':0}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--require-full-repository',action='store_true')
    p.add_argument('--output',type=Path);a=p.parse_args();result=verify(a.require_full_repository)
    if a.output:
        if a.output.exists():raise FileExistsError('Choose a new report file')
        a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))
