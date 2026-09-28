"""Restore the reviewed 140-file Case011 and its byte-identical Case010 subset."""
from pathlib import Path
import argparse,json,shutil
from integrity import CASE,read,sha,safe,verify_original,verify_support,original_file

def restore(output, case=CASE):
    case=Path(case);output=Path(output)
    if output.exists():raise FileExistsError('Use a new external restoration directory')
    if output.resolve().is_relative_to(case.parents[1].resolve()):
        raise ValueError('Restoration must be outside the repository')
    verify_original(case);verify_support(case)
    identity=read(case/'publication/original_identity.json')
    restored=output/'cases'/case.name
    for name,item in identity['research_files'].items():
        target=safe(restored,name);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(original_file(case,name),target)
        if sha(target)!=item['sha256']:raise ValueError('Restoration byte mismatch')
    support=read(case/'publication/reference_support.json')['files']
    for item in support:
        target=safe(output,item['path']);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(safe(case.parents[1],item['path']),target)
    shutil.copy2(case.parents[1]/'LICENSE',output/'LICENSE')
    return {'status':'PASS','case_files':len(identity['research_files']),'reference_files':len(support),
            'scope':'original review paths and bytes; no private checkpoints or upstream tree'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(restore(a.output)))
