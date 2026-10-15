"""Restore an exact historical review view using the immutable path/hash map."""
import argparse,hashlib,json,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def restore(case,stage,output):
    case=Path(case).resolve();out=Path(output).resolve()
    if out.exists() or out.is_relative_to(case):raise ValueError('fresh external output required')
    entry=json.loads((case/'publication/provenance/STAGE_ARCHIVES.json').read_text())[stage];out.mkdir(parents=True)
    for rel,meta in entry['files'].items():
        a=Path(rel);b=Path(meta['retained_path'])
        if a.is_absolute() or b.is_absolute() or '..' in a.parts or '..' in b.parts:raise ValueError('unsafe mapped path')
        src=case/b;dst=out/a
        if src.is_symlink() or not src.is_file() or src.stat().st_size!=meta['bytes'] or sha(src)!=meta['sha256']:raise ValueError('historical file identity mismatch: '+rel)
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    manifest_name={'P0':'PUBLIC_MANIFEST.json','P1':'P1_PUBLIC_MANIFEST.json','P1b':'REVIEW_MANIFEST.json','P2':'REVIEW_MANIFEST.json'}[stage]
    original=json.loads((out/manifest_name).read_text())['files']
    expected=set(original)|{manifest_name};actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file()}
    if actual!=expected:raise ValueError('historical closed file set mismatch')
    for rel,meta in original.items():
        h=meta['sha256'] if isinstance(meta,dict) else meta
        if sha(out/rel)!=h:raise ValueError('historical manifest mismatch: '+rel)
    return {'stage':stage,'status':'PASS','files':len(actual),'original_manifest':manifest_name,'closed_set':True,'large_private_artifacts':'external identities only'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--stage',choices=['P0','P1','P1b','P2'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(restore(a.case,a.stage,a.output),indent=2))
