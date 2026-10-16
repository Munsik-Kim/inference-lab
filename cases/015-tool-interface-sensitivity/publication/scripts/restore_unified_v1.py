"""Restore the closed public v1 view; v2 never loosens its original manifest."""
import argparse
import json
from pathlib import Path,PurePosixPath
import hashlib
CASE=Path(__file__).resolve().parents[2]


def restore(case,output):
    case=Path(case);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    entries=json.loads((case/'publication/provenance/unified_v1_map.json').read_text())['files']
    v2_map=case/'publication/provenance/unified_v2_map.json'
    previous=json.loads(v2_map.read_text())['files'] if v2_map.exists() else {}
    for name,e in entries.items():
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:raise ValueError('unsafe v1 path')
        retained=e['retained_path'];p=case/previous.get(retained,{'retained_path':retained})['retained_path'];data=p.read_bytes()
        if p.is_symlink() or len(data)!=e['bytes'] or hashlib.sha256(data).hexdigest()!=e['sha256']:raise ValueError('v1 source changed: '+name)
        dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    return {'status':'PASS','original_v1_files':len(entries),'original_v1_manifest_unchanged':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(restore(CASE,a.output)))
