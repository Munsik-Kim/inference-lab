"""Restore the exact original closed P0/P1 view without relaxing its verifier."""
import json
import hashlib
from pathlib import Path,PurePosixPath
import argparse
CASE=Path(__file__).resolve().parents[2]


def restore(case,output):
    case=Path(case);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    entries=json.loads((case/'publication/provenance/p1_original_map.json').read_text())['files']
    for original,e in entries.items():
        for name in [original,e['retained_path']]:
            p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError('unsafe original map path')
        source=case/e['retained_path'];data=source.read_bytes()
        if source.is_symlink() or len(data)!=e['bytes'] or hashlib.sha256(data).hexdigest()!=e['sha256']:raise ValueError('original changed: '+original)
        dest=output/original;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    return {'status':'PASS','restored_original_files':len(entries),'closed_view':'original relative paths and bytes; additions excluded'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(restore(CASE,a.output)))
