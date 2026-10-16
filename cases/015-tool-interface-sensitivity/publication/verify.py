"""New unified public identity, separate from the frozen P1 closed-file manifest."""
import argparse
import hashlib
import json
from pathlib import Path
CASE=Path(__file__).resolve().parents[1]


def verify(case=CASE):
    case=Path(case);name='publication/UNIFIED_PUBLIC_MANIFEST.json';manifest=json.loads((case/name).read_text());expected=manifest['files']
    actual={str(p.relative_to(case)) for p in case.rglob('*') if p.is_file() and '__pycache__' not in p.parts and str(p.relative_to(case))!=name}
    if actual!=set(expected):raise ValueError('unified public file set mismatch')
    for rel,e in expected.items():
        p=case/rel
        if p.is_symlink() or p.stat().st_size!=e['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=e['sha256']:raise ValueError('unified public bytes changed: '+rel)
        if p.suffix in {'.safetensors','.rpi4','.pt','.npy','.npz','.bin'}:raise ValueError('private tensor/payload in public bundle')
    original=json.loads((case/'publication/provenance/p1_original_map.json').read_text())['files']
    for rel,e in original.items():
        p=case/e['retained_path']
        if p.stat().st_size!=e['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=e['sha256']:raise ValueError('P1 original mismatch: '+rel)
    return {'status':'PASS','public_files':len(actual),'original_P1_files_preserved':len(original),'tensor_payloads_included':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=verify()
    with a.output.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
    print(json.dumps(v))
