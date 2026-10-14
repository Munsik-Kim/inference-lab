"""Verify the separate follow-up file set and redacted export identities."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def verify(study):
    study = Path(study)
    manifest = json.loads((study / 'PUBLIC_MANIFEST.json').read_text())
    actual = {str(p.relative_to(study)): p for p in study.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    if set(actual) != set(manifest['files']) | {'PUBLIC_MANIFEST.json'}:
        raise ValueError('followup public file-set mismatch')
    private = re.compile('/' + 'home' + r'/[^/\s]+/|/mnt/[a-z]/|gh[pousr]_[A-Za-z0-9]{20,}')
    for name, path in actual.items():
        if path.is_symlink() or path.suffix in ('.rpi4','.bin','.safetensors','.pt','.npz','.npy','.arrow'):
            raise ValueError('unsafe/private payload in public study')
        data = path.read_bytes()
        if name != 'PUBLIC_MANIFEST.json':
            expected = manifest['files'][name]
            if path.stat().st_size != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise ValueError('changed public study file: ' + name)
        if private.search(data.decode('utf-8')):
            raise ValueError('private path/credential in public study: ' + name)
    mapping = json.loads((study / 'records/export_map.json').read_text())
    for row in mapping['files']:
        path = study / 'records' / row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['public_sha256']:
            raise ValueError('changed redacted export: ' + row['path'])
        if row['operation'] == 'BYTE_IDENTICAL' and row['original_sha256'] != row['public_sha256']:
            raise ValueError('false original/public equality')
    return {'status':'PASS','files':len(actual),'bytes':sum(p.stat().st_size for p in actual.values()),
            'scope':'Separate follow-up inventory/redaction/readback; Case013 manifest unchanged',
            'new_GPU_runs':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.study)))
