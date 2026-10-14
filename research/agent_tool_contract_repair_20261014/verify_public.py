"""Read-only publication file identities, redaction and measurement linkage."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(study):
    study = Path(study)
    manifest = json.loads((study / 'PUBLIC_MANIFEST.json').read_text())
    files = {str(p.relative_to(study)): p for p in study.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts}
    if set(files) != set(manifest['files']) | {'PUBLIC_MANIFEST.json'}:
        raise ValueError('public file set mismatch')
    private = re.compile('/' + 'home' + r'/[^/\s]+/|/mnt/[a-z]/|gh[pousr]_[A-Za-z0-9]{20,}')
    for name, path in files.items():
        if path.is_symlink() or path.suffix in ('.rpi4','.safetensors','.bin','.pt','.npy','.npz','.arrow'):
            raise ValueError('private/model payload not permitted')
        if name != 'PUBLIC_MANIFEST.json':
            identity = manifest['files'][name]
            if path.stat().st_size != identity['bytes'] or digest(path) != identity['sha256']:
                raise ValueError('changed published bytes: ' + name)
        if private.search(path.read_text()):
            raise ValueError('private path/credential: ' + name)
    mapping = json.loads((study / 'publication/export_map.json').read_text())
    for row in mapping['files']:
        if digest(study / row['export_path']) != row['export_sha256']:
            raise ValueError('changed public raw export')
        if not row['path_redacted'] and row['source_sha256'] != row['export_sha256']:
            raise ValueError('false byte-identical export claim')
    registration = json.loads((study / 'records/preregistration.json').read_text())
    for name, h in registration['code_hashes'].items():
        if digest(study / name) != h:
            raise ValueError('measured execution code mismatch: ' + name)
    archive = json.loads((study / 'publication/archive_metadata.json').read_text())
    p = study.parents[1] / archive['repository_path']
    if p.stat().st_size != archive['bytes'] or digest(p) != archive['sha256']:
        raise ValueError('review ZIP mismatch')
    return {'status':'PASS','files':len(files),'bytes':sum(p.stat().st_size for p in files.values()),
            'measurement_code':'MATCHES_REGISTRATION','raw_export':'PASS','archive':'PASS',
            'scope':'CPU identities and redaction; no model/GPU rerun or certification of absent weights'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        result=verify(a.study)
    except Exception as e:
        a.output.write_text(json.dumps({'status':'FAIL','error':str(e)},indent=2)+'\n');raise
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
