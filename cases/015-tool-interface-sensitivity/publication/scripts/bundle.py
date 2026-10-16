"""Small public ZIP with content identity; safe fresh extraction and CRC checks."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import stat
import zipfile
from restore_p1 import CASE


def validate_names(items):
    seen=set()
    for item in items:
        name=item.filename;p=PurePosixPath(name)
        if name in seen or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or str(p)!=name or stat.S_ISLNK(item.external_attr>>16):raise ValueError('unsafe/duplicate/symlink ZIP member')
        seen.add(name)
    return seen


def create(case,output):
    case=Path(case);output=Path(output)
    if output.exists():raise FileExistsError(output)
    files={str(p.relative_to(case)):p for p in sorted(case.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    identity={'version':1,'case_folder':'case015','files':{'case015/'+name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for name,p in files.items()},
        'included':'code, owned synthetic fixtures, lightweight measured responses/events/scalars, frozen stage manifests and bilingual documentation',
        'excluded':['model','INT4 payload','K','activations','full logits','environment/cache','credentials/private paths']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,p in files.items():z.write(p,'case015/'+name)
        z.writestr('ARCHIVE_CONTENTS.json',json.dumps(identity,ensure_ascii=False,indent=2)+'\n')
    return {'filename':output.name,'bytes':output.stat().st_size,'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'case_files':len(files),'version':'unified-publication-v2','model_payload_included':False}


def extract(archive,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        names=validate_names(z.infolist())
        if z.testzip() is not None:raise ValueError('ZIP CRC failure')
        manifest=json.loads(z.read('ARCHIVE_CONTENTS.json'))
        if names!=set(manifest['files'])|{'ARCHIVE_CONTENTS.json'}:raise ValueError('ZIP content inventory mismatch')
        for name,e in manifest['files'].items():
            data=z.read(name)
            if len(data)!=e['bytes'] or hashlib.sha256(data).hexdigest()!=e['sha256']:raise ValueError('ZIP file identity mismatch')
            p=output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
        (output/'ARCHIVE_CONTENTS.json').write_bytes(z.read('ARCHIVE_CONTENTS.json'))
    return {'status':'PASS','CRC':'PASS','safe_paths_duplicates_symlinks':'PASS','files_hashed':len(manifest['files']),'extracted_case':'case015'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    c=s.add_parser('create');c.add_argument('--output',type=Path,required=True)
    c=s.add_parser('extract');c.add_argument('--archive',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(create(CASE,a.output) if a.command=='create' else extract(a.archive,a.output)))
