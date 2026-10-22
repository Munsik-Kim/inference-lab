"""Closed public manifest and bounded, hash-checked regular-object restoration."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import tarfile
import io
import re

CASE = Path(__file__).resolve().parents[1]

def sha(b):
    return hashlib.sha256(b).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def safe(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or any(x in ('', '.', '..') for x in name.split('/')) or '\\' in name or ':' in name:
        raise ValueError('Unsafe archive/mapping path')
    return p

def verify(case=CASE):
    m = read(case/'publication/PUBLIC_MANIFEST.json')
    actual = {p.relative_to(case).as_posix() for p in case.rglob('*') if p.is_file()}
    if actual != {r['path'] for r in m['files']} | {'publication/PUBLIC_MANIFEST.json'}:
        raise ValueError('Public file coverage differs')
    for r in m['files']:
        p = case/safe(r['path'])
        if p.is_symlink() or p.stat().st_size != r['bytes'] or sha(p.read_bytes()) != r['sha256']:
            raise ValueError('Public bytes differ: '+r['path'])
    return {'status':'PASS', 'files':len(actual), 'bytes':sum(r['bytes'] for r in m['files']),
            'self_hash_excluded':True}

def check_archive(case=CASE, blob=None):
    mapping=read(case/'publication/provenance/original_mapping.json')
    a=mapping['archive'];blob=blob if blob is not None else (case/a['path']).read_bytes()
    if not blob.startswith(b'\xfd7zXZ\x00') or len(blob)!=a['bytes'] or sha(blob)!=a['sha256']:
        raise ValueError('Wrong evidence XZ signature/bytes/hash')
    expected={r['member']:(r['bytes'],r['sha256']) for r in mapping['files']}
    sensitive=re.compile(r'/home/[A-Za-z0-9_.-]+/|/mnt/[a-z]/Users/|[A-Za-z]:[\\/]+Users[\\/]|(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,})|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\s*\n')
    seen=set()
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:xz') as tar:
        for m in tar:
            safe(m.name)
            if not m.isfile() or m.name in seen or m.name not in expected:raise ValueError('Unexpected evidence member')
            n,h=expected[m.name]
            if m.size!=n or n>16*1024*1024:raise ValueError('Evidence member size bound')
            b=tar.extractfile(m).read(n+1)
            if len(b)!=n or sha(b)!=h or sensitive.search(b.decode('utf-8')):raise ValueError('Evidence hash/privacy pattern check')
            seen.add(m.name)
    if seen!=set(expected):raise ValueError('Evidence coverage differs')
    for r in mapping['files']:
        if ('/src/' in r['path'] or re.match(r'case/results/p[12]_r[15]/episodes/',r['path'])) and r['transforms']:
            raise ValueError('Frozen source or model episode was transformed')
    return {'objects':len(seen),'logical_files':len(mapping['files']),'privacy':'Pattern check only, not a security certification'}

def restore(case, output):
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(case.resolve()):
        raise ValueError('Restoration requires a fresh external output')
    mapping = read(case/'publication/provenance/original_mapping.json')
    a = mapping['archive']; p = case/safe(a['path'])
    if p.stat().st_size != a['bytes'] or sha(p.read_bytes()) != a['sha256']:
        raise ValueError('Archive bytes/hash differ')
    expected, paths = {}, set()
    for r in mapping['files']:
        safe(r['path']); safe(r['member'])
        if r['path'] in paths:
            raise ValueError('Duplicate logical restoration path')
        paths.add(r['path'])
        old = expected.setdefault(r['member'], (r['bytes'],r['sha256']))
        if old != (r['bytes'],r['sha256']):
            raise ValueError('Inconsistent alias identity')
    seen, data = set(), {}
    # Never extractall: reject links/devices/directories and check sizes before read.
    with tarfile.open(p,'r:xz') as tar:
        for member in tar:
            safe(member.name)
            if not member.isfile() or member.name in seen or member.name not in expected:
                raise ValueError('Unexpected/duplicate/nonregular archive member')
            n,h = expected[member.name]
            if member.size != n:
                raise ValueError('Archive size mismatch')
            b = tar.extractfile(member).read(n+1)
            if len(b) != n or sha(b) != h:
                raise ValueError('Archive object hash mismatch')
            seen.add(member.name);data[member.name]=b
    if seen != set(expected):
        raise ValueError('Missing archive objects')
    output.mkdir(parents=True)
    for r in mapping['files']:
        dst=output/safe(r['path']);dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes(data[r['member']])
    for r in mapping.get('additional_public_files',[]):
        safe(r['path'])
        if r['path'] in paths:raise ValueError('Duplicate additional restoration path')
        src=case/safe(r['public_path']);b=src.read_bytes()
        if src.is_symlink() or len(b)!=r['bytes'] or sha(b)!=r['sha256']:raise ValueError('Additional original bytes differ')
        dst=output/safe(r['path']);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(b);paths.add(r['path'])
    return {'status':'PASS','files':len(paths),'objects':len(seen),
            'redacted_files':mapping['redacted_files'],'archive':a,
            'scope':mapping['scope']}
