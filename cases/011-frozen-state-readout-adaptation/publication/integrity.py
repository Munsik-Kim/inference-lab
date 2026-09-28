"""Separate fixed review identity from the editable publication inventory."""
from pathlib import Path, PurePosixPath
import hashlib
import json

CASE = Path(__file__).resolve().parents[1]
ROOT = CASE.parents[1]
ANCHORS = {
    'original_identity.json': 'd08cc444193ffa499c3abad20340b2d67991375cc683f0be75ef7cbb653f8aac',
    'reference_support.json': 'f7bf3aeba80d0aed9ac53c1c04f799991f0f6b9dc8544267e2ffdf51d8ab4a98',
    'protected_inventory.json': '44d1e89f504b16dde471ff7d158daa38f262ecbd649ec5df5be61e68a664b1eb',
}
EDITABLE = frozenset(('README.md','README.ko.md','REPORT.md','REPORT.ko.md',
                     'METHODS.md','REPRODUCTION.md','NOTICE.md'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    def pairs(items):
        out = {}
        for key,value in items:
            if key in out: raise ValueError('Duplicate JSON key: '+key)
            out[key]=value
        return out
    def bad(value): raise ValueError('Nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=bad)

def safe(root, name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('Unsafe relative path: '+name)
    target = Path(root)/name
    if target.is_symlink() or not target.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Unsafe file: '+name)
    return target

def anchored(case=CASE):
    pub=Path(case)/'publication'
    for name,expected in ANCHORS.items():
        if sha(safe(pub,name)) != expected: raise ValueError('Historical identity changed: '+name)
    return read(pub/'original_identity.json')

def original_file(case, name):
    if name in EDITABLE:
        return safe(Path(case)/'publication/original_docs',name)
    return safe(case,name)

def verify_original(case=CASE):
    case=Path(case); identity=anchored(case)
    for name,item in identity['research_files'].items():
        p=original_file(case,name)
        if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise ValueError('Original review file changed: '+name)
    return {'status':'PASS','research_files':len(identity['research_files']),
            'archive_sha256':identity['archive_sha256'],'editable_documents':sorted(EDITABLE)}

def verify_support(case=CASE):
    case=Path(case); repo=case.parents[1]; anchored(case)
    entries=read(case/'publication/reference_support.json')['files']
    for item in entries:
        p=safe(repo,item['path'])
        if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise ValueError('Case010 reference changed: '+item['path'])
    return len(entries)

def verify_protected(case=CASE, required=False):
    case=Path(case); repo=case.parents[1];anchored(case)
    expected=read(case/'publication/protected_inventory.json')['files']
    missing=[n for n in expected if not safe(repo,n).is_file()]
    # A standalone ZIP has a declared Case010 subset, not the whole repository.
    if missing and not required:
        return {'status':'NOT_RUN_FULL_REPOSITORY','reason':'standalone subset; full clone required',
                'reference_subset_checked':verify_support(case),'absent_protected_files':len(missing)}
    if missing: raise ValueError('Protected files missing: '+repr(missing[:3]))
    for name,h in expected.items():
        if sha(safe(repo,name))!=h:raise ValueError('Protected file changed: '+name)
    return {'status':'PASS','files':len(expected)}

def verify_manifest(case=CASE):
    case=Path(case);mf=case/'publication/PUBLICATION_SHA256SUMS'
    expected={}
    for line in mf.read_text().splitlines():
        h,name=line.split('  ',1)
        if name in expected:raise ValueError('Duplicate manifest path')
        if len(h)!=64:raise ValueError('Invalid SHA256')
        expected[name]=h
    actual={p.relative_to(case).as_posix() for p in case.rglob('*') if p.is_file() and p!=mf}
    if actual!=set(expected):raise ValueError('Publication inventory mismatch')
    for name,h in expected.items():
        if sha(safe(case,name))!=h:raise ValueError('Publication file changed: '+name)
    return {'status':'PASS','files':len(expected),'manifest_sha256':sha(mf)}
