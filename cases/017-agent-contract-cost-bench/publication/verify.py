"""Verify public packaging and restore a byte-identical research view. CPU only."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import tarfile

CASE = Path(__file__).resolve().parents[1]
REPO = CASE.parents[1]

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def safe(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or any(x in ('..', '.', '') for x in name.split('/')) or '\\' in name or ':' in name:
        raise ValueError('Unsafe member path')
    return p

def original_bytes(case=CASE):
    mapping = read(case / 'publication/provenance/original_mapping.json')
    a = mapping['archive']
    archive = case / safe(a['path'])
    if archive.stat().st_size != a['bytes'] or sha(archive.read_bytes()) != a['sha256']:
        raise ValueError('Compressed evidence changed')
    expected = {r['member']: r for r in mapping['files'] if r.get('encoding') == 'tar.xz'}
    seen = set()
    with tarfile.open(archive, 'r:xz') as tar:
        for entry in tar:
            safe(entry.name)
            if not entry.isfile() or entry.name in seen or entry.name not in expected:
                raise ValueError('Unexpected/duplicate/nonregular evidence member')
            row = expected[entry.name]
            if entry.size != row['bytes']:
                raise ValueError('Archive member size differs')
            b = tar.extractfile(entry).read()
            if sha(b) != row['sha256']:
                raise ValueError('Original archived bytes differ')
            seen.add(entry.name)
            yield row, b
    if seen != set(expected):
        raise ValueError('Missing evidence members')
    for row in mapping['files']:
        if row.get('encoding') != 'identity':
            continue
        p = case / safe(row['storage'])
        if p.is_symlink():
            raise ValueError('Symlink in retained evidence')
        b = p.read_bytes()
        if len(b) != row['bytes'] or sha(b) != row['sha256']:
            raise ValueError('Original retained bytes differ: ' + row['path'])
        yield row, b

def verify(case=CASE):
    m = read(case / 'publication/PUBLIC_MANIFEST.json')
    actual = {p.relative_to(case).as_posix() for p in case.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    expected = {r['path'] for r in m['files']} | {'publication/PUBLIC_MANIFEST.json'}
    if actual != expected:
        raise ValueError('Public closed file set differs: ' + str(sorted(actual ^ expected)))
    for row in m['files']:
        p = case / safe(row['path'])
        if p.is_symlink() or p.stat().st_size != row['bytes'] or sha(p.read_bytes()) != row['sha256']:
            raise ValueError('Public member changed: ' + row['path'])
    return {'status': 'PASS', 'public_files': len(m['files']), 'manifest_excludes_itself': True}

def restore(case, output):
    """Output is a fresh repository-shaped view, never an existing research tree."""
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(case.resolve()):
        raise ValueError('Choose a fresh external restoration directory')
    output.mkdir(parents=True)
    target_case = output / 'cases/017-agent-contract-cost-bench'
    count = 0
    for row, b in original_bytes(case):
        p = target_case / safe(row['path'])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
        count += 1
    # Frozen imports are supplied by the existing repository, hash checked first.
    freeze = read(target_case / 'provenance/execution_freeze_p2.json')
    for row in freeze['files']:
        if 'root_path' not in row:
            continue
        name = safe(row['root_path'])
        b = (case.parents[1] / name).read_bytes()
        if sha(b) != row['sha256'] or len(b) != row['bytes']:
            raise ValueError('Read-only Case016 import differs')
        p = output / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    (output / 'LICENSE').write_bytes((case.parents[1] / 'LICENSE').read_bytes())
    return {'original_files_restored': count, 'case': str(target_case), 'readonly_upstream_imports': 5,
            'scope': 'All 256 complete raw episodes, original sources/specs/inputs and retained receipts; no model payload.'}
