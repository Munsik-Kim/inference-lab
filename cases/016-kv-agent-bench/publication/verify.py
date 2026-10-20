"""Closed public inventory and losslessly retained historical evidence."""
import gzip
import hashlib
import json
import pathlib
import shutil
import stat
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe(name):
    path = pathlib.PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or not name:
        raise ValueError('Unsafe relative member: ' + name)
    return path


def verify(case):
    manifest = json.loads((case / 'publication/PUBLIC_MANIFEST.json').read_text())
    actual = {str(p.relative_to(case)) for p in case.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    expected = {r['path'] for r in manifest['files']} | {'publication/PUBLIC_MANIFEST.json'}
    if actual != expected:
        raise ValueError('Public inventory differs: ' + str(sorted(actual ^ expected)))
    for row in manifest['files']:
        path = case / safe(row['path'])
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes'] or sha(path.read_bytes()) != row['sha256']:
            raise ValueError('Public member changed: ' + row['path'])
    mapping = json.loads((case / 'publication/provenance/original_mapping.json').read_text())
    retained = 0
    for row in mapping['files']:
        if row['retained_path'] is None:
            continue
        path = case / safe(row['retained_path'])
        data = gzip.decompress(path.read_bytes()) if row['encoding'] == 'gzip' else path.read_bytes()
        if len(data) != row['bytes'] or sha(data) != row['sha256']:
            raise ValueError('Original bytes differ: ' + row['path'])
        retained += 1
    technical = case / 'followups/nvfp4-technical-r1'
    prior_analysis = json.loads((technical / 'provenance/analysis_freeze.json').read_text())
    for row in prior_analysis['files']:
        original = case / 'publication/originals/technical-analysis-v1/audit.py' if row['path'] == 'src/nvfp4_technical/audit.py' else technical / safe(row['path'])
        if sha(original.read_bytes()) != row['sha256']:
            raise ValueError('Retained pre-compression analysis source differs: ' + row['path'])
    storage = json.loads((technical / 'provenance/publication_storage.json').read_text())
    if sha((technical / 'src/nvfp4_technical/audit.py').read_bytes()) != storage['postprocessing_audit_public_sha256']:
        raise ValueError('Publication decompression adapter changed')
    return {'status': 'PASS', 'public_files': len(manifest['files']), 'original_files_retained': retained, 'original_helpers_omitted': 2}


def restore(case, output):
    output.mkdir(parents=True, exist_ok=False)
    mapping = json.loads((case / 'publication/provenance/original_mapping.json').read_text())
    for row in mapping['files']:
        if row['retained_path'] is None:
            continue
        source, target = case / safe(row['retained_path']), output / safe(row['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        data = gzip.decompress(source.read_bytes()) if row['encoding'] == 'gzip' else source.read_bytes()
        assert len(data) == row['bytes'] and sha(data) == row['sha256']
        target.write_bytes(data)


def extract_baseline(case, output):
    archive = case / 'followups/baseline-audit-r1/Case016_baseline_audit_v1.zip'
    assert sha(archive.read_bytes()) == '48ca3cc46c16740bd397c2379efc425651836c20699d6de5fe0db254604aa958'
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names = z.namelist()
        assert len(names) == len(set(names))
        for entry in z.infolist():
            safe(entry.filename)
            assert not stat.S_ISLNK(entry.external_attr >> 16)
            z.extract(entry, output)
    return output / 'Case016_baseline_audit_v1'
