"""Verify immutable research identities and the separate current integration inventory."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

CASE = Path(__file__).resolve().parents[1]
SNAPSHOT_SHA256 = '9fdd342a9d8da6b216fb06e2458ac1ba79d6fe4b56c13f7511df7346d49d436e'
COUNTS = {'v1': 1075, 'v2': 387}
ROOTS = {'v1': 'cases/010-ckda-finite-precision-memory-horizon/', 'v2': 'research/case010-failure-aware-v2/'}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def safe_path(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value or re.match(r'^[A-Za-z]:', value):
        raise ValueError(f'unsafe path: {value!r}')
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or '.' in p.parts or str(p) != value:
        raise ValueError(f'unsafe path: {value!r}')
    return p

def inventory(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError('symlink root')
    result = []
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'symlink: {p.relative_to(root)}')
        if p.is_file():
            result.append(p.relative_to(root).as_posix())
        elif not p.is_dir():
            raise ValueError('non-regular entry')
    return result

def verify_snapshots(case=CASE):
    case = Path(case)
    manifest_path = case / 'provenance/snapshot_manifest.json'
    if sha(manifest_path) != SNAPSHOT_SHA256:
        raise ValueError('original snapshot identity changed; rehashing cannot authorize research edits')
    manifest = json.loads(manifest_path.read_text())
    expected = {}
    originals = set()
    counts = dict.fromkeys(COUNTS, 0)
    for row in manifest['files']:
        path, original, version = row['path'], row['original_path'], row['version']
        safe_path(path); safe_path(original)
        if version not in ROOTS or not path.startswith(f'versions/{version}/'):
            raise ValueError('snapshot version/path mismatch')
        if original != ROOTS[version] + path.removeprefix(f'versions/{version}/'):
            raise ValueError('historical mapping mismatch')
        if path in expected or original in originals:
            raise ValueError('duplicate mapping')
        expected[path] = row
        originals.add(original)
        counts[version] += 1
    if counts != COUNTS:
        raise ValueError('snapshot inventory counts differ')
    actual = {'versions/' + p for p in inventory(case / 'versions')}
    if actual != set(expected):
        raise ValueError(f'snapshot inventory mismatch: missing={sorted(set(expected)-actual)[:5]}, extra={sorted(actual-set(expected))[:5]}')
    for path, row in expected.items():
        p = case / path
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError(f'protected snapshot changed: {path}')
    return manifest

def read_current_manifest(case):
    rows = {}
    for line in (case / 'UNIFIED_SHA256SUMS').read_text().splitlines():
        digest, path = line.split('  ', 1)
        safe_path(path)
        if path == 'UNIFIED_SHA256SUMS' or path in rows or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('invalid or duplicate current manifest entry')
        rows[path] = digest
    return rows

def verify(case=CASE, *, current=True, derived=True):
    case = Path(case)
    verify_snapshots(case)
    paths = inventory(case)
    if current:
        expected = read_current_manifest(case)
        if set(paths) - {'UNIFIED_SHA256SUMS'} != set(expected):
            raise ValueError('current integration manifest inventory mismatch')
        for path, digest in expected.items():
            if sha(case/path) != digest:
                raise ValueError(f'current integration file changed: {path}')
    if derived:
        from derive_summary import derive
        if json.loads((case/'summary/project.json').read_text()) != derive(case):
            raise ValueError('display summary differs from frozen scientific scalars')
    return {'status': 'PASS', 'snapshots': COUNTS, 'snapshot_manifest_sha256': SNAPSHOT_SHA256,
            'current_inventory_verified': current, 'derived_summary_verified': derived,
            'inventory': {'files': paths, 'count': len(paths)}, 'new_model_runs': 0}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, default=CASE)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.case)
    if args.output:
        with args.output.open('x') as f: json.dump(result, f, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != 'inventory'}))
if __name__ == '__main__':
    main()
