"""Case021 identities and files. Importing this module never imports Torch."""
from pathlib import Path
import hashlib
import json
import os
from datetime import datetime, timezone

CASE = Path(__file__).resolve().parents[1]

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def byte_sha(value):
    return hashlib.sha256(value).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value, *, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())

def file_row(path):
    p = Path(path)
    return {'bytes': p.stat().st_size, 'sha256': sha(p)}

def freeze_files(paths, root=CASE):
    root = Path(root)
    files = {str(Path(p).relative_to(root)): file_row(p) for p in sorted(paths)}
    digest = byte_sha(json.dumps(files, sort_keys=True, separators=(',', ':')).encode())
    return {'files': files, 'bundle_sha256': digest}

def verify_files(freeze, root=CASE):
    files = freeze['files']
    digest = byte_sha(json.dumps(files, sort_keys=True, separators=(',', ':')).encode())
    if digest != freeze['bundle_sha256']:
        raise ValueError('FREEZE_BUNDLE_MISMATCH')
    for name, expected in files.items():
        p = (Path(root) / name).resolve()
        if not p.is_relative_to(Path(root).resolve()) or file_row(p) != expected:
            raise ValueError('SOURCE_INPUT_CONFIG_MISMATCH: ' + name)

def require_new(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
