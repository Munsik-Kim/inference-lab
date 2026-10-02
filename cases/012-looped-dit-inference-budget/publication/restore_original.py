"""Restore the original Case012 subtree into a new external directory."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = {'README.md', 'README.ko.md', 'REPORT.md', 'REPORT.ko.md', 'REPRODUCTION.md'}


def original_bytes(root, name):
    path = root / 'publication/original_docs' / (name + '.txt') if name in DOCS else root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Unsafe original source: ' + name)
    return path.read_bytes()


def restore(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or output == root or output.is_relative_to(root):
        raise ValueError('Output must be a new directory outside the current Case012 tree')
    inventory = json.loads((root / 'publication/original_inventory.json').read_text())['files']
    blobs = {}
    for name, meta in inventory.items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('Unsafe original path: ' + name)
        blob = original_bytes(root, name)
        if len(blob) != meta['bytes'] or hashlib.sha256(blob).hexdigest() != meta['sha256']:
            raise ValueError('Original identity mismatch: ' + name)
        blobs[name] = blob
    partial = output.with_name(output.name + '.partial')
    if partial.exists():
        raise ValueError('Unfinished restore already exists: ' + str(partial))
    partial.mkdir(parents=True)
    for name, blob in blobs.items():
        path = partial / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    partial.rename(output)
    return {'status': 'PASS', 'original_files': len(blobs), 'original_bytes': sum(len(b) for b in blobs.values()), 'scope': 'Original Case012 subtree only; no weights/environment/whole review ZIP'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(restore(a.root, a.output), indent=2))
