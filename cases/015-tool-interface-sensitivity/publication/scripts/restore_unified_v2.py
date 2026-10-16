"""Restore the exact closed public v2 view before reader-facing explanations changed."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

CASE = Path(__file__).resolve().parents[2]


def restore(case, output):
    case = Path(case)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    entries = json.loads((case / 'publication/provenance/unified_v2_map.json').read_text())['files']
    for name, entry in entries.items():
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('unsafe v2 path')
        source = case / entry['retained_path']
        data = source.read_bytes()
        if source.is_symlink() or len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('v2 source changed: ' + name)
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return {'status': 'PASS', 'original_v2_files': len(entries), 'original_v2_manifest_unchanged': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(CASE, args.output)))
