"""Verify combined publication files without changing historical scientific checks."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path, PurePosixPath


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe(name: str) -> bool:
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name


def verify(case: Path) -> dict:
    case = case.resolve()
    record = json.loads((case / 'provenance/publication_snapshot.json').read_text())
    actual = {p.relative_to(case).as_posix(): p for p in case.rglob('*') if p.is_file()}
    if any(p.is_symlink() for p in case.rglob('*')):
        raise ValueError('Symlinks are not publication material')
    checks = {}
    for line in (case / 'PUBLICATION_SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        if not safe(name) or name in checks:
            raise ValueError('Unsafe or duplicate manifest path')
        if name not in actual or sha(actual[name]) != expected:
            raise ValueError('Publication checksum mismatch: ' + name)
        checks[name] = expected
    if set(checks) != set(actual) - {'PUBLICATION_SHA256SUMS'}:
        raise ValueError('Publication inventory does not cover the exact combined tree')
    editorial = record['original_editorial_changes']
    original = record['original_review_files']
    for name, expected in original.items():
        target = editorial[name]['published_sha256'] if name in editorial else expected
        if sha(case / name) != target:
            raise ValueError('Original reviewed evidence changed: ' + name)
        if name in editorial and editorial[name]['reviewed_sha256'] != expected:
            raise ValueError('Editorial provenance mismatch')
    prefix = 'supplemental/readout-ties-v1/'
    for name, expected in record['supplement_review_files'].items():
        if sha(case / prefix / name) != expected:
            raise ValueError('Approved supplement changed: ' + name)
    expected_added = set(record['publication_added_files']) | {'PUBLICATION_SHA256SUMS'}
    if set(actual) != set(original) | {prefix + n for n in record['supplement_review_files']} | expected_added:
        raise ValueError('Files outside approved scientific and publication scope')
    for name, expected in record['frozen_identities'].items():
        if sha(case / name) != expected:
            raise ValueError('Frozen protocol/raw identity changed: ' + name)
    for name, p in actual.items():
        if p.suffix.lower() in {'.pt', '.pth', '.npy', '.npz', '.safetensors', '.bin', '.so', '.zip', '.pyc'}:
            raise ValueError('Excluded payload: ' + name)
        if any(part in {'.git', '.venv', '__pycache__', 'cache'} for part in p.relative_to(case).parts):
            raise ValueError('Excluded working directory')
        if p.stat().st_size > 64_000_000:
            raise ValueError('Unexpected oversized file')
    return {'status': 'PASS', 'publication_files': len(actual), 'original_review_files': len(original),
            'original_scientific_files_unchanged': len(original) - len(editorial),
            'presentation_changes': sorted(editorial), 'supplement_files_unchanged': len(record['supplement_review_files']),
            'frozen_identities_verified': len(record['frozen_identities']),
            'scope': 'Packaging and reviewed-byte identity; not independent GPU reproduction'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.case.resolve()):
        parser.error('Choose a new output file outside the case')
    result = verify(args.case)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
