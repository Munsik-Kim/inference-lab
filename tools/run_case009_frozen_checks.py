"""Run the immutable Case009/diova-compare0.1.1 verifier in restored scratch.

New package releases have their own checks. This wrapper leaves the old exact
inventory, tests and wheel checks intact, rather than rewriting their manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ANCHOR = '27294fc5cfde575cbed638c17b9a547e126a2f01'
ARCHIVE_SHA256 = '0eeb58401ba548fc64d12929b4a8ce8cd0ec35ed7981541e5110e487c27700d2'
PATHS = ('cases/009-q-serving-quality', 'packages/diova-compare',
         'downloads/diova_compare-0.1.1-py3-none-any.whl', 'downloads/diova_compare-0.1.1.metadata.json')


def restore_frozen(root, scratch):
    root, scratch = Path(root), Path(scratch)
    metadata = json.loads((root / 'downloads/case009_serving_quality_reviewed_publication_v2.json').read_text())
    archive = root / 'downloads' / metadata['filename']
    if archive.is_symlink() or metadata['sha256'] != ARCHIVE_SHA256 \
            or hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256 \
            or archive.stat().st_size != metadata['bytes']:
        raise ValueError('protected Case009 archive identity mismatch')
    scratch.mkdir()
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(set(z.namelist())) or z.testzip() is not None:
            raise ValueError('duplicate/corrupt frozen archive')
        for member in z.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or '..' in relative.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('unsafe frozen archive path')
            if not any(member.filename == name or member.filename.startswith(name + '/') for name in PATHS):
                raise ValueError('unexpected frozen archive member')
        z.extractall(scratch)
    def inventory(base):
        if any(p.is_symlink() for p in base.rglob('*')):
            raise ValueError('symlink in protected live/restored Case009')
        return {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if inventory(root / PATHS[0]) != inventory(scratch / PATHS[0]):
        raise ValueError('live Case009 differs from protected archive')
    for name in PATHS[2:]:
        if (root / name).read_bytes() != (scratch / name).read_bytes():
            raise ValueError('historical download changed: ' + name)
    return {'archive_sha256': ARCHIVE_SHA256, 'historical_package_version': '0.1.1',
            'live_case_and_old_downloads_match': True}


def verify_current_case009(root):
    """Exact old verifier on restored bytes, plus live Case009/download identity."""
    import importlib.util
    with tempfile.TemporaryDirectory(prefix='case009-frozen-docs-') as d:
        restored = Path(d) / 'restored'
        receipt = restore_frozen(root, restored)
        spec = importlib.util.spec_from_file_location('case009_immutable_verify', restored / PATHS[0] / 'publication/verify_publication.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return {'restoration': receipt, 'original_verifier': module.verify(restored)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root = a.root.resolve(); output = a.output.resolve()
    if output.is_relative_to(root):
        raise ValueError('external fresh scratch required')
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / 'restored'
    restoration = restore_frozen(root, scratch)
    case = Path(PATHS[0])
    command = [sys.executable, '-B', str(scratch / case / 'publication/run_checks.py'),
               '--root', str(scratch), '--output', str(output / 'checks')]
    result = subprocess.run(command, cwd=output)
    receipt = {'status': 'PASS' if result.returncode == 0 else 'FAIL', 'anchor': ANCHOR, **restoration,
               'Case009_current_bytes_match_anchor': True, 'historical_package_version': '0.1.1',
               'command': command, 'exit_code': result.returncode,
               'scope': 'Unmodified historical source/package/wheel checks; current package checked separately'}
    (output / 'restoration_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
