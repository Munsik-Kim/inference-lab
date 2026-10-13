"""Keep Case011's full historical package ledger intact in external scratch."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

CASE = 'cases/011-frozen-state-readout-adaptation'


def restore(root, scratch):
    root, scratch = Path(root).resolve(), Path(scratch).resolve()
    spec = importlib.util.spec_from_file_location('case009_frozen_dependency', Path(__file__).parent / 'run_case009_frozen_checks.py')
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    old.restore_frozen(root, scratch)
    expected = json.loads((root / CASE / 'publication/protected_inventory.json').read_text())['files']
    changed_package = []
    for name, digest in expected.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('unsafe historical protected path')
        source, target = root / relative, scratch / relative
        if source.is_symlink() or not source.is_file():
            raise ValueError('missing/unsafe live protected path: ' + name)
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if name.startswith('packages/diova-compare/'):
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError('frozen package does not match Case011 ledger')
            if actual != digest:
                changed_package.append(name)
        else:
            if actual != digest:
                raise ValueError('protected scientific/support file changed: ' + name)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    shutil.copytree(root / CASE, scratch / CASE, ignore=shutil.ignore_patterns('__pycache__'))
    return {'protected_files': len(expected), 'current_scientific_support_files_match': True,
            'historical_package_version': '0.1.1', 'evolving_package_changes': changed_package,
            'scope': 'Original package bytes restored; every other protected live file still matches its original hash'}


def verify_current_case011(root):
    with tempfile.TemporaryDirectory(prefix='case011-frozen-docs-') as d:
        scratch = Path(d) / 'restored'
        receipt = restore(root, scratch)
        command = [sys.executable, '-B', str(scratch / CASE / 'publication/verify_publication.py'), '--require-full-repository']
        process = subprocess.run(command, capture_output=True, text=True, cwd=scratch)
        if process.returncode:
            raise ValueError('unchanged Case011 verifier failed: ' + process.stderr[-2000:])
        return {'restoration': receipt, 'original_verifier': json.loads(process.stdout)}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--synthetic-only', action='store_true')
    a = p.parse_args(); out = a.output.resolve()
    if out.is_relative_to(a.root.resolve()):
        raise ValueError('fresh external output required')
    out.mkdir(parents=True, exist_ok=False)
    receipt = restore(a.root, out / 'frozen')
    command = [sys.executable, '-B', str(out / 'frozen' / CASE / 'publication/run_checks.py'),
               '--require-full-repository', '--output', str(out / 'checks')]
    if a.synthetic_only:
        command.append('--synthetic-only')
    process = subprocess.run(command, cwd=out)
    receipt.update(status='PASS' if process.returncode == 0 else 'FAIL', command=command, exit_code=process.returncode)
    (out / 'restoration_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    raise SystemExit(process.returncode)
