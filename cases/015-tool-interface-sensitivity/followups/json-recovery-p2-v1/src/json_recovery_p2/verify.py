"""Verify original bytes, registered data and raw replay, never a model result."""
import shutil
import subprocess
import sys
import re
from pathlib import Path, PurePosixPath
from .evidence import derive
from .io import read, sha, write
from .upstream import CASE, ROOT, digest


def originals():
    manifest = read(ROOT / 'provenance/original_case_files.json')
    for name, entry in manifest['files'].items():
        p = CASE / name
        if p.is_symlink() or not p.is_file() or p.stat().st_size != entry['bytes'] or sha(p) != entry['sha256']:
            raise ValueError('original Case015 file changed: ' + name)
    return {'status': 'PASS', 'original_case_files_unchanged': len(manifest['files'])}


def validate():
    preserved = originals()
    source = read(ROOT / 'provenance/source_manifest.json')
    for name, entry in source['files'].items():
        if sha(CASE / name) != entry['sha256']:
            raise ValueError('pinned source changed: ' + name)
    frozen = read(ROOT / 'provenance/code_freeze.json')
    for name, entry in frozen['files'].items():
        p = ROOT / name
        if p.is_symlink() or not p.is_file() or p.stat().st_size != entry['bytes'] or sha(p) != entry['sha256']:
            raise ValueError('P2 code/spec/data freeze changed: ' + name)
    checkpoints = read(ROOT / 'data/checkpoints.json')
    snapshots = read(ROOT / 'data/model_request_snapshots.json')
    if len(checkpoints) != 8 or len(snapshots) != 8 or len({c['checkpoint_id'] for c in checkpoints}) != 8:
        raise ValueError('missing/duplicate checkpoint')
    by_id = {s['checkpoint_id']: s for s in snapshots}
    if len(by_id) != 8:
        raise ValueError('duplicate request snapshot')
    for point in checkpoints:
        if digest(by_id[point['checkpoint_id']]) != point['model_input_snapshot_sha256']:
            raise ValueError('checkpoint/request mismatch')
    cells = read(ROOT / 'results/cpu_preparation/recovery_results.json')
    expected = {(c['checkpoint_id'], m, arm) for c in checkpoints for m in ('BF16', 'RTN', 'D16') for arm in ('C0', 'G', 'N', 'E')}
    actual = {(c['checkpoint_id'], c['method'], c['condition']) for c in cells}
    if len(cells) != 96 or actual != expected or any(c['status'] != 'NOT_RUN' or c['functional_success'] is not None for c in cells):
        raise ValueError('CPU preparation must not contain recovery measurements')
    protocol = read(ROOT / 'specs/protocol.json')
    if protocol['new_model_inference_authorized'] or protocol['remote_publication_authorized']:
        raise ValueError('CPU-only scope changed')
    return {'status': 'PASS', **preserved, 'checkpoint_count': 8, 'recovery_cells': '96 NOT_RUN, not 96 measured episodes',
            'GPU_execution': 'CLOSED_SCOPE', 'new_model_forwards': 0}


def audit(output):
    output = Path(output)
    value = validate()
    regenerated = derive(output)
    for name in ['checkpoints.json', 'model_request_snapshots.json', 'historical_continuations.json']:
        if read(output / name) != read(ROOT / 'data' / name):
            raise ValueError('raw-derived data differs: ' + name)
    for name in ['cpu_diagnostic_summary.json', 'malformed_slots.json']:
        if read(output / name) != read(ROOT / 'results/cpu_preparation' / name):
            raise ValueError('raw-derived diagnostics differ: ' + name)
    write(output / 'verification.json', {'status': 'PASS', 'preservation': value, 'raw_replay': regenerated, 'new_model_forwards': 0})
    return regenerated


def baseline(output, tests=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    originals()
    view = output / 'original_case015_view'; view.mkdir()
    manifest = read(ROOT / 'provenance/original_case_files.json')
    for name, entry in manifest['files'].items():
        if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('unsafe preserved path')
        dest = view / name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CASE / name, dest)
    command = [sys.executable, '-B', str(view / 'publication/scripts/run_checks.py'), '--output', str(output / 'receipt')]
    if tests:
        command.append('--tests')
    result = subprocess.run(command, capture_output=True, text=True)
    (output / 'baseline.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise ValueError('preserved baseline checks failed; see baseline.log')
    value = read(output / 'receipt/verification.json')
    test_runs = {}
    if tests:
        for suite in ['p1-tests', 'a1-tests', 'publication-tests']:
            log = (output / 'receipt' / (suite + '.log')).read_text()
            names = re.findall(r'^test_\w+ \(([^)]+)\) \.\.\. ok$', log, re.MULTILINE)
            count = re.search(r'^Ran (\d+) tests? in ', log, re.MULTILINE)
            if not count or int(count.group(1)) != len(names) or len(set(names)) != len(names):
                raise ValueError('test count/name receipt mismatch: ' + suite)
            test_runs[suite] = {'unique_test_count': len(names), 'test_names': names}
    write(output / 'verification.json', {'status': 'PASS', 'original_closed_view': len(manifest['files']),
        'test_runs': test_runs, 'counts_derived_from_executed_test_logs': True,
        'original_manifest_was_not_relaxed': True, 'receipt': value})
    return value

