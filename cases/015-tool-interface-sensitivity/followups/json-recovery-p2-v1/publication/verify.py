"""Separate P2 public manifest and exact restoration of older closed views."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess

P2 = Path(__file__).resolve().parents[1]
CASE = P2.parents[1]
P2_REL = P2.relative_to(CASE).as_posix()
MANIFEST = P2_REL + '/publication/PUBLIC_MANIFEST.json'
MAP = P2_REL + '/publication/preservation_map.json'
GUIDES = {'README.ko.md', 'README.md', 'REPORT.ko.md', 'REPORT.md', 'REPRODUCE.ko.md', 'REPRODUCE.md'} | {
    P2_REL + '/' + name for name in ['README.ko.md', 'REPORT.ko.md', 'REPRODUCE.ko.md', 'HANDOFF.ko.md']}


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def safe(case, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError('unsafe retained path')
    path = Path(case) / name
    if any(p.is_symlink() for p in [path, *path.parents] if p == Path(case) or p.is_relative_to(Path(case))) or not path.is_file():
        raise ValueError('missing/symlink retained file: ' + name)
    return path


def retained(case, original):
    case = Path(case)
    mapping = read(case / MAP)['files'] if (case / MAP).exists() else {}
    if not set(mapping) <= GUIDES:
        raise ValueError('preservation map may only relocate the explicit guide allowlist')
    entry = mapping.get(original)
    path = safe(case, entry['retained_path'] if entry else original)
    if entry and identity(path) != {k: entry[k] for k in ['bytes', 'sha256']}:
        raise ValueError('retained guide bytes differ: ' + original)
    return path


def preserve(case=CASE):
    case = Path(case); p2 = case / P2_REL
    original = read(p2 / 'provenance/original_case_files.json')['files']
    for name, expected in original.items():
        if identity(retained(case, name)) != expected:
            raise ValueError('original P1/A1/publication file changed: ' + name)
    cpu = read(p2 / 'execution/cpu_original_files.json')['files']
    for name, expected in cpu.items():
        if identity(retained(case, P2_REL + '/' + name)) != expected:
            raise ValueError('original CPU preparation file changed: ' + name)
    execution = read(p2 / 'execution/source_freeze.json')
    for name, expected in execution['files'].items():
        if identity(safe(case, P2_REL + '/' + name)) != expected:
            raise ValueError('frozen GPU execution source/input changed: ' + name)
    git_checked = False
    repo = case.parents[1]
    detected = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=case, capture_output=True, text=True)
    if detected.returncode == 0 and Path(detected.stdout.strip()).resolve() == repo.resolve():
        groups = [(read(p2 / 'provenance/original_case_files.json')['base_commit'], original),
                  (read(case / MAP)['CPU_original_commit'], {P2_REL + '/' + k: v for k, v in cpu.items()}),
                  (read(p2 / 'results/gpu_r2/identity.json')['execution_source_commit'], {P2_REL + '/' + k: v for k, v in execution['files'].items()})]
        for commit, entries in groups:
            tree = subprocess.check_output(['git', 'ls-tree', '-rz', commit, '--', case.relative_to(repo).as_posix()], cwd=repo)
            objects = {}
            for line in tree.split(b'\0'):
                if not line:continue
                info, name = line.split(b'\t', 1)
                objects[name.decode()] = info.split()[2].decode()
            for name in entries:
                data = retained(case, name).read_bytes()
                blob_sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
                full = case.relative_to(repo).as_posix() + '/' + name
                if objects.get(full) != blob_sha:
                    raise ValueError('retained/frozen bytes differ from source commit: ' + commit + ':' + name)
        git_checked = True
    return {'status': 'PASS', 'original_case015_files': len(original), 'original_CPU_preparation_files': len(cpu),
            'intentional_guide_changes': sorted(read(case / MAP)['files']) if (case / MAP).exists() else [],
            'old_manifests_modified': False, 'Git_source_blobs_checked': git_checked,
            'execution_source_commit': read(p2 / 'results/gpu_r2/identity.json')['execution_source_commit'] if (p2 / 'results/gpu_r2/identity.json').exists() else None}


def restore(case, output, include_cpu=False):
    case = Path(case); output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    preserve(case)
    p2 = case / P2_REL
    files = read(p2 / 'provenance/original_case_files.json')['files']
    if include_cpu:
        files = {**files, **{P2_REL + '/' + k: v for k, v in read(p2 / 'execution/cpu_original_files.json')['files'].items()}}
        for name, entry in read(p2 / 'execution/source_freeze.json')['files'].items():
            files[P2_REL + '/' + name] = entry
        # Source freeze is self-excluded, so copy it explicitly.
        files[P2_REL + '/execution/source_freeze.json'] = identity(p2 / 'execution/source_freeze.json')
    for name, expected in files.items():
        source = retained(case, name)
        if identity(source) != expected:
            raise ValueError('restoration identity differs: ' + name)
        dest = output / name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    return {'status': 'PASS', 'restored_files': len(files), 'include_CPU_preparation_and_execution': include_cpu}


def verify(case=CASE):
    case = Path(case); preservation = preserve(case)
    p2 = case / P2_REL
    protocol = read(p2 / 'specs/protocol.json')
    checkpoints = {c['checkpoint_id']: c for c in read(p2 / 'data/checkpoints.json')}
    expected_cells = {method + '__' + checkpoint + '__' + arm for method in protocol['methods'] for checkpoint in checkpoints for arm in ('C0', 'G', 'N', 'E')}
    observed = set()
    execution_commit = read(p2 / 'results/gpu_r2/identity.json')['execution_source_commit']
    for path in (p2 / 'results/gpu_r2/cells').glob('*.json'):
        row = read(path)
        cell = row['method'] + '__' + row['checkpoint_id'] + '__' + row['arm']
        if cell not in expected_cells or cell in observed or path.stem != cell:
            raise ValueError('unregistered/duplicate measured cell')
        point = checkpoints[row['checkpoint_id']]
        if row['api_version'] != point['api_version'] or row['scenario_id'] != point['scenario_id'] or row['execution_source_commit'] != execution_commit:
            raise ValueError('cell metadata/source differs from registration')
        observed.add(cell)
    not_applicable = set()
    for path in (p2 / 'results/gpu_r2/not_applicable').glob('*.json'):
        row = read(path); cell = row['cell_id']
        if cell not in expected_cells or cell in observed | not_applicable or row['status'] != 'NOT_APPLICABLE':
            raise ValueError('invalid inapplicable cell')
        not_applicable.add(cell)
    if observed | not_applicable != expected_cells:
        raise ValueError('missing registered cell status')
    manifest = read(case / MANIFEST)
    actual = {p.relative_to(case).as_posix() for p in case.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.relative_to(case).as_posix() != MANIFEST}
    if actual != set(manifest['files']):
        raise ValueError('P2 public file set differs: missing=' + str(sorted(set(manifest['files']) - actual)) + ', extra=' + str(sorted(actual - set(manifest['files']))))
    for name, expected in manifest['files'].items():
        path = safe(case, name)
        if identity(path) != expected:
            raise ValueError('P2 public bytes differ: ' + name)
        if path.suffix.lower() in {'.safetensors', '.rpi4', '.pt', '.pth', '.npy', '.npz', '.bin'}:
            raise ValueError('private tensor/model payload in public tree')
    return {**preservation, 'public_files': len(actual), 'measured_cells': len(observed), 'not_applicable_cells': len(not_applicable),
            'scope': 'P2 followup plus retained older public views; not GPU inference reproduction'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); value = verify()
    with args.output.open('x') as f:json.dump(value, f, indent=2);f.write('\n')
    print(json.dumps(value))
