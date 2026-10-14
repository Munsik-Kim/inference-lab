"""Sequential owned worker launch and conservative wall-budget accounting."""
import argparse
import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from contracts import atomic, read, sha, METHODS


def launch(result, stage, method, attempt):
    result = Path(result)
    records = read(result / 'gpu_commands.json') if (result / 'gpu_commands.json').exists() else []
    remaining = 7200 - sum(r['worker_wall_seconds'] for r in records)
    if remaining <= 1:
        raise SystemExit('registered worker budget exhausted')
    out = result / 'workers' / stage / method / ('attempt' + str(attempt).zfill(2))
    if out.exists():
        raise SystemExit('attempt already exists; preserve it')
    worker = Path(__file__).resolve().parent / 'worker.py'
    registration = read(result / 'preregistration.json')
    actual_hashes = {p.name: sha(p) for p in worker.parent.glob('*.py')}
    for name, digest in registration['followup_code_hashes'].items():
        if actual_hashes[name] != digest:
            raise SystemExit('registered code changed; an explicit amendment is required: ' + name)
    command = [sys.executable, '-B', str(worker), '--result', str(result), '--stage', stage,
               '--method', method, '--worker-budget-seconds', str(remaining), '--attempt', str(attempt)]
    log = result / 'attempts' / (stage + '-' + method + '-' + str(attempt).zfill(2) + '.log')
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    start = time.monotonic()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with log.open('x') as stream:
        try:
            p = subprocess.run(command, cwd=worker.parent, env=env, stdout=stream, stderr=subprocess.STDOUT,
                               timeout=remaining)
            exit_code, status = p.returncode, 'COMPLETED_PROCESS'
        except subprocess.TimeoutExpired:
            exit_code, status = -9, 'BUDGET_TIMEOUT_OWNED_WORKER_KILLED'
    receipt = {'command': command, 'cwd': str(worker.parent), 'stage': stage, 'method': method,
               'attempt': attempt, 'started_utc': started_utc, 'exit_code': exit_code, 'status': status,
               'worker_wall_seconds': time.monotonic() - start, 'log': str(log.relative_to(result)),
               'code_hashes': actual_hashes, 'budget_remaining_at_launch_seconds': remaining,
               'scope': 'load/import/artifact read/serialization/idle included; not kernel busy time'}
    records.append(receipt)
    atomic(result / 'gpu_commands.json', records)
    print(receipt)
    raise SystemExit(0 if exit_code == 0 else 1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--stage', choices=('first', 'common', 'p2'), required=True)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--attempt', type=int, choices=(1, 2), default=1)
    a = p.parse_args()
    launch(a.result, a.stage, a.method, a.attempt)
