"""Bounded sequential launch; preserves all process failures and attempts."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--method', choices=('BF16', 'RTN', 'D16', 'D4'), required=True)
    p.add_argument('--attempt', type=int, default=1)
    args = p.parse_args()
    record_path = args.result / 'gpu_commands.json'
    previous = json.loads(record_path.read_text()) if record_path.exists() else []
    reg = json.loads((args.result / 'preregistration.json').read_text())
    remaining = reg['worker_wall_budget_seconds'] - sum(x['wall_seconds'] for x in previous)
    if remaining < 5:
        raise RuntimeError('budget exhausted')
    if (args.result / 'workers' / args.method / ('attempt' + str(args.attempt).zfill(2))).exists():
        raise FileExistsError('existing worker attempt')
    cmd = [sys.executable, '-B', str(Path(__file__).parent / 'runner.py'), '--result', str(args.result),
           '--method', args.method, '--attempt', str(args.attempt), '--budget-seconds', str(remaining)]
    log = args.result / 'attempts' / (args.method + '-' + str(args.attempt).zfill(2) + '.log')
    env = os.environ.copy(); env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    start = time.monotonic(); date = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with log.open('x') as f:
        try:
            child = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                   cwd=Path(__file__).parent, timeout=remaining)
            code = child.returncode
        except subprocess.TimeoutExpired:
            code = -9
    receipt = {'method': args.method, 'attempt': args.attempt, 'command': cmd,
        'started_utc': date, 'exit_code': code, 'wall_seconds': time.monotonic() - start,
        'log': str(log.relative_to(args.result)), 'worker_budget_at_launch_seconds': remaining}
    previous.append(receipt)
    temp = record_path.with_suffix('.partial'); temp.write_text(json.dumps(previous, indent=2) + '\n'); temp.replace(record_path)
    print(json.dumps(receipt)); raise SystemExit(0 if code == 0 else 1)


if __name__ == '__main__':
    main()
