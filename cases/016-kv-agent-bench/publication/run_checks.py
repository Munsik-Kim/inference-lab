"""Fresh-output CPU route: preserve, restore, raw replay and regenerate displays."""
import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
from verify import verify, restore, extract_baseline
from derive import project

CASE = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--tests', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(CASE):
        raise ValueError('Choose a fresh external output directory')
    output.mkdir(parents=True)
    started = time.monotonic()
    verified = verify(CASE)
    restore(CASE, output / 'agent-original')
    baseline = extract_baseline(CASE, output / 'baseline-view')
    technical = CASE / 'followups/nvfp4-technical-r1'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    commands = [('historical-source-views', [sys.executable, '-B', str(output / 'agent-original/scripts/verify_sources.py'), '--output', str(output / 'source-views')], None),
                ('historical-baseline-replay', [sys.executable, '-B', str(baseline / 'src/verify.py'), '--output', str(output / 'baseline-replay')], str(baseline / 'src')),
                ('technical-raw-replay', [sys.executable, '-B', '-m', 'nvfp4_technical.audit', '--root', str(technical), '--output', str(output / 'technical-replay')], str(technical / 'src')),
                ('figures-from-csv', [sys.executable, '-B', '-m', 'nvfp4_technical.plots', '--analysis', str(output / 'technical-replay'), '--output', str(output / 'figures')], str(technical / 'src'))]
    if args.tests:
        commands.append(('technical-contract-tests', [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(technical / 'tests'), '-v'], str(technical / 'src')))
    receipts = []
    for label, command, pythonpath in commands:
        child = env.copy()
        if pythonpath:
            child['PYTHONPATH'] = pythonpath
        t = time.monotonic()
        with (output / (label + '.log')).open('w') as stream:
            result = subprocess.run(command, env=child, stdout=stream, stderr=subprocess.STDOUT)
        receipts.append({'check': label, 'exit_code': result.returncode, 'wall_seconds': time.monotonic() - t})
        if result.returncode:
            raise RuntimeError('CPU check failed: ' + label)
    from request_contracts import check
    contract = check(CASE)
    (output / 'request_contracts.json').write_text(json.dumps(contract, indent=2) + '\n')
    data = project(CASE)
    for figure in (technical / 'figures').glob('*.svg'):
        assert figure.read_bytes() == (output / 'figures' / figure.name).read_bytes(), 'Figure differs from public CSV: ' + figure.name
    expected = json.loads((CASE / 'publication/summary/project.json').read_text())
    assert data == expected, 'Raw to display projection differs'
    (output / 'project.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    assert not {'torch', 'transformers', 'vllm'} & set(sys.modules)
    receipt = {'status': 'PASS', 'preservation': verified, 'commands': receipts, 'display_exactly_regenerated': True,
               'new_gpu_requests': 0, 'scope': 'CPU scalar/SSE/fixture replay only. No GPU inference reproduction.', 'wall_seconds': time.monotonic() - started}
    (output / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
