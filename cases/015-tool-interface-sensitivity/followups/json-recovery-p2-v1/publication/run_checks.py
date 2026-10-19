"""Model-free P2 publication checks with original P1/A1/CPU closed views."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from verify import CASE, P2, P2_REL, restore, verify, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--tests', action='store_true')
    parser.add_argument('--grammar-python', help='existing CPU dependency Python; no installation by this command')
    args = parser.parse_args(); output = args.output.resolve()
    if output.exists() or output.is_relative_to(CASE.resolve()):
        raise ValueError('choose a fresh external output')
    output.mkdir(parents=True)
    preservation = verify()
    legacy = output / 'legacy_case015_view'; restore(CASE, legacy)
    cpu = output / 'cpu_execution_view'; restore(CASE, cpu, include_cpu=True)
    cpup2 = cpu / P2_REL
    commands = [('old-P1-A1-publication', sys.executable, legacy,
                 [str(legacy / 'publication/scripts/run_checks.py'), '--output', str(output / 'legacy')] + (['--tests'] if args.tests else []), None)]
    if args.tests:
        commands += [('cpu20', sys.executable, cpu, ['-m', 'unittest', 'discover', '-s', str(cpup2 / 'tests'), '-p', 'test_cpu_contracts.py', '-v'], cpup2 / 'src'),
                     ('gpu-boundaries9', sys.executable, cpu, ['-m', 'unittest', 'discover', '-s', str(cpup2 / 'execution/tests'), '-v'], cpup2 / 'src'),
                     ('publication-tests', sys.executable, CASE, ['-m', 'unittest', 'discover', '-s', str(P2 / 'publication/tests'), '-v'], None)]
        if args.grammar_python:
            commands.append(('grammar5', args.grammar_python, cpu, ['-m', 'unittest', 'discover', '-s', str(cpup2 / 'tests'), '-p', 'test_grammar_contracts.py', '-v'], cpup2 / 'src'))
    commands.append(('new-raw-audit', sys.executable, CASE,
                     [str(P2 / 'execution/audit.py'), '--results', str(P2 / 'results/gpu_r2'), '--output', str(output / 'p2-audit')], None))
    commands.append(('regenerate-public-tables', sys.executable, CASE,
                     [str(P2 / 'publication/derive.py'), '--results', str(P2 / 'results/gpu_r2'), '--output', str(output / 'derived')], None))
    receipts = []
    for name, python, cwd, command, source in commands:
        env = os.environ.copy(); env.pop('PYTHONPATH', None)
        env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        if source:env['PYTHONPATH'] = str(source)
        started = time.monotonic(); result = subprocess.run([python, '-B', *command], cwd=cwd, env=env, capture_output=True, text=True)
        log = output / (name + '.log'); log.write_text(result.stdout + result.stderr)
        receipts.append({'check': name, 'exit_code': result.returncode, 'wall_seconds': time.monotonic() - started, 'log': name + '.log', **identity(log)})
        print(name, result.returncode, flush=True)
        if result.returncode:break
    passed = len(receipts) == len(commands) and all(r['exit_code'] == 0 for r in receipts)
    if passed:
        recomputed = json.loads((output / 'p2-audit/summary.json').read_text())
        stored = json.loads((P2 / 'results/gpu_r2/analysis/summary.json').read_text())
        if recomputed != stored:
            raise ValueError('raw-derived P2 summary differs from stored display')
        if json.loads((output / 'derived/decision.json').read_text()) != json.loads((P2 / 'results/gpu_r2/decision.json').read_text()):
            raise ValueError('raw-derived P2 decision differs')
        for name in ['method_api_arm.csv', 'paired_outcomes.csv', 'C0_reproduction.csv', 'request_costs.csv', 'display.json']:
            if (output / 'derived' / name).read_bytes() != (P2 / 'results/gpu_r2/tables' / name).read_bytes():
                raise ValueError('raw-derived public table differs: ' + name)
    value = {'status': 'PASS' if passed else 'FAIL', 'preservation': preservation, 'checks': receipts,
             'optional_grammar_tests': 'RUN' if args.grammar_python else 'NOT_RUN; use --grammar-python with existing dependencies',
             'new_GPU_model_forwards_in_checks': 0, 'scope': 'CPU raw/fixture replay and scalar display; no GPU logits reproduction',
             'older_regressions_counted_separately': True}
    (output / 'verification.json').write_text(json.dumps(value, indent=2) + '\n')
    return 0 if passed else 1


if __name__ == '__main__':raise SystemExit(main())
