"""Freeze a fresh mitigation experiment and record immutable input identities."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def inventory(root, private=False):
    output = {}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        st = p.stat()
        output[str(p.relative_to(root))] = {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns} if private and p.suffix in ('.rpi4', '.npy', '.npz') else {'bytes': st.st_size, 'sha256': sha(p)}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.result.exists():
        raise FileExistsError('use a fresh result suffix')
    args.result.mkdir(parents=True)
    (args.result / 'attempts').mkdir()
    old = args.research / 'results/diova_agent_followup_r1'
    runtime = json.loads((old / 'runtime_manifest.json').read_text())
    write(args.result / 'runtime_manifest.json', runtime)
    original = Path(runtime['original_result_root'])
    inputs = {name: {'sha256': sha(original / 'inputs' / name), 'bytes': (original / 'inputs' / name).stat().st_size}
              for name in ('agent_tasks.json', 'agent_initial_messages.json', 'tool_schema.json')}
    write(args.result / 'input_manifest.json', inputs)
    registration = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'phase': 'MITIGATION_DEV: same eight previously observed DEV tasks, no formal or held-out claims',
        'methods': {'BF16': ['BASELINE'], 'RTN': ['BASELINE'],
                    'D16': ['BASELINE', 'FEEDBACK', 'ENUM', 'BOTH'],
                    'D4': ['BASELINE', 'FEEDBACK', 'ENUM', 'BOTH']},
        'primary': 'D16 BOTH minus newly measured D16 BASELINE; paired task success and unregistered name events',
        'secondary': 'D4 and factorial feedback/enum effects; all eight tasks and all conditions retained',
        'feedback': 'separate unknown-name vs invalid-arguments errors, allowed exact names, no automatic aliases',
        'enum': 'only top-level tool-call name values masked to schema enum; arguments/final answers unrestricted; no forced calls',
        'gold_not_available_to_decoder': True,
        'generation': {'greedy': True, 'max_new_tokens': 192, 'max_model_calls': 8,
            'max_tool_calls': 6, 'deadline_seconds': 900, 'context_limit': 24576, 'seed': 610132,
            'within_request_KV': True, 'prefix_cache_across_turns': False},
        'arithmetic': 'original BF16 reference, TF32 false, BF16 reduced precision reduction false, SDPA',
        'task_order': 'original task order, once per condition; no prompt/rubric/quantizer selection',
        'conditions_order': ['BASELINE', 'FEEDBACK', 'ENUM', 'BOTH'],
        'worker_wall_budget_seconds': 3600,
        'runtime_sha256': sha(args.result / 'runtime_manifest.json'),
        'input_manifest_sha256': sha(args.result / 'input_manifest.json'),
        'code_hashes': {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'new_fitting': 0, 'new_downloads': 0, 'formal_serving_benefit': 'NOT_ESTABLISHED',
        'completion': 'all scheduled episodes, original scorer, CPU event/config/paired audit; report residual failures without retries to improve scores'}
    write(args.result / 'preregistration.json', registration)
    protected = {'cases': inventory(root / 'cases'), 'original_r2': inventory(original, private=True),
                 'prior_followup': inventory(old, private=True), 'original_source': inventory(args.research / 'source'),
                 'frozen_followup_source': inventory(root / 'research/agent_stability_20261013')}
    write(args.result / 'protected_before.json', protected)
    identity = {'repo': str(root), 'research': str(args.research), 'result': str(args.result),
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip(),
        'new_GPU_experiment_authorized_by': 'user: 문제를 해결해서 실험 완료해줘', 'remote_work': 0}
    write(args.result / 'identity.json', identity)
    print(json.dumps({'status': 'REGISTERED', 'result': str(args.result), 'protected_case_files': len(protected['cases'])}))


if __name__ == '__main__':
    main()
