"""Authorized GPU stage controller; CPU-only parent, isolated read-only worker."""
import argparse
import copy
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from state import CASE, P2, REPO, MeasuredSession, append, digest, inputs, read, sha, write


def freeze_gate():
    frozen = read(P2 / 'execution/source_freeze.json')
    binding = {k: frozen[k] for k in ['files', 'upstream_files', 'dependencies']}
    if digest(binding) != frozen['execution_source_sha256']:
        raise ValueError('execution freeze root hash differs')
    import hashlib
    manifest_blob = subprocess.check_output(['git', 'show', 'HEAD:' + (P2 / 'execution/source_freeze.json').relative_to(REPO).as_posix()], cwd=REPO)
    if hashlib.sha256(manifest_blob).hexdigest() != sha(P2 / 'execution/source_freeze.json'):
        raise ValueError('execution freeze itself must be committed')
    for rel, identity in frozen['upstream_files'].items():
        if (CASE / rel).stat().st_size != identity['bytes'] or sha(CASE / rel) != identity['sha256']:
            raise ValueError('original upstream evidence/source changed: ' + rel)
    for rel, identity in frozen['files'].items():
        path = P2 / rel
        if path.is_symlink() or path.stat().st_size != identity['bytes'] or sha(path) != identity['sha256']:
            raise ValueError('execution source/input freeze differs: ' + rel)
        blob = subprocess.check_output(['git', 'show', 'HEAD:' + path.relative_to(REPO).as_posix()], cwd=REPO)
        import hashlib
        if hashlib.sha256(blob).hexdigest() != identity['sha256']:
            raise ValueError('execution source must be committed before any response: ' + rel)
    cpu = read(P2 / 'execution/cpu_original_files.json')['files']
    for rel, identity in cpu.items():
        if (P2 / rel).stat().st_size != identity['bytes'] or sha(P2 / rel) != identity['sha256']:
            raise ValueError('original CPU preparation changed before GPU: ' + rel)
    return {'execution_source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            'execution_source_sha256': frozen['execution_source_sha256'],
            'generation_config_hash': digest(read(P2 / 'specs/protocol.json')['generation'])}


class Worker:
    def __init__(self, python, runtime, method, directory, deadline, journal):
        self.directory = directory; self.journal = journal
        self.log = (directory.parent / (directory.name + '.stderr.log')).open('x')
        command = [str(python), '-B', str(P2 / 'execution/gpu_worker.py'), '--runtime', str(runtime),
                   '--method', method, '--output', str(directory), '--worker-deadline', str(deadline)]
        env = os.environ.copy(); env.pop('PYTHONPATH', None)
        env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        self.process = subprocess.Popen(command, cwd=REPO, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1)
        ready = self.receive()
        if ready['kind'] != 'READY':
            raise RuntimeError('worker failed before READY')
        append(journal, {'event': 'WORKER_READY', 'method': method, 'gate': ready['gate']})

    def receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('GPU worker exited; see retained stderr and journal')
        return json.loads(line)

    def generate(self, payload, request_id, grammar, deadline, expected_hash=None):
        command = {'command': 'GENERATE', 'payload': payload, 'request_id': request_id, 'grammar': grammar,
                   'episode_deadline': deadline, 'expected_input_hash': expected_hash}
        self.process.stdin.write(json.dumps(command, ensure_ascii=False) + '\n'); self.process.stdin.flush()
        result = self.receive()
        if result['kind'] == 'ERROR':
            raise GPUError(result)
        if result['kind'] != 'RESPONSE' or result['response']['request_id'] != request_id:
            raise RuntimeError('GPU worker response identity differs')
        return result['response']

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"command":"STOP"}\n'); self.process.stdin.flush()
            self.process.wait(timeout=60)
        self.log.close()
        if self.process.returncode:
            raise RuntimeError('GPU worker exit code ' + str(self.process.returncode))


class GPUError(Exception):
    def __init__(self, record):
        self.record = record
        super().__init__(record['label'] + ': ' + record['message'])


def cell_id(method, point, arm):
    return method + '__' + point['checkpoint_id'] + '__' + arm


def run(args):
    start = time.monotonic()
    identity = freeze_gate()
    spec = read(P2 / 'execution/spec.json')
    if args.output.resolve().is_relative_to(REPO):
        raise ValueError('private worker output must be outside the Git tree')
    args.output.mkdir(parents=True, exist_ok=False)
    previous = []; previous_hashes = set()
    for receipt_path in args.prior_receipt:
        fingerprint = sha(receipt_path)
        if fingerprint in previous_hashes:
            raise ValueError('duplicate prior worker receipt')
        previous_hashes.add(fingerprint)
        value = read(receipt_path)
        if value.get('actual_new_model_request_attempts', 0) != 0:
            raise ValueError('this revised source may only resume pre-response infrastructure failures; do not mix generated source identities')
        previous.append({'sha256': fingerprint, 'worker_wall_seconds': value['total_worker_wall_seconds'],
                         'status': value['status'], 'execution_source_commit': value.get('execution_source_commit')})
    prior_wall = sum(r['worker_wall_seconds'] for r in previous)
    # Earlier uninstrumented source-validation overhead is conservatively charged,
    # separately from the measured wall. It never expands the registered cap.
    prior_overhead = 1.0 if previous else 0.0
    deadline = start + spec['worker_wall_limit_seconds'] - prior_wall - prior_overhead
    if time.monotonic() >= deadline:
        raise ValueError('cumulative worker budget already exhausted')
    journal = args.output / 'attempt_journal.jsonl'; journal.touch(exist_ok=False)
    request_journal = args.output / 'requests.jsonl'; request_journal.touch(exist_ok=False)
    write(args.output / 'identity.json', {**identity, 'execution_spec_sha256': sha(P2 / 'execution/spec.json'),
          'cpu_protocol_sha256': sha(P2 / 'specs/protocol.json'), 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'run_id': spec['execution_id'], 'worker_wall_limit_seconds': spec['worker_wall_limit_seconds'], 'smoke_model_requests': 0})
    write(args.output / 'prior_attempts.json', {'prior_receipts': previous, 'prior_worker_wall_seconds': prior_wall,
          'conservative_prior_source_validation_overhead_seconds': prior_overhead,
          'cumulative_cap_seconds': spec['worker_wall_limit_seconds']})
    # Reuse the byte-pinned CPU gates, now under the newly authorized execution.
    from json_recovery_p2.local_inputs import check
    from json_recovery_p2.tokenizer_cpu import retokenize, check_grammar
    # Optional CPU dependencies are checked in a separate existing Python process;
    # this controller remains stdlib-only and cannot import a model framework.
    runtime = read(args.runtime)
    manifest = read(Path(runtime['original_result_root']) / 'experiment_manifest.json')
    if Path(manifest['models']['4b']['path']).resolve() != Path(runtime['model']['path']).resolve():
        raise ValueError('runtime and original Worker snapshot differ')
    preflight_cmd = [str(args.gpu_python), '-B', str(P2 / 'execution/preflight.py'), '--runtime', str(args.runtime), '--output', str(args.output / 'preflight')]
    result = subprocess.run(preflight_cmd, capture_output=True, text=True, cwd=REPO, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    (args.output / 'preflight.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('CPU preflight failed; no model response generated')
    points, snapshots, scenarios, historical = inputs()
    c0_sessions = {}; reproduction = []; rows = []; request_counter = 0
    planned = [{'cell_id': cell_id(m, p, a), 'method': m, 'checkpoint_id': p['checkpoint_id'], 'arm': a,
                'status': 'NOT_RUN', 'trigger': 'new eligible C0 malformed-only' if a in ('N', 'E') else 'registered'}
               for m in spec['method_order'] for p in points for a in ('C0', 'G', 'N', 'E')]
    write(args.output / 'planned_cells.json', planned)

    def session(method, point, arm):
        obj = MeasuredSession(point, snapshots[point['checkpoint_id']], scenarios[point['scenario_id']],
                              args.output / 'fixtures' / cell_id(method, point, arm), method=method, condition=arm)
        obj.active_boundary = time.monotonic()
        return obj

    def step(worker, obj):
        nonlocal request_counter
        if time.monotonic() >= deadline:
            raise GPUError({'label': 'WORKER_BUDGET', 'message': 'cumulative registered wall cap', 'kind': 'ERROR'})
        payload, sidecar = obj.next_request()
        host_since_boundary = time.monotonic() - obj.active_boundary
        remaining = obj.budget.before_request() - host_since_boundary
        if remaining <= 0:
            obj.terminal = 'TIMEOUT'; return
        request_counter += 1
        if request_counter > 408:
            raise RuntimeError('absolute registered request bound exceeded')
        rid = 'p2-r1-request-' + format(request_counter, '04d')
        append(journal, {'event': 'REQUEST_STARTED', 'request_id': rid, 'method': obj.method, 'arm': obj.condition,
            'checkpoint_id': obj.checkpoint['checkpoint_id'], 'model_calls_used': obj.budget.model_calls,
            'tool_attempts_used': obj.budget.tool_attempts, 'messages_tools_sha256': digest(payload),
            'common_observation_sha256': sidecar['common_observation_sha256'], 'state_sha256': sidecar['state_sha256'],
            'grammar_active': sidecar['grammar_active']})
        try:
            gpu = worker.generate(payload, rid, sidecar['grammar_active'], time.monotonic() + remaining,
                    obj.checkpoint['recorded_input_token_sha256'] if obj.budget.model_calls == 2 else None)
        except GPUError as e:
            append(journal, {'event': 'REQUEST_ERROR', 'request_id': rid, 'error': e.record})
            obj.observation_status = 'UNOBSERVED_INFRA'; obj.infra_error = e.record; obj.terminal = e.record['label']
            raise
        write(args.output / 'gpu_response_receipts' / (rid + '.json'), gpu)
        elapsed = time.monotonic() - obj.active_boundary
        response = obj.consume_measured(gpu, elapsed_worker_seconds=elapsed)
        obj.active_boundary = time.monotonic()
        response['method'] = obj.method
        append(request_journal, {'checkpoint_id': obj.checkpoint['checkpoint_id'], 'arm': obj.condition, **response})
        append(journal, {'event': 'REQUEST_COMPLETED', 'request_id': rid, 'output_tokens': gpu['output_tokens']})

    def complete(worker, obj):
        failed = None
        try:
            while obj.terminal is None:
                step(worker, obj)
        except GPUError as error:
            failed = error
        row = obj.result(identity)
        write(args.output / 'cells' / (cell_id(obj.method, obj.checkpoint, obj.condition) + '.json'), row)
        rows.append(row)
        print(json.dumps({'phase': obj.condition, 'method': obj.method, 'checkpoint': obj.checkpoint['checkpoint_id'],
                          'success': row['VERIFIED_WORKFLOW_SUCCESS'], 'new_logical_requests': len(row['responses'])}), flush=True)
        if failed:
            raise failed

    error = None
    try:
        # All original C0 requests are generated before any recovery intervention.
        for method in spec['method_order']:
            append(journal, {'event': 'WORKER_LOAD_STARTED', 'phase': 'C0', 'method': method})
            worker = Worker(args.gpu_python, args.runtime, method, args.output / ('C0-' + method), deadline, journal)
            try:
                for point in points:
                    obj = session(method, point, 'C0'); step(worker, obj)
                    if not obj.responses:
                        raise RuntimeError('C0 focal response unavailable')
                    focal = obj.responses[0]; old = historical[(point['checkpoint_id'], method)]
                    same = focal['output_token_ids'] == old['output_token_ids']
                    match = {'method': method, 'checkpoint_id': point['checkpoint_id'],
                        'input_hash_match': focal['input_token_sha256'] == old['input_token_sha256'],
                        'output_ids_match': same, 'historical_raw_sha256': old['raw_response_sha256'],
                        'new_raw_sha256': focal['raw_utf8_sha256'], 'request_id': focal['request_id']}
                    reproduction.append(match)
                    write(args.output / 'focal_C0' / (cell_id(method, point, 'C0') + '.json'), focal)
                    complete(worker, obj); c0_sessions[(method, point['checkpoint_id'])] = obj
            finally:
                worker.close()
        write(args.output / 'reproduction.json', {'status': 'PASS' if len(reproduction) == 24 and all(r['input_hash_match'] and r['output_ids_match'] for r in reproduction) else 'REPRODUCTION_BLOCKED', 'rows': reproduction})
        gate = read(args.output / 'reproduction.json')
        if gate['status'] != 'PASS':
            raise GPUError({'label': 'REPRODUCTION_BLOCKED', 'message': 'C0 focal IDs differ; recovery interventions not run', 'kind': 'ERROR'})
        for method in spec['method_order']:
            append(journal, {'event': 'WORKER_LOAD_STARTED', 'phase': 'RECOVERY', 'method': method})
            worker = Worker(args.gpu_python, args.runtime, method, args.output / ('RECOVERY-' + method), deadline, journal)
            try:
                for point in points:
                    obj = session(method, point, 'G'); complete(worker, obj)
                    base = c0_sessions[(method, point['checkpoint_id'])]
                    for arm in ('N', 'E'):
                        if base.terminal != 'PARSE_ERROR' or not base.c0_observation:
                            write(args.output / 'not_applicable' / (cell_id(method, point, arm) + '.json'),
                                  {'cell_id': cell_id(method, point, arm), 'status': 'NOT_APPLICABLE', 'reason': 'actual new C0 was not malformed-only'})
                            continue
                        try:
                            branch = base.fork(arm, args.output / 'fixtures' / cell_id(method, point, arm))
                        except ValueError as e:
                            write(args.output / 'not_applicable' / (cell_id(method, point, arm) + '.json'),
                                  {'cell_id': cell_id(method, point, arm), 'status': 'NOT_APPLICABLE', 'reason': 'C0 malformed but ineligible remaining budget', 'detail': str(e)})
                            continue
                        branch.active_boundary = time.monotonic()
                        complete(worker, branch)
            finally:
                worker.close()
    except Exception as e:
        error = getattr(e, 'record', {'kind': 'ERROR', 'label': 'INFRA_ERROR', 'type': type(e).__name__, 'message': str(e)})
        append(journal, {'event': 'RUN_STOPPED', 'error': error})
        print(json.dumps({'RUN_STOPPED': error}), flush=True)
    wall = time.monotonic() - start
    write(args.output / 'receipt.json', {**identity, 'status': 'COMPLETED' if error is None else (error.get('label', 'PARTIAL')),
          'error': error, 'completed_cells': len(rows), 'C0_reproduction_rows': len(reproduction),
          'actual_new_model_request_attempts': request_counter, 'smoke_model_requests': 0,
          'total_worker_wall_seconds': wall, 'worker_wall_limit_seconds': spec['worker_wall_limit_seconds'],
          'prior_measured_worker_wall_seconds': prior_wall,
          'cumulative_measured_worker_wall_seconds': wall + prior_wall,
          'budget_debited_worker_wall_seconds': wall + prior_wall + prior_overhead,
          'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'model_downloads': 0, 'solver_calls': 0, 'encode_calls': 0, 'artifact_writes': 0})
    return 0 if error is None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True); parser.add_argument('--gpu-python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-receipt', action='append', type=Path, default=[], help='debit preserved pre-response infrastructure attempts from the same cumulative cap')
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
