"""Registered P0 and 36 fixed technical workload blocks; no agent generation."""
import argparse
import asyncio
import importlib.metadata
import json
import os
import pathlib
import subprocess
import time
import traceback
from .common import load, save, sha_file, digest, utc
from .server import OwnedServer, rpc, metrics, gpu_sample, support
from .client import request


def freeze(root, runtime):
    paths = sorted(list((root / 'src').rglob('*.py')) + list((root / 'tests').rglob('*.py')) + list((root / 'specs').glob('*.json')) + list((root / 'data').glob('*.json')))
    rows = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': sha_file(p)} for p in paths]
    old = pathlib.Path(runtime['original_case'])
    for name in ('worker_inspection.py', 'profiler_extension.py', 'kernel_check.py', 'common.py'):
        path = old / 'src/kv_agent_bench' / name
        rows.append({'original_path': 'src/kv_agent_bench/' + name, 'bytes': path.stat().st_size, 'sha256': sha_file(path)})
    env = {name: importlib.metadata.version(name) for name in ('vllm', 'torch', 'flashinfer-python', 'transformers', 'tokenizers', 'httpx')}
    result = {'created_utc': utc(), 'files': rows, 'execution_source_sha256': digest(rows), 'dependencies': env,
              'git_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime['original_case'], text=True).strip(),
              'first_new_model_request_before_freeze': 0, 'runtime_private_excluded': True}
    save(root / 'provenance/execution_freeze.json', result)
    return result


def validate_model(root, runtime):
    original = load(pathlib.Path(runtime['original_case']) / 'provenance/model_identity.json')
    rows = []
    for row in original['files']:
        path = pathlib.Path(runtime['snapshot']) / row['file']
        current = {'file': row['file'], 'bytes': path.stat().st_size, 'sha256': sha_file(path)}
        if any(current[k] != row[k] for k in ('file', 'bytes', 'sha256')):
            raise AssertionError('Existing snapshot identity changed: ' + row['file'])
        rows.append(current)
    cfg = load(pathlib.Path(runtime['snapshot']) / 'config.json')
    if (cfg['num_hidden_layers'], cfg['num_key_value_heads'], cfg['head_dim']) != (36, 8, 128):
        raise AssertionError('Model geometry differs')
    save(root / 'provenance/model_identity.json', {'model': original['model'], 'revision': original['revision'], 'files': rows, 'local_full_file_hashes_match': True, 'D_origin_required': False, 'official_external_digest_verified': False})


async def batch(base, prompts, identities, concurrency, out_tokens, ignore_eos, out_dir, metrics_on=False):
    import httpx
    out_dir.mkdir(parents=True, exist_ok=False)
    if metrics_on:
        (out_dir / 'metrics_before.txt').write_text(await asyncio.to_thread(metrics, base))
        await asyncio.to_thread(rpc, base, 'technical_reset_peaks')
    start = time.monotonic()
    stop = asyncio.Event()
    samples = []
    async def sample():
        while not stop.is_set():
            row = {'elapsed': time.monotonic() - start}
            try:
                row['metrics'] = await asyncio.to_thread(metrics, base)
                row['gpu'] = await asyncio.to_thread(gpu_sample)
            except Exception as exc:
                row['error_type'] = type(exc).__name__
            samples.append(row)
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.5)
            except TimeoutError:
                pass
    sampler = asyncio.create_task(sample()) if metrics_on else None
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=8, max_keepalive_connections=8), trust_env=False) as client:
        sem = asyncio.Semaphore(concurrency)
        rows = await asyncio.gather(*(request(client, base, p, i, sem, start, out_tokens, ignore_eos) for p, i in zip(prompts, identities)))
    elapsed = time.monotonic() - start
    stop.set()
    if sampler:
        await sampler
    # Every response and unsuccessful attempt is retained before any gate fails.
    for i, row in enumerate(rows):
        save(out_dir / f'request_{i:02}.json', row)
    save(out_dir / 'block.json', {'identities': identities, 'concurrency': concurrency, 'queue_launch_utc_basis': 'elapsed monotonic offset from common eight-request launch',
                                 'makespan_seconds': max(r['terminal'] for r in rows), 'outer_batch_seconds': elapsed,
                                 'requests': len(rows), 'model_request_attempts': sum(r['model_request_attempts'] for r in rows),
                                 'responses_received': sum(r['responses_received'] for r in rows), 'status': 'COMPLETED' if all(r['status'] == 'SUCCESS' for r in rows) else 'PARTIAL_OR_INFRA',
                                 'metric_sampling_timed_overhead': metrics_on})
    if metrics_on:
        save(out_dir / 'samples.json', samples)
        (out_dir / 'metrics_after.txt').write_text(await asyncio.to_thread(metrics, base))
        save(out_dir / 'cache_after.json', await asyncio.to_thread(rpc, base, 'technical_inspect'))
        deadline = time.monotonic() + 30
        while True:
            text = await asyncio.to_thread(metrics, base)
            from .metrics import metric_scalar
            active = {k: metric_scalar(text, k) for k in ('vllm:num_requests_running', 'vllm:num_requests_waiting')}
            if all(v == 0 for v in active.values()):
                save(out_dir / 'drain.json', {'status': 'DRAINED', 'active': active})
                break
            if time.monotonic() > deadline:
                save(out_dir / 'drain.json', {'status': 'UNRESOLVED', 'active': active})
                raise RuntimeError('Drain unresolved; next workload blocked')
            await asyncio.sleep(0.5)
    return rows


def run(root, runtime):
    if (root / 'results/worker_receipt.json').exists():
        raise RuntimeError('Existing attempt receipt; explicit identity audit required before resume')
    validate_model(root, runtime)
    source = freeze(root, runtime)
    prompts = {p['prompt_id']: p for p in load(root / 'data/prompts.json')}
    planned = load(root / 'data/planned_blocks.json')
    start = time.monotonic()
    receipt = {'started_utc': utc(), 'execution_source_sha256': source['execution_source_sha256'], 'limit_seconds': 7200, 'attempts': [], 'status': 'RUNNING'}
    save(root / 'results/worker_started.json', receipt)
    def budget():
        left = 7200 - (time.monotonic() - start)
        if left < 30:
            raise TimeoutError('Cumulative worker budget exhausted')
        return left
    def use_server(name, kv, mode, action):
        budget()
        server = OwnedServer(runtime, name, kv, mode)
        a = {'name': name, 'kv': kv, 'mode': mode, 'started_utc': utc(), 'status': 'UNKNOWN'}
        t = time.monotonic()
        try:
            server.start(min(480, budget()))
            a['ready_seconds'] = server.ready_seconds
            before = rpc(server.base, 'technical_inspect')
            support(before, kv)
            save(root / 'results/servers' / name / 'cache_before.json', before)
            a['status'] = 'READY'
            action(server)
            a['status'] = 'COMPLETED'
        except Exception as exc:
            a.update(status='SUPPORT_OR_INFRA_FAILURE', error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            server.stop()
            a['wall_seconds'] = time.monotonic() - t
            receipt['attempts'].append(a)
            save(root / 'results/servers' / name / 'attempt.json', a)
            print(json.dumps(a), flush=True)
    try:
        # The existing fixed small test is reused without editing its tolerance.
        env = os.environ.copy()
        env.update(PYTHONPATH=runtime['original_case'] + '/src', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        python = pathlib.Path(runtime['python'])
        cuda = python.parent.parent / 'lib/python3.12/site-packages/nvidia/cu13'
        env.update(PATH=str(python.parent) + os.pathsep + env['PATH'], CUDA_HOME=str(cuda),
                   LIBRARY_PATH=str(python.parent.parent.parent / 'setup/jit_link_libs') + ':/usr/lib/wsl/lib',
                   LD_LIBRARY_PATH=str(cuda / 'lib') + ':/usr/lib/wsl/lib',
                   FLASHINFER_WORKSPACE_BASE=str(pathlib.Path(runtime['private_root']) / 'flashinfer_cache'))
        result = subprocess.run([runtime['python'], '-B', '-m', 'kv_agent_bench.kernel_check', '--output', str(root / 'results/kernel_check.json')], env=env, capture_output=True, text=True, timeout=min(300, budget()))
        (pathlib.Path(runtime['private_root']) / 'kernel_check_private.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError('Small fixed NVFP4 kernel test failed; private log retained')
        for kv in ('B16', 'F4'):
            def numeric(s, kv=kv):
                rpc(s.base, 'technical_finite_start')
                rpc(s.base, 'case016_profile_start')
                rows = asyncio.run(batch(s.base, [prompts['P-SHORT']], [{'phase': 'p0_numeric', 'kv': kv}], 1, 32, True, root / 'results/p0' / kv))
                profile = rpc(s.base, 'case016_profile_stop')
                finite = rpc(s.base, 'technical_finite_stop')
                save(root / 'results/p0' / kv / 'profile.json', profile)
                save(root / 'results/p0' / kv / 'finite.json', finite)
                if not finite['all36_finite'] or rows[0]['status'] != 'SUCCESS':
                    raise AssertionError('Actual attention finite/raw observation gate failed')
            use_server('p0_' + kv, kv, 'EAGER', numeric)
        graph = []
        for kv in ('B16', 'F4'):
            def graph_probe(s, kv=kv):
                rows = asyncio.run(batch(s.base, [prompts['P-SHORT']] * 4, [{'phase': 'p0_graph', 'kv': kv, 'slot': i} for i in range(4)], 4, 16, True, root / 'results/graph' / kv))
                if any(r['status'] != 'SUCCESS' for r in rows):
                    raise AssertionError('Graph support raw/token gate failed')
            try:
                use_server('graph_' + kv, kv, 'GRAPH', graph_probe)
                graph.append({'kv': kv, 'status': 'PASS'})
            except Exception as exc:
                graph.append({'kv': kv, 'status': 'FAIL', 'reason': type(exc).__name__})
        mode = 'GRAPH' if all(r['status'] == 'PASS' for r in graph) else 'EAGER'
        save(root / 'specs/final_configuration.json', {'created_utc': utc(), 'mode': mode, 'graph_support': graph, 'timed_requests_before_freeze': 0,
                                                     'common_configuration': True, 'selection_reason': 'Both graph gates passed' if mode == 'GRAPH' else 'Support failure; registered common eager fallback',
                                                     'execution_source_sha256': source['execution_source_sha256']})
        if mode == 'EAGER':
            save(root / 'specs/amendment01_common_eager.json', {'created_utc': utc(), 'cause': graph, 'timed_requests_before_amendment': 0, 'applies_to': ['B16', 'F4'], 'quality_or_speed_selection': False})
        for rnd, kvs in ((1, ('B16', 'F4')), (2, ('F4', 'B16')), (3, ('B16', 'F4'))):
            for kv in kvs:
                def measured(s, rnd=rnd, kv=kv):
                    for length in (8192, 16384):
                        budget()
                        rows = asyncio.run(batch(s.base, [prompts[f'B-L{length}-00']], [{'phase': 'warmup', 'round': rnd, 'kv': kv, 'length': length}], 1, 128, True, root / 'results/warmup' / f'R{rnd}_{kv}_{length}'))
                        if rows[0]['status'] != 'SUCCESS':
                            raise RuntimeError('Warmup request invalid')
                    for block in [b for b in planned if b['round'] == rnd and b['kv'] == kv]:
                        budget()
                        rows = asyncio.run(batch(s.base, [prompts[i] for i in block['prompt_ids']], [{'phase': 'timed', **{k: v for k, v in block.items() if k != 'prompt_ids'}, 'request_index': i} for i in range(8)], block['concurrency'], 128, True, root / 'results/blocks' / block['block_id'], True))
                        print(json.dumps({'completed_block': block['block_id'], 'successful': sum(r['status'] == 'SUCCESS' for r in rows)}), flush=True)
                        if any(r['status'] != 'SUCCESS' for r in rows):
                            raise RuntimeError('Observed infrastructure/raw mismatch; attempt retained, no quality retry')
                    if rnd == 1:
                        for length in (8192, 16384):
                            budget()
                            ids = [f'S-L{length}-{i:02}' for i in range(8)]
                            asyncio.run(batch(s.base, [prompts[i] for i in ids], [{'phase': 'sanity', 'kv': kv, 'length': length, 'request_index': i} for i in range(8)], 1, 64, False, root / 'results/sanity' / f'{kv}_{length}'))
                use_server(f'R{rnd}_{kv}', kv, mode, measured)
        receipt['status'] = 'COMPLETED'
    except Exception as exc:
        receipt.update(status='PARTIAL_OR_BLOCKED', error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        print(json.dumps({'worker_error': type(exc).__name__, 'message': str(exc)}), flush=True)
    finally:
        receipt['wall_seconds'] = time.monotonic() - start
        receipt['finished_utc'] = utc()
        save(root / 'results/worker_receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=pathlib.Path, required=True, help='Private runtime JSON; not published')
    args = parser.parse_args()
    runtime = load(args.runtime)
    result = run(pathlib.Path(runtime['technical_root']), runtime)
    raise SystemExit(0 if result['status'] == 'COMPLETED' else 2)


if __name__ == '__main__':
    main()
