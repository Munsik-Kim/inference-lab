"""Owned loopback servers. No attachment to or termination of other services."""
import json
import os
import pathlib
import signal
import socket
import subprocess
import time
import urllib.request
from .common import save, utc


def rpc(base, method):
    request = urllib.request.Request(base + '/collective_rpc', data=json.dumps({'method': method, 'timeout': 90}).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=100) as response:
        return json.load(response)['results'][0]


def metrics(base):
    with urllib.request.urlopen(base + '/metrics', timeout=10) as response:
        return response.read().decode()


def gpu_sample():
    fields = 'timestamp,memory.used,memory.free,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu'
    result = subprocess.run(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
    return {'utc': utc(), 'fields': fields, 'csv': result.stdout.strip(), 'exit_code': result.returncode}


class OwnedServer:
    def __init__(self, runtime, name, kv, mode):
        self.runtime = runtime
        self.kv, self.mode = kv, mode
        self.root = pathlib.Path(runtime['private_root']) / 'servers' / name
        self.proc = None
        self.port = runtime['port']
        self.base = f'http://127.0.0.1:{self.port}'

    def command(self):
        r = self.runtime
        cmd = [r['python'], '-B', '-m', 'vllm.entrypoints.openai.api_server', '--model', r['snapshot'], '--tokenizer', r['snapshot'],
               '--served-model-name', 'case016-tech-qwen3-4b', '--host', '127.0.0.1', '--port', str(self.port), '--dtype', 'bfloat16',
               '--tensor-parallel-size', '1', '--max-model-len', '28672', '--max-num-seqs', '4', '--max-num-batched-tokens', '2048',
               '--kv-cache-memory-bytes', '4294967296', '--gpu-memory-utilization', '0.85', '--kv-cache-dtype', {'B16': 'auto', 'F4': 'nvfp4'}[self.kv],
               '--attention-backend', 'FLASHINFER', '--enable-chunked-prefill', '--no-enable-prefix-caching', '--no-async-scheduling',
               '--generation-config', 'vllm', '--worker-extension-cls', 'nvfp4_technical.worker.TechnicalInspector', '--seed', '6102002']
        if self.mode == 'EAGER':
            cmd.append('--enforce-eager')
        return cmd

    def start(self, ready_seconds=480):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', self.port)) == 0:
                raise RuntimeError('Occupied port: refusing to attach or stop its owner')
        self.root.mkdir(parents=True, exist_ok=False)
        self.log = (self.root / 'server.log').open('wb')
        python = pathlib.Path(self.runtime['python'])
        env = os.environ.copy()
        old_private = python.parent.parent.parent
        cuda = python.parent.parent / 'lib/python3.12/site-packages/nvidia/cu13'
        env.update({'PATH': str(python.parent) + os.pathsep + env['PATH'],
                    'PYTHONPATH': self.runtime['technical_root'] + '/src:' + self.runtime['original_case'] + '/src',
                    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                    'PYTHONDONTWRITEBYTECODE': '1', 'VLLM_SERVER_DEV_MODE': '1', 'NVIDIA_TF32_OVERRIDE': '0',
                    'VLLM_CACHE_ROOT': str(pathlib.Path(self.runtime['private_root']) / 'engine_cache'),
                    'FLASHINFER_WORKSPACE_BASE': str(pathlib.Path(self.runtime['private_root']) / 'flashinfer_cache'),
                    'CUDA_HOME': str(cuda),
                    'LIBRARY_PATH': str(old_private / 'setup/jit_link_libs') + ':/usr/lib/wsl/lib:' + env.get('LIBRARY_PATH', ''),
                    'LD_LIBRARY_PATH': str(cuda / 'lib') + ':/usr/lib/wsl/lib:' + env.get('LD_LIBRARY_PATH', '')})
        self.started = time.monotonic()
        cmd = self.command()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        save(self.root / 'launch_private.json', {'command': cmd, 'pid': self.proc.pid, 'utc': utc(), 'prelaunch_gpu': gpu_sample()})
        while time.monotonic() - self.started < ready_seconds:
            if self.proc.poll() is not None:
                raise RuntimeError(f'Owned server exited {self.proc.returncode}; private attempt log retained')
            try:
                with urllib.request.urlopen(self.base + '/health', timeout=2) as response:
                    if response.status == 200:
                        self.ready_seconds = time.monotonic() - self.started
                        return
            except Exception:
                pass
            time.sleep(1)
        raise TimeoutError('Readiness deadline; not a model-quality failure')

    def stop(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=45)
            except subprocess.TimeoutExpired as exc:
                save(self.root / 'shutdown.json', {'status': 'NOT_DRAINED', 'owned_pid': self.proc.pid})
                raise RuntimeError('Owned server did not stop normally; port reuse blocked') from exc
        self.log.close()
        save(self.root / 'shutdown.json', {'status': 'STOPPED', 'exit_code': self.proc.returncode, 'owned_pid': self.proc.pid})


def support(receipt, kv):
    if receipt['cuda_capability'] != [12, 0] or len(receipt['layers']) != 36:
        raise AssertionError('SM120 and all36 actual attention layers required')
    if receipt['cache_config']['enable_prefix_caching']:
        raise AssertionError('Prefix caching must be off')
    expected = 'torch.bfloat16' if kv == 'B16' else 'torch.uint8'
    if not receipt['cache_tensors'] or any(t['dtype'] != expected for t in receipt['cache_tensors']):
        raise AssertionError('Actual storage dtype differs from registered KV')
    if kv == 'F4':
        if len(receipt['nvfp4_views']) != 36 or not all(r['is_kvcache_nvfp4'] and r['nvfp4_fa2'] for r in receipt['layers']):
            raise AssertionError('NVFP4 FA2 compressed-cache path absent')
        if any(v['scale_dtype'] != ['torch.float8_e4m3fn'] * 2 for v in receipt['nvfp4_views']):
            raise AssertionError('Actual FP8 block scales absent')
    return True
