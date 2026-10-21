"""Launch only owned, isolated loopback vLLM processes; no environment writes.

Adapted from Case016 nvfp4_technical.server.OwnedServer. This manager intentionally
omits native tool parsing; all arms use the same content JSON contract.
"""
import json
import os
import pathlib
import signal
import socket
import subprocess
import time
import urllib.parse
import urllib.request


def write_once(path, data):
    path = pathlib.Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def rpc(base, method):
    from .client import loopback
    request = urllib.request.Request(loopback(base) + '/collective_rpc',
            data=json.dumps({'method': method, 'timeout': 90}).encode(),
            headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=100) as response:
        return json.load(response)['results'][0]


def metrics(base):
    from .client import loopback
    with urllib.request.urlopen(loopback(base) + '/metrics', timeout=10) as response:
        return response.read().decode()


def gpu_sample():
    fields = 'timestamp,name,memory.used,memory.free,temperature.gpu,clocks.sm,power.draw,utilization.gpu'
    result = subprocess.run(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=5)
    return {'fields': fields, 'csv': result.stdout.strip(), 'exit_code': result.returncode}


class OwnedServer:
    def __init__(self, runtime, kv, name, mode='GRAPH'):
        if kv not in ('K0', 'K1') or mode not in ('GRAPH', 'EAGER'):
            raise ValueError('Registered K0/K1 and common GRAPH/EAGER mode required')
        self.runtime, self.kv, self.mode = runtime, kv, mode
        self.port = int(runtime.get('port', 8173))
        if not 1024 <= self.port <= 65535 or self.port == 8152:
            raise ValueError('A separate nonprivileged Case017 port is required')
        if pathlib.Path(name).name != name or name in ('.', '..'):
            raise ValueError('Server attempt name must be one local component')
        self.root = pathlib.Path(runtime['private_root']) / 'servers' / name
        self.base = f'http://127.0.0.1:{self.port}'
        self.proc = self.log = None

    def command(self):
        r = self.runtime
        return [r['python'], '-B', '-m', 'vllm.entrypoints.openai.api_server',
                '--model', r['snapshot'], '--tokenizer', r['snapshot'],
                '--served-model-name', 'case017-qwen3-4b', '--host', '127.0.0.1',
                '--port', str(self.port), '--dtype', 'bfloat16', '--tensor-parallel-size', '1',
                '--max-model-len', str(r.get('max_model_len', 28672)), '--max-num-seqs', '4',
                '--max-num-batched-tokens', '2048', '--kv-cache-memory-bytes', str(r.get('pool_bytes', 4294967296)),
                '--gpu-memory-utilization', '0.85', '--kv-cache-dtype', {'K0': 'auto', 'K1': 'nvfp4'}[self.kv],
                '--attention-backend', 'FLASHINFER', '--enable-chunked-prefill', '--no-enable-prefix-caching',
                '--no-async-scheduling', '--generation-config', 'vllm',
                '--structured-outputs-config', '{"backend":"xgrammar"}',
                '--worker-extension-cls', 'agent_contract_cost.p0_inspector.ContractInspector',
                '--seed', str(r.get('seed', 6102007))] + (['--enforce-eager'] if self.mode == 'EAGER' else [])

    def public_command(self):
        cmd = self.command()
        replacements = {self.runtime['python']: '<EXISTING_VLLM_PYTHON>', self.runtime['snapshot']: '<PINNED_LOCAL_SNAPSHOT>'}
        return [replacements.get(v, v) for v in cmd]

    def environment(self):
        r, env = self.runtime, os.environ.copy()
        python = pathlib.Path(r['python'])
        envroot = python.parent.parent
        cuda = envroot / 'lib/python3.12/site-packages/nvidia/cu13'
        sources = [str(pathlib.Path(r['case_root']) / 'src'),
                   str(pathlib.Path(r['original_case']) / 'followups/nvfp4-technical-r1/src'),
                   str(pathlib.Path(r['original_case']) / 'src')]
        env.update({'PATH': str(python.parent) + os.pathsep + env['PATH'], 'PYTHONPATH': os.pathsep.join(sources),
                    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                    'PYTHONDONTWRITEBYTECODE': '1', 'VLLM_SERVER_DEV_MODE': '1', 'NVIDIA_TF32_OVERRIDE': '0',
                    'VLLM_NO_USAGE_STATS': '1', 'DO_NOT_TRACK': '1',
                    'VLLM_CACHE_ROOT': str(pathlib.Path(r['private_root']) / 'engine_cache'),
                    'FLASHINFER_WORKSPACE_BASE': str(pathlib.Path(r['private_root']) / 'flashinfer_cache'),
                    'CUDA_HOME': str(cuda),
                    'LIBRARY_PATH': str(envroot.parent / 'setup/jit_link_libs') + ':/usr/lib/wsl/lib:' + env.get('LIBRARY_PATH', ''),
                    'LD_LIBRARY_PATH': str(cuda / 'lib') + ':/usr/lib/wsl/lib:' + env.get('LD_LIBRARY_PATH', '')})
        return env

    def start(self, ready_seconds=600):
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1', self.port)) == 0:
                raise RuntimeError('Occupied port: refusing to attach or stop its owner')
        for key in ('python', 'snapshot', 'case_root', 'original_case'):
            if not pathlib.Path(self.runtime[key]).exists():
                raise FileNotFoundError('Required existing runtime input missing: ' + key)
        if self.runtime.get('max_model_len', 28672) < 24576:
            raise ValueError('Server cannot support registered client context')
        self.root.mkdir(parents=True, exist_ok=False)
        self.log = (self.root / 'server.log').open('xb')
        cmd = self.command()
        self.started = time.monotonic()
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     env=self.environment(), start_new_session=True)
        write_once(self.root / 'launch_private.json', {'command': cmd, 'pid': self.proc.pid,
                   'owned_process_group': self.proc.pid, 'prelaunch_gpu': gpu_sample(), 'wall_started': time.time()})
        while time.monotonic() - self.started < ready_seconds:
            if self.proc.poll() is not None:
                raise RuntimeError(f'Owned server exited {self.proc.returncode}; attempt log retained')
            try:
                with urllib.request.urlopen(self.base + '/health', timeout=2) as response:
                    if response.status == 200:
                        self.ready_seconds = time.monotonic() - self.started
                        return self
            except (OSError, TimeoutError):
                pass
            time.sleep(1)
        raise TimeoutError('Server readiness deadline; not a model quality failure')

    def stop(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            if os.getpgid(self.proc.pid) != self.proc.pid:
                raise RuntimeError('Owned process group identity changed; refusing signal')
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=45)
            except subprocess.TimeoutExpired as exc:
                write_once(self.root / 'shutdown.json', {'status': 'NOT_DRAINED', 'owned_pid': self.proc.pid})
                raise RuntimeError('Owned server did not stop normally; no forced kill') from exc
        self.log.close()
        write_once(self.root / 'shutdown.json', {'status': 'STOPPED', 'exit_code': self.proc.returncode,
                                               'owned_pid': self.proc.pid})
