"""Fixed loopback workload, streamed request records, fresh server-process rounds.

Timing convention follows vLLM 0.29 serve: TTFT is first choice-bearing SSE event;
latency ends at the last such event; TPOT=(latency-TTFT)/(usage.output_tokens-1).
SSE events are NOT counted as generated tokens. No model download or weight write.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading
import urllib.request


def sha(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(x, indent=2, allow_nan=False) + '\n')


def make_requests(tokenizer_path, output):
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    rows = []
    for length in (128, 1024):
        for i in range(256):
            text = f'Work item {i}: Explain the following record in plain language. '
            text += ' '.join(f'Station {j} recorded {(i*13+j*17)%997} units on day {(j%28)+1}. Compare entries and describe their sequence.' for j in range(150))
            ids = t.encode(text, add_special_tokens=False)[:length]
            if len(ids) != length:
                raise ValueError('wrong token length')
            rows.append({'id': f'Q009-L{length}-{i:03}', 'input_length': length, 'token_ids': ids, 'prompt_hash': sha(ids)})
    write(output, rows)


def command(python, model, port, eager=False, profile_dir=None, kv=4*1024**3):
    cmd = [python, '-B', '-m', 'vllm.entrypoints.cli.main', 'serve', model,
           '--host', '127.0.0.1', '--port', str(port), '--served-model-name', 'diova-q',
           '--dtype', 'bfloat16', '--kv-cache-dtype', 'bfloat16', '--tensor-parallel-size', '1',
           '--max-model-len', '2048', '--max-num-seqs', '32', '--max-num-batched-tokens', '2048',
           '--kv-cache-memory-bytes', str(kv), '--gpu-memory-utilization', '0.87',
           '--no-enable-prefix-caching', '--no-enable-chunked-prefill', '--no-async-scheduling',
           '--attention-config', '{"backend":"FLASH_ATTN"}', '--generation-config', 'vllm', '--seed', '909220']
    if eager:
        cmd.append('--enforce-eager')
    if profile_dir:
        cmd += ['--profiler-config', json.dumps({'profiler': 'torch', 'torch_profiler_dir': str(profile_dir), 'torch_profiler_with_stack': False, 'torch_profiler_record_shapes': True, 'torch_profiler_with_memory': True})]
    return cmd


def environment(work):
    e = os.environ.copy()
    e.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1', VLLM_USE_V2_MODEL_RUNNER='0', VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_CACHE_ROOT=str(work/'runtime-cache'))
    return e


async def request(session, url, row, semaphore, metadata, output_tokens):
    async with semaphore:
        start = time.perf_counter()
        first = last = None
        usage = None
        finish = None
        token_ids = []
        events = 0
        error = None
        try:
            body = {'model': 'diova-q', 'prompt': row['token_ids'], 'max_tokens': output_tokens,
                    'temperature': 0, 'ignore_eos': True, 'stream': True, 'stream_options': {'include_usage': True}, 'return_token_ids': True}
            async with session.post(url+'/v1/completions', json=body) as response:
                if response.status != 200:
                    raise ValueError(f'HTTP {response.status}: ' + (await response.text())[:500])
                async for line in response.content:
                    line = line.strip()
                    if not line.startswith(b'data: '):
                        continue
                    content = line[6:]
                    if content == b'[DONE]':
                        continue
                    x = json.loads(content)
                    if x.get('choices'):
                        now = time.perf_counter()
                        first = now if first is None else first
                        last = now
                        choice = x['choices'][0]
                        events += 1
                        token_ids.extend(choice.get('token_ids') or [])
                        finish = choice.get('finish_reason') or finish
                    if x.get('usage'):
                        usage = x['usage']
            if first is None or not usage:
                raise ValueError('missing stream timestamps/usage')
            if usage['prompt_tokens'] != row['input_length'] or usage['completion_tokens'] != output_tokens:
                raise ValueError('unexpected actual token length')
            if len(token_ids) != output_tokens:
                raise ValueError('returned token-ID count disagrees with usage')
            if finish != 'length':
                raise ValueError('unexpected finish reason')
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        latency = last-start if last is not None else None
        ttft = first-start if first is not None else None
        return {**metadata, 'request_id': row['id'], 'prompt_hash': row['prompt_hash'],
                'input_tokens': usage.get('prompt_tokens') if usage else None,
                'output_tokens': usage.get('completion_tokens') if usage else None,
                'latency_seconds': latency, 'ttft_seconds': ttft,
                'tpot_seconds': (latency-ttft)/(usage['completion_tokens']-1) if error is None else None,
                'http_complete_seconds': time.perf_counter()-start,
                'stream_events': events, 'generated_ids_hash': sha(token_ids),
                'finish_reason': finish, 'status': 'OK' if error is None else 'ERROR', 'error': error}


async def block(url, rows, concurrency, metadata, output_tokens=256):
    import aiohttp
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=600)
    start = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        records = await asyncio.gather(*(request(session, url, r, sem, metadata, output_tokens) for r in rows))
    return {'elapsed_seconds': time.perf_counter()-start, 'records': records}


def metrics(url):
    with urllib.request.urlopen(url+'/metrics', timeout=10) as response:
        return response.read().decode()


def start_server(cmd, work, directory):
    directory.mkdir(parents=True, exist_ok=False)
    write(directory/'command.json', {'argv': cmd})
    log = (directory/'server.log').open('w')
    process = subprocess.Popen(cmd, env=environment(work), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process, log


def stop_server(process, log):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=20)
    log.close()


def ready(process, url):
    deadline = time.monotonic()+600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('server exited before readiness')
        try:
            with urllib.request.urlopen(url+'/health', timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError('server startup budget exceeded')



def telemetry(stop, path):
    """One-Hz whole-device samples, including desktop/background allocations."""
    with path.open('x') as f:
        while not stop.is_set():
            r = subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu,power.draw','--format=csv,noheader,nounits'], capture_output=True, text=True)
            f.write(json.dumps({'monotonic': time.perf_counter(), 'values': r.stdout.strip(), 'returncode': r.returncode})+'\n'); f.flush()
            stop.wait(1)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='mode',required=True)
    m=sub.add_parser('requests');m.add_argument('--tokenizer',required=True);m.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('sweep');s.add_argument('--private-config',type=Path,required=True);s.add_argument('--protocol',type=Path,required=True);s.add_argument('--work',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='requests':
        make_requests(a.tokenizer,a.output);return
    cfg=json.loads(a.private_config.read_text());protocol=json.loads(a.protocol.read_text())
    if protocol['status']!='FROZEN':raise ValueError('Protocol not frozen')
    rows=json.loads(Path(cfg['requests']).read_text())
    if sha(rows)!=protocol['request_manifest_hash']:raise ValueError('Request manifest changed')
    a.work.mkdir(parents=True,exist_ok=True)
    url='http://127.0.0.1:8099'
    for run in protocol['process_order']:
        name=f"round{run['round']}-{run['arm']}-{run['execution']}"
        d=a.work/name
        if (d/'complete.json').exists():
            print('PRESERVED',name,flush=True);continue
        if d.exists():raise RuntimeError('Existing failed/incomplete attempt requires diagnosis; no automatic retry: '+name)
        cmd=command(cfg['runtime_python'],cfg['models'][run['arm']],8099,run['execution']=='eager',kv=protocol['kv_bytes'])
        proc,log=start_server(cmd,a.work,d)
        monitor_stop=threading.Event()
        monitor=threading.Thread(target=telemetry,args=(monitor_stop,d/'telemetry.jsonl'))
        monitor.start()
        try:
            ready(proc,url)
            for condition in run['conditions']:
                length,c=condition;key=f'L{length}-C{c}'
                items=[r for r in rows if r['input_length']==length][:max(64,8*c)]
                meta={'round':run['round'],'arm':run['arm'],'execution':run['execution'],'condition':key,'client_concurrency':c,'process_id':name,'warmup':True}
                warm=asyncio.run(block(url,items[:2],min(c,2),meta))
                write(d/(key+'-warmup.json'),warm)
                if any(r['status']!='OK' for r in warm['records']):raise ValueError('Warmup validity failed')
                (d/(key+'-metrics-before.txt')).write_text(metrics(url))
                meta['warmup']=False
                result=asyncio.run(block(url,items,c,meta))
                write(d/(key+'.json'),result)
                (d/(key+'-metrics-after.txt')).write_text(metrics(url))
                print(name,key,'success',sum(r['status']=='OK' for r in result['records']),'/',len(items),flush=True)
                if any(r['status']!='OK' for r in result['records']):raise ValueError('Cell failed; retained without dropping requests')
            write(d/'complete.json',{'status':'COMPLETE','protocol_sha256':hashlib.sha256(a.protocol.read_bytes()).hexdigest()})
        except Exception as e:
            write(d/'failure.json',{'error':type(e).__name__+': '+str(e)})
            raise
        finally:
            monitor_stop.set(); monitor.join()
            stop_server(proc,log)


if __name__=='__main__':main()
