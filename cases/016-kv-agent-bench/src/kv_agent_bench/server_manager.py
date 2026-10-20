"""Start/stop only our loopback process group; never attach to an occupied port."""
import os
import pathlib
import signal
import socket
import subprocess
import time
import urllib.request
from .common import save,utc

class Server:
    def __init__(self,python,snapshot,kv,max_len,pool_bytes,log_root,source_dir,port=8152,extension='kv_agent_bench.worker_inspection.CacheInspector',log_requests=False):
        self.python=str(python);self.snapshot=str(snapshot);self.kv=kv;self.max_len=max_len;self.pool_bytes=pool_bytes
        self.log_root=pathlib.Path(log_root);self.source_dir=str(source_dir);self.port=port;self.proc=None;self.extension=extension;self.log_requests=log_requests
    @property
    def base(self):return f'http://127.0.0.1:{self.port}'
    def start(self,ready_seconds=900):
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1',self.port))==0:raise RuntimeError('port is occupied; refusing to touch existing service')
        self.log_root.mkdir(parents=True,exist_ok=False);self.log=(self.log_root/'server.log').open('wb');started=time.monotonic()
        self.cmd=[self.python,'-m','vllm.entrypoints.openai.api_server','--model',self.snapshot,'--tokenizer',self.snapshot,
          '--served-model-name','case016-qwen3-4b','--host','127.0.0.1','--port',str(self.port),'--dtype','bfloat16',
          '--tensor-parallel-size','1','--max-model-len',str(self.max_len),'--max-num-seqs','4','--max-num-batched-tokens','2048',
          '--kv-cache-memory-bytes',str(self.pool_bytes),'--gpu-memory-utilization','0.85','--kv-cache-dtype',self.kv,'--attention-backend','FLASHINFER',
          '--enable-chunked-prefill','--no-enable-prefix-caching','--no-async-scheduling','--enforce-eager',
          '--enable-auto-tool-choice','--tool-call-parser','hermes','--generation-config','vllm',
          '--worker-extension-cls',self.extension,'--seed','6101902']
        if self.log_requests:self.cmd.append('--enable-log-requests')
        env=os.environ.copy();env.update({'PATH':str(pathlib.Path(self.python).parent)+os.pathsep+env['PATH'],
          'PYTHONPATH':self.source_dir,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1',
          'PYTHONDONTWRITEBYTECODE':'1','VLLM_SERVER_DEV_MODE':'1','NVIDIA_TF32_OVERRIDE':'0',
          'VLLM_ENFORCE_STRICT_TOOL_CALLING':'0',
          'VLLM_CACHE_ROOT':str(self.log_root.parent.parent/'engine_cache'),'FLASHINFER_WORKSPACE_BASE':str(self.log_root.parent.parent/'flashinfer_cache'),
          'CUDA_HOME':str(pathlib.Path(self.python).parent.parent/'lib/python3.12/site-packages/nvidia/cu13')})
        private_root=pathlib.Path(self.python).parent.parent.parent
        env['LIBRARY_PATH']=str(private_root/'setup/jit_link_libs')+':/usr/lib/wsl/lib:'+env.get('LIBRARY_PATH','')
        env['LD_LIBRARY_PATH']=str(pathlib.Path(self.python).parent.parent/'lib/python3.12/site-packages/nvidia/cu13/lib')+':/usr/lib/wsl/lib:'+env.get('LD_LIBRARY_PATH','')
        # All new JIT/cache writes are inside Case016 private scratch.
        self.proc=subprocess.Popen(self.cmd,stdout=self.log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        save(self.log_root/'launch_private.json',{'command':self.cmd,'pid':self.proc.pid,'started_utc':utc(),'max_ready_seconds':ready_seconds})
        while time.monotonic()-started<ready_seconds:
            if self.proc.poll() is not None:raise RuntimeError(f'server exited {self.proc.returncode}; see attempt log')
            try:
                with urllib.request.urlopen(self.base+'/health',timeout=2) as r:
                    if r.status==200:self.ready_seconds=time.monotonic()-started;return
            except Exception:pass
            time.sleep(1)
        raise TimeoutError('server readiness deadline')
    def stop(self):
        if self.proc is None:return
        if self.proc.poll() is None:
            os.killpg(self.proc.pid,signal.SIGTERM)
            try:self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # No forced killing. Mark the owned group unresolved and block reuse.
                save(self.log_root/'shutdown.json',{'status':'NOT_DRAINED','owned_pid':self.proc.pid});raise RuntimeError('owned server did not terminate normally')
        self.log.close();save(self.log_root/'shutdown.json',{'status':'STOPPED','exit_code':self.proc.returncode,'owned_pid':self.proc.pid})
