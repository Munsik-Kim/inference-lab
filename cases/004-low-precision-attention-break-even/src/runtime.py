"""Small GPU runtime helpers; imported only by explicit GPU commands."""
import importlib.metadata,json,os,platform,subprocess,threading,time
from pathlib import Path
from .common import fingerprint,sha


def environment():
    import torch
    import transformers
    from transformers.models.qwen3 import modeling_qwen3
    versions={n:importlib.metadata.version(n) for n in ['torch','triton','transformers','numpy']}
    try:versions['sageattention']=importlib.metadata.version('sageattention')
    except importlib.metadata.PackageNotFoundError:versions['sageattention']=None
    driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()
    binaries={}
    try:
        import sageattention
        for p in Path(sageattention.__file__).parent.glob('*.so'):binaries[p.name]=sha(p)
        core_sha=sha(Path(sageattention.__file__).parent/'core.py')
    except ImportError:core_sha=None
    return dict(gpu=torch.cuda.get_device_name(),sm=list(torch.cuda.get_device_capability()),vram_bytes=torch.cuda.get_device_properties(0).total_memory,driver=driver,os=platform.system(),kernel=platform.release(),wsl='microsoft' in platform.release().lower(),python=platform.python_version(),versions=versions,torch_runtime=torch.__version__,cuda_build=torch.version.cuda,sage_binary_sha256=binaries,sage_core_sha256=core_sha,qwen_modeling_sha256=sha(Path(modeling_qwen3.__file__)),tf32=False,compile=False,cuda_graph=False)


class Sampler:
    def __init__(self):
        self.samples=[];self.stop_event=threading.Event();self.error=None
        try:
            import pynvml
            self.nv=pynvml;pynvml.nvmlInit();self.handle=pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:self.error=type(e).__name__;self.nv=None
    def sample(self):
        if not self.nv:return None
        try:
            n=self.nv;h=self.handle;m=n.nvmlDeviceGetMemoryInfo(h);u=n.nvmlDeviceGetUtilizationRates(h)
            x=dict(monotonic=time.monotonic(),device_used_bytes=m.used,free_bytes=m.free,gpu_utilization=u.gpu,temperature_c=n.nvmlDeviceGetTemperature(h,n.NVML_TEMPERATURE_GPU),graphics_clock_mhz=n.nvmlDeviceGetClockInfo(h,n.NVML_CLOCK_GRAPHICS))
            self.samples.append(x);return x
        except Exception as e:self.error=type(e).__name__;return None
    def start(self):
        self.sample()
        def run():
            while not self.stop_event.wait(.1):self.sample()
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self
    def stop(self):
        self.stop_event.set();self.thread.join();self.sample()
        return dict(interval_seconds=.1,scope='whole-device sampled peak; WSL/Windows background use included; not process allocation',error=self.error,samples=self.samples,peak_bytes=max((s['device_used_bytes'] for s in self.samples),default=None))


def profile_call(call):
    import torch
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        call();torch.cuda.synchronize()
    ops=sorted(set(e.name for e in p.events() if 'scaled_dot' in e.name or 'sageattention' in e.name))
    kernels=sorted(set(e.name for e in p.events() if str(e.device_type).endswith('CUDA')))
    return dict(ops=ops,cuda_kernels=kernels,fused_bf16=any('flash' in n or 'cudnn' in n or 'efficient' in n for n in ops) and not any('math' in n for n in ops) and bool(kernels),low_precision=any('sageattention' in n for n in ops) and any('attn' in n.lower() for n in kernels))
