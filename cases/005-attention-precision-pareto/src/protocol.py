"""Two immutable local freezes and runtime guards."""
import json,subprocess
from pathlib import Path
from .common import CASE,sha,fingerprint

def verify_phase(phase):
 p=CASE/f'configs/phase_{phase}.json';spec=json.loads(p.read_text());digest=sha(p)
 assert digest==(CASE/f'configs/phase_{phase}.sha256').read_text().strip()
 for name,h in spec['frozen_files'].items():assert sha(CASE/name)==h,name
 if phase=='b':assert sha(CASE/'configs/phase_a.json')==spec['phase_a_sha256']
 return spec,digest

def gpu_guard():
 import torch
 # WSL may not expose every Windows process; preserve telemetry as well.
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if apps:raise RuntimeError('GPU_BUSY: another compute process is active')
 if not torch.cuda.is_available():raise RuntimeError('GPU_NOT_AVAILABLE')
 free,total=torch.cuda.mem_get_info()
 if free<4*2**30:raise RuntimeError('GPU_BUSY: less than 4 GiB available')
 torch.cuda.set_per_process_memory_fraction(13*2**30/total)
 return {'free_bytes':free,'total_bytes':total,'external_compute_processes_before_init':apps or 'none reported'}

def configure():
 import torch
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False

def current_environment():
 import torch
 from .runtime import environment
 e=environment();e['reference_settings']={n:getattr(torch.backends.cuda.matmul,n) for n in ['allow_tf32','allow_fp16_reduced_precision_reduction','allow_bf16_reduced_precision_reduction']};e['float32_matmul_precision']=torch.get_float32_matmul_precision();return e
