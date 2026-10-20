"""Optional P0-only profiler through the official worker extension interface."""
from .worker_inspection import CacheInspector

class ProfileInspector(CacheInspector):
    def case016_profile_start(self):
        import torch
        self._case016_profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False)
        self._case016_profile.start();return {'started':True,'scope':'P0 only'}
    def case016_profile_stop(self):
        import torch
        from collections import Counter
        torch.cuda.synchronize();self._case016_profile.stop()
        names=Counter(e.name for e in self._case016_profile.events() if str(e.device_type).endswith('CUDA'))
        rows=[{'kernel':name,'calls':count}for name,count in sorted(names.items())]
        del self._case016_profile
        return {'gpu_kernels':rows,'unique_gpu_kernels':len(rows),'attention_kernels':[r for r in rows if any(s in r['kernel'].lower()for s in ['prefill','decode','attention','append','cache'])],'model_parameters_modified':False}
