"""Faithful Euler write-back adapter; derived from pinned MIT diffusion.py.
Original vendored files remain byte-identical. No timestep argument is added.
"""
import sys, time, hashlib
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor'))
from looped_dit.pipeline import load_model, TextEncoder, to_pil, generate
from looped_dit.diffusion import euler_sample

def tensor_sha(x):
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

def make_noise(seed,device,dtype,size=512):
    g=torch.Generator(device=device).manual_seed(seed)
    return torch.randn(1,3,size,size,device=device,dtype=dtype,generator=g)*2.0

@torch.no_grad()
def sample_from_noise(model,text,mask,noise,steps,loops,diagnostic=False):
    x=noise; device=text.device; b=text.shape[0]
    ts=torch.linspace(0.,1.,steps+1,device=device)
    null_mask=torch.zeros_like(mask); was_training=model.training; model.eval()
    trace=[]
    try:
        for i in range(steps):
            t0,t1=ts[i],ts[i+1]; t=torch.full((b,),float(t0),device=device)
            with torch.autocast('cuda',dtype=text.dtype,enabled=x.is_cuda and text.dtype in (torch.bfloat16,torch.float16)):
                pred=model(x,text,mask,num_loops=loops)
                uncond=model(x,text,null_mask,num_loops=loops)
                pred=uncond+(pred-uncond)*6.0
            v=(pred-x)/(1.0-t[:,None,None,None]).clamp_min(.05)
            x=x+v*(t1-t0)
            if diagnostic:
                good=bool(torch.isfinite(pred).all() and torch.isfinite(x).all())
                trace.append({'step':i,'state_dtype':str(x.dtype),'finite':good})
                if not good: raise FloatingPointError('Nonfinite sampler step '+str(i))
        return x.clamp(-1,1),x,trace
    finally: model.train(was_training)

class Engine:
    def __init__(self,checkpoint,encoder):
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
        self.device=torch.device('cuda:0')
        start=time.perf_counter()
        self.model,self.config=load_model(checkpoint,self.device,torch.bfloat16,'ema')
        if self.model.num_loops!=4 or (self.model.pre,self.model.core,self.model.post)!=(6,5,6) or self.model.patch_size!=32:
            raise ValueError('Unexpected B32 architecture')
        self.text_encoder=TextEncoder(str(encoder),self.config.prompt_length,self.device)
        torch.cuda.synchronize(); self.load_seconds=time.perf_counter()-start
        self.identity={'device':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'torch':torch.__version__,
            'cuda':torch.version.cuda,'encoder_dtype':str(next(self.text_encoder.model.parameters()).dtype),
            'denoiser_dtype':str(next(self.model.parameters()).dtype),'prompt_length':self.config.prompt_length,
            'model_parameters':sum(p.numel() for p in self.model.parameters()),'encoder_parameters':sum(p.numel() for p in self.text_encoder.model.parameters()),'load_seconds':self.load_seconds,
            'compile':False,'tf32_matmul':True,'tf32_cudnn':True,'cpu_threads':2}
        if self.identity['encoder_dtype']!='torch.float32': raise ValueError('Original T5 default dtype was changed')

    @torch.no_grad()
    def request(self,prompt,seed,setting,diagnostic=False):
        counters={}; handles=[]
        def hook(name):
            def h(*args): counters[name]=counters.get(name,0)+1
            return h
        if diagnostic:
            handles.append(self.model.register_forward_hook(hook('model')))
            for i,b in enumerate(self.model.blocks):
                kind='pre' if i<self.model.pre else ('core' if i<self.model.pre+self.model.core else 'post')
                handles.append(b.register_forward_hook(hook(kind)))
            for b in self.model.txt_blocks: handles.append(b.register_forward_hook(hook('text_preamble')))
            handles.append(self.model.final.register_forward_hook(hook('decode')))
        try:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            free_before,total=torch.cuda.mem_get_info()
            start=time.perf_counter()
            ids,mask=self.text_encoder.tokenize([prompt])
            text=self.text_encoder.encode(ids,mask).to(torch.bfloat16)
            noise=make_noise(seed,self.device,text.dtype)
            ev0,ev1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            ev0.record()
            image,raw,trace=sample_from_noise(self.model,text,mask,noise,setting['steps'],setting['loops'],diagnostic)
            ev1.record()
            pil=to_pil(image)[0]
            torch.cuda.synchronize(); complete=time.perf_counter()-start
            # These checks/hashes occur outside the performance timer. Diagnostic runs have hooks inside and are not timing evidence.
            finite=bool(torch.isfinite(raw).all() and torch.isfinite(text).all())
            if not finite: raise FloatingPointError('Nonfinite pre-clamp image or text')
            meta={'complete_seconds':complete,'sampler_gpu_ms':ev0.elapsed_time(ev1),'finite':finite,'initial_noise_sha256':tensor_sha(noise),
                'initial_noise_dtype':str(noise.dtype),'initial_noise_shape':list(noise.shape),'input_ids_sha256':tensor_sha(ids),'attention_mask_sha256':tensor_sha(mask),
                'prompt_tokens':int(mask.sum()),'truncated':False,'pre_pil_sha256':tensor_sha(image),'pre_pil_dtype':str(image.dtype),'raw_final_dtype':str(raw.dtype),
                'image_shape':[512,512,3],'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'device_free_before_bytes':free_before,'device_total_bytes':total,'device_used_after_bytes':total-torch.cuda.mem_get_info()[0],
                'measurement_kind':'DIAGNOSTIC_HOOKS_NOT_TIMING' if diagnostic else 'WARMED_COMPLETE_REQUEST','actual_counts':counters or None,'finite_trace':trace or None}
            return pil,meta
        finally:
            for h in handles: h.remove()

    @torch.no_grad()
    def parity(self,prompt,seed,loops):
        ids,mask=self.text_encoder.tokenize([prompt]); text=self.text_encoder.encode(ids,mask).to(torch.bfloat16)
        with torch.random.fork_rng(devices=[0]):
            torch.cuda.manual_seed(seed)
            original=euler_sample(self.model,text,mask,512,2,6.,2.,loops)
        ours,raw,_=sample_from_noise(self.model,text,mask,make_noise(seed,self.device,text.dtype),2,loops,True)
        with torch.random.fork_rng(devices=[0]):
            torch.cuda.manual_seed(seed)
            official_png=generate(self.model,self.text_encoder,[prompt],512,2,6.,loops,2.)[0]
        p,q=to_pil(original)[0],to_pil(ours)[0]
        return {'loops':loops,'steps':2,'pre_pil_bitwise':bool(torch.equal(original,ours)),
            'max_absolute_difference':float((original-ours).abs().max()),'png_pixels_equal':p.tobytes()==q.tobytes(),'official_generate_png_equal':official_png.tobytes()==q.tobytes(),
            'finite':bool(torch.isfinite(raw).all()),'noise_sha256':tensor_sha(make_noise(seed,self.device,text.dtype))},q
