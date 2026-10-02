"""Additional native parity at the frozen MAIN step counts, no quality selection.
Serial, charged against the same 320-generation / 4-hour ledger.
"""
import argparse,os,time,io
from pathlib import Path
from source.contracts import load,dump,sha
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--encoder',required=True);a=p.parse_args()
    dest=ROOT/'results/full-parity.json'
    if dest.exists():raise ValueError('Full parity receipt already exists; do not overwrite')
    lock=ROOT/'results/GPU_RUNNING.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    bpath=ROOT/'results/budget.json';budget=load(bpath);start=time.perf_counter();pid=f'full-parity-{time.time_ns()}'
    def charge(n):
        if budget['full_generation_starts']+n>320 or budget['generation_process_seconds']+time.perf_counter()-start>=14400:raise RuntimeError('BUDGET_EXHAUSTED')
        budget['full_generation_starts']+=n;dump(bpath,budget)
    try:
        identity=load(ROOT/'provenance/models.json');assert sha(a.checkpoint)==identity['model']['sha256'];assert sha(Path(a.encoder)/'model.safetensors')==identity['encoder']['encoder_only_sha256']
        import torch
        from source.adapter import Engine,make_noise,sample_from_noise,tensor_sha,euler_sample,generate,to_pil
        engine=Engine(a.checkpoint,a.encoder);settings=load(ROOT/'configs/main_settings.json')['settings']
        prompt=next(p['english'] for p in load(ROOT/'configs/prompts.json')['prompts'] if p['prompt_id']=='parity-1');seed=2372001
        charge(1);engine.request(prompt,71999,{'loops':4,'steps':2})
        rows=[]
        with torch.no_grad():
            for s in settings:
                charge(3);ids,mask=engine.text_encoder.tokenize([prompt]);text=engine.text_encoder.encode(ids,mask).to(torch.bfloat16)
                with torch.random.fork_rng(devices=[0]):
                    torch.cuda.manual_seed(seed);old=euler_sample(engine.model,text,mask,512,s['steps'],6.,2.,s['loops'])
                noise=make_noise(seed,engine.device,text.dtype)
                ours,raw,trace=sample_from_noise(engine.model,text,mask,noise,s['steps'],s['loops'],True)
                with torch.random.fork_rng(devices=[0]):
                    torch.cuda.manual_seed(seed);original_png=generate(engine.model,engine.text_encoder,[prompt],512,s['steps'],6.,s['loops'],2.)[0]
                adapter_png=to_pil(ours)[0];buff=io.BytesIO();adapter_png.save(buff,format='PNG')
                rows.append({'setting':s['id'],'loops':s['loops'],'steps':s['steps'],'noise_sha256':tensor_sha(noise),'pre_pil_bitwise':bool(torch.equal(old,ours)),
                  'max_absolute_difference':float((old-ours).abs().max()),'official_generate_png_equal':original_png.tobytes()==adapter_png.tobytes(),'finite_every_step':all(t['finite'] for t in trace),'adapter_png_sha256':__import__('hashlib').sha256(buff.getvalue()).hexdigest(),
                  'model_calls_each_sampler':2*s['steps'],'purpose':'native parity only, no new quality sample'})
        dump(dest,{'status':'PASS' if all(r['pre_pil_bitwise'] and r['official_generate_png_equal'] and r['finite_every_step'] for r in rows) else 'FAIL','rows':rows,'environment':engine.identity,'settings_sha256':sha(ROOT/'configs/main_settings.json'),'tolerance':'bitwise required; fixed before full parity'})
        if load(dest)['status']!='PASS':raise RuntimeError('FULL_PARITY_FAILED')
        print('FULL_NATIVE_PARITY_PASS',flush=True)
    finally:
        dt=time.perf_counter()-start;budget['generation_process_seconds']+=dt;budget['processes'].append({'id':pid,'phase':'full-parity','elapsed_seconds':dt});dump(bpath,budget);lock.unlink(missing_ok=True)
if __name__=='__main__':main()
