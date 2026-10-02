"""Bounded serial GPU runner. Resume committed rows only; never overwrite attempts."""
import argparse, os, sys, time, random, platform, traceback
from pathlib import Path
from source.contracts import load,dump,sha,digest,verify_freeze,counts,commit_attempt,verify_attempt
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--encoder',required=True)
    p.add_argument('--phase',choices=['pilot','parity','trace','dev','verify','smoke','main','repeat'],required=True)
    p.add_argument('--round',type=int,choices=[0,1,2],default=0)
    args=p.parse_args(); verify_freeze(ROOT)
    protocol=load(ROOT/'configs/protocol.json'); ps=load(ROOT/'configs/prompts.json')['prompts']
    prompt_map={x['prompt_id']:(i,x) for i,x in enumerate(ps)}
    result=ROOT/'results'; result.mkdir(exist_ok=True)
    lock=result/'GPU_RUNNING.lock'
    try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError: raise RuntimeError('GPU runner lock exists; inspect owner, do not start concurrent GPU work')
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    start=time.perf_counter(); budgetpath=result/'budget.json'
    budget=load(budgetpath) if budgetpath.exists() else {'full_generation_starts':0,'generation_process_seconds':0.,'limit_count':320,'limit_seconds':14400,'processes':[]}
    process_id=f'{args.phase}-{args.round}-{time.time_ns()}'
    def charge(n=1):
        if budget['full_generation_starts']+n>320 or budget['generation_process_seconds']+time.perf_counter()-start>=14400:
            raise RuntimeError('BUDGET_EXHAUSTED')
        budget['full_generation_starts']+=n; dump(budgetpath,budget)
    def request(job, engine, diagnostic=False):
        key=digest(job)[:24]; parent=result/'attempts'/key; parent.mkdir(parents=True,exist_ok=True)
        for old in sorted(parent.glob('attempt-*')):
            if not old.name.endswith('.partial') and (old/'record.json').exists():
                rec=verify_attempt(old)
                if rec['job']!=job: raise ValueError('Job hash collision')
                if rec['status']=='SUCCESS': return rec
        num=len(list(parent.glob('attempt-*')))+1; partial=parent/f'attempt-{num}.partial'; partial.mkdir()
        final=parent/f'attempt-{num}'; charge()
        rec={'schema':'case012-record-v1','job_id':key,'job':job,'process_id':process_id,'attempt':num,'status':'STARTED','quality':None,
             'protocol_sha256':sha(ROOT/'configs/protocol.json'),'job_manifest_sha256':sha(ROOT/'results'/f'jobs-{args.phase}-{args.round}.json')}
        dump(partial/'started.json',rec)
        try:
            pil,meta=engine.request(job['english'],job['noise_seed'],job['setting'],diagnostic)
            pil.save(partial/'image.png')
            rec.update(meta); rec.update(status='SUCCESS',image_sha256=sha(partial/'image.png'),image_bytes=(partial/'image.png').stat().st_size)
            if diagnostic:
                actual=dict(rec['actual_counts']); actual['joint_blocks']=actual.get('pre',0)+actual.get('core',0)+actual.get('post',0)
                if actual!=counts(job['setting']): raise ValueError('Observed module counts do not match proxy')
            commit_attempt(partial,final,rec)
            print(f"{args.phase} {key} L{job['setting']['loops']} S{job['setting']['steps']} {rec['complete_seconds']:.3f}s",flush=True)
            return rec
        except Exception as e:
            rec.update(status='FAILED',error_type=type(e).__name__,error=str(e))
            commit_attempt(partial,final,rec)
            raise
    def job(pid,label,setting,purpose):
        i,x=prompt_map[pid]
        return {'prompt_id':pid,'prompt_index':i,'english':x['english'],'seed_label':label,'noise_seed':label+100000*i,
            'setting':setting,'phase':args.phase,'round':args.round,'purpose':purpose,'cfg':6,'size':512,'noise_scale':2,'weights':'ema'}
    try:
        jobs=[]
        if args.phase=='pilot': jobs=[job('latency-dev-1',71000+r,{'id':'pilot_C','loops':4,'steps':50},'PILOT_TIME_ONLY') for r in range(3)]
        elif args.phase=='trace': jobs=[job('parity-1',72001,s,'DIAGNOSTIC_COUNTERS') for s in protocol['block_proxy']]
        elif args.phase=='dev':
            jobs=[job(x['prompt_id'],72201+k,{'id':f'dev_{l}_{s}','loops':l,'steps':s},'LATENCY_ONLY') for l in (1,2,4) for s in (25,50,75) for k,x in enumerate([p for p in ps if p['split']=='LATENCY_DEV'])]
        elif args.phase=='verify':
            pred=load(ROOT/'configs/latency_prediction.json')
            jobs=[job(x['prompt_id'],72201+k,s,'LATENCY_VERIFY_ONLY') for s in pred['predictions'] for k,x in enumerate([p for p in ps if p['split']=='LATENCY_DEV'])]
        elif args.phase in ('smoke','main'):
            settings=protocol['block_proxy'] if args.phase=='smoke' else load(ROOT/'configs/main_settings.json')['settings']
            split='SMOKE' if args.phase=='smoke' else 'MAIN'; seeds=protocol['data']['smoke_seeds' if args.phase=='smoke' else 'main_seeds']
            jobs=[job(x['prompt_id'],s,c,split) for x in ps if x['split']==split for s in seeds for c in settings]
        elif args.phase=='repeat':
            settings={s['id']:s for s in load(ROOT/'configs/main_settings.json')['settings']}
            jobs=[job(x['prompt_id'],x['seed_label'],settings[s],'REPEATED_TIMING_NOT_NEW_QUALITY_SAMPLE') for x in protocol['repeat_timing']['main_pairs'] for s in protocol['repeat_timing']['setting_orders'][args.round]]
        if args.phase not in ('repeat','parity'): random.Random(72401+['pilot','trace','dev','verify','smoke','main'].index(args.phase)).shuffle(jobs)
        manifest={'phase':args.phase,'round':args.round,'protocol_sha256':sha(ROOT/'configs/protocol.json'),'jobs':jobs}
        mf=result/f'jobs-{args.phase}-{args.round}.json'
        if mf.exists() and load(mf)!=manifest: raise ValueError('Resume manifest mismatch')
        if not mf.exists(): dump(mf,manifest)
        model_identity=load(ROOT/'provenance/models.json')
        if sha(args.checkpoint)!=model_identity['model']['sha256'] or sha(Path(args.encoder)/'model.safetensors')!=model_identity['encoder']['encoder_only_sha256']: raise ValueError('Artifact hash mismatch')
        from source.adapter import Engine
        engine=Engine(args.checkpoint,args.encoder)
        dump(result/f'environment-{process_id}.json',dict(engine.identity,python=platform.python_version(),phase=args.phase,round=args.round))
        # One unscored warmup per fresh process, charged to budget; first call is separately labelled cold.
        warm=job('parity-1',71999,{'id':'warmup','loops':4,'steps':2},'WARMUP_NOT_QUALITY_SAMPLE')
        wm=result/f'jobs-warmup-{process_id}.json'; dump(wm,{'jobs':[warm]})
        charge(); pil,wmeta=engine.request(warm['english'],warm['noise_seed'],warm['setting'])
        dump(result/f'warmup-{process_id}.json',dict(wmeta,purpose='FIRST_SAMPLE_COLD_EXCLUDED',setting=warm['setting'],quality=None))
        if args.phase=='parity':
            receipt=[]
            for l in (1,2,4):
                charge(3); row,pil=engine.parity(prompt_map['parity-1'][1]['english'],72001+100000*prompt_map['parity-1'][0],l); receipt.append(row)
            dump(result/'parity.json',{'comparison':'original Euler versus dedicated-noise adapter, actual B32/512/GPU','rows':receipt,'tolerance':'bitwise required; no adaptive tolerance'})
            if any(not(r['pre_pil_bitwise'] and r['png_pixels_equal'] and r['official_generate_png_equal'] and r['finite']) for r in receipt): raise RuntimeError('PARITY_FAILED')
        else:
            for j in jobs: request(j,engine,args.phase=='trace')
    except Exception as e:
        dump(result/f'process-failure-{process_id}.json',{'phase':args.phase,'error_type':type(e).__name__,'error':str(e),'status':'PARTIAL_EXECUTION','new_training':0})
        raise
    finally:
        elapsed=time.perf_counter()-start
        budget['generation_process_seconds']+=elapsed
        budget['processes'].append({'id':process_id,'phase':args.phase,'round':args.round,'elapsed_seconds':elapsed})
        dump(budgetpath,budget); lock.unlink(missing_ok=True)

if __name__=='__main__': main()
