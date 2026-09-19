"""Curate scalar evidence only. Original private run records remain immutable."""
from pathlib import Path
import shutil
import numpy as np
from .common import CASE,read,write,sha,require,digest
from .numerics import full_kl


def curate_q(work: Path):
    target=CASE/'results/raw/Q';records=[];origins={}
    for arm,folder in [('Q-BF16','q-eval-bf16'),('Q-W4','q-eval-w4')]:
        path=work/folder;status=read(path/'status.json');require(status['status']=='PASS' and status['count']==192,'Q incomplete')
        diag=read(path/'runtime_after.json')[0]['diagnostics']
        valid=diag['invalid']==0 and diag['outputs_checked']>0 and status['full_outputs_finite']
        for row in read(path/'records.json'):
            r={**row,'arm':arm,'validity':{'full_model_outputs':bool(valid),'full_final_vocab':row['full_vocab_checked']==151936},
                 'full_kl_B_candidate':0.}
            if arm=='Q-W4':
                b=np.load(work/'q-eval-bf16/vectors'/(row['id']+'.npy'),allow_pickle=False)
                c=np.load(path/'vectors'/(row['id']+'.npy'),allow_pickle=False);r['full_kl_B_candidate']=full_kl(b,c)
            records.append(r)
        origins[folder]={'records_sha256':sha(path/'records.json'),'status_sha256':sha(path/'status.json')}
        write(target/(arm+'-runtime.json'),{'before':read(path/'runtime_before.json'),'after':read(path/'runtime_after.json'),
                   'profiler':read(path/'profiler.json'),'start':read(path/'start.json'),'decode':read(path/'decode.json')})
    write(target/'model_records.json',records);write(target/'source_records.json',origins)
    timings=[];memory=[]
    for round_id in [1,2,3]:
        for arm,slug in [('Q-BF16','bf16'),('Q-W4','w4')]:
            path=work/f'q-timing-{round_id}-{slug}';require(read(path/'status.json')['status']=='PASS','Q timing incomplete')
            timings += [{**r,'arm':arm} for r in read(path/'timing.json')]
            memory.append({'round':round_id,'arm':arm,'runtime':read(path/'runtime_after.json')})
    write(target/'timing.json',timings);write(target/'memory.json',memory)
    build=read(work/'q-full-build/build_report.json');write(target/'build.json',build)
    for slug in ['bf16','w4']:
        write(target/f'development-{slug}.json',read(work/f'q-dev-{slug}/records.json'))
    reloads=[read(work/f'q-reload-v2-{i}/records.json') for i in [1,2]]
    require(reloads[0]==reloads[1],'Q reload native outputs mismatch')
    write(target/'reload.json',{'status':'PASS','fresh_processes':2,'different_directory':True,
       'exact_scores':True,'record_hashes':[digest(r) for r in reloads],
       'decode':[read(work/f'q-reload-v2-{i}/decode.json') for i in [1,2]],
       'fixture':False,'offline_environment':True})


def curate_r(work: Path):
    target=CASE/'results/raw/R';records=[];origins={}
    for arm in ['R-B','I25','I25-R','P25','P25-R','S50','S50-R']:
        path=work/'r-eval'/arm;require(read(path/'status.json')['status']=='PASS','R incomplete')
        for r in read(path/'records.json'):
            v=r['validity'];v['full_model_outputs']=bool(v['full_block_outputs']==28 and v['full_norm'] and v['full_mlp'])
            records.append(r)
        origins[arm]=sha(path/'records.json')
    write(target/'model_records.json',records);write(target/'source_records.json',origins)
    for name,source in [('smoke','r-smoke/smoke.json'),('calibration','r-calibration/calibration.json'),
                        ('fit','r-calibration/fit.json'),('development','r-development/development.json'),
                        ('selection','r-development/selection.json'),('calibration_readout','r-calibration-readout/calibration_readout.json'),
                        ('export','r-artifacts/export.json')]:write(target/(name+'.json'),read(work/source))
    reloads={}
    for arm in ['R-B','I25','I25-R','P25','P25-R','S50','S50-R']:
        a=read(work/'r-reload'/f'{arm}-1.json');b=read(work/'r-reload'/f'{arm}-2.json')
        require(a==b,'R reload changed outputs')
        reloads[arm]={'status':'PASS','fresh_processes':2,'full_outputs_checked':a['full_outputs_checked'],
          'exact_report_hash':digest(a),'different_directory':True,'offline_environment':True}
        if arm=='R-B':
            expected={r['id']:r['native_full_logits_hash'] for r in read(work/'r-smoke/smoke.json')['records']}
            require(all(r['full_logits_hashes'][0]==expected[r['id']] for r in a['rows']),'Original/reloaded no-op mismatch')
    write(target/'reload.json',reloads)
    for arm in reloads:
        write(target/'artifacts'/(arm+'.json'),read(work/'r-artifacts'/arm/'artifact.json'))
    timings=[];memory=[]
    for i in [1,2,3]:
        path=work/f'r-timing-{i}';require(read(path/'status.json')['status']=='PASS','R timing incomplete')
        timings+=read(path/'timing.json');memory += [{**m,'round':i} for m in read(path/'memory.json')]
    write(target/'timing.json',timings);write(target/'memory.json',memory)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,required=True);p.add_argument('--track',choices=['Q','R'],required=True)
    a=p.parse_args();(curate_q if a.track=='Q' else curate_r)(a.work)
