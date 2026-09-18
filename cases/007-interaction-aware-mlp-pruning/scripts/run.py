"""Serial GPU study stages; outputs must be new private paths outside the case."""
import argparse,sys,time,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import *
from src.runtime import Runtime,resource_gate,array,thash
from src.surgery import contributions,masked,replace_mlp

def verify_freeze(case):
 spec=load(case/'configs/protocol.json')
 for n,h in spec['code_hashes'].items():
  if file_hash(case/n)!=h:raise ValueError('Frozen code changed: '+n)
 for split,h in spec['inputs'].items():
  if file_hash(case/f'inputs/{split}.json')!=h:raise ValueError('Frozen input changed')
 return spec

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['smoke','calibrate','development','heldout','timing'],required=True);p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--round',type=int);a=p.parse_args();case=Path(__file__).resolve().parents[1]
 if a.work.resolve().is_relative_to(case):raise ValueError('Private output required')
 a.work.mkdir(parents=True,exist_ok=True);spec=verify_freeze(case);stage=a.stage+(''+str(a.round) if a.stage=='timing' else '');outdir=a.work/stage
 if outdir.exists():raise FileExistsError('No automatic overwrite/resume: '+str(outdir))
 outdir.mkdir();write_new(outdir/'resource_before.json',resource_gate());rt=Runtime(a.snapshot);torch=rt.torch
 try:
  if a.stage=='smoke':
   sels={f'S{b}_{i}':{'removed':v} for b in [4,8] for i,v in enumerate(spec['random_sets'][str(b)][:3])};rt.add(sels);result=[]
   for row in load(case/'inputs/SMOKE.json'):
    rec,z,x,o,fullx=rt.diagnose(row);rec2,z2,*_=rt.diagnose(row)
    if not np.array_equal(z,z2):raise ValueError('Native repeat mismatch')
    q,qd,cs,fp,h=rt.q(x)
    with torch.inference_mode():
     total_error=norm_stats(array(cs.sum(1)),array(fp))['relative_error']
     if total_error>1e-5:raise ValueError('FP32 decomposition failure')
     identity=[]
     for sel in sels.values():
      removed=sel['removed'];hh=h.clone();hh[:,[j for i in removed for j in rt.mapping[i]]]=0
      retained=torch.nn.functional.linear(hh,rt.original.down_proj.weight.float())
      err=norm_stats(array(fp-retained),array(cs[:,removed].sum(1)))['relative_error']
      if err>1e-5:raise ValueError('Direct deletion identity failure')
      identity.append(err)
    eq=rt.equivalence(fullx,sels)
    result.append({'id':row['id'],'B_repeat_bitwise':True,'decomposition_error':total_error,'max_identity_error':max(identity),'equivalence':eq,'Q_diagnostics':qd,'valid':rec['valid']})
   write_new(outdir/'summary.json',{'status':'PASS','rows':result,'gpu':torch.cuda.get_device_name(),'SM':list(torch.cuda.get_device_capability()),'native_attention':'forced FLASH_ATTENTION; no math fallback','new_model_forwards':12})
  elif a.stage=='calibrate':
   if load(a.work/'smoke/summary.json')['status']!='PASS':raise ValueError('No smoke validity')
   records=[];(outdir/'private_x').mkdir()
   for row in load(case/'inputs/CALIBRATION.json'):
    rec,z,x,o,fullx=rt.diagnose(row);q,qd,*_=rt.q(x);rec.update({'Q':q.tolist(),'Q_diagnostics':qd,'reference':norm_stats(array(o),array(o))});write_new(outdir/(row['id']+'.json'),rec);torch.save({'x':x.cpu(),'o':o.cpu()},outdir/'private_x'/(row['id']+'.pt'));records.append(rec)
    print(row['id'],flush=True)
   q=calibration_q(records);sels={}
   for b in spec['budgets']:
    for method in ['INDEPENDENT','PAIRWISE']:sels[f'{method}_{b}']={**select(q,b,method),'budget':b,'method':method}
   freeze={'kind':'SELECTION_FREEZE_BEFORE_DEVELOPMENT_AND_HELD_OUT','created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'protocol_sha256':file_hash(case/'configs/protocol.json'),'calibration_ids':[r['id'] for r in records],'calibration_raw_hashes':{r['id']:file_hash(outdir/(r['id']+'.json')) for r in records},'Q':q.tolist(),'selections':sels,'heldout_accessed':False}
   write_new(a.work/'selection.json',freeze);print(json.dumps(sels))
  elif a.stage in ['development','heldout']:
   freeze=load(a.work/'selection.json')
   if freeze['protocol_sha256']!=file_hash(case/'configs/protocol.json'):raise ValueError('Selection protocol mismatch')
   if a.stage=='heldout':
    if load(a.work/'development/summary.json')['status']!='PASS':raise ValueError('Development validity gate failed')
    if file_hash(a.work/'selection.json')!=load(a.work/'development/summary.json')['selection_sha256']:raise ValueError('Selector changed after development')
   sels=freeze['selections'];all_sels=dict(sels)
   for b in spec['budgets']:
    for i,removed in enumerate(spec['random_sets'][str(b)]):all_sels[f'RANDOM_{b}_{i:02d}']={'removed':removed,'budget':b,'method':'RANDOM_FIXED'}
   rt.add(all_sels);rows=load(case/('inputs/HELD_OUT.json' if a.stage=='heldout' else 'inputs/CALIBRATION.json'))
   if a.stage=='development':rows+=load(case/'inputs/DEVELOPMENT.json')
   eqmax=0.;count=0
   for row in rows:
    if row['split']=='CALIBRATION':
     old=load(a.work/'calibrate'/(row['id']+'.json'));t=torch.load(a.work/'calibrate/private_x'/(row['id']+'.pt'),weights_only=True);x=t['x'].cuda();o=t['o'].cuda();rec=old;fullx=None
    else:
     rec,z,x,o,fullx=rt.diagnose(row);q,qd,*_=rt.q(x);rec.update({'Q':q.tolist(),'Q_diagnostics':qd,'reference':norm_stats(array(o),array(o))})
    local=[]
    with torch.inference_mode():
     for name,sel in all_sels.items():
      out=rt.modules[name](x);stats=norm_stats(array(out),array(o));token=[norm_stats(array(out[i]),array(o[i])) for i in range(len(x))]
      if stats['relative_error'] is None:raise ValueError('Undefined local norm')
      local.append({'method':name,'removed':sel['removed'],'budget':sel['budget'],**stats,'token_metrics':token})
     if a.stage=='development':
      eq=rt.equivalence(x,sels);eqmax=max(eqmax,max(e['relative_error'] for e in eq))
    rec['local']=local;rec['selection_sha256']=file_hash(a.work/'selection.json');write_new(outdir/(row['id']+'--B.json'),rec)
    if a.stage=='heldout':
     for name in sels:
      cand,cz,cx,co,*_=rt.diagnose(row,name,rec['mlp_input_hash']);cand['full_vocab_kl_B_to_candidate']=kl(z,cz);cand['transition']=transition(rec['score'],cand['score']);cand['selection_sha256']=rec['selection_sha256']
      # Integrated output must agree with standalone sliced on the same inputs up to the frozen tolerance.
      with torch.inference_mode():eq=norm_stats(array(co),array(rt.modules[name](x)))
      if eq['relative_error'] is None or eq['relative_error']>.01:raise ValueError('Integrated/sliced mismatch')
      cand['integrated_vs_standalone']=eq;write_new(outdir/(row['id']+'--'+name+'.json'),cand)
    count+=1;print(json.dumps({'stage':a.stage,'prompt':row['id'],'completed':count}),flush=True)
   write_new(outdir/'summary.json',{'status':'PASS','prompts':count,'max_equivalence_relative':eqmax if a.stage=='development' else None,'selection_sha256':file_hash(a.work/'selection.json'),'model_forwards':48 if a.stage=='development' else 192*5,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
  else:
   from scripts.timing import run_timing
   run_timing(rt,case,a.work,outdir,spec,a.round)
 except Exception as exc:
  write_new(outdir/'failure.json',{'status':'BLOCKED_RESOURCE' if isinstance(exc,torch.OutOfMemoryError) else 'BLOCKED_IMPLEMENTATION','type':type(exc).__name__,'message':str(exc),'stage':a.stage});raise
 print(json.dumps({'stage':stage,'completed':True}),flush=True)
if __name__=='__main__':main()
