"""CPU analysis of actual measurements; includes every fixed configuration."""
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import CASE,sha,write_json
from src.protocol import verify_phase
from src.statistics import timing,error_ci
from src.decision import numerical,choose_dev,empirical_pareto,final_decision
from src.backends import IDS

def analyze(stage):
 s,digest=verify_phase('a' if stage=='dev' else 'b');docs=json.loads((CASE/f'inputs/{"dev" if stage=="dev" else "fresh"}_manifest.json').read_text())['documents'];dids=[d['document_id'] for d in docs];nr=2 if stage=='dev' else 5;nb=5 if stage=='dev' else 20
 cfg=s['executable_configs'] if stage=='dev' else ['B','A_PUBLIC',s['finalist_id']];lengths=[4096] if stage=='dev' else [4096,512,2048]
 raws=[];sources={}
 for ri in range(nr):
  p=CASE/f'results/{stage}/round-{ri}.json';r=json.loads(p.read_text());sources[str(p.relative_to(CASE))]=sha(p)
  assert r['status']=='COMPLETED' and not r['errors'] and r['evidence_kind']=='gpu_measurement' and not r['mock']
  assert r['phase_sha256']==digest and r['round']==ri and r['started_at_utc']>s['frozen_at_utc']
  assert r['environment_fingerprint']==s['environment_fingerprint']
  raws.append(r)
 records=[m for r in raws for m in r['measurements']];index={(m['document_id'],m['round'],m['length'],m['config_id']):m for m in records}
 assert len(index)==len(records)==len(dids)*nr*len(lengths)*len(cfg)
 for m in records:
  assert m['split']==stage and m['status']=='OK' and m['inputs_unchanged'] and m['query_positions']==[i*(m['length']-1)//31 for i in range(32)]
  assert len(m['wall_ms'])==len(m['event_ms'])==nb and all(math.isfinite(t) and t>0 for t in m['wall_ms']+m['event_ms'])
  assert m['input_sha256']==index[m['document_id'],m['round'],m['length'],'B']['input_sha256']
 rows=[]
 for n in lengths:
  base=np.asarray([[index[d,r,n,'B']['wall_ms'] for r in range(nr)] for d in dids])
  for cid in (IDS if stage=='dev' else cfg):
   if cid not in cfg:
    rows.append(dict(config_id=cid,length=n,execution_status='IMPLEMENTATION_ERROR',reason='Excluded before DEV: constant-V smoothing gives zero scales and NaN outputs',numerical={'status':'NOT_RUN','median':None,'p95':None},speedup=None,wall_median_ms=None));continue
   ms=[index[d,r,n,cid] for d in dids for r in range(nr)];primary=[index[d,0,n,cid] for d in dids];units=[u for m in primary for u in m['units']]
   err=numerical(units,len(dids)*16);latency=np.asarray([[index[d,r,n,cid]['wall_ms'] for r in range(nr)] for d in dids]);t=timing(base,latency)
   ci=error_ci([[u['relative_output_error'] for u in m['units']] for m in primary]) if err['undefined_units']==0 else None
   maxbad=sum(m['full_output_nonfinite'] for m in ms);numrepeat=all(index[d,r,n,cid]['units']==index[d,0,n,cid]['units'] for d in dids for r in range(nr))
   rows.append(dict(config_id=cid,length=n,execution_status='EXECUTABLE_VERIFIED',numerical=err,error_ci=ci,**t,wall_median_ms=float(np.median(latency)),wall_p95_ms=float(np.quantile(latency,.95)),event_median_ms=float(np.median([t for m in ms for t in m['event_ms']])),full_output_nonfinite=maxbad,numerically_identical_rounds=numrepeat,peak_allocated_bytes=max(m['peak_allocated_bytes'] for m in ms),round_wall_medians_ms=[float(np.median(latency[:,r])) for r in range(nr)],worst_units=sorted([dict(document_id=m['document_id'],**u) for m in primary for u in m['units']],key=lambda u:u['relative_output_error'] or -1,reverse=True)[:5]))
 finalist=choose_dev(rows) if stage=='dev' else s['finalist_id']
 if stage=='dev':decision='FINALIST_SELECTED' if finalist else 'STOP_DEV_SCREEN'
 else:
  primary=next(r for r in rows if r['config_id']==finalist and r['length']==4096);decision=final_decision(finalist,True,True,primary['numerical'],primary['speedup'],primary['ci95'])
 return dict(stage=stage,phase_sha256=digest,source_hashes=sources,documents=len(dids),process_rounds=nr,primary_numeric_round=0,finalist_id=finalist,final_decision=decision,rows=rows,dominated_by=empirical_pareto([r for r in rows if r['length']==4096]),scope='Sampled-query local fidelity, full-query complete call; pointwise intervals; synthetic document family')

def render(result,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out.mkdir(parents=True,exist_ok=True);write_json(out/'summary.json',result)
 with (out/'comparison.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['config','length','execution','median_error','p95_error','speedup','ci_low','ci_high','wall_ms'])
  for r in result['rows']:w.writerow([r['config_id'],r['length'],r['execution_status'],r['numerical']['median'],r['numerical']['p95'],r['speedup'],*(r.get('ci95',[None,None])),r['wall_median_ms']])
 if result['stage']=='dev':
  for metric,limit in [('median',1),('p95',3)]:
   fig,ax=plt.subplots(figsize=(7,4.5))
   for r in result['rows']:
    if r['speedup'] is None:continue
    x=r['speedup'];y=100*r['numerical'][metric];ax.scatter(x,y);ax.annotate(r['config_id'],(x,y),xytext=(5,5),textcoords='offset points')
   ax.axvline(1.5,color='gray',linestyle='--',label='DEV speedup gate 1.50×');ax.axhline(limit,color='red',linestyle='--',label=f'Local {metric} error limit {limit}%');ax.set_xlabel('Paired complete-call wall speedup vs BF16');ax.set_ylabel(f'{metric} relative output error (%)');ax.set_title('DEV: 8 documents, L4096, layer13, RTX 5080');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/f'dev_{metric}_pareto.png',dpi=150);plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['dev','fresh'],default='dev');p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=analyze(a.stage);render(r,a.output);print(json.dumps({'final_decision':r['final_decision'],'finalist_id':r['finalist_id'],'rows':[{k:v for k,v in x.items() if k in ['config_id','speedup','numerical','ci95']} for x in r['rows']]},indent=2))
if __name__=='__main__':main()
