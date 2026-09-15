"""Post-measurement audit. Independent scalar formulas and bootstrap indexing.

No imports from the shared analysis, metrics, statistics or decision modules.
This verifies retained summaries, not independent GPU output reproduction.
"""
import argparse,hashlib,json,math,statistics,sys
from pathlib import Path
import numpy as np
CASE=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def quantile(xs,q):
 a=sorted(xs);x=(len(a)-1)*q;lo=math.floor(x);hi=math.ceil(x);return a[lo]+(a[hi]-a[lo])*(x-lo)
def same(x,y):assert math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-12),(x,y)
def main():
 p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 s=json.loads((CASE/'configs/phase_a.json').read_text());digest=sha(CASE/'configs/phase_a.json');assert digest==(CASE/'configs/phase_a.sha256').read_text().strip()
 for f,h in s['frozen_files'].items():assert sha(CASE/f)==h,f
 summary=json.loads(a.summary.read_text());assert summary['phase_sha256']==digest
 docs=json.loads((CASE/'inputs/dev_manifest.json').read_text())['documents'];dids=[d['document_id'] for d in docs]
 assert len(dids)==len(set(dids))==8
 historical=json.loads((CASE/'inputs/historical_index.json').read_text())['documents'];assert len(historical)==16
 assert not set(dids)&{d['document_id'] for d in historical}
 for d in docs:
  assert sha(CASE/d['text_file'])==d['text_sha256']
  for n in [512,2048,4096]:
   h=hashlib.sha256(json.dumps(d['token_ids_4096'][:n],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();assert h==d['prefix_sha256'][str(n)]
 records=[]
 for r in range(2):
  p=CASE/f'results/dev/round-{r}.json';raw=json.loads(p.read_text());assert summary['source_hashes'][str(p.relative_to(CASE))]==sha(p)
  assert raw['status']=='COMPLETED' and raw['round']==r and raw['started_at_utc']>s['frozen_at_utc'] and raw['phase_sha256']==digest and not raw['mock']
  records+=raw['measurements']
 index={(m['document_id'],m['round'],m['config_id']):m for m in records};assert len(index)==len(records)==80
 count=0
 for m in records:
  assert m['full_output_nonfinite']==m['full_invalid_rows']==0 and m['inputs_unchanged']
  assert m['query_positions']==[i*4095//31 for i in range(32)]
  if m['config_id']=='B':assert m['native_bitwise_equal']
  for u in m['units']:
   e2=math.fsum(v*v for v in u['row_error_norms']);r2=math.fsum(v*v for v in u['row_reference_norms'])
   same(e2,u['error_squared_sum']);same(r2,u['reference_squared_sum']);same(math.sqrt(e2/r2),u['relative_output_error']);same(math.sqrt(e2/4096),u['absolute_rms_error']);same(math.sqrt(r2),u['reference_norm']);same(u['dot_sum']/math.sqrt(r2*u['actual_squared_sum']),u['cosine']);count+=1
 compared=0;eligible=[]
 for row in summary['rows']:
  cid=row['config_id']
  if cid=='V3':assert row['speedup'] is None and row['numerical']['status']=='NOT_RUN';continue
  errors=[[math.sqrt(math.fsum(x*x for x in u['row_error_norms'])/math.fsum(x*x for x in u['row_reference_norms'])) for u in index[d,0,cid]['units']] for d in dids]
  flat=[x for d in errors for x in d];med=statistics.median(flat);tail=quantile(flat,.95);same(med,row['numerical']['median']);same(tail,row['numerical']['p95'])
  ratios=[[[x/y for x,y in zip(index[d,r,'B']['wall_ms'],index[d,r,cid]['wall_ms'])] for r in range(2)] for d in dids]
  vals=[v for ds in ratios for rs in ds for v in rs];point=statistics.median(vals);same(point,row['speedup'])
  lat=[v for d in dids for r in range(2) for v in index[d,r,cid]['wall_ms']];same(statistics.median(lat),row['wall_median_ms']);same(quantile(lat,.95),row['wall_p95_ms'])
  # Same recorded seed/design, implemented with scalar cell lookups rather than vectorized shared code.
  rng=np.random.default_rng(505901);boots=[]
  for _ in range(5000):
   ds=rng.integers(8,size=8);ps=rng.integers(2,size=2);bs=rng.integers(5,size=(8,2,5));sample=[]
   for di,d in enumerate(ds):
    for ri,r in enumerate(ps):sample.extend(ratios[d][r][int(b)] for b in bs[di,ri])
   boots.append(statistics.median(sample))
  for x,y in zip([quantile(boots,.025),quantile(boots,.975)],row['ci95']):same(x,y)
  rng=np.random.default_rng(505902);mb=[];tb=[]
  for _ in range(5000):
   sample=[v for d in rng.integers(8,size=8) for v in errors[d]];mb.append(statistics.median(sample));tb.append(quantile(sample,.95))
  for k,arr in [('median_ci95',mb),('p95_ci95',tb)]:
   for x,y in zip([quantile(arr,.025),quantile(arr,.975)],row['error_ci'][k]):same(x,y)
  if cid.startswith('V') and med<=.01 and tail<=.03 and point>=1.5:eligible.append(cid)
  compared+=1
 assert not eligible and summary['finalist_id'] is None and summary['final_decision']=='STOP_DEV_SCREEN'
 assert not (CASE/'inputs/fresh_manifest.json').exists() and not (CASE/'configs/phase_b.json').exists() and not list((CASE/'results/fresh').glob('*.json'))
 result={'status':'PASS','scope':'Independent scalar aggregation and bootstrap implementation from retained row norms/timing; not independent GPU reproduction','frozen_files':len(s['frozen_files']),'phase_a_sha256':digest,'unit_recomputations':count,'paired_timing_cells':8*2*compared,'config_summary_and_ci_comparisons':compared,'bootstrap_replicates_per_comparison':5000,'final_decision':'STOP_DEV_SCREEN','fresh_gpu_not_run':True,'missing_dev_records':0,'source_hashes':summary['source_hashes']}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
