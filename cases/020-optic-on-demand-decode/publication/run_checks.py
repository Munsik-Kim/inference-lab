"""One CPU entrypoint: quick recorded-data checks or full historical audit + figures."""
from pathlib import Path
import argparse,json,sys,os,traceback,subprocess,importlib.util,shutil,tempfile,math
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from verify import CASE,ARCHIVE,verify,read,check,sha,npz_integrity

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def equal(a,b,path='root'):
 if isinstance(a,dict):
  check(isinstance(b,dict) and a.keys()==b.keys(),'KEYS '+path)
  for k in a:equal(a[k],b[k],path+'.'+k)
 elif isinstance(a,list):
  check(isinstance(b,list) and len(a)==len(b),'LENGTH '+path)
  for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+f'[{i}]')
 elif isinstance(a,(float,int)) and not isinstance(a,bool):check(isinstance(b,(float,int)) and math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-9),'NUMERIC '+path)
 else:check(a==b,'VALUE '+path)

def docs(case):
 import re
 checked=0;contracts=read(case/'publication/display_contract.json')
 for rel,expected in contracts.items():
  text=(case/rel).read_text()
  for token in expected:check(token in text,'DOC_FACT '+rel+' '+token)
 facts=read(case/'summary/facts.json')
 for suffix in ['.ko.md','.md']:
  overview=(case/('README'+suffix)).read_text();report=(case/('REPORT'+suffix)).read_text()
  check(f"{facts['P4_C_faster_blocks']}/{facts['P4_C_B_block_comparisons']}" in overview,'DOC_RAW_WIN_COUNT')
  check(f"{facts['memory']['A_to_B_allocated_peak_reduction']*100:.0f}%" in overview,'DOC_RAW_MEMORY_REDUCTION')
  for key in ['session_wall_seconds','worker_wall_seconds']:check(f"{facts['N4096_A_reduction_ratio_of_method_medians'][key]*100:.1f}%" in report,'DOC_RAW_SESSION_PERCENT')
 for rel in ['README.md','README.ko.md','REPORT.md','REPORT.ko.md','REPRODUCE.md','REPRODUCE.ko.md','SOURCE_MAP.md','REUSE_MAP.md','NOTICE.md','publication/HISTORY.md']:
  p=case/rel;text=re.sub(r'```.*?```','',p.read_text(),flags=re.S)
  for target in re.findall(r'!?\[[^\]\n]*\]\(([^\s()]+)\)',text):
   if '://' in target:continue
   base,_,fragment=target.partition('#');dest=(p.parent/base).resolve() if base else p.resolve();check(dest.is_relative_to(case.parents[1].resolve()) and dest.exists(),'DOC_LINK '+rel+' '+target);checked+=1
  check('case020.html' not in text,'UNDEPLOYED_SITE_LINK')
 return {'relative_links_checked':checked,'display_contracts_checked':len(contracts),'scope':'Path existence and selected raw-derived literal facts; translation meaning reviewed separately.'}

def full_audits(case,out,data):
 hist=case/ARCHIVE;p4=hist/'followup/p4_speed_cost'
 # Exact original747-file scope, no edits or weaker expectations in the old verifier.
 original=read(p4/'provenance/baseline_inventory.json')['files']
 with tempfile.TemporaryDirectory(prefix='P2-original-scope-',dir=out) as temp:
  root=Path(temp)
  for rel in original:
   p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(hist/rel,p)
  old=load('case020_historical_P2',root/'src/audit_v2.py');p2=old.collect(root);old.save_tables(root,out/'historical_P2_summary',p2)
 newer=load('case020_historical_P4',p4/'src/audit.py');p4data=newer.collect(p4,portable=True);newer.save(out/'historical_P4_summary',p4data)
 # Alternative NumPy statistics path must agree with the historical pure-Python path.
 idx={(r['Z'],r['condition'],r['block'],r['method'],r['pass'],r['clock']):r for r in p4data['controlled_latency']}
 for row in data['all_P4_clock_statistics']:
  oldrow=idx[row['Z'],row['input'],row['block'],row['method'],row['pass'],row['clock']]
  for key in ['mean_us','median_us','p95_us','min_us','IQR_us']:equal(row[key],oldrow[key],key)
 # Independently audit all saved P2/P4 numbers; archived outside paths may be absent,
 # while the publication mapping verifies exact copies of those19 external originals.
 return {'historical_P2':p2['decision'],'historical_P4':p4data['decision'],'alternative_statistics_records_compared':len(idx),'outside_original_paths':'Missing machine paths are disclosed by portable P4; exact public copies independently checked by preservation mapping.'}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['quick','full'],default='quick');p.add_argument('--output',type=Path,required=True,help='A new directory outside the checkout');p.add_argument('--case',type=Path,default=CASE);p.add_argument('--tests',action='store_true');a=p.parse_args();case=a.case.resolve();out=a.output.resolve();repo=case.parents[1]
 if out.exists() or out.is_relative_to(repo):p.error('Choose a nonexistent output outside the repository')
 out.mkdir(parents=True);receipt={'mode':a.mode,'new_GPU_runs':0,'scope':'Stored-record CPU verification, not new GPU/model inference'}
 try:
  receipt['inventory']=verify(case)
  from analyze import collect,save
  data=collect(case);save(data,out/'summary')
  # Recompute supplied publication summaries; never trust a saved PASS flag.
  for name,value in data.items():
   if isinstance(value,dict):equal(value,read(case/'summary'/f'{name}.json'),name)
   else:
    import csv
    old=list(csv.DictReader((case/'summary'/f'{name}.csv').open()));check(len(value)==len(old),'SUMMARY_ROWS '+name)
    for fresh,saved in zip(value,old):
     check(fresh.keys()==saved.keys(),'SUMMARY_COLUMNS '+name)
     for key,v in fresh.items():equal(v,float(saved[key]) if isinstance(v,(float,int)) and not isinstance(v,bool) else saved[key],name+'.'+key)
  receipt['docs']=docs(case);receipt['tiny_reference']=data['tiny_reference'];receipt['raw_P2_clock_records']=len(data['P2_clock_statistics']);receipt['raw_P4_clock_records']=len(data['all_P4_clock_statistics']);receipt['facts']=data['facts']
  if a.mode=='full':
   for path in (case/ARCHIVE).rglob('*.npz'):npz_integrity(path)
   receipt['full']=full_audits(case,out,data)
   from figures import render
   render(data,out/'figures');receipt['figure_scope']='Four publication PNG/SVG pairs regenerated; historical figures remain exact preserved bytes.'
  else:receipt['full_historical_numerical_audit']='NOT_RUN_IN_QUICK; full mode covers full P1/P2 raw and P4 preflight samples/layers'
  if a.tests:
   r=subprocess.run([sys.executable,'-I','-B','-m','unittest','discover','-s',str(case/'tests'),'-v'],capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});(out/'tests.log').write_text(r.stdout+r.stderr);check(r.returncode==0,'UNIT_TESTS');receipt['unit_tests']='PASS'
  check('torch' not in sys.modules,'CPU_PATH_IMPORTED_TORCH');receipt.update(status='PASS',torch_imported=False,environment={'python':sys.version,'executable':sys.executable,'numpy':__import__('numpy').__version__})
  (out/'verification.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps({'status':'PASS','mode':a.mode,'output':str(out),'P2_cell_blocks':90,'P4_method_blocks':96,'new_GPU_runs':0,'torch_imported':False}));return 0
 except Exception as exc:
  receipt.update(status='FAIL',error=repr(exc),traceback=traceback.format_exc());(out/'failure.json').write_text(json.dumps(receipt,indent=2)+'\n');(out/'failure.log').write_text(traceback.format_exc());print(traceback.format_exc(),file=sys.stderr);return 1
if __name__=='__main__':sys.exit(main())
