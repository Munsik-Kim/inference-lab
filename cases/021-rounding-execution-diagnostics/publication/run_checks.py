"""CPU public package checks. --full regenerates public tables and figures only."""
import argparse,json,os,sys,time,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from verify import verify,CASE,read,require
from derive import projection,numerical

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['quick','full'],default='quick');p.add_argument('--tests',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();require(not out.exists() and not out.is_relative_to(CASE.resolve()),'NEW_EXTERNAL_OUTPUT_REQUIRED');out.mkdir(parents=True);start=time.perf_counter();cpu=time.process_time();result={'status':'FAIL','mode':a.mode,'GPU_calls':0};env=os.environ.copy();env.pop('PYTHONPATH',None);env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg',MPLCONFIGDIR=str(out/'cache'))
 def child(name,args):
  cmd=[sys.executable,'-B',*map(str,args)];p=subprocess.run(cmd,env=env,capture_output=True,text=True);(out/(name+'.log')).write_text(p.stdout+p.stderr);require(p.returncode==0,'CHILD:'+name)
 try:
  result['identity']=verify(CASE);derived=projection(CASE);require(derived==read(CASE/'summary/facts.json'),'DERIVED_FACTS');result['numerical']=numerical(CASE);(out/'facts.json').write_text(json.dumps(derived,indent=2,ensure_ascii=False)+'\n')
  child('synthetic_and_p5',[CASE/'history/p5/source/public_cpu.py','--root',CASE/'history/p5','--output',out/'p5_public',*(['--quick'] if a.mode=='quick' else [])])
  if a.tests:child('publication_tests',['-m','unittest','discover','-s',HERE/'tests','-v']);child('historical_p5_contracts',[CASE/'history/p5/tests/test_contracts_v3.py','--output',out/'p5_tests'])
  if a.mode=='full':child('overview_figures',[HERE/'figures.py','--case',CASE,'--output',out/'figures'])
  require('torch' not in sys.modules,'TORCH_IMPORTED');result.update(status='PASS',torch_imported=False,private_raw_KL='NOT_RUN',scope='Public aggregate reanalysis/bootstrap and synthetic BF16/ULP checks; full adds figures. Not GPU inference or private logits audit.',sys_path_excludes_private_research=not any('case021-rounding-followup-' in x or 'case021-p5-final-' in x for x in sys.path))
 finally:
  result.update(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu);(out/'receipt.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 print(json.dumps(result,ensure_ascii=False));return 0
if __name__=='__main__':main()
