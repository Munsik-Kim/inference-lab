"""Fresh-directory, standard-library replay of all 256 public recorded episodes."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from verify import CASE, read, restore, verify
from derive import export

class CPUOnly:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','transformers','vllm','xgrammar'}:
            raise RuntimeError('CPU publication path forbids model runtime imports: '+fullname)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tests',action='store_true',help='Run unchanged 84+8 source tests and new publication contracts')
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or out.is_relative_to(CASE): raise ValueError('Use a fresh external output directory')
    out.mkdir(parents=True);start=time.monotonic();sys.meta_path.insert(0,CPUOnly())
    public=verify(CASE);restored=restore(CASE,out/'original-view')
    case=Path(restored['case'])
    # This guard is injected into child processes before any source module import.
    guard=out/'import-guard';guard.mkdir()
    (guard/'sitecustomize.py').write_text('import sys\nclass Guard:\n def find_spec(self,fullname,path=None,target=None):\n  if fullname.split(".")[0] in {"torch","transformers","vllm","xgrammar"}: raise RuntimeError("CPU import blocked: "+fullname)\nsys.meta_path.insert(0,Guard())\n')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join([str(guard),str(case/'src'),str(HERE)]))
    commands=[('all-256-raw-replay',['-m','agent_contract_cost.verify','--output',str(out/'replay')]),
              ('terminal-independent',['postprocess/check_terminal_independence.py','--output',str(out/'terminal')])]
    if a.tests:
        commands.extend([('original-unit',['-m','unittest','discover','-s','tests','-v']),
                         ('original-postprocess-unit',['-m','unittest','discover','-s','postprocess/tests','-v']),
                         ('publication-unit',['-m','unittest','discover','-s',str(HERE/'tests'),'-v'])])
    results=[]
    for label,args in commands:
        t=time.monotonic()
        result=subprocess.run([sys.executable,'-B',*args],cwd=case,env=env,capture_output=True,text=True,timeout=600)
        (out/(label+'.log')).write_text(result.stdout+result.stderr)
        results.append({'label':label,'exit_code':result.returncode,'seconds':time.monotonic()-t})
        if result.returncode:
            (out/'failed_commands.json').write_text(json.dumps(results,indent=2)+'\n')
            raise RuntimeError('CPU check failed: '+label)
    data=export(CASE,out/'derived')
    if data!=read(HERE/'summary/project.json'): raise ValueError('Raw-to-display projection changed')
    # Compare delivered lightweight review ledger independently with newly derived rows.
    import csv,io
    text=(HERE/'reviews/CASE017_REVIEW_ADDENDUM_KO.sanitized.txt').read_text()
    block=text.split('BEGIN_EPISODE_LEDGER\n',1)[1].split('END_EPISODE_LEDGER',1)[0]
    ledger=list(csv.DictReader(io.StringIO(block)))
    lookup={(r['stage'],r['task_id'],r['kv'],r['policy']):r for r in data['rows']}
    if len(ledger)!=256: raise ValueError('Review ledger row count differs')
    for r in ledger:
        v=lookup[(r['phase'],r['task_id'],r['kv'],r['policy'])]
        for k in ('verified','state_correct','final_correct','requests','output_tokens'):
            if int(r[k])!=v[k]: raise ValueError('Review ledger scalar differs')
        if int(r['fresh_validation'])!=v['fresh_validation']: raise ValueError('Posthoc Fresh differs')
    receipt={'status':'PASS','scope':'Lossless public bytes, all256 raw/SSE/parser/new-fixture/independent-endpoint replay, original paired bootstrap, publication projection and review-ledger comparison.',
             'public':public,'original_files_restored':restored['original_files_restored'],
             'commands':results,'display_exact':True,'review_ledger_rows':len(ledger),
             'new_model_requests':0,'GPU':'NOT_RUN','retokenization':'NOT_RUN; saved IDs hashes and usage checked',
             'CPU_model_import_guard':True,'seconds':time.monotonic()-start}
    (out/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
