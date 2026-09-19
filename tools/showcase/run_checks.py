"""Run the bounded CPU publication/display checks used by the prepared workflow."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
from common import ROOT, C6, C7, SUP, json_text, new_output, sha

def protected(root: Path) -> dict:
    paths=[root/'LICENSE']
    for folder in ('cases','downloads','notes'):
        paths.extend(p for p in (root/folder).rglob('*') if p.is_file())
    return {p.relative_to(root).as_posix():sha(p) for p in paths}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True,help='New external report/build directory')
    a=p.parse_args();out=new_output(a.repo,a.output);out.mkdir(parents=True)
    start=protected(a.repo);py=sys.executable
    commands=[
      ('docs-tests',['-m','unittest','discover','-s','docs/_checks','-v']),
      ('docs',['docs/_checks/check_docs.py','--output',str(out/'docs.json')]),
      ('showcase-tests',['-m','unittest','discover','-s','tests/showcase','-v']),
      ('selector',['tools/showcase/replay_selection.py','--output',str(out/'selection.json')]),
      ('case006-package',[C6+'/scripts/verify_publication.py','--case',C6,'--output',str(out/'case006-package.json')]),
      ('case007-package',[C7+'/scripts/verify_publication.py','--case',C7,'--output',str(out/'case007-package.json')]),
      ('case007-scalars',[C7+'/scripts/verify.py','--raw',C7+'/results/raw','--summary',C7+'/results/derived/summary.json','--output',str(out/'case007-scalars.json')]),
      ('readout-scalars',[SUP+'/scripts/verify_scalar.py','--case',SUP,'--summary',SUP+'/results/derived/summary.json','--output',str(out/'readout-scalars.json')]),
      ('build',['tools/showcase/build.py','--output',str(out/'site'),'--base-path','/inference-lab/']),
      ('site',['tools/showcase/check.py','--site',str(out/'site'),'--output',str(out/'site.json')])]
    results=[]
    for name,args in commands:
        command=[py,'-B',*args]
        r=subprocess.run(command,cwd=a.repo,text=True,capture_output=True)
        (out/(name+'.log')).write_text(r.stdout+r.stderr)
        results.append({'check':name,'exit_code':r.returncode})
        print(name+': '+('PASS' if r.returncode==0 else 'FAIL'),flush=True)
        if r.returncode:break
    unchanged=start==protected(a.repo)
    ok=len(results)==len(commands) and all(r['exit_code']==0 for r in results) and unchanged
    report={'status':'PASS' if ok else 'FAIL','checks':results,'protected_files':len(start),
            'evidence_unchanged':unchanged,'new_gpu_runs':0,'remote_ci':'NOT_RUN_LOCALLY',
            'scope':'Package preservation, public scalar audits, docs, CPU selector, display extraction and build; browser is separate. No original GPU tests or full-vector replay.'}
    (out/'checks.json').write_text(json_text(report)+'\n')
    raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
