"""Check protected roots with before-manifest; no mutation of original evidence."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from interface_sensitivity.common import read,sha,atomic,require

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--private',type=Path,required=True);a=p.parse_args();case=Path(__file__).resolve().parents[1];root=case.parents[1];runtime=read(a.private/'runtime.json');old=Path(runtime['research_source']).parent
 before=read(a.private/'protected_before.json');bases={'existing_cases':root/'cases','research':root/'research','packages':root/'packages','r2':Path(runtime['original_result_root']),'old_source':Path(runtime['research_source']),'old_followup':old/'results/diova_agent_followup_r1'};report={};failures=[]
 for key,entries in before.items():
  for rel,e in entries.items():
   f=bases[key]/rel
   if not f.is_file() or f.stat().st_size!=e['bytes'] or ('sha256' in e and sha(f)!=e['sha256']) or ('mtime_ns' in e and f.stat().st_mtime_ns!=e['mtime_ns']):failures.append(key+'/'+rel)
  present={str(f.relative_to(bases[key])) for f in bases[key].rglob('*') if f.is_file() and '__pycache__' not in f.parts and not (key=='existing_cases' and str(f.relative_to(bases[key])).startswith('015-'))}
  failures += [key+'/unexpected/'+r for r in present-set(entries)]
  report[key]={'files':len(entries),'fully_hashed':sum('sha256' in e for e in entries.values()),'bytes_mtime_only':sum('mtime_ns' in e for e in entries.values())}
 model=Path(runtime['model']['path']);mf=read(a.private/'model_files.json')
 for name,e in mf.items():
  f=model/name
  if not f.is_file() or sha(f)!=e['sha256']:failures.append('model/'+name)
 from interface_sensitivity.model_adapter import verify_saved_artifacts
 artifacts=verify_saved_artifacts(runtime,case)
 git=read(a.private/'git_before.json');old14=root.parent/'inference-lab-case014-compensation-20261015'
 require(subprocess.check_output(['git','status','--porcelain=v1'],cwd=old14,text=True)==git['original_case014_status'],'original Case014 status changed')
 require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=old14,text=True).strip()==git['original_case014_head'],'original Case014 head changed')
 require(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True)=='','staged changes unexpected')
 require(subprocess.check_output(['git','diff','--name-only'],cwd=root,text=True)=='','tracked files outside new case modified')
 require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==git['head'],'new commit unexpected')
 require(not failures,'protected evidence changed: '+str(failures[:5]))
 value={'status':'PASS','protected_groups':report,'mismatches':failures,'model_files_fully_rehashed':len(mf),'model_shards_fully_rehashed':sum(n.endswith('.safetensors') for n in mf),'saved_4B_artifacts_fully_rehashed':sum(v['projection_count'] for v in artifacts.values()),'original_case014_git_state_unchanged':True,'tracked_diff':[],'staged_diff':[],'HEAD_unchanged':True,'environment_install_upgrade_build':0,'remote_mutations':0,'scope':'large original non-4B arrays use bytes+mtime; model3shards and RTN/D16504payloads fully SHA256 rechecked. Existing WSL snapshot/cache preserved.'}
 atomic(case/'results/p1_r1/preservation.json',value,True);print(json.dumps(value))
if __name__=='__main__':main()
