"""Create a model-free review ZIP, safely extract and run real CPU checks."""
import argparse,hashlib,json,os,stat,subprocess,sys,zipfile
from pathlib import Path,PurePosixPath

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(2**20),b''):h.update(data)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--zip',type=Path,required=True);p.add_argument('--extract',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
 if a.zip.exists() or a.extract.exists() or a.receipt.exists():raise FileExistsError('all output paths must be new')
 excluded={'.safetensors','.rpi4','.npy','.npz','.pt','.pth','.gguf','.bin','.pyc'}
 files=[]
 for f in sorted(a.case.rglob('*')):
  if '__pycache__' in f.parts:continue
  if f.is_symlink():raise ValueError('symlink in review source')
  if f.is_file():
   if f.suffix in excluded:raise ValueError('forbidden private payload: '+f.name)
   files.append(f)
 entries={str(f.relative_to(a.case)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in files}
 manifest={'scope':'all selected archive content; excludes this generated manifest itself','files':entries}
 a.zip.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(a.zip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for f in files:z.write(f,'case015/'+str(f.relative_to(a.case)))
  z.writestr('case015/ARCHIVE_CONTENTS.json',json.dumps(manifest,indent=2)+'\n')
 with zipfile.ZipFile(a.zip) as z:
  names=z.namelist()
  if len(names)!=len(set(names)):raise ValueError('duplicate archive member')
  for info in z.infolist():
   name=PurePosixPath(info.filename)
   if name.is_absolute() or '..' in name.parts or '\\' in info.filename or stat.S_ISLNK(info.external_attr>>16):raise ValueError('unsafe archive path/type')
  if z.testzip() is not None:raise ValueError('CRC mismatch')
  z.extractall(a.extract)
 root=a.extract/'case015'
 for rel,e in entries.items():
  f=root/rel
  if f.stat().st_size!=e['bytes'] or sha(f)!=e['sha256']:raise ValueError('extracted content mismatch: '+rel)
 env=os.environ.copy();env.update(PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 commands=[['-m','unittest','discover','-s','tests','-v'],['-m','interface_sensitivity.cli','validate'],
 ['-m','interface_sensitivity.cli','oracle-check','--output',str(a.extract/'oracle.json')],
 ['-m','interface_sensitivity.cli','analyze','--records','results/p1_r1/episodes.jsonl','--output',str(a.extract/'analysis.json')],
 ['-m','interface_sensitivity.cli','audit','--records','results/p1_r1/episodes.jsonl','--output',str(a.extract/'audit.json')],
 ['-m','interface_sensitivity.cli','render','--records','results/p1_r1/episodes.jsonl','--output',str(a.extract/'explorer')],
 ['scripts/audit_public.py','--output',str(a.extract/'public_audit.json')],
 ['scripts/check_review.py','--output',str(a.extract/'static_review.json')]]
 receipts=[]
 for i,args in enumerate(commands):
  command=[sys.executable,'-B',*args];r=subprocess.run(command,cwd=root,env=env,capture_output=True,text=True)
  (a.extract/('check'+str(i)+'.log')).write_text(r.stdout+r.stderr)
  receipts.append({'command':['python3','-B',*args],'exit_code':r.returncode})
  if r.returncode:raise RuntimeError('extracted CPU verification failed: '+str(i))
 value={'status':'PASS','zip_bytes':a.zip.stat().st_size,'zip_sha256':sha(a.zip),'selected_files':len(entries),'CRC':True,'safe_paths':True,'no_symlinks':True,'duplicate_members':0,'all_extracted_hashes_checked':True,'commands':receipts,'model_forward':0}
 a.receipt.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
