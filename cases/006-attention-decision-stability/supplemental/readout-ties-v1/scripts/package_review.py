"""Package supplement-only files safely, without staging or touching original archives."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,stat,zipfile

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,required=True);p.add_argument('--zip',type=Path,required=True);p.add_argument('--record',type=Path,required=True);a=p.parse_args();root=a.case.resolve()
 if a.zip.resolve().is_relative_to(root) or a.record.resolve().is_relative_to(root) or a.zip.exists() or a.record.exists():p.error('New external output locations required')
 files=sorted(f for f in root.rglob('*') if f.is_file());assert not any(f.is_symlink() for f in root.rglob('*'))
 assert all(f.suffix not in ['.npy','.npz','.pt','.pth','.safetensors','.so','.zip','.pyc'] for f in files)
 expected={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (root/'SHA256SUMS').read_text().splitlines()}
 assert set(expected)=={str(f.relative_to(root)) for f in files if f.name!='SHA256SUMS'}
 for name,h in expected.items():assert digest(root/name)==h,name
 with zipfile.ZipFile(a.zip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for f in files:z.write(f,'readout-ties-v1/'+str(f.relative_to(root)))
 with zipfile.ZipFile(a.zip) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(files)
  for i in z.infolist():
   path=PurePosixPath(i.filename);assert not path.is_absolute() and '..' not in path.parts and '\\' not in i.filename
   assert not stat.S_ISLNK(i.external_attr>>16);assert z.read(i)==(root/Path(*path.parts[1:])).read_bytes()
 result={'filename':a.zip.name,'files':len(files),'bytes':a.zip.stat().st_size,'sha256':digest(a.zip),'crc':'PASS','safe_paths':'PASS','duplicates':0,'symlinks':0,'source_byte_correspondence':'PASS','contents':'Supplement only; original archive and full tensors excluded'}
 with a.record.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
