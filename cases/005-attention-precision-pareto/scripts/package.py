"""Package this case only. No Git, external upload or model files."""
import argparse,hashlib,json,stat,zipfile
from pathlib import Path,PurePosixPath
CASE=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True);p.add_argument('--record',type=Path,required=True);a=p.parse_args();assert not a.zip.exists() and not a.record.exists()
 assert not a.zip.resolve().is_relative_to(CASE) and not a.record.resolve().is_relative_to(CASE),'Write outside the case'
 files=sorted(p for p in CASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 allowed={'.md','.py','.json','.csv','.png','.txt','.sha256','.log'}
 for p in files:
  assert not p.is_symlink() and (p.suffix in allowed or p.name in ['.gitattributes','SHA256SUMS','LICENSE','SageAttention_LICENSE']),p
  assert p.stat().st_size<12*2**20,(p,'Unexpected large public file')
 source=[p for p in files if p.name!='SHA256SUMS']
 expected=''.join(sha(p)+'  '+p.relative_to(CASE).as_posix()+'\n' for p in source)
 assert (CASE/'SHA256SUMS').read_text()==expected,'Checksum mismatch; packaging never rewrites evidence'
 files=sorted(source+[CASE/'SHA256SUMS']);a.zip.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(a.zip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for p in files:
   name=CASE.name+'/'+p.relative_to(CASE).as_posix();info=zipfile.ZipInfo(name,date_time=(2026,9,15,0,0,0));info.external_attr=(stat.S_IFREG|0o644)<<16;info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes(),compresslevel=9)
 with zipfile.ZipFile(a.zip) as z:
  assert z.testzip() is None and len(z.namelist())==len(files) and len(set(z.namelist()))==len(files)
  for i in z.infolist():
   name=PurePosixPath(i.filename);assert not name.is_absolute() and '..' not in name.parts and not stat.S_ISLNK(i.external_attr>>16)
   assert z.read(i)==(CASE/Path(*name.parts[1:])).read_bytes()
 result={'files':len(files),'bytes':a.zip.stat().st_size,'sha256':sha(a.zip),'crc':'PASS','safe_paths':'PASS','source_byte_correspondence':'PASS','contains':'Case005 only; no model weights or full activations'}
 a.record.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
