"""Package a checksummed case into a new external ZIP; never overwrites."""
import argparse,hashlib,json,stat,zipfile
from pathlib import Path,PurePosixPath

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--zip',type=Path,required=True);p.add_argument('--record',type=Path,required=True);a=p.parse_args();case=Path(__file__).resolve().parents[1]
 if a.zip.exists() or a.record.exists() or a.zip.resolve().is_relative_to(case) or a.record.resolve().is_relative_to(case):raise ValueError('New external destinations required')
 expected={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (case/'SHA256SUMS').read_text().splitlines()};files={str(f.relative_to(case)):f for f in case.rglob('*') if f.is_file()}
 if set(files)-{'SHA256SUMS'}!=set(expected):raise ValueError('Exact file inventory mismatch')
 for n,f in files.items():
  if f.is_symlink() or '..' in PurePosixPath(n).parts:raise ValueError('Unsafe source')
  if n in expected and hashlib.sha256(f.read_bytes()).hexdigest()!=expected[n]:raise ValueError('Hash mismatch '+n)
 a.zip.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(a.zip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for n,f in sorted(files.items()):
   info=zipfile.ZipInfo(case.name+'/'+n,date_time=(2026,9,18,0,0,0));info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,f.read_bytes())
 with zipfile.ZipFile(a.zip) as z:
  assert z.testzip() is None and len(set(z.namelist()))==len(files)
  for info in z.infolist():
   n=PurePosixPath(info.filename);assert not n.is_absolute() and '..' not in n.parts and not stat.S_ISLNK(info.external_attr>>16);assert z.read(info)==files[str(PurePosixPath(*n.parts[1:]))].read_bytes()
 record={'filename':a.zip.name,'files':len(files),'bytes':a.zip.stat().st_size,'sha256':hashlib.sha256(a.zip.read_bytes()).hexdigest(),'crc':'PASS','safe_paths':'PASS','byte_correspondence':'PASS','scientific_status':json.loads((case/'RUN_STATE.json').read_text())['status'],'publication':'NOT_PUBLISHED_BY_THIS_TASK'}
 a.record.parent.mkdir(parents=True,exist_ok=True)
 with a.record.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
 print(json.dumps(record))
if __name__=='__main__':main()
