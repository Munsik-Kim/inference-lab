"""Closed public inventory and exact historical mapping. Stdlib only; no network/GPU."""
from pathlib import Path,PurePosixPath
import hashlib,json,zipfile
CASE=Path(__file__).resolve().parents[1]
ARCHIVE='publication/originals/pre-publication'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(ok,msg):
 if not ok:raise ValueError(msg)
def safe(name):
 check(isinstance(name,str) and bool(name),'PATH_TYPE');p=PurePosixPath(name)
 check(not p.is_absolute() and all(x not in ('','.','..') for x in name.split('/')) and not any(x in name for x in '\\:\0\r\n'),'UNSAFE_PATH');return p
def inventory(root):
 return {p.relative_to(root).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(Path(root).rglob('*')) if p.is_file()}
def check_rows(root,rows):
 for rel,v in rows.items():
  p=root/safe(rel);check(p.is_file() and not p.is_symlink() and p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],'BYTES: '+str(rel))
def verify(case=CASE,require_manifest=True):
 case=Path(case);hist=case/ARCHIVE;mapping=read(case/'publication/preservation_map.json');paths=set();originals=set()
 for row in mapping['files']:
  safe(row['public_path']);check(row['public_path'] not in paths and row['original_path'] not in originals,'DUPLICATE_MAPPING');paths.add(row['public_path']);originals.add(row['original_path']);check(row['transformation']=='NONE; exact bytes','UNDECLARED_TRANSFORM');check_rows(case,{row['public_path']:row})
 check(not mapping['omitted_original_files'] and not mapping['redacted_or_transformed_files'],'ORIGINAL_OMISSION')
 case_rows={r['case_relative_path']:r for r in mapping['files'] if r['kind']=='original_case'};check(set(case_rows)==set(inventory(hist)),'ORIGINAL_SCOPE')
 old=read(hist/'PUBLIC_MANIFEST.json');p4=hist/'followup/p4_speed_cost';p4m=read(p4/'PUBLIC_MANIFEST_P4.json')
 check_rows(hist,old['files']);check_rows(p4,p4m['files']);check(set(inventory(p4))==set(p4m['files'])|{'PUBLIC_MANIFEST_P4.json'},'P4_HISTORICAL_COVERAGE')
 for row in read(hist/'provenance/source_manifest.json')['files']:check_rows(hist,{row['path']:row})
 p2f=read(hist/'provenance/freeze.json');check_rows(hist,p2f['main_files']);p4f=read(p4/'provenance/freeze.json');check_rows(p4,p4f['files'])
 check(hashlib.sha256(json.dumps(p4f['files'],sort_keys=True).encode()).hexdigest()==p4f['bundle_sha256'],'P4_EXECUTION_BUNDLE')
 manifest=case/'PUBLIC_MANIFEST.json'
 if require_manifest:
  pub=read(manifest);actual=inventory(case);actual.pop('PUBLIC_MANIFEST.json',None);check(actual==pub['files'],'PUBLIC_COVERAGE_OR_BYTES')
 check(all(p.is_file() and not p.is_symlink() for p in case.rglob('*') if p.is_file() or p.is_symlink()),'SYMLINK')
 allfiles=inventory(case);check(max(v['bytes'] for v in allfiles.values())<=25*2**20,'FILE_SIZE_CAP');check(sum(v['bytes'] for v in allfiles.values())<=128*2**20,'CASE_SIZE_CAP')
 return {'status':'PASS','public_files':len(allfiles),'public_bytes':sum(v['bytes'] for v in allfiles.values()),'preserved_case_files':len(case_rows),'preserved_external_files':len(paths)-len(case_rows),'P2_original_manifest_sha256':sha(hist/'PUBLIC_MANIFEST.json'),'P4_original_manifest_sha256':sha(p4/'PUBLIC_MANIFEST_P4.json'),'new_GPU_runs':0,'scope':'Closed bytes/hash inventory, mapping, historical source/manifests; numerical audit is separate.'}
def npz_integrity(path):
 with zipfile.ZipFile(path) as z:
  names=z.namelist();check(len(names)==len(set(names)) and all(not n.endswith('/') and safe(n) for n in names),'NPZ_MEMBERS');check(z.testzip() is None,'NPZ_CRC')
  for name in names:
   with z.open(name) as stream:check(stream.read(6)==b'\x93NUMPY','NPY_MAGIC')
