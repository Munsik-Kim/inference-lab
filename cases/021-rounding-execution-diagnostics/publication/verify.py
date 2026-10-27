"""Closed public inventory, mapping, links, exclusions. Stdlib; no GPU imports."""
from pathlib import Path,PurePosixPath
import hashlib,json,re,sys
CASE=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def safe(name):
 p=PurePosixPath(name);require(not p.is_absolute() and all(x not in ('','.','..') for x in name.split('/')) and not any(x in name for x in '\\:\0\r\n'),'UNSAFE_PATH');return p
def identity(p):return {'bytes':p.stat().st_size,'sha256':sha(p)}
def inventory(case):return {str(p.relative_to(case)):identity(p) for p in sorted(case.rglob('*')) if p.is_file()}
def verify(case=CASE,require_manifest=True):
 case=Path(case);files=inventory(case);require(not any(p.is_symlink() for p in case.rglob('*')),'SYMLINK');require(sum(x['bytes'] for x in files.values())<=50*2**20,'50MiB_CAP')
 for name,row in files.items():
  safe(name);require(row['bytes']<=25*2**20,'FILE_SIZE');require(Path(name).suffix not in ['.npy','.npz','.rpi4','.safetensors','.bin','.pt','.pth','.zip','.pdf'],'PRIVATE_BINARY_PAYLOAD')
  if name.endswith(('.py','.json','.csv','.md','.txt','.jsonl')):
   text=(case/name).read_text();require(not re.search(r'/home/[A-Za-z0-9_]+/|/mnt/[a-z]/Users/[A-Za-z0-9_]+/',text),'PRIVATE_PATH:'+name)
   require(not re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|ghp_[A-Za-z0-9]{36}|hf_[A-Za-z0-9]{30,}',text),'CREDENTIAL_PATTERN:'+name)
 if require_manifest:
  m=read(case/'PUBLIC_MANIFEST.json');actual={k:v for k,v in files.items() if k!='PUBLIC_MANIFEST.json'};require(set(actual)==set(m['files']),'PUBLIC_COVERAGE')
  for name,row in m['files'].items():require(actual[name]=={k:row[k] for k in ['bytes','sha256']},'PUBLIC_BYTES:'+name)
 for row in read(case/'publication/preservation_map.json')['files']:
  require(identity(case/safe(row['target']))==row['target_identity'],'MAPPING_TARGET');require(row['transformation'] in ['EXACT_BYTES','NPY_INTEGER_IDS_TO_JSON; dtype/shape/value preserved','USER_PATH_REDACTION_ONLY; original bytes retained in read-only research root'],'MAPPING_TRANSFORM')
 hist=case/'history/p5';hm=read(hist/'PUBLIC_MANIFEST.json');ha=inventory(hist);ha.pop('PUBLIC_MANIFEST.json');require(ha==hm['files'],'P5_HISTORICAL_BUNDLE')
 links=0
 for f in case.rglob('*.md'):
  for link in re.findall(r'\]\(([^)]+)\)',f.read_text()):
   if link.startswith(('http://','https://','mailto:','#')):continue
   path,_,anchor=link.partition('#');target=(f.parent/path).resolve();require(target.is_relative_to(case.resolve()) and target.exists(),'LINK:'+str(f.relative_to(case))+':'+link)
   if anchor:
    text=target.read_text();explicit=re.findall(r'<a id="([^"]+)"',text);headings=[re.sub(r'[^\w\- ]','',x.strip().lower()).replace(' ','-') for x in re.findall(r'^#+\s+(.+)$',text,re.M)];require(anchor in explicit+headings,'ANCHOR:'+link)
   links+=1
 return {'status':'PASS','public_files':len(files),'public_bytes':sum(x['bytes'] for x in files.values()),'mapping_entries':len(read(case/'publication/preservation_map.json')['files']),'links':links,'scope':'Closed public files, declared transformations and current relative links. Historical text snapshots are plain .txt; private evidence excluded. No security certification.'}
if __name__=='__main__':print(json.dumps(verify(),indent=2))
