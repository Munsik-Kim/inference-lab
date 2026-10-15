"""Closed public inventory, historical preservation and raw-display checks. stdlib."""
import argparse,hashlib,importlib.util,json,re,stat,tempfile,zipfile
from pathlib import Path,PurePosixPath

MANIFEST='publication/provenance/PUBLICATION_MANIFEST.json'
ARCHIVE='case014_compensation_unified_publication_v1.zip'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(case,downloads=True):
    case=Path(case).resolve();m=json.loads((case/MANIFEST).read_text());actual={str(p.relative_to(case)) for p in case.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual!=set(m['files'])|{MANIFEST}:raise ValueError('Case014 public closed file set mismatch')
    private=re.compile('/'+'home'+r'/[A-Za-z0-9_.-]+/|/'+'mnt'+r'/[a-z]/|[A-Z]:\\Users\\[A-Za-z0-9_.-]+|gh[pousr]_[A-Za-z0-9]{25,}|hf_[A-Za-z0-9]{25,}')
    banned={'.int4','.safetensors','.bin','.pt','.pth','.npy','.npz','.arrow','.parquet','.pdf','.zip'}
    for rel,meta in m['files'].items():
        p=case/rel;q=PurePosixPath(rel)
        if q.is_absolute() or '..' in q.parts or p.is_symlink() or not p.is_file() or p.suffix in banned:raise ValueError('unsafe public member: '+rel)
        if p.stat().st_size!=meta['bytes'] or sha(p)!=meta['sha256']:raise ValueError('Case014 changed file: '+rel)
        if p.suffix=='.png':
            if not p.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('invalid PNG')
        elif private.search(p.read_text()):raise ValueError('private path/credential candidate: '+rel)
    old=json.loads((case/'publication/provenance/ORIGINAL_MAP.json').read_text());retained=0
    for row in old['source_files']:
        p=case/row['retained_path']
        if sha(p)!=row['retained_sha256'] or p.stat().st_size!=row['retained_bytes']:raise ValueError('original retained file changed: '+row['old_relative_path'])
        retained+=row['byte_identical']
    if retained!=old['byte_identical_files'] or len(old['sanitized_files'])!=1:raise ValueError('original/sanitized mapping count')
    def module(name,path):
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    scripts=case/'publication/scripts'
    restore=module('case014_publication_restore',scripts/'restore_stage.py').restore
    derive=module('case014_publication_derive',scripts/'derive.py').derive
    with tempfile.TemporaryDirectory(prefix='case014-originals-') as td:views=[restore(case,s,Path(td)/s) for s in ['P0','P1','P1b','P2']]
    data=json.loads((case/'publication/summary/project.json').read_text())
    if data!=derive(case):raise ValueError('Case014 source-to-display mismatch')
    metadata_path=case.parents[1]/'downloads'/ARCHIVE.replace('.zip','.json');zip_status='NOT_AVAILABLE_IN_CASE_ONLY_VIEW'
    if downloads and ((case.parents[1]/'.git').exists() or metadata_path.parent.exists()) and not metadata_path.is_file():raise ValueError('missing repository download metadata')
    if downloads and metadata_path.is_file():
        meta=json.loads(metadata_path.read_text());archive=metadata_path.parent/meta['filename']
        if archive.is_symlink() or meta['filename']!=ARCHIVE or archive.stat().st_size!=meta['bytes'] or sha(archive)!=meta['sha256'] or meta['case_manifest_sha256']!=sha(case/MANIFEST):raise ValueError('download identity mismatch')
        with zipfile.ZipFile(archive) as z:
            if z.testzip():raise ValueError('ZIP CRC')
            names=set()
            for r in z.infolist():
                p=PurePosixPath(r.filename)
                if p.is_absolute() or '..' in p.parts or '\\' in r.filename or r.filename in names or stat.S_ISLNK(r.external_attr>>16):raise ValueError('unsafe ZIP member')
                names.add(r.filename)
            prefix='case014_compensation_unified_publication_v1/cases/014-quantization-compensation/'
            if {n[len(prefix):] for n in names if n.startswith(prefix)}!=actual:raise ValueError('ZIP Case014 coverage')
            for rel in actual:
                if hashlib.sha256(z.read(prefix+rel)).hexdigest()!=sha(case/rel):raise ValueError('ZIP file mismatch')
        zip_status='PASS'
    return {'status':'PASS','public_files':len(actual),'research_original_files':len(old['source_files']),'byte_identical_retained':retained,'sanitized_metadata_receipts':1,'sanitized_scope':'model_config._name_or_path only; original private file unchanged',
        'closed_stage_views':views,'display_raw_counts':{'P1':data['p1']['raw_rows'],'P1b':data['p1b']['raw_rows'],'P2':data['p2']['raw_rows']},'download':zip_status,'new_model_forwards':0,
        'limits':'metadata/hash/scalar verification; original GPU payload decode and model logits are historical execution receipts, not repeated in publication'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=verify(a.case);a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2)
    print(json.dumps(r,indent=2))
