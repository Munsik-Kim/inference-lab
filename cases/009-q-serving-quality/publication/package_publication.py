"""Package only the verified Case009/CLI/wheel tree with repository-relative paths."""
import argparse,hashlib,json,stat,zipfile
from pathlib import Path,PurePosixPath
from verify_publication import verify,inventory,CASE

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);a=p.parse_args();root=a.root.resolve();verify(root)
    names=sorted(list(inventory(root))+[CASE+'/publication/PUBLICATION_SHA256SUMS'])
    if a.archive.exists() or a.metadata.exists():raise ValueError('Never overwrite a publication version')
    with zipfile.ZipFile(a.archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in names:
            f=root/name
            if f.is_symlink() or f.stat().st_nlink!=1:raise ValueError('unsafe link')
            z.write(f,name)
    with zipfile.ZipFile(a.archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(names)
        for i in z.infolist():
            p=PurePosixPath(i.filename);assert not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename and not stat.S_ISLNK(i.external_attr>>16)
            assert z.read(i)==(root/i.filename).read_bytes()
    meta={'filename':a.archive.name,'bytes':a.archive.stat().st_size,'sha256':hashlib.sha256(a.archive.read_bytes()).hexdigest(),'members':len(names),'publication_manifest_sha256':hashlib.sha256((root/CASE/'publication/PUBLICATION_SHA256SUMS').read_bytes()).hexdigest(),'review_archive_sha256':'147c74832f266d5defc3238863656ad1710d472432ef122cfdf3366caa7eec87','scope':'Repository-relative Case009 plus corrected CPU package source/tests/examples/license and wheel. Standalone current/original scalar audits and CLI; full-site and historical Case008 example need full clone.','start_here':CASE+'/publication/README.md','excluded':['model weights','full tensors/logits','benchmark prose','generated rationales','environments','private logs'],'CRC':'PASS','member_byte_correspondence':'PASS','old_full_split_retries':0,'new_lifecycle_process_attempts':24}
    a.metadata.write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
if __name__=='__main__':main()
