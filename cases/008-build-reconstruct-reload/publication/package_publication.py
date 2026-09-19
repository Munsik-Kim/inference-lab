"""Create or check a case/code-only ZIP; never includes models or the repository root."""
from pathlib import Path
import argparse
import hashlib
import json
import stat
import zipfile
from verify_publication import CASE_PATH,ARCHIVE_SHA256,package_files,verify,require,safe,sha

def check_archive(root: Path,path: Path):
    expected=package_files(root);name=CASE_PATH+'/PUBLICATION_SHA256SUMS';expected[name]=sha(root/name)
    with zipfile.ZipFile(path) as z:
        require(z.testzip() is None,'CRC failed');names=z.namelist()
        require(len(names)==len(set(names))==len(expected) and set(names)==set(expected),'ZIP file inventory')
        require(sum(i.file_size for i in z.infolist())<100_000_000,'Unexpected expanded size')
        for i in z.infolist():
            safe(i.filename);require(not stat.S_ISLNK(i.external_attr>>16) and not i.is_dir(),'ZIP link/directory')
            require(hashlib.sha256(z.read(i)).hexdigest()==expected[i.filename],'ZIP member bytes differ')
    return {'filename':path.name,'files':len(expected),'bytes':path.stat().st_size,'sha256':sha(path),
            'publication_manifest_sha256':sha(root/name),'original_archive_sha256':ARCHIVE_SHA256,
            'scope':'Case008 reports, original code/recipes/inputs/scalars, publication checks and historical restoration; root LICENSE. No weights, private H/Y or full vectors.',
            'zip_crc':'PASS','safe_paths':'PASS','file_byte_correspondence':'PASS'}

def package(root: Path,path: Path,record: Path):
    require(not path.exists() and not record.exists(),'Refusing to overwrite package/metadata')
    require(not path.resolve().is_relative_to((root/CASE_PATH).resolve()),'ZIP must be outside case')
    verify(root);paths=sorted([*package_files(root),CASE_PATH+'/PUBLICATION_SHA256SUMS'])
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in paths:z.write(root/name,name)
    result=check_archive(root,path)
    with record.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--zip',type=Path,required=True);p.add_argument('--record',type=Path);p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if a.check_only:print(check_archive(a.root.resolve(),a.zip))
    else:
        require(a.record is not None,'--record required');print(package(a.root.resolve(),a.zip,a.record))
