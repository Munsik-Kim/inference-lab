"""Create a new case-only ZIP after reviewed-original and current-tree checks."""
import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
from verify_publication import verify, file_map, safe_name, sha, require

def verify_zip(path, case):
    expected=file_map(case)
    with zipfile.ZipFile(path) as z:
        infos=z.infolist();names=z.namelist()
        require(len(names)==len(set(names))==len(expected),'ZIP count/duplicate mismatch')
        require(sum(i.file_size for i in infos)<150*1024**2,'Abnormal expanded size')
        for info in infos:
            safe_name(info.filename);p=PurePosixPath(info.filename)
            require(p.parts[0]==case.name and len(p.parts)>1 and not info.is_dir(), 'ZIP root mismatch')
            require(not stat.S_ISLNK(info.external_attr>>16),'ZIP symlink')
            name=str(PurePosixPath(*p.parts[1:]))
            require(name in expected and info.file_size<15*1024**2,'Unexpected/oversized ZIP member')
            require(hashlib.sha256(z.read(info)).hexdigest()==expected[name],'ZIP byte mismatch')
        require(z.testzip() is None,'ZIP CRC failure')
    return len(expected)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--zip',type=Path,required=True)
    p.add_argument('--record',type=Path,required=True);a=p.parse_args()
    for target in [a.zip,a.record]:
        require(not target.exists() and not target.resolve().is_relative_to(a.case.resolve()),
                'New external destinations required')
    require(a.case.name=='007-interaction-aware-mlp-pruning','Wrong case root')
    verify(a.case);files=file_map(a.case)
    a.zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.zip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in sorted(files):
            info=zipfile.ZipInfo(a.case.name+'/'+name,date_time=(2026,9,18,0,0,0))
            info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(a.case/name).read_bytes())
    count=verify_zip(a.zip,a.case)
    record={'filename':a.zip.name,'files':count,'bytes':a.zip.stat().st_size,'sha256':sha(a.zip),
            'publication_manifest_sha256':sha(a.case/'PUBLICATION_SHA256SUMS'),
            'original_review_sha256':'811c392b7efd36c4ed186a5766a396b7ce70adcb68363cde4342f9da6fbcfa53',
            'scientific_status':'COMPLETED_NO_CLEAR_TRANSFER','deployment':'NOT_ASSESSED',
            'crc':'PASS','safe_paths':'PASS','byte_correspondence':'PASS',
            'scope':'Case-only reviewed publication package; original experimental files preserved; new GPU runs 0'}
    with a.record.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print(json.dumps(record))

if __name__=='__main__':main()
