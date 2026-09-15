"""Package only manifest-listed public files; refuse existing archives."""
import argparse
import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import CASE, sha, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--record',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists(),'Do not overwrite an existing ZIP'
    assert not a.output.resolve().is_relative_to(CASE.resolve())
    entries={}
    for line in (CASE/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1);q=PurePosixPath(name)
        assert not q.is_absolute() and '..' not in q.parts
        assert name not in entries and not (CASE/name).is_symlink()
        assert (CASE/name).suffix not in ['.pt','.pth','.safetensors','.gguf','.so','.zip']
        assert sha(CASE/name)==digest,name
        entries[name]=digest
    entries['SHA256SUMS']=sha(CASE/'SHA256SUMS')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name in sorted(entries):
            info=zipfile.ZipInfo(CASE.name+'/'+name,date_time=(2026,9,15,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=(stat.S_IFREG|0o644)<<16
            archive.writestr(info,(CASE/name).read_bytes())
    with zipfile.ZipFile(a.output) as archive:
        assert archive.testzip() is None
        assert len(archive.infolist())==len(entries)
        for info in archive.infolist():
            q=PurePosixPath(info.filename)
            assert not q.is_absolute() and '..' not in q.parts
            assert stat.S_IFMT(info.external_attr>>16)!=stat.S_IFLNK
            name=str(q.relative_to(CASE.name))
            assert hashlib.sha256(archive.read(info)).hexdigest()==entries[name]
    result=dict(name=a.output.name,files=len(entries),bytes=a.output.stat().st_size,
                sha256=sha(a.output),crc='PASS',safe_paths='PASS',byte_correspondence='PASS')
    write_json(a.record,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
