"""Package only the checksummed public case; never overwrite an archive."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path, PurePosixPath
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storage import write_new


def safe_name(name):
    p=PurePosixPath(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--zip',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
    a=p.parse_args()
    if a.zip.exists() or a.record.exists() or any(x.resolve().is_relative_to(a.case.resolve()) for x in (a.zip,a.record)):
        p.error('Archive and record must be new paths outside the case')
    rows=[line.split('  ',1) for line in (a.case/'SHA256SUMS').read_text().splitlines()]
    expected={name:sha for sha,name in rows}
    if len(expected)!=len(rows): raise ValueError('Duplicate checksum path')
    expected['SHA256SUMS']=hashlib.sha256((a.case/'SHA256SUMS').read_bytes()).hexdigest()
    a.zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.zip,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,sha in sorted(expected.items()):
            file=a.case/name
            # Embedded measured-data HTML and scalar JSON can exceed 10 MB.
            # Tensor/binary/model formats remain excluded regardless of size.
            if (not safe_name(name) or file.is_symlink() or not file.is_file()
                    or file.stat().st_size>64_000_000
                    or file.suffix.lower() in {'.pt','.pth','.safetensors','.npy','.npz','.so','.bin'}
                    or any(part in {'.git','.venv','__pycache__','cache'} for part in file.parts)):
                raise ValueError('Unsafe/oversized case member')
            data=file.read_bytes()
            if hashlib.sha256(data).hexdigest()!=sha: raise ValueError('Checksum mismatch')
            info=zipfile.ZipInfo('006-attention-decision-stability/'+name,date_time=(2026,9,16,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            z.writestr(info,data)
    with zipfile.ZipFile(a.zip) as z:
        if z.testzip() is not None or len(z.infolist())!=len(expected): raise ValueError('CRC/count failure')
        for info in z.infolist():
            if not safe_name(info.filename) or ((info.external_attr>>16)&0o170000)==0o120000: raise ValueError('Unsafe ZIP member')
            name=info.filename.split('/',1)[1]
            if z.read(info.filename)!=(a.case/name).read_bytes(): raise ValueError('ZIP byte mismatch')
    record={'filename':a.zip.name,'files':len(expected),'bytes':a.zip.stat().st_size,
            'sha256':hashlib.file_digest(a.zip.open('rb'),'sha256').hexdigest(),
            'crc':'PASS','safe_paths':'PASS','byte_correspondence':'PASS','symlinks':0}
    write_new(a.record,record);print(json.dumps(record,indent=2))


if __name__=='__main__': main()
