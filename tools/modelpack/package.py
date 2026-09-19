"""Public case/code review bundle; weights and private execution data excluded."""
from pathlib import Path,PurePosixPath
import stat,zipfile
from .common import ROOT,CASE,inventory,sha,write,read,require,safe_name


def sources():return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'tools/modelpack').glob('*.py'))}


def manifest():
    write(CASE/'provenance/tool_source_hashes.json',sources())
    rows=inventory(CASE,('SHA256SUMS',))
    with (CASE/'SHA256SUMS').open('x') as f:f.write(''.join(f'{h}  {n}\n' for n,h in rows.items()))


def verify_case():
    entries={}
    for line in (CASE/'SHA256SUMS').read_text().splitlines():
        h,n=line.split('  ',1);safe_name(n);require(n not in entries and n!='SHA256SUMS','Duplicate/self manifest');entries[n]=h
    require(entries==inventory(CASE,('SHA256SUMS',)),'Case inventory/hash mismatch')
    require(read(CASE/'provenance/tool_source_hashes.json')==sources(),'Tool source mismatch')
    for track in ['Q','R']:
        freeze=read(CASE/f'configs/{track}_evaluation_freeze.json')
        for name,value in freeze['input_files'].items():require(sha(CASE/name)==value,'Changed frozen inputs')
        for name,value in freeze['code'].items():require(sha(ROOT/name)==value,'Changed frozen code')
    return {'status':'PASS','case_files':len(entries)+1,'tool_files':len(sources()),'manifest_sha256':sha(CASE/'SHA256SUMS')}


def package(path: Path,record: Path):
    require(not path.exists() and not record.exists(),'Refusing existing review package')
    verify=verify_case();paths=sorted([p for p in CASE.rglob('*') if p.is_file()])+sorted((ROOT/'tools/modelpack').glob('*.py'))+[ROOT/'LICENSE']
    expected={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:
            require(not p.is_symlink(),'Symlink rejected');z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(path) as z:
        require(z.testzip() is None,'ZIP CRC failed');names=z.namelist();require(len(names)==len(set(names))==len(expected),'ZIP duplicate/count')
        import hashlib
        for info in z.infolist():
            name=safe_name(info.filename);require(not stat.S_ISLNK(info.external_attr>>16),'ZIP symlink')
            require(hashlib.sha256(z.read(info)).hexdigest()==expected[name],'Repository/ZIP byte mismatch')
    result={**verify,'archive':path.name,'files':len(expected),'bytes':path.stat().st_size,'sha256':sha(path),
       'scope':'local review bundle: new case and modelpack code; weights and previous case packages excluded',
       'zip_crc':'PASS','safe_paths':'PASS','file_byte_correspondence':'PASS'}
    write(record,result);return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['manifest','verify','zip'])
    p.add_argument('--zip',type=Path);p.add_argument('--record',type=Path);a=p.parse_args()
    if a.action=='manifest':manifest()
    elif a.action=='verify':print(verify_case())
    else:print(package(a.zip,a.record))
