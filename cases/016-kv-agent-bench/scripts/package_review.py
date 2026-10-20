#!/usr/bin/env python3
"""Build an allowlisted archive, verify safe extraction/CRC/every file hash."""
import argparse
import hashlib
import json
import pathlib
import stat
import sys
import zipfile
CASE=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(CASE/'src'));sys.path.insert(0,str(CASE/'scripts'))
from kv_agent_bench.common import load,save,sha_file
from check_review import selected

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=pathlib.Path,default=CASE);p.add_argument('--zip',type=pathlib.Path,required=True);p.add_argument('--extract',type=pathlib.Path,required=True);p.add_argument('--receipt',type=pathlib.Path,required=True);args=p.parse_args();case=args.case.resolve()
    if args.zip.exists()or args.extract.exists()or args.receipt.exists():raise ValueError('fresh zip/extraction/receipt required')
    if args.zip.resolve().is_relative_to(case)or args.extract.resolve().is_relative_to(case):raise ValueError('archive/extraction must be outside case')
    manifest=case/'provenance/review_file_manifest.json';files=[f for f in selected(case)if f!=manifest]
    rows=[{'path':str(f.relative_to(case)),'bytes':f.stat().st_size,'sha256':sha_file(f)}for f in files]
    save(manifest,{'schema':'case016-review-v1','files':rows,'manifest_itself_excluded':True,'model_environment_kv_payloads_included':False})
    args.zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9)as z:
        for f in files+[manifest]:z.write(f,'case016/'+str(f.relative_to(case)))
    args.extract.mkdir(parents=True);seen=set();entries=0
    with zipfile.ZipFile(args.zip)as z:
        if z.testzip()is not None:raise AssertionError('CRC failure')
        for info in z.infolist():
            rel=pathlib.PurePosixPath(info.filename)
            if info.filename in seen or rel.is_absolute()or'..'in rel.parts or'\\'in info.filename or not rel.parts or rel.parts[0]!='case016':raise ValueError('unsafe/duplicate archive path')
            if stat.S_ISLNK(info.external_attr>>16):raise ValueError('archive symlink')
            seen.add(info.filename);entries+=1
        z.extractall(args.extract)
    extracted=args.extract/'case016';stored=load(extracted/'provenance/review_file_manifest.json')
    for r in stored['files']:
        f=extracted/r['path']
        if not f.is_file()or f.stat().st_size!=r['bytes']or sha_file(f)!=r['sha256']:raise AssertionError('extracted file identity mismatch')
    if len(seen)!=len(rows)+1:raise AssertionError('archive file set mismatch')
    save(args.receipt,{'status':'PASS','zip_name':args.zip.name,'zip_bytes':args.zip.stat().st_size,'zip_sha256':sha_file(args.zip),'entries':entries,'selected_content_bytes':sum(r['bytes']for r in rows),'path_symlink_duplicate_crc_checks':'PASS','content_hash_checks':len(rows),'manifest_self_hash_recorded_separately':sha_file(manifest),'new_model_requests':0,'extracted_case_label':'NEW_EXTERNAL_REVIEW_VIEW/case016'})
    print({'status':'PASS','zip_bytes':args.zip.stat().st_size,'sha256':sha_file(args.zip),'files':entries})

if __name__=='__main__':main()
