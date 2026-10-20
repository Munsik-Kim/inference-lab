#!/usr/bin/env python3
"""Restore closed source views to fresh scratch and verify every frozen byte."""
import argparse
import pathlib
import shutil
import sys
CASE=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(CASE/'src'))
from kv_agent_bench.common import load,save,sha_file,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',type=pathlib.Path,default=CASE);p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args();case=args.case.resolve()
    if args.output.exists():raise ValueError('fresh source-view directory required')
    args.output.mkdir(parents=True);mapping=load(case/'provenance/execution_source_map.json');count=0;sets=[]
    for entry in mapping['frozen_sets']:
        original=load(case/entry['freeze'])
        if original['source_sha256']!=entry['original_source_sha256']or digest(original['files'])!=original['source_sha256']:raise AssertionError('original freeze identity mismatch')
        if [{k:r[k]for k in('path','bytes','sha256')}for r in entry['files']]!=original['files']:raise AssertionError('source map differs from original file set')
        view=args.output/entry['freeze'].replace('/','_').replace('.json','');rows=[]
        for r in entry['files']:
            rel=pathlib.PurePosixPath(r['path']);retained=(case/r['retained_path']).resolve()
            if rel.is_absolute()or'..'in rel.parts or not retained.is_relative_to(case):raise ValueError('unsafe source map')
            if sha_file(retained)!=r['sha256']or retained.stat().st_size!=r['bytes']:raise AssertionError('retained source mismatch')
            target=view/pathlib.Path(*rel.parts);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(retained,target)
            if sha_file(target)!=r['sha256']:raise AssertionError('restored source mismatch')
            rows.append(r['path']);count+=1
        sets.append({'freeze':entry['freeze'],'files':len(rows),'status':'PASS'})
    save(args.output/'verification.json',{'status':'PASS','source_sets':sets,'file_hash_comparisons':count,'gpu_execution_replayed':False,'historical_manifests_modified':False})
    print({'status':'PASS','source_sets':len(sets),'file_hash_comparisons':count})

if __name__=='__main__':main()
