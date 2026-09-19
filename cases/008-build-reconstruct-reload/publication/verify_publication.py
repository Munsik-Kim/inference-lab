"""Verify current publication against independently pinned original snapshot hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

HERE=Path(__file__).resolve().parent
CASE_PATH='cases/008-build-reconstruct-reload'
ORIGINAL_INVENTORY_SHA256='09aca5e06ae66232cb6538a91b7289394617431311db86200f246bfca64282b0'
ARCHIVE_SHA256='49cc2640b63d7662cf3d40c168eaf036505bbf5f7f5f73de8de48099d452daca'
EDITABLE=('README.md','README.ko.md','REPRODUCTION.md','ANALYSIS.md','LIMITATIONS.md','PORTFOLIO.md')
sys.path.insert(0,str(HERE/'posthoc'))
from recalculate import read, sha, require, analyze

def safe(name: str) -> str:
    p=PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name and str(p)==name,'Unsafe path')
    return name

def package_files(root: Path) -> dict[str,str]:
    paths=[root/'LICENSE']
    for folder in [root/CASE_PATH,root/'tools/modelpack']:
        require(folder.is_dir() and not folder.is_symlink(),'Missing/linked package directory')
        for p in folder.rglob('*'):
            require(not p.is_symlink(),'Symlink in package')
            require(not any(part in ('__pycache__','.venv','venv','.cache','node_modules') for part in p.relative_to(root).parts),'Cache/environment in package')
            require(p.suffix not in ('.pyc','.pyo','.safetensors','.pt','.pth','.npy','.npz','.pkl'),'Binary tensor/cache in package')
            if p.is_file():paths.append(p)
    out={}
    for p in sorted(paths):
        require(not p.is_symlink(),'Linked package file')
        name=safe(p.relative_to(root).as_posix())
        if name != CASE_PATH+'/PUBLICATION_SHA256SUMS':out[name]=sha(p)
    return out

def original_inventory(root: Path):
    p=root/CASE_PATH/'publication/original_inventory.json'
    require(sha(p)==ORIGINAL_INVENTORY_SHA256,'Original inventory identity changed')
    data=read(p)
    require(len(data)==105,'Original package count')
    for n,h in data.items():safe(n);require(bool(re.fullmatch('[0-9a-f]{64}',h)),'Invalid original digest')
    return data

def check_original(root: Path):
    c=root/CASE_PATH;old=original_inventory(root);rec=read(c/'publication/change_record.json')
    require(rec['original_archive']['sha256']==ARCHIVE_SHA256,'Original archive mismatch')
    docs=rec['edited_documents'];require(set(docs)==set(EDITABLE),'Changed edit allowlist')
    for name,h in old.items():
        rel=name.removeprefix(CASE_PATH+'/')
        if rel in EDITABLE:
            d=docs[rel];require(d['reviewed_sha256']==h and d['reason'].strip(),'Missing original edit provenance')
            require(d['archived_path']=='publication/original_docs/'+rel,'Changed archived document path')
            require(sha(c/d['archived_path'])==h,'Archived original modified')
            require(sha(root/name)==d['published_sha256'],'Published editorial hash mismatch')
        else:require((root/name).is_file() and sha(root/name)==h,'Protected original changed: '+name)
    require(set(p.relative_to(root).as_posix() for p in (root/'tools/modelpack').rglob('*') if p.is_file())=={n for n in old if n.startswith('tools/modelpack/')},'Original tool inventory changed')
    additions=rec['added_files'];require(len(additions)==len(set(additions)),'Duplicate added files')
    for name in additions:safe(name);require(name.startswith(CASE_PATH+'/') and name not in old,'Invalid addition')
    require(set(package_files(root))==set(old)|set(additions),'Unexpected/missing public file')
    return old

def check_manifest(root: Path):
    data={};p=root/CASE_PATH/'PUBLICATION_SHA256SUMS'
    for line in p.read_text().splitlines():
        h,name=line.split('  ',1);safe(name)
        require(name not in data and re.fullmatch('[0-9a-f]{64}',h),'Duplicate/invalid manifest entry')
        require(name!=CASE_PATH+'/PUBLICATION_SHA256SUMS','Self-referential manifest')
        data[name]=h
    require(data==package_files(root),'Current publication checksum/inventory mismatch')

def check_posthoc(case: Path):
    actual=analyze(case);require(actual==read(case/'publication/posthoc/metrics.json'),'Post-hoc metrics stale')
    review=read(case/'publication/posthoc/posthoc_review_metrics.json')
    fields={'correct':'correct_forced_choice','full_argmax_allowed_recorded':'full_argmax_allowed',
            'mean_label_mass':'mass_mean','mean_negative_log_mass':'mean_outside_label_penalty',
            'delta_full':'delta_full_nll','delta_choice':'delta_choice_nll','delta_mass_nll':'delta_outside_label_penalty',
            'regression':'regressions_vs_baseline','gain':'gains_vs_baseline'}
    for track,groups in actual['tracks'].items():
        for task,group in groups.items():
            for arm,row in group['arms'].items():
                r=review['tracks'][track][arm][task];require(r['n']==group['n'],'Review denominator')
                for k,v in fields.items():require(math.isclose(row[k],r[v],rel_tol=1e-10,abs_tol=1e-10),'Review scalar discrepancy: '+k)
                require({k:v for k,v in row['prediction_histogram'].items() if v}==r['predicted_label_counts'],'Review label histogram')
    codes=read(case/'publication/posthoc/code_task_structure_review.json')['checks']
    for track,splits in actual['code_structure'].items():
        for split,v in splits.items():
            r=codes[track][split];require(v['code_prompts']==r['n_code']==r['count_gold_matches']==r['gold_is_smallest_option'],'Code cue discrepancy')
    for track in ['Q-BF16','Q-W4']:
        r=read(case/f'results/raw/Q/{track}-runtime.json');b,a=r['before'][0],r['after'][0]
        require(b['parameters']==a['parameters'],'Recorded runtime inventory changed')
        counts={}
        for route in a['routes'].values():k=route['kernel'] or 'null';counts[k]=counts.get(k,0)+1
        target=read(case/'publication/posthoc/recorded_runtime_review.json')['tracks'][track]
        require(counts==target['route_counts'],'Recorded kernel route counts differ')
        require(a['diagnostics']['invalid']==target['invalid_count']==0,'Recorded invalid outputs')
    exports={n:read(case/f'results/raw/R/artifacts/{n}.json') for n in ['I25','I25-R','P25','P25-R','S50','S50-R']}
    for n in ['I25','P25','S50']:
        a,b=exports[n]['weights'],exports[n+'-R']['weights'];require(set(a)==set(b),'Repair keys changed')
        require([k for k in a if a[k]!=b[k]]==['model.layers.13.mlp.down_proj.weight'],'Repair changed more than down weight')
        require(all(a[k]['shape']==b[k]['shape'] and a[k]['bytes']==b[k]['bytes'] for k in a),'Repair shapes/size changed')

def check_documents(root: Path):
    from tables import validate_tables
    case=root/CASE_PATH;validate_tables(case)
    sections=['background','hypotheses','theory','methods','experiments','results','analysis','conclusions','references']
    for name in ['REPORT.md','REPORT.ko.md']:
        text=(case/name).read_text();anchors=re.findall(r'<a id="([\w-]+)"></a>',text)
        require(anchors==sections,'Report section order/anchors')
        require(all('](#'+s+')' in text for s in sections),'Incomplete report contents')
        require('](REPORT.ko.md)' in text if name=='REPORT.md' else '](REPORT.md)' in text,'Language counterpart')
        require('NOT_ASSESSED' in text and 'COMPLETED' in text,'Missing study scope')
    mapping=read(case/'publication/source_map.json');seen=set()
    for entry in mapping['claims']:
        require(entry['id'] not in seen,'Duplicate source mapping');seen.add(entry['id'])
        p=root/safe(entry['source_path']);require(p.is_file() and sha(p)==entry['source_sha256'],'Source map hash mismatch')
        locator=entry['locator']
        if locator.startswith('/'):
            value=read(p)
            for key in locator[1:].split('/'):value=value[int(key)] if isinstance(value,list) else value[key.replace('~1','/').replace('~0','~')]
        else:require(locator in p.read_text(),'Missing text source locator')
        for doc in entry['documents']:require((case/safe(doc)).is_file(),'Missing mapped report')
    return len(seen)

def verify(root: Path):
    old=check_original(root);check_manifest(root);check_posthoc(root/CASE_PATH);claims=check_documents(root)
    return {'status':'PASS','original_files_verified':len(old),'current_files':len(package_files(root))+1,
            'source_mappings':claims,'new_gpu_runs':0,'scope':'CPU scalar/document/manifest consistency. Original 19-source snapshot preserved; no private vector or GPU replication.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=HERE.parents[2]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=a.output.resolve()
    require(not out.exists() and not out.is_relative_to(root),'New external output required')
    value=verify(root);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(value))

if __name__=='__main__':main()
