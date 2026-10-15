"""Read-only comparison against the pre-P2 inventories; writes only a new receipt."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
from p2_common import save,sha,utc

def main():
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);p.add_argument('--output',required=True);a=p.parse_args();start=time.monotonic()
    before=Path(a.private)/'preservation_before.private.json';d=json.loads(before.read_text());case=Path(d['roots']['case']['path']);repo=case.parents[1];changed=[];counts={}
    def check(label,path,expected):
        if not path.is_file() or path.stat().st_size!=expected['bytes'] or sha(path)!=expected['sha256']:changed.append(label)
    for label,root in d['roots'].items():
        counts[label]=len(root['files'])
        for rel,meta in root['files'].items():check(label+'/'+rel,Path(root['path'])/rel,meta)
    for rel,expected in d['tracked'].items():
        if not (repo/rel).is_file() or sha(repo/rel)!=expected:changed.append('tracked/'+rel)
    for rel,expected in d['model']['files'].items():check('model/'+rel,Path(d['model']['model_path'])/rel,expected)
    check('P1b_review_zip',Path(d['P1b_review_zip']['path']),d['P1b_review_zip'])
    def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
    actual={'branch':git('branch','--show-current'),'HEAD':git('rev-parse','HEAD'),'status':git('status','--short'),'index':git('diff','--cached','--raw'),'remotes':git('remote','-v')}
    for key in ['branch','HEAD','status','index','remotes']:
        if actual[key]!=d['git'][key]:changed.append('git/'+key)
    freeze=json.loads((case/'results/p2_r1/source_freeze.json').read_text())
    for rel,expected in freeze['files'].items():
        if sha(case/rel)!=expected:changed.append('execution_source/'+rel)
    receipt={'status':'PRESERVED' if not changed else 'MISMATCH','checked_utc':utc(),'pre_P2_manifest_sha256':sha(before),'preexisting_files_checked':counts,'tracked_files_checked':len(d['tracked']),
       'model_files_checked':len(d['model']['files']),'full_local_weight_file_hash_checked':True,'official_external_digest_or_D_origin_comparison':False,'historical_P1b_review_zip_checked':True,
       'execution_source_files_checked':len(freeze['files']),'branch':actual['branch'],'base_HEAD':actual['HEAD'],'git_index_unchanged':actual['index']==d['git']['index'],
       'git_remotes_unchanged':actual['remotes']==d['git']['remotes'],'git_status_unchanged':actual['status']==d['git']['status'],'mismatches':changed,'wall_seconds':time.monotonic()-start,
       'environment_scope':'Existing environments reused without installation/upgrade/build. Package-tree full hashing was not performed. No cache cleanup or other process termination.'}
    save(a.output,receipt);print(json.dumps(receipt,indent=2));raise SystemExit(0 if not changed else 1)
if __name__=='__main__':main()
