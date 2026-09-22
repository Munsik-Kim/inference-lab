"""Current public identity plus immutable original map; CPU only."""
import argparse,hashlib,json,sys
from pathlib import Path,PurePosixPath
CASE='cases/009-q-serving-quality'
ORIGINAL_MAP_ANCHOR='ad1d0351e50915ee3093ad27adf1eba046274175854f3341d20f2b401412a621'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def safe(name):
    p=PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name and str(p)==name,'unsafe path')
def inventory(root):
    paths=[]
    for base in [root/CASE,root/'packages/diova-compare']:
        for p in base.rglob('*'):
            require(not p.is_symlink(),'symlink')
            if p.is_file():paths.append(p)
    paths.extend([root/'downloads/diova_compare-0.1.1-py3-none-any.whl',root/'downloads/diova_compare-0.1.1.metadata.json'])
    return {p.relative_to(root).as_posix():sha(p) for p in paths if p!=root/CASE/'publication/PUBLICATION_SHA256SUMS'}

def verify(root):
    pub=root/CASE/'publication';i=json.loads((pub/'original_identity.json').read_text())
    maps={'case':i['original_case_files'],'package':i['original_package_files']}
    require(hashlib.sha256(json.dumps(maps,sort_keys=True,separators=(',',':')).encode()).hexdigest()==ORIGINAL_MAP_ANCHOR,'original map identity changed')
    require(i['original_archive']['sha256']=='147c74832f266d5defc3238863656ad1710d472432ef122cfdf3366caa7eec87','review archive')
    edits=i['edited_documents'];allowed={CASE+'/'+n for n in ['README.md','README.ko.md','REPORT.md','REPORT.ko.md','REPRODUCTION.md']}
    require(set(edits)==allowed,'editorial allowlist')
    for n,h in maps['case'].items():
        safe(n)
        if n in edits:
            e=edits[n];require(e['reviewed_sha256']==h and e['reason'] and sha(pub/'original_overlay'/n)==h,'editorial original copy')
            require(sha(root/n)==e['published_sha256'],'editorial published hash')
        else:require(sha(root/n)==h,'protected original changed: '+n)
    for n,h in maps['package'].items():require(sha(pub/'original_overlay'/n)==h,'old package overlay changed')
    require(sha(pub/'original_overlay/diova_compare-0.1.0-py3-none-any.whl')==i['old_wheel_sha256'],'old wheel changed')
    current_pkg={p.relative_to(root).as_posix():sha(p) for p in (root/'packages/diova-compare').rglob('*') if p.is_file()}
    require(current_pkg==i['current_package_files'],'current package ledger')
    actual=inventory(root);listed={}
    for line in (pub/'PUBLICATION_SHA256SUMS').read_text().splitlines():
        h,n=line.split('  ',1);safe(n);require(n not in listed,'duplicate manifest');listed[n]=h
    require(actual==listed,'current exact inventory/hash')
    current_case={n for n in actual if n.startswith(CASE+'/')}
    require(current_case-set(maps['case'])==set(i['added_case_files']),'unapproved case additions')
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('posthoc_publication',pub/'posthoc/analyze.py');mod=module_from_spec(spec);spec.loader.exec_module(mod)
    require(mod.analyze(root/CASE)==json.loads((pub/'posthoc/analysis.json').read_text()),'posthoc regeneration')
    life=json.loads((root/CASE/'supplemental/lifecycle-v1/summary.json').read_text())
    require(life['original_full_split_retries']==0 and life['attempts']<=24,'lifecycle boundary')
    return {'status':'PASS','current_files':len(actual),'original_case_files':len(maps['case']),'original_package_files':len(maps['package']),'original_full_quality_status':'4 clean / 4 failed, unchanged','lifecycle_status':life['status'],'scope':'Current bytes, immutable reviewed source map, overlay and posthoc scalars. Full GPU benchmark not rerun.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=verify(a.root)
    with a.output.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
if __name__=='__main__':main()
