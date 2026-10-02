"""Verify an extracted repository-relative review package without models."""
import argparse,hashlib,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);a=p.parse_args();root=a.bundle.resolve();manifest=root/'REVIEW_SHA256SUMS'
    expected={}
    for line in manifest.read_text().splitlines():
        h,name=line.split('  ',1)
        if name in expected:raise ValueError('Duplicate manifest member')
        target=root/name
        if not target.resolve().is_relative_to(root) or target.is_symlink():raise ValueError('Unsafe path')
        expected[name]=h
    actual={str(x.relative_to(root)) for x in root.rglob('*') if x.is_file() and not any(part in ('__pycache__','.pytest_cache') for part in x.parts)}
    if actual!=set(expected)|{'REVIEW_SHA256SUMS'}:raise ValueError('Review exact inventory mismatch')
    for name,h in expected.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('Review bytes mismatch: '+name)
    print(json.dumps({'status':'PASS','files_verified':len(expected),'manifest_self_excluded':True}))
if __name__=='__main__':main()
