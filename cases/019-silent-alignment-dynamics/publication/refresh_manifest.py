"""Explicit publication-only reseal; never changes historic scientific files."""
import json
from pathlib import Path
from verify import file_rows,canonical
import hashlib

def refresh(case):
    rows=file_rows(case,exclude=('publication/PUBLIC_MANIFEST.json',))
    m={'schema':1,'self_excluded':'publication/PUBLIC_MANIFEST.json','files':rows,
       'public_bundle_sha256':hashlib.sha256(canonical(rows)).hexdigest(),
       'identity_scope':'current public files; separate from historic science bundles and Git/build identities'}
    (case/'publication/PUBLIC_MANIFEST.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    return m
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];m=refresh(root);print(len(m['files']),m['public_bundle_sha256'])
