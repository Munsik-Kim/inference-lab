"""Create a separate publication inventory; never rewrites historical manifests."""
import hashlib
import json
from pathlib import Path
CASE=Path(__file__).resolve().parents[1]
def main():
    target=CASE/'publication/PUBLIC_MANIFEST.json'
    files=[]
    for p in sorted(CASE.rglob('*')):
        if p==target or not p.is_file() or '__pycache__' in p.parts:continue
        if p.is_symlink():raise ValueError('Refuse symlink')
        b=p.read_bytes();files.append({'path':p.relative_to(CASE).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
    target.write_text(json.dumps({'schema':1,'version':'case017-publication-v1-local','files':files,
        'self_excluded':True,'original_manifest_rewritten':False,
        'scope':'Public packaging hash set. Original execution freeze and current uncommitted source identity are separate.'},indent=2)+'\n')
    print(json.dumps({'files':len(files),'bytes':sum(r['bytes'] for r in files)}))
if __name__=='__main__':main()
