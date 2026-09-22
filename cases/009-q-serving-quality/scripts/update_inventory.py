"""Hash the new Case009 only; historical manifests are not touched."""
import argparse
import hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
paths=sorted(x for x in a.case.rglob('*') if x.is_file() and x.name!='SHA256SUMS')
if any(x.is_symlink() for x in a.case.rglob('*')):raise ValueError('symlink')
(a.case/'SHA256SUMS').write_text(''.join(hashlib.sha256(x.read_bytes()).hexdigest()+'  '+x.relative_to(a.case).as_posix()+'\n' for x in paths))
