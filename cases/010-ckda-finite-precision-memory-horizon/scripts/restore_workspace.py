"""Restore both immutable research roots to a new external workspace."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import os
from verify_unified import CASE, sha, verify_snapshots

def restore(case, output):
    case, output = Path(case).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('output must not exist')
    if output.resolve().is_relative_to(case):
        raise ValueError('restore output must be outside the case')
    manifest = verify_snapshots(case)
    # Verify all inputs before writing. An interrupted partial directory is never selected.
    partial = output.with_name(output.name + '.partial')
    partial.mkdir(parents=True, exist_ok=False)
    for row in manifest['files']:
        source, target = case / row['path'], partial / row['original_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        if target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
            raise ValueError(f'restored bytes differ: {row["path"]}')
    receipt = {'status': 'PASS', 'files': len(manifest['files']), 'counts': {'v1':1075,'v2':387},
               'operation': 'copy unchanged files to historical paths; no model execution'}
    (partial/'restoration_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    if output.exists():
        raise ValueError('output appeared during restoration')
    os.rename(partial, output)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,default=CASE)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(restore(a.case,a.output)))
if __name__=='__main__': main()
