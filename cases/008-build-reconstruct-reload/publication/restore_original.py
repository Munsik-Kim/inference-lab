"""Restore the exact historical case/code bundle into a new external directory."""
from pathlib import Path
import argparse
import shutil
from verify_publication import CASE_PATH, EDITABLE, check_original, require, sha

def restore(root: Path, output: Path):
    root=root.resolve();output=output.resolve()
    require(not output.exists() and not output.is_relative_to(root),'Use a nonexistent external directory')
    old=check_original(root)
    sources={}
    for name,h in old.items():
        rel=name.removeprefix(CASE_PATH+'/')
        src=root/CASE_PATH/'publication/original_docs'/rel if rel in EDITABLE else root/name
        require(not src.is_symlink() and sha(src)==h,'Invalid historical source')
        sources[name]=src
    output.mkdir(parents=True)
    for name,src in sources.items():
        dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        require(sha(dest)==old[name],'Restored bytes differ')
    return {'status':'PASS','files':len(old),'original_case_files':85,'original_modelpack_sources':19}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(restore(a.root,a.output))
