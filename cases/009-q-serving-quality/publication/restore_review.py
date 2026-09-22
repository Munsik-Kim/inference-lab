"""Restore the immutable reviewed case and package into a NEW external directory."""
import argparse,hashlib,json,shutil
from pathlib import Path,PurePosixPath

def restore(root,output):
    pub=root/'cases/009-q-serving-quality/publication';identity=json.loads((pub/'original_identity.json').read_text())
    if output.resolve().is_relative_to(root.resolve()):raise ValueError('Restore into a new external directory')
    for group in ['original_case_files','original_package_files']:
        for name in identity[group]:
            p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or str(p)!=name:raise ValueError('unsafe restoration path')
    output.mkdir(parents=True,exist_ok=False)
    for group in ['original_case_files','original_package_files']:
        for name,expected in identity[group].items():
            overlay=pub/'original_overlay'/name;source=overlay if overlay.is_file() else root/name
            if source.is_symlink() or not source.resolve().is_relative_to(root.resolve()):raise ValueError('unsafe restoration source')
            if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise ValueError('review identity mismatch: '+name)
            dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();restore(a.root,a.output)
if __name__=='__main__':main()
