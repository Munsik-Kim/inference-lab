"""Verify the exact public file set, export identities and display derivation."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, re
CASE=Path(__file__).resolve().parents[1]

def verify(case=CASE):
    case=Path(case); inventory=json.loads((case/'provenance/PUBLIC_MANIFEST.json').read_text())
    files={p.relative_to(case).as_posix():p for p in case.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    expected=set(inventory['files'])|{'provenance/PUBLIC_MANIFEST.json'}
    if set(files)!=expected:raise ValueError('Public file-set mismatch')
    # Compose the home marker to avoid matching this scanner's own regex literal.
    private=re.compile('/'+'home'+r'/[^/\s]+/|/mnt/[a-z]/|[A-Za-z]:\\Users\\|gh[pousr]_[A-Za-z0-9]{20,}')
    for name,path in files.items():
        if path.is_symlink() or '..' in Path(name).parts:raise ValueError('Unsafe file')
        if name!='provenance/PUBLIC_MANIFEST.json':
            row=inventory['files'][name]
            if path.stat().st_size!=row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Changed public file: '+name)
        if path.suffix in ('.py','.md','.json','.jsonl','.txt','.csv','.log'):
            if private.search(path.read_text()):raise ValueError('Private path/credential in '+name)
        if path.suffix in ('.rpi4','.safetensors','.bin','.pt','.pth','.arrow'):raise ValueError('Unexpected weight/full-data artifact')
    spec=importlib.util.spec_from_file_location('case013_derive_check',case/'publication/derive.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    derived=m.derive(case)
    if derived!=json.loads((case/'summary/project.json').read_text()):raise ValueError('Display summary changed from raw records')
    figures=json.loads((case/'figures/figure_manifest.json').read_text())
    if figures['source_identity']!=derived['source_identity']:raise ValueError('Stale figure source identity')
    if figures['source_script_sha256']!=hashlib.sha256((case/'publication/plot_results.py').read_bytes()).hexdigest():raise ValueError('Changed plot script')
    if set(figures['files'])!={'ppl.svg','dev-success.svg','packed-gates.svg'}:raise ValueError('Missing figure')
    for name,digest in figures['files'].items():
        if hashlib.sha256((case/'figures'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed figure: '+name)
    return {'status':'PASS','files':len(files),'bytes':sum(p.stat().st_size for p in files.values()),
            'new_model_runs':0,'scope':'Public allowlist/hash/path checks; compact raw scalar and pairing readback; source-linked display'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',type=Path,default=CASE);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.case.resolve()):raise SystemExit('Choose a new external output')
    r=verify(a.case);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
