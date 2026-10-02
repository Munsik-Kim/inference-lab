from pathlib import Path
import ast,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def test_vendor_original_identity():
    prov=json.loads((ROOT/'provenance/upstream.json').read_text())
    files=prov['files'] if 'files' in prov else prov['source_files']
    for name,meta in files.items():
        if name.startswith('looped_dit/') and (ROOT/'vendor'/name).exists():
            h=meta['sha256'] if isinstance(meta,dict) else meta
            assert hashlib.sha256((ROOT/'vendor'/name).read_bytes()).hexdigest()==h

def test_unique_split_and_prompts():
    ps=json.loads((ROOT/'configs/prompts.json').read_text())['prompts']
    assert len(ps)==24 and len({p['prompt_id'] for p in ps})==24
    assert sum(p['split']=='MAIN' for p in ps)==16
    assert sum(p['split']=='SMOKE' for p in ps)==4
    assert all(len(p['constraints'])>0 for p in ps if p['split'] in ('MAIN','SMOKE'))
    assert len({p['english'] for p in ps})==24

def test_source_compiles():
    for folder in ['source','analysis','demo']:
        for p in (ROOT/folder).glob('*.py'):ast.parse(p.read_text())

def test_generation_source_frozen():
    f=json.loads((ROOT/'provenance/generation-source-freeze.json').read_text())
    for name,m in f['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==m['sha256']
