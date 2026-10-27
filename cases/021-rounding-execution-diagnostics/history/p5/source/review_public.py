"""Public file coverage, bilingual facts, local links and exclusion checks."""
import argparse,csv,hashlib,json,re,xml.etree.ElementTree as ET
from pathlib import Path

def identity(p):return {'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def inspect(root):
    r=Path(root).resolve();facts=json.loads((r/'facts.json').read_text());checks=[]
    for f in r.rglob('*'):
        if f.is_symlink():raise ValueError('PUBLIC_SYMLINK:'+str(f))
        if not f.is_file():continue
        assert f.suffix not in {'.npy','.npz','.rpi4','.safetensors','.pt','.pth','.bin','.zip'},str(f)
        assert f.stat().st_size<25*2**20
        if f.suffix in ['.md','.json','.csv','.py','.txt']:
            content=f.read_text();assert not re.search(r'/home/[A-Za-z0-9_]+/|/mnt/[a-z]/Users/[A-Za-z0-9_]+/',content),'PRIVATE_PATH:'+str(f)
            assert not re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|ghp_[A-Za-z0-9]{36}|hf_[A-Za-z0-9]{30,}',content),str(f)
        if f.suffix=='.svg':ET.parse(f)
        if f.suffix=='.md':
            for link in re.findall(r'\]\(([^)]+)\)',f.read_text()):
                if link.startswith(('https://','http://','#')):continue
                target=(f.parent/link.split('#')[0]).resolve();assert target.is_relative_to(r) and target.exists(),(str(f),link)
    for lang in ['.ko','']:
        for stem in ['README_PUBLIC','P5_REPORT','CONSOLIDATED_REPORT']:
            doc=(r/(stem+lang+'.md')).read_text();assert 'INCONCLUSIVE' in doc;assert f"{facts['B']['candidate_fit_totals_nonoverlapping_all_valid_P']['A_only']:,}" in doc
        doc=(r/('CONSOLIDATED_REPORT'+lang+'.md')).read_text()
        for value in [facts['original_primary']['mean_E0'],facts['original_primary']['mean_E1'],facts['original_primary']['primary512']['mean_delta'],facts['original_primary']['primary512']['ci_low'],facts['original_primary']['primary512']['ci_high']]:assert f'{value:.13f}' in doc
        doc=(r/('P5_REPORT'+lang+'.md')).read_text()
        for item in facts['C']['first_observation']:
            sequence=' / '.join(str(v['numeric_mismatch']) for v in item['conditions']);assert sequence in doc
    entries=json.loads((r/'records/ledger.json').read_text())['entries'];budget=json.loads((r/'summary/budget.json').read_text())
    import math
    for kind in ['CPU','GPU']:
        assert math.fsum(x['parent_wall_seconds'] for x in entries if x['kind']==kind)==budget[kind]['parent_wall_seconds']
    assert budget['old_campaign_GPU_parent_seconds']+budget['GPU']['parent_wall_seconds']==budget['campaign_GPU_parent_seconds']
    return {'relative_links':'PASS','bilingual_display_facts':'PASS','ledger_recomputed':'PASS','exclusion_scan':'No listed patterns found; not a security certification','SVG':'XML parsed, not browser rendered','files':sum(p.is_file() for p in r.rglob('*')),'bytes':sum(p.stat().st_size for p in r.rglob('*') if p.is_file())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--write-manifest',action='store_true');a=p.parse_args();r=a.root.resolve();result=inspect(r)
    if a.write_manifest:
        files={str(f.relative_to(r)):identity(f) for f in sorted(r.rglob('*')) if f.is_file() and f.name!='PUBLIC_MANIFEST.json'}
        (r/'PUBLIC_MANIFEST.json').write_text(json.dumps({'scope':'P5 local public aggregate/synthetic candidate only; excludes private raw and original P0-P4 trees','self_excluded':True,'files':files},indent=2)+'\n')
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    with a.receipt.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result)
if __name__=='__main__':main()
