"""Static local links/privacy/source identities. No remote writes."""
import argparse,json,re,hashlib
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
ROOT=Path(__file__).resolve().parents[1]
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set()
    def handle_starttag(self,t,attrs):
        d=dict(attrs)
        if 'id'in d:self.ids.add(d['id'])
        if t=='a'and'href'in d:self.links.append(d['href'])
        if t in ('img','script','link'):
            key='src'if t in ('img','script') else 'href'
            if key in d:self.links.append(d[key])
def check(root):
    errors=[];checked=0
    for p in root.rglob('*'):
        if not p.is_file()or any(x in p.parts for x in ['__pycache__','.pytest_cache']):continue
        if p.suffix in ('.py','.md','.json','.txt','.html','.js','.css'):
            text=p.read_text()
            if re.search(r'/home/[a-z][a-z0-9_-]*/|/mnt/[cd]/(?:Users|AI)/|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{25,}|BEGIN (?:OPENSSH|RSA) PRIVATE KEY',text):errors.append('Private path/credential candidate: '+str(p.relative_to(root)))
        if p.suffix=='.html':
            parser=Links();parser.feed(p.read_text())
            for ref in parser.links:
                u=urlsplit(ref)
                if u.scheme or ref.startswith('//'):continue
                target=(p.parent/unquote(u.path)).resolve()if u.path else p
                if not target.exists():errors.append('Missing HTML target '+str(p.relative_to(root))+': '+ref)
                elif u.fragment and target.suffix=='.html':
                    other=Links();other.feed(target.read_text())
                    if unquote(u.fragment)not in other.ids:errors.append('Missing HTML anchor '+ref)
                checked+=1
        if p.suffix=='.md':
            for ref in re.findall(r'\]\(([^)]+)\)',p.read_text()):
                u=urlsplit(ref.strip())
                if u.scheme or ref.startswith('#'):continue
                if not(p.parent/unquote(u.path)).exists():errors.append('Missing Markdown target '+str(p.relative_to(root))+': '+ref)
                checked+=1
    # Pinned upstream and input identities remain exact.
    freeze=json.loads((root/'configs/input_freeze.json').read_text())
    for name,h in freeze['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:errors.append('Frozen input changed '+name)
    for file in ['generation-source-freeze.json','full-parity-source.json']:
        f=json.loads((root/'provenance'/file).read_text())
        pairs=f['files']if'files'in f else {f['source']:{'sha256':f['sha256']}}
        for name,m in pairs.items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=m['sha256']:errors.append('Source changed '+name)
    status={'status':'PASS'if not errors else'FAIL','local_links_checked':checked,'errors':errors,'private_path_scan':'public text only; local execution logs excluded','source_and_inputs':'fixed identities checked','remote_requests':0}
    return status
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--output',type=Path);a=ap.parse_args();r=check(a.root)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2));raise SystemExit(0 if r['status']=='PASS'else 1)
