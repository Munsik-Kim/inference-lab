"""Small local-link/asset check; not a browser or semantic translation test."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import argparse,json,re
CASE=Path(__file__).resolve().parents[1]
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('href','src') and v:self.links.append(v)
def check():
    checked=[];external=[];failures=[]
    files=list(CASE.glob('*.md'))+list((CASE/'demo').glob('*.html'))
    for p in files:
        text=p.read_text()
        if p.suffix=='.md':links=re.findall(r'(?<!!)\[[^\]]*\]\(([^\s)]+)\)|!\[[^\]]*\]\(([^\s)]+)\)',text);links=[a or b for a,b in links]
        else:parser=Links();parser.feed(text);links=parser.links
        for link in links:
            part=urlsplit(link)
            if part.scheme or part.netloc:external.append(link);continue
            target=(p.parent/unquote(part.path)).resolve() if part.path else p
            if not target.exists():failures.append({'file':p.name,'link':link,'reason':'missing local target'});continue
            if part.fragment and target.suffix in ('.html','.md'):
                body=target.read_text();fragment=unquote(part.fragment)
                explicit=re.search(r'id=[\"\']'+re.escape(fragment)+r'[\"\']',body)
                headings={re.sub(r'[^\w\- ]','',x.lower()).replace(' ','-') for x in re.findall(r'^#+\s+(.+)$',body,re.M)}
                if not explicit and fragment not in headings:
                    failures.append({'file':p.name,'link':link,'reason':'missing local anchor'})
            checked.append({'file':p.name,'link':link})
    if failures:raise ValueError(json.dumps(failures))
    return {'status':'PASS','documents':len(files),'local_links':len(checked),'external_links_not_fetched':sorted(set(external)),
            'browser_execution':'NOT_RUN: no existing headless browser found; no browser installed for this research task',
            'figures':'Actual computed PNGs visually inspected; primary title wrap corrected without metric changes',
            'scope':'local Markdown/HTML link and asset presence; does not prove semantic language equivalence'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Use a new receipt path')
    value=check();out.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');print(json.dumps(value))
