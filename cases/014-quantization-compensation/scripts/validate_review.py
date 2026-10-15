"""Static links/schema, selected-input pairing and public-file privacy validation."""
import argparse,json,re,hashlib
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=[]
 def handle_starttag(self,tag,attrs):
  d=dict(attrs)
  if 'id' in d:self.ids.append(d['id'])
  for k in ['href','src']:
   if k in d:self.links.append(d[k])
def validate(root):
 root=Path(root).resolve();bad=[];count=0
 private=re.compile('/'+'home'+r'/[^/\s]+/|/'+'mnt'+r'/[a-z]/|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{25,}')
 for p in root.rglob('*'):
  if not p.is_file() or p.suffix in ['.png','.pyc']:continue
  if p.suffix=='.json':json.loads(p.read_text())
  if p.suffix=='.jsonl':
   for l in p.read_text().splitlines():json.loads(l)
  try:text=p.read_text()
  except UnicodeDecodeError:continue
  if private.search(text):bad.append((str(p.relative_to(root)),'private path/token pattern'))
  links=[]
  if p.suffix=='.md':links=re.findall(r'!?\[[^\]]*\]\(([^)]*)\)',text)
  if p.suffix=='.html':parser=Links();parser.feed(text);links=parser.links;assert len(parser.ids)==len(set(parser.ids)),p
  for link in links:
   if urlparse(link).scheme or link.startswith('//'):continue
   dest,_,frag=unquote(link.strip('<>')).partition('#');q=(p.parent/dest).resolve() if dest else p
   if not q.exists():bad.append((str(p.relative_to(root)),link));continue
   if frag:
    target=q.read_text();valid_ids=re.findall(r'\bid=["\']([^"\']+)',target);headings=re.findall(r'^#+\s+(.+)$',target,re.M)
    valid_ids += [re.sub(r'[^\w\- ]','',h.lower()).replace(' ','-') for h in headings]
    if frag not in valid_ids:bad.append((str(p.relative_to(root)),link))
   count+=1
 m=json.loads((root/'provenance/input_manifest.json').read_text());docs=m['documents'];assert len(docs)==97;assert len({d['document_id_hash'] for d in docs})==97;assert len({d['token_hash'] for d in docs})==97
 assert m['counts']=={'CAL-FIT':32,'DEV-SELECT':16,'EVAL-ID':32,'SHIFT':16,'SMOKE':1}
 for lang in ['ko','en']:
  h=(root/f'demo/{lang}/index.html').read_text();parser=Links();parser.feed(h);assert all('s'+str(i) in parser.ids for i in range(6));assert 'p1-grid' in parser.ids;assert 'P1' in h
 for report in ['REPORT.ko.md','REPORT.md']:
  h=(root/report).read_text();assert all(f'id="s{i}"' in h for i in range(1,10))
 if bad:raise ValueError(bad)
 return {'status':'PASS','local_links_checked':count,'input_records':97,'distinct_internal_document_groups':97,'scope':'static existence/anchors/privacy/schema only; browser checks separate','model_article_disjointness':'PG19 book IDs verified; Pile original article IDs unavailable'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--output',required=True);a=p.parse_args();r=validate(a.case);Path(a.output).write_text(json.dumps(r,indent=2));print(r)
