"""Check local document/HTML links, required files and review-only payload safety."""
import argparse,ast,json,re,sys
from pathlib import Path
from urllib.parse import unquote,urlsplit
from html.parser import HTMLParser
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from interface_sensitivity.common import atomic,sha,read,require

class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=set();self.scripts=[];self.in_script=False
 def handle_starttag(self,t,attrs):
  d=dict(attrs)
  if 'id' in d:self.ids.add(d['id'])
  for k in ('href','src'):
   if k in d:self.links.append(d[k])
  self.in_script=t=='script'
 def handle_endtag(self,t):
  if t=='script':self.in_script=False
 def handle_data(self,text):
  if self.in_script:self.scripts.append(text)

def anchors(p):
 if p.suffix=='.html':x=Links();x.feed(p.read_text());return x.ids
 s=p.read_text();out=set(re.findall(r'<a\s+id="([^"]+)"',s))
 for line in s.splitlines():
  if re.match(r'^#{1,6} ',line):
   h=re.sub(r'[^\w -]','',line.lstrip('# ').strip()).lower().replace(' ','-');out.add(h)
 return out

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();case=Path(__file__).resolve().parents[1]
 banned={'.rpi4','.safetensors','.npy','.npz','.pt','.pth','.gguf','.bin'};files=[];link_count=0
 for f in sorted(case.rglob('*')):
  if '__pycache__' in f.parts:continue
  require(not f.is_symlink(),'symlink not permitted in review')
  if not f.is_file():continue
  require(f.suffix not in banned,'private model/tensor payload in case')
  require(f.stat().st_size<5*2**20,'unexpected oversized public file')
  files.append(f)
  if f.suffix=='.py':ast.parse(f.read_text(),filename=str(f.relative_to(case)))
  if f.suffix in ('.md','.html'):
   text=f.read_text()
   if f.suffix=='.html':parser=Links();parser.feed(text);links=parser.links
   else:links=re.findall(r'\]\(([^)]+)\)',text)
   for link in links:
    parsed=urlsplit(link.strip('<>'))
    if parsed.scheme or parsed.netloc:continue
    path=unquote(parsed.path)
    target=(f.parent/path).resolve() if path else f
    require(target.is_relative_to(case),'link escapes model-free review root: '+str(f.relative_to(case)))
    require(target.exists(),'broken file link: '+str(f.relative_to(case))+' '+link)
    if parsed.fragment and target.suffix in ('.md','.html'):
     require(unquote(parsed.fragment) in anchors(target),'broken anchor: '+str(f.relative_to(case))+' '+link)
    link_count+=1
  if f.suffix in ('.md','.txt','.py','.json','.jsonl','.log','.csv','.html','.cjs'):
   text=f.read_text()
   private_pattern=r'/'+'home'+r'/[^/\s]+/|/mnt/[a-z]/Users/|(?:gh[pousr]_|hf_)[A-Za-z0-9]{20,}'
   require(not re.search(private_pattern,text),'personal path/credential candidate: '+str(f.relative_to(case)))
 protocol=read(case/'specs/protocol.json');require(protocol['expected_primary_episodes']==96,'protocol count mismatch')
 required=['README.md','README.ko.md','REPORT.md','REPORT.ko.md','REPRODUCE.md','PROTOCOL.md','ROADMAP.md','REUSE_MAP.md','NOTICE.md','PLAN.ko.txt','demo/ko.html','demo/en.html','results/p1_r1/decision.json','results/p1_r1/independent_audit.json']
 require(all((case/p).is_file() for p in required),'missing required review output')
 value={'status':'PASS','file_count':len(files),'relative_links_checked':link_count,'AST_python_syntax':True,'personal_path_and_credential_pattern_check':True,'no_model_or_tensor_payload':True,'no_symlinks':True,'max_file_bytes':max(p.stat().st_size for p in files),'total_file_bytes':sum(p.stat().st_size for p in files),'browser':'separate actual Edge receipt, not inferred from this static check'}
 atomic(a.output,value,True);print(json.dumps(value))
if __name__=='__main__':main()
