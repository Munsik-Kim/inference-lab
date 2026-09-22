"""Check an explicit site artifact against its retained evidence and local links."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from common import ROOT, read, require, sha, json_text, new_output
from data import load
from case008 import load_case008, render_case008
from case009 import load_case009, render_case009, C9, SOURCE_COPIES
from study import SECTION_IDS, result_html
from layout import report8_url, C8_PATH, C8_REVISION

EXPECTED = {'en/case008.html','ko/case008.html','data/case008.json','assets/case008.css','assets/case008.js','index.html','en/index.html','ko/index.html','en/guide.html','ko/guide.html','en/case006.html','ko/case006.html',
            'en/case007.html','ko/case007.html','assets/icon.svg','assets/app.js','assets/state.js','assets/style.css','assets/study.css',
            'assets/en.js','assets/ko.js','data/case006.js','data/case007.js',
            'data/case006.json','data/case007.json','build_manifest.json'}
class Links(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[];self.ids=set()
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if 'id' in a:
            require(a['id'] not in self.ids, 'Duplicate HTML id');self.ids.add(a['id'])
        for key in ('src','href'):
            if a.get(key):self.links.append((tag,a[key]))

class Regions(HTMLParser):
    """Collect visible text and links in named generated HTML regions."""
    VOID = {'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}
    def __init__(self):
        super().__init__(); self.stack=[]; self.blocks={}
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if 'id' in a:
            self.blocks[a['id']]={'text':[], 'links':[]}
        active=[key for _,key in self.stack if key is not None]
        if a.get('id'): active.append(a['id'])
        if tag=='a':
            for key in active:self.blocks[key]['links'].append(a.get('href',''))
        if tag not in self.VOID:self.stack.append((tag,a.get('id')))
    def handle_endtag(self, tag):
        if self.stack and self.stack[-1][0]==tag:self.stack.pop()
    def handle_data(self, data):
        for _,key in self.stack:
            if key is not None:self.blocks[key]['text'].append(data)


def check_home_outputs(html: str, data8: dict, lang: str) -> None:
    """Check outcome units/scopes next to home metrics, not a scientific verdict."""
    p=Regions();p.feed(html)
    require(all(key in p.blocks for key in ('output-q','output-r','project-007','project-006')), 'Missing outcome region')
    q=data8['tracks']['Q'];r=data8['tracks']['R']
    qb,rb=p.blocks['output-q'],p.blocks['output-r']
    qt=' '.join(qb['text']);rt=' '.join(rb['text'])
    for value in q['weight_bytes'].values():
        require(f'{value/10**9:.3f} GB' in qt, 'Home weight-file decimal GB mismatch')
    require(f"{100*q['file_reduction_fraction']:.1f}%" in qt, 'Home file reduction mismatch')
    require(q['model'].split('/')[-1] in qt and 'GPTQ W4A16' in qt, 'Home Q model/precision scope missing')
    require(('Weight files' if lang=='en' else '가중치 파일') in qt, 'Home Q file scope missing')
    values=[100*v['recovery']['pooled_recovery'] for v in r['structures'].values()]
    require('strict' in rt and ('meta skeleton' if lang=='en' else 'meta skeleton') in rt, 'Home R loader implementation missing')
    scope=('One MLP layer','192 short synthetic held-out prompts') if lang=='en' else ('한 층 MLP','짧은 합성 평가 입력 192개')
    require(r['model'].split('/')[-1] in rt and all(term in rt for term in scope), 'Home R error/model/input scope missing')
    for key in ('q','r'):
        require('case008.html#'+key+'-evaluation' in p.blocks['output-'+key]['links'], 'Missing direct quality/runtime link')
    for case,scope in [('007',('One MLP layer','16 channel groups') if lang=='en' else ('한 층 MLP','16개 채널 그룹')),('006',('One attention layer','Fixed synthetic questions') if lang=='en' else ('한 층 attention','고정 합성 질문'))]:
        block=p.blocks['project-'+case]
        require(all(term in ' '.join(block['text']) for term in scope), 'Missing tool scope')
        require('case'+case+'.html#design' in block['links'], 'Missing detailed study link')


def check(root: Path, site: Path) -> dict:
    require(site.is_dir(), 'Missing built site')
    paths={}
    for p in site.rglob('*'):
        require(not p.is_symlink(), 'Site symlink')
        if p.is_file():
            require(p.stat().st_nlink == 1, 'Site hardlink')
            paths[p.relative_to(site).as_posix()]=p
    data9=load_case009(root)
    expected=EXPECTED|({'en/case009.html','ko/case009.html','data/case009.json','data/case009.js','assets/case009.js','assets/serving-L128.svg','assets/serving-L1024.svg',
      'sources/diova-compare-core.py.txt','sources/diova-compare-tests.py.txt','sources/diova-compare-edge-tests.py.txt','sources/case009-serving.py.txt','sources/case009-quality.py.txt','sources/case009-report-en.md.txt','sources/case009-report-ko.md.txt','sources/case009-notice.md.txt'} if data9 else set())
    require(set(paths)==expected, 'Unexpected/missing deploy artifact member')
    m=read(site/'build_manifest.json')
    require(set(m['files'])==expected-{'build_manifest.json'}, 'Manifest coverage')
    for name,path in paths.items():
        if name!='build_manifest.json':require(sha(path)==m['files'][name], 'Changed build member: '+name)
    require(m['source_manifest_sha256']==sha(root/'presentation/source_manifest.json'), 'Wrong source manifest')
    for name,digest in m['presentation_files'].items():
        path=(root/name).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file() and sha(path)==digest, 'Stale presentation build')
    data8=load_case008(root)
    if data9:
        require(read(site/'data/case009.json')==data9,'Case009 numeric/source mismatch')
        wrapped=(site/'data/case009.js').read_text()
        prefix='window.CASE009_EVIDENCE = '
        require(wrapped.startswith(prefix) and wrapped.endswith(';\n'),'Case009 offline data wrapper')
        require(json.loads(wrapped[len(prefix):-2])==data9,'Case009 offline numeric/source mismatch')
        require(m['case009_units']['source_files']==data9['sources'],'Case009 manifest source identity')
        for name,path in SOURCE_COPIES.items():
            require((site/'sources'/name).read_bytes()==(root/path).read_bytes(),'Changed source copy: '+name)
        for length in (128,1024):
            require((site/f'assets/serving-L{length}.svg').read_bytes()==(root/C9/f'figures/serving-L{length}.svg').read_bytes(),'Changed serving figure')
        for lang in ('en','ko'):
            require(render_case009(data9,lang) in (site/lang/'case009.html').read_text(),'Case009 rendering/scope mismatch')
    require(read(site/'data/case008.json') == data8, 'Case008 display mismatch')
    require(m['case008_source_manifest_sha256']==sha(root/'presentation/case008_sources.json'), 'Wrong Case008 source manifest')
    for lang in ('en','ko'):
        require(render_case008(root,data8,lang) in (site/lang/'case008.html').read_text(), 'Case008 rendered values/scope mismatch')
    payloads=load(root)
    for case,expected in payloads.items():
        require(read(site/f'data/{case}.json')==expected, 'Display numeric/source mismatch')
        text=(site/f'data/{case}.js').read_text()
        require(text.startswith('window.EVIDENCE = ') and text.endswith(';\n'), 'Wrong JS data wrapper')
        require(json.loads(text[len('window.EVIDENCE = '):-2])==expected, 'JS/JSON mismatch')
        require('<' not in text and '\u2028' not in text and '\u2029' not in text, 'Unsafe embedded JSON')
    for case, data in payloads.items():
        for lang in ('en','ko'):
            html=(site/f'{lang}/{case}.html').read_text()
            positions=[html.find('id="'+section+'"') for section in (*SECTION_IDS,'explorer')]
            require(min(positions)>=0 and positions==sorted(positions), 'Study reading order/coverage')
            require(result_html(data,lang) in html, 'Study table values/scope differ from retained evidence')
    linked=m['linked_reports']['case008']
    require(linked['revision']==C8_REVISION,'Wrong Case008 report revision')
    require(linked['files']=={C8_PATH+'/'+name:sha(root/C8_PATH/name) for name in ('REPORT.md','REPORT.ko.md')},'Changed linked report')
    for lang in ('en','ko'):
        home=(site/lang/'index.html').read_text()
        check_home_outputs(home,data8,lang)
        require('id="case008-report"' in home and '#case-008' in home,'Missing Case008 home/archive entry')
        for page in ('index','guide'):
            require('href="'+report8_url(lang)+'"' in (site/lang/(page+'.html')).read_text(),'Missing same-language Case008 report link')
    n=0
    for name,path in paths.items():
        if path.suffix!='.html':continue
        parser=Links();parser.feed(path.read_text())
        for tag,url in parser.links:
            u=urlsplit(url)
            if u.scheme:
                require(tag=='a' and u.scheme=='https' and u.hostname=='github.com', 'External runtime dependency/unsafe link')
                continue
            require(not url.startswith(('/', '//')) and '\\' not in url, 'Nonportable local URL')
            dest=(path.parent/unquote(u.path)).resolve() if u.path else path
            require(dest.is_relative_to(site.resolve()) and dest.is_file(), 'Broken site link: '+url)
            if u.fragment:
                target=Links();target.feed(dest.read_text());require(unquote(u.fragment) in target.ids, 'Broken site anchor')
            n+=1
    # Static privacy bounds supplement the retained-source allowlist, not arbitrary redaction.
    forbidden=('ghp_','hf_','sk-proj-','BEGIN PRIVATE KEY')
    private_path=re.compile(r'/home/[^/\s]+/|/mnt/[a-z]/|[A-Za-z]:\\Users\\')
    for path in paths.values():
        text=path.read_text()
        require(not private_path.search(text) and not any(marker in text for marker in forbidden), 'Private path/credential marker in site')
    return {'status':'PASS','files':len(paths),'local_links':n,'records':m['record_counts'],
            'evidence_revision':m['evidence_revision'], 'browser_check':'SEPARATE',
            'scope':'Display projection equality, source hashes, artifact allowlist, local paths and static privacy checks'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=ROOT)
    p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=new_output(a.repo,a.output);result=check(a.repo,a.site)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json_text(result)+'\n');print(json_text(result))
if __name__=='__main__':main()
