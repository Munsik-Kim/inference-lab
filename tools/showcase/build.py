"""Build a small bilingual, file:// compatible display of preserved evidence."""
from __future__ import annotations
import argparse
from html import escape
from pathlib import Path
from string import Template
import hashlib
from common import ROOT, C6, C7, SUP, json_text, read, new_output, require, sha, script_json, source_manifest
from data import load, source_url
from layout import home, language_index, report8_url, C8_PATH, C8_REVISION
from study import render as study_render, method_labels
from case008 import load_case008, render_case008
from case009 import load_case009, render_case009, C9, SOURCE_COPIES

def link(url, text, cls=''):
    return f'<a class="{cls}" href="{escape(url, quote=True)}">{escape(text)}</a>'

def explorer_body(t: dict, case: str, introduction: str) -> str:
    c7 = case == 'case007'
    h = introduction
    h += '<p id="view-note" class="note"></p>'
    h += f'<section class="panel filter-panel"><h2>{t["filters"]}</h2><div class="filters">'
    for key in ('set','task','arm','budget' if c7 else 'readout','outcome'):
        h += f'<label>{t[key]}<select id="{key}"></select></label>'
    h += f'<label>{t["query"]}<input id="query" type="search" autocomplete="off"></label></div>'
    h += f'<div class="actions"><button id="reset" class="secondary">{t["reset"]}</button><span id="count" class="status" role="status" aria-live="polite"></span></div><div id="stats" class="stats"></div>'
    h += f'<label>{t["items"]}<select id="items"></select></label><p class="small">{t["guided"]}</p><div id="guided"></div></section>'
    h += f'<section class="panel" id="detail"><h2>{t["comparison"]}</h2><p id="item-title" class="id"></p><div class="scroll" id="comparison" tabindex="0" role="region" aria-label="{t['comparison']}"></div><h3>{t["probability"]}</h3><div id="probabilities" class="prob-grid"></div><p class="scope">{t["probHelp"]}</p><p class="scope">{t["tieHelp7" if c7 else "tieHelp"]}</p>'
    h += f'<details><summary>{t["prompt"]}</summary><pre id="prompt"></pre><a id="prompt-source">{t["source"]}</a></details><details><summary>{t["precision"]}</summary><pre id="record"></pre></details><div class="actions"><button id="download">{t["download"]}</button><a id="record-source">{t["source"]}</a></div></section>'
    h += f'<section class="panel"><h2>{t["cost"]}</h2><p>{t["costNote7" if c7 else "costNote6"]}</p><div id="cost" class="scroll"></div></section>'
    h += f'<p class="scope">{t["c7limits" if c7 else "c6limits"]}</p><p class="scope">{t["localHelp"]}</p><p class="scope">{t["scope"]}</p><button id="download-all" class="secondary">{t["downloadAll"]}</button>'
    return h

def build(root: Path, output: Path, base_path: str = '/') -> dict:
    require(base_path in ('/', '/inference-lab/'), 'Supported base paths: / or /inference-lab/')
    output = new_output(root, output)
    payloads = load(root)
    data8 = load_case008(root)
    data9 = load_case009(root)
    manifest = source_manifest(root)
    rev = manifest['evidence_revision']
    languages = {lang:read(root/f'presentation/content/{lang}.json') for lang in ('en','ko')}
    require(set(languages['en']) == set(languages['ko']), 'Missing language counterpart key')
    template = Template((root/'presentation/templates/page.html').read_text())
    files = {}
    for name in ('style.css','study.css','case008.css','case008.js','app.js','state.js','icon.svg'):
        files['assets/'+name] = (root/'presentation/assets'/name).read_bytes()
    for case, data in payloads.items():
        files['data/'+case+'.js'] = ('window.EVIDENCE = '+script_json(data)+';\n').encode()
        files['data/'+case+'.json'] = (json_text(data)+'\n').encode()
    files['data/case008.json'] = (json_text(data8)+'\n').encode()
    if data9:
        files['data/case009.json'] = (json_text(data9)+'\n').encode()
        files['data/case009.js'] = ('window.CASE009_EVIDENCE = '+script_json(data9)+';\n').encode()
        files['assets/case009.js'] = (root/'presentation/assets/case009.js').read_bytes()
        # Ship read-only source copies matching this build for offline inspection.
        for name,path in SOURCE_COPIES.items():files['sources/'+name]=(root/path).read_bytes()
        for length in (128,1024):
            files[f'assets/serving-L{length}.svg']=(root/C9/f'figures/serving-L{length}.svg').read_bytes()
    files['index.html'] = language_index().encode()
    case_paths = sorted(p.parent.relative_to(root).as_posix() for p in (root/'cases').glob('*/README.md'))
    names = {'en':['Execution compatibility','BF16 / FP8 extraction','Softmax approximation','Complete attention cost','Precision–cost settings','Answer decisions and ties','Structured MLP pruning','Model build, reload and MLP reconstruction'],
             'ko':['실행 호환성','BF16 / FP8 문서 추출','Softmax 수치 근사','Attention 전체 호출 비용','정밀도와 비용의 절충','답변 선택과 동률','구조화 MLP 압축','모델 제작·재실행과 MLP 출력 복구']}
    if len(case_paths)==9:
        require(data9 is not None,'Case009 README requires measured display records')
        names['en'].append('Graph serving and official quality evaluation')
        names['ko'].append('Graph 요청 비용과 공식 품질 평가')
    for lang,t in languages.items():
        t['methodLabels']=method_labels(lang)
        files[f'assets/{lang}.js'] = ('window.TEXT = '+script_json(t)+';\n').encode()
        other = 'ko' if lang == 'en' else 'en'
        for page in ('index','case008','case007','case006','guide')+(('case009',) if data9 else ()):
            scripts = ''
            if page == 'index':
                body = home(t, lang, payloads, case_paths, names[lang], rev, data8, data9)
            elif page == 'case009':
                body = render_case009(data9,lang)
                scripts = '<script defer src="../data/case009.js"></script><script defer src="../assets/case009.js"></script>'
            elif page == 'case008':
                body = render_case008(root, data8, lang)
                scripts = '<script defer src="../assets/case008.js"></script>'
            elif page == 'guide':
                command='OUT=$(mktemp -d)\npython3 -B tools/showcase/build.py --output "$OUT/site"\npython3 -B tools/showcase/check.py --site "$OUT/site" --output "$OUT/site-check.json"\npython3 -B tools/showcase/replay_selection.py --output "$OUT/selection.json"'
                body=f'<h1>{t["guide"]}</h1><p>{t["guide_intro"]}</p><pre>{escape(command)}</pre><p>{t["guide_result"]}</p><p>{t["guide_browser"]}</p>'
                body+=link('https://github.com/Munsik-Kim/inference-lab/blob/main/docs/'+lang+'/GETTING_STARTED.md',t['guide'])
                body+='<div class="actions">'+link('https://github.com/Munsik-Kim/inference-lab/blob/main/docs/'+lang+'/START_HERE.md',t['beginner'])+' · '+link('https://github.com/Munsik-Kim/inference-lab/blob/main/docs/'+lang+'/GLOSSARY.md',t['glossary'])+'</div>'
                body+='<div class="actions">'+link(source_url(rev,C7+'/src/surgery.py'),t['code'])+' · '+link(source_url(rev,C6+'/src/intervention.py'),t['code'])+'</div>'
                body+='<h2>Case 008</h2><p>'+escape(t['homeIntro8'])+'</p>'+link(report8_url(lang),t['report8'])
            else:
                body=explorer_body(t,page,study_render(root,payloads[page],lang))
                path=C7 if page=='case007' else C6
                body += '<div class="actions">'+link(source_url(rev,path+'/REPRODUCTION.md'),t['original'])+'</div>'
                scripts=f'<script defer src="../assets/{lang}.js"></script><script defer src="../data/{page}.js"></script><script defer src="../assets/state.js"></script><script defer src="../assets/app.js"></script>'
            page_rev = 'Case009 reviewed snapshot (file hashes in JSON)' if page=='case009' else C8_REVISION if page == 'case008' else rev
            body+=f'<details class="provenance"><summary>{t["evidence"]}</summary><p class="small">{t["evidence"]}: <code>{page_rev}</code> · '+link('../build_manifest.json',t['presentation'])+'</p></details>'
            navigation = ''.join('<a'+(' aria-current="page"' if page==target else '')+' href="'+target+'.html">'+escape(label)+'</a>' for target,label in [('case008','Case 008'),('case007','Case 007'),('case006','Case 006'),('guide',t['navGuide'])])
            if page == 'index':
                navigation = ''.join(link('#'+target,t[key]) for target,key in [('capabilities','navCapabilities'),('tech-stack','navStack'),('projects','navProjects')])
            notice = '../sources/case009-notice.md.txt' if page=='case009' else source_url(C8_REVISION,C8_PATH+'/NOTICE.md') if page=='case008' else source_url(rev,C7+'/NOTICE.md')
            html=template.substitute(lang=lang,title=escape('Case 009 · Serving and quality' if page=='case009' else t['siteTitle'] if page=='index' else t['guide'] if page=='guide' else t['homeTitle8'] if page=='case008' else t['case007' if page=='case007' else 'case006']),brand=escape(t['brand']),brand_expansion=escape(t['brandExpansion']),navigation=navigation,navlabel=t['navigation'],asset_prefix='..',case=page,skip='Skip to content' if lang=='en' else '본문으로 이동',other=other,page=page+'.html',language=t['language'],body=body,footer=t['foot'],license=source_url(rev,'LICENSE'),notice=notice,scripts=scripts)
            if page in ('case006','case007'):
                html=html.replace('</head>', '<link rel="stylesheet" href="../assets/study.css"></head>')
            if page == 'case008':
                html=html.replace('</head>', '<link rel="stylesheet" href="../assets/case008.css"></head>')
            files[f'{lang}/{page}.html']=html.encode()
    # Explicit file set: no recursive copy of repository data, caches or archives.
    source_files = [p for folder in ('presentation','tools/showcase') for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    presentation_hashes={p.relative_to(root).as_posix():sha(p) for p in sorted(source_files)}
    identity=hashlib.sha256(json_text(presentation_hashes).encode()).hexdigest()
    report={'schema':1,'build_base_path':base_path,'url_policy':'relative resources; file:// and hosted subpath use the same bytes',
            'evidence_revision':rev,'source_manifest_sha256':sha(root/'presentation/source_manifest.json'),
            'case008_source_manifest_sha256':sha(root/'presentation/case008_sources.json'),
            'linked_reports':{'case008':{'revision':C8_REVISION,'files':{C8_PATH+'/'+name:sha(root/C8_PATH/name) for name in ('REPORT.md','REPORT.ko.md')}}},
            'presentation_identity':identity,'presentation_files':presentation_hashes,
            'record_counts':{k:len(v['records']) for k,v in payloads.items()},
            'independent_scenarios':{'case007':192,'case006_standard':192,'case006_selected_stress':46,'case008_Q':192,'case008_R':192},
            'files':{name:hashlib.sha256(blob).hexdigest() for name,blob in sorted(files.items())}}
    if data9:
        report['case009_units'] = {
            'timing': 'three paired server rounds per condition; concurrent requests share a scheduler',
            'quality': 'task-specific paired items; MMLU subject clusters; WikiText document denominators',
            'source_identity': data9['source_identity'],
            'source_files': data9['sources'],
        }
    files['build_manifest.json']=(json_text(report)+'\n').encode()
    output.mkdir(parents=True)
    for name,blob in files.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(blob)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True,help='New external directory')
    p.add_argument('--base-path',default='/',choices=['/','/inference-lab/'])
    a=p.parse_args();r=build(a.repo,a.output,a.base_path)
    print(json_text({'status':'BUILT','files':len(r['files'])+1,'records':r['record_counts'],'presentation_identity':r['presentation_identity']}))
if __name__=='__main__':main()
