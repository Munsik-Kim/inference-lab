"""Build a small bilingual, file:// compatible display of preserved evidence."""
from __future__ import annotations
import argparse
from html import escape
from pathlib import Path
from string import Template
import hashlib
import subprocess
from common import ROOT, C6, C7, SUP, json_text, read, new_output, require, sha, script_json, source_manifest
from data import load, source_url
from layout import home, language_index, report8_url, site_navigation, related_cases, C8_PATH, C8_REVISION
from study import render as study_render, method_labels
from case008 import load_case008, render_case008
from case009 import load_case009, render_case009, C9, SOURCE_COPIES
from case010 import load_case010, render_case010, C10, SOURCE_COPIES as C10_SOURCES, FIGURE_SOURCE, FIGURE_ASSET
from case011 import load_case011, load_items, render_case011, C11, SOURCE_COPIES as C11_SOURCES, figures as case011_figures
from case012_review import review_copy
from case012_pages import load_case012, render_case012, render_annotation, annotation_data, image_viewer, C12, SOURCE_COPIES as C12_SOURCES
from case013 import load_case013, render_case013, C13, SOURCE_COPIES as C13_SOURCES, FIGURES as C13_FIGURES
from case014 import load_case014, render_case014, C14, SOURCE_COPIES as C14_SOURCES, FIGURES as C14_FIGURES, ARCHIVE as C14_ARCHIVE, TITLES as C14_TITLES
from practical import load_practical, render as practical_render, PAGES as GUIDE_PAGES, TITLES as GUIDE_TITLES

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
    data10 = load_case010(root)
    data11 = load_case011(root)
    data12 = load_case012(root)
    data13 = load_case013(root)
    data14 = load_case014(root)
    practical = load_practical(root, data8, data9, data13)
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
    files['data/practical-guide.json'] = (json_text(practical)+'\n').encode()
    files['assets/practical-guide.css'] = (root/'presentation/assets/practical-guide.css').read_bytes()
    if data9:
        files['data/case009.json'] = (json_text(data9)+'\n').encode()
        files['data/case009.js'] = ('window.CASE009_EVIDENCE = '+script_json(data9)+';\n').encode()
        files['assets/case009.js'] = (root/'presentation/assets/case009.js').read_bytes()
        # Ship read-only source copies matching this build for offline inspection.
        for name,path in SOURCE_COPIES.items():files['sources/'+name]=(root/path).read_bytes()
        for length in (128,1024):
            files[f'assets/serving-L{length}.svg']=(root/C9/f'figures/serving-L{length}.svg').read_bytes()
    if data10:
        files['data/case010.json'] = (json_text(data10)+'\n').encode()
        files[FIGURE_ASSET] = (root/FIGURE_SOURCE).read_bytes()
        for name,path in C10_SOURCES.items():
            files['sources/'+name] = (root/path).read_bytes()
        files['assets/case010.js'] = (root/'presentation/assets/case010.js').read_bytes()
    if data11:
        items11 = load_items(root)
        files['data/case011.json'] = (json_text(data11)+'\n').encode()
        files['data/case011-items.json'] = (json_text(items11)+'\n').encode()
        files['data/case011-items.js'] = ('window.CASE011_ITEMS = '+script_json(items11)+';\n').encode()
        for name,path in C11_SOURCES.items():
            files['sources/'+name] = (root/path).read_bytes()
        for name in ('case011.js','case011.css'):
            files['assets/'+name] = (root/'presentation/assets'/name).read_bytes()
        files.update(case011_figures(root))
    if data12:
        human12 = annotation_data(root, data12)
        copy12 = review_copy(root,data12)
        files['data/case012-review-copy.json'] = (json_text(copy12)+'\n').encode()
        files['data/case012-review-copy.js'] = ('window.CASE012_REVIEW_COPY = '+script_json(copy12)+';\n').encode()
        files['data/case012.json'] = (json_text(data12)+'\n').encode()
        files['data/case012-annotation.json'] = (json_text(human12)+'\n').encode()
        files['data/case012-annotation.js'] = ('window.CASE012_ANNOTATION = '+script_json(human12)+';\n').encode()
        for name,path in C12_SOURCES.items(): files['sources/'+name] = (root/path).read_bytes()
        for asset,path in data12['images'].items(): files[asset] = (root/path).read_bytes()
        for name in ('case012.css', 'case012-annotation.js', 'case012-images.js', 'case012-groups.js'):
            files['assets/'+name] = (root/'presentation/assets'/name).read_bytes()
        files['assets/case012-review.css'] = (root/C12/'demo/style.css').read_bytes()
    if data13:
        files['data/case013.json'] = (json_text(data13)+'\n').encode()
        for name,path in C13_SOURCES.items(): files['sources/'+name] = (root/path).read_bytes()
        for name,path in C13_FIGURES.items(): files[name] = (root/path).read_bytes()
    if data14:
        files['data/case014.json'] = (json_text(data14)+'\n').encode()
        files['data/case014-download.json'] = (root/'downloads'/C14_ARCHIVE.replace('.zip','.json')).read_bytes()
        files['assets/case014.css'] = (root/'presentation/assets/case014.css').read_bytes()
        for name,path in C14_SOURCES.items(): files['sources/'+name] = (root/path).read_bytes()
        for name,path in C14_FIGURES.items(): files[name] = (root/path).read_bytes()
    def versioned_case012(name):
        return '../'+name+'?v='+hashlib.sha256(files[name]).hexdigest()[:16]
    files['index.html'] = language_index().encode()
    case_paths = sorted(p.parent.relative_to(root).as_posix() for p in (root/'cases').glob('*/README.md'))
    names = {'en':['Execution compatibility','BF16 / FP8 extraction','Softmax approximation','Complete attention cost','Precision–cost settings','Answer decisions and ties','Structured MLP pruning','Model build, reload and MLP reconstruction'],
             'ko':['실행 호환성','BF16 / FP8 문서 추출','Softmax 수치 근사','Attention 전체 호출 비용','정밀도와 비용의 절충','답변 선택과 동률','구조화 MLP 압축','모델 제작·재실행과 MLP 출력 복구']}
    # IDs, translated archive labels, and verified projections are explicit.
    ids = [int(Path(path).name.split('-')[0]) for path in case_paths]
    require(ids == list(range(1, len(ids)+1)) and 8 <= len(ids) <= 14,
            'Unexpected case identity/count; extend the explicit display registry')
    registry = {
        9: (data9, 'Graph serving and official quality evaluation', 'Graph 요청 비용과 공식 품질 평가'),
        10: (data10, 'Recurrent-state storage, restart and readout horizon', '반복 상태의 저장·재시작과 판독 수명'),
        11: (data11, 'Same state, different readout', '같은 상태, 다른 판독기'),
        # Current AI assessment and blank human form retain the historical scope.
        12: (data12, 'Image-generation budget: loops vs steps', '이미지 생성 시간: 반복과 단계'),
        13: (data13, 'Weight error, real answers and tool use', '가중치 오차와 실제 답변·도구 사용'),
        14: (data14, 'Same-budget INT4 compensation and whole-model quality', '같은 INT4 저장량과 전층 품질 보정'),
    }
    for case_id in ids[8:]:
        data, en, ko = registry[case_id]
        require(data is not None, f'Case{case_id:03} README requires verified display records')
        names['en'].append(en); names['ko'].append(ko)
    for lang,t in languages.items():
        t['methodLabels']=method_labels(lang)
        files[f'assets/{lang}.js'] = ('window.TEXT = '+script_json(t)+';\n').encode()
        other = 'ko' if lang == 'en' else 'en'
        for page in ('index','case008','case007','case006','guide')+GUIDE_PAGES+(('case009',) if data9 else ())+(('case010',) if data10 else ())+(('case011',) if data11 else ())+(('case012','case012-annotate') if data12 else ())+(('case013',) if data13 else ())+(('case014',) if data14 else ()):
            scripts = ''
            if page == 'index':
                body = home(t, lang, payloads, case_paths, names[lang], rev, data8, data9, data10, data11, data12, data13, practical_render(root, practical, lang, 'featured'), data14=data14)
            elif page in GUIDE_PAGES:
                body = practical_render(root, practical, lang, page)
            elif page == 'case014':
                body = render_case014(data14, lang)
            elif page == 'case013':
                body = render_case013(data13, lang)
            elif page == 'case012':
                body = render_case012(data12, lang)
            elif page == 'case012-annotate':
                body = render_annotation(lang)
                scripts = '<script defer src="../data/case012-annotation.js"></script><script defer src="../data/case012-review-copy.js"></script><script defer src="../assets/case012-groups.js"></script><script defer src="../assets/case012-annotation.js"></script>'
                for asset in ('data/case012-annotation.js','data/case012-review-copy.js','assets/case012-groups.js','assets/case012-annotation.js'):
                    scripts=scripts.replace('../'+asset,versioned_case012(asset))
            elif page == 'case011':
                body = render_case011(data11,lang)
                scripts = '<script defer src="../data/case011-items.js"></script><script defer src="../assets/case011.js"></script>'
            elif page == 'case010':
                body = render_case010(data10,lang)
                scripts = '<script defer src="../assets/case010.js"></script>'
            elif page == 'case009':
                body = render_case009(data9,lang)
                scripts = '<script defer src="../data/case009.js"></script><script defer src="../assets/case009.js"></script>'
            elif page == 'case008':
                body = render_case008(root, data8, lang)
                scripts = '<script defer src="../assets/case008.js"></script>'
            elif page == 'guide':
                command='OUT=$(mktemp -d)\npython3 -B tools/showcase/build.py --output "$OUT/site" --base-path /inference-lab/\npython3 -B tools/showcase/check.py --site "$OUT/site" --output "$OUT/site-check.json"\npython3 -B tools/showcase/replay_selection.py --output "$OUT/selection.json"'
                body=f'<h1>{t["guide"]}</h1><p>{t["guide_intro"]}</p>'
                body+='<div class="guide-role-links">'+link('llm-guide.html',t['navModels'])+' · '+link('measurements.html',t['navMeasurements'])+'</div>'
                body+='<h2>'+escape(t['guideBuildTitle'])+'</h2><pre>'+escape(command)+'</pre>'
                body+='<details><summary>'+escape(t['guideSelectorTitle'])+'</summary><p>'+escape(t['guide_result'])+'</p></details><p>'+escape(t['guide_browser'])+'</p>'
                body+=link('https://github.com/Munsik-Kim/inference-lab/blob/main/docs/'+lang+'/GETTING_STARTED.md',t['guide'])
                body+='<div class="actions">'+link('https://github.com/Munsik-Kim/inference-lab/blob/main/docs/'+lang+'/START_HERE.md',t['beginner'])+' · '+link('https://github.com/Munsik-Kim/inference-lab/blob/main/docs/'+lang+'/GLOSSARY.md',t['glossary'])+'</div>'
                body+='<div class="actions">'+link(source_url(rev,C7+'/src/surgery.py'),t['code'])+' · '+link(source_url(rev,C6+'/src/intervention.py'),t['code'])+'</div>'
                body+='<h2>Case 008</h2><p>'+escape(t['homeIntro8'])+'</p>'+link(report8_url(lang),t['report8'])
            else:
                body=explorer_body(t,page,study_render(root,payloads[page],lang))
                path=C7 if page=='case007' else C6
                body += '<div class="actions">'+link(source_url(rev,path+'/REPRODUCTION.md'),t['original'])+'</div>'
                scripts=f'<script defer src="../assets/{lang}.js"></script><script defer src="../data/{page}.js"></script><script defer src="../assets/state.js"></script><script defer src="../assets/app.js"></script>'
            page_rev = 'Case013 recorded reference variant (file hashes in JSON)' if page=='case013' else 'Case012 recorded PNGs with separate AI and grouped human assessments (file hashes in JSON)' if page.startswith('case012') else 'Case011 original snapshot and post-hoc file hashes (JSON)' if page=='case011' else 'Case010 v1/v2 recorded snapshots (file hashes in JSON)' if page=='case010' else 'Case009 reviewed snapshot (file hashes in JSON)' if page=='case009' else C8_REVISION if page == 'case008' else rev
            if page=='case014':page_rev='Case014 P0/P1/P1b/P2 recorded scalar sources; execution and build revisions are separate'
            if page in GUIDE_PAGES:
                page_rev = 'Recorded public evidence '+practical['evidence_revision']
            elif page == 'index':
                page_rev = 'Per-case source hashes in build manifest; practical guide evidence '+practical['evidence_revision']
            body+=related_cases(lang,page)
            body+=f'<details class="provenance"><summary>{t["evidence"]}</summary><p class="small">{t["evidence"]}: <code>{page_rev}</code> · '+link('../build_manifest.json',t['presentation'])+'</p></details>'
            navigation = site_navigation(lang,page)
            notice = '../sources/case013-notice.md.txt' if page=='case013' else '../sources/case012-notice.md.txt' if page.startswith('case012') else '../sources/case011-notice.md.txt' if page=='case011' else '../sources/case010-notice.md.txt' if page=='case010' else '../sources/case009-notice.md.txt' if page=='case009' else source_url(C8_REVISION,C8_PATH+'/NOTICE.md') if page=='case008' else source_url(rev,C7+'/NOTICE.md')
            if page=='case014':notice='../sources/case014-notice.md.txt'
            if page in GUIDE_PAGES:
                notice = 'qwen3-4b.html#implementation'
            language_label = ('English summary' if lang=='ko' else '한국어 상세') if page in GUIDE_PAGES else t['language']
            titles = {'index':t['homeHeroTitle'], 'guide':t['guide'], 'case006':t['case006'], 'case007':t['case007'], 'case008':t['homeTitle8'],
                      'case009':'Case 009 · Serving and quality' if lang=='en' else 'Case 009 · 요청 비용과 품질',
                      'case010':'Case 010 · Recurrent memory' if lang=='en' else 'Case 010 · 반복 상태와 기억 수명',
                      'case011':'Case 011 · Same state, different readout' if lang=='en' else 'Case 011 · 같은 상태, 다른 판독기',
                      'case013':'Case 013 · Weight error and real answers' if lang=='en' else 'Case 013 · 가중치 오차와 실제 답변',
                      'case014':'Case 014 · '+C14_TITLES[lang],
                      'case012':'Case 012 · Loops vs steps' if lang=='en' else 'Case 012 · 이미지 생성 시간 배분',
                      'case012-annotate':'Case 012 · Human image annotation' if lang=='en' else 'Case 012 · 직접 정답 체크하기',
                      **dict(zip(GUIDE_PAGES,GUIDE_TITLES[lang]))}
            page_title = titles[page]
            description = t['pageDescriptions'].get(page,t['pageDescriptions']['case'])
            html=template.substitute(lang=lang,title=escape(page_title),description=escape(description,quote=True),brand=escape(t['brand']),brand_expansion=escape(t['brandExpansion']),navigation=navigation,navlabel=t['navigation'],asset_prefix='..',case=page,skip='Skip to content' if lang=='en' else '본문으로 이동',other=other,page=page+'.html',language=language_label,body=body,footer=t['foot'],license=source_url(rev,'LICENSE'),notice=notice,scripts=scripts)
            if page in ('index', *GUIDE_PAGES):
                html=html.replace('</head>', '<link rel="stylesheet" href="../assets/practical-guide.css"></head>')
            if page in ('case006','case007','case010','case011','case013','case014'):
                html=html.replace('</head>', '<link rel="stylesheet" href="../assets/study.css"></head>')
            if page in ('index','case014') and data14:
                html=html.replace('</head>', '<link rel="stylesheet" href="../assets/case014.css"></head>')
            if page.startswith('case012'):
                html = html.replace('</head>', '<link rel="stylesheet" href="../assets/study.css"><link rel="stylesheet" href="'+versioned_case012('assets/case012.css')+'"></head>')
            if page in ('index','case011') and data11:
                html=html.replace('</head>', '<link rel="stylesheet" href="../assets/case011.css"></head>')
            if page == 'case008':
                html=html.replace('</head>', '<link rel="stylesheet" href="../assets/case008.css"></head>')
            files[f'{lang}/{page}.html']=html.encode()
        if data12:
            files[f'{lang}/case012-images.html'] = image_viewer(root, data12, lang).encode()
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
    report['practical_guide_units'] = {'source_files': practical['sources'], 'field_map': practical['field_map'],
        'evidence_revision': practical['evidence_revision'], 'evidence_kind': practical['evidence_kind'],
        'source_manifest_sha256': sha(root/'presentation/practical_sources.json'),
        'rounding': 'Decimal GB 3 places; file reduction 1 place; ratios and times 3 places; exact values in data/practical-guide.json',
        'new_gpu_runs': 0, 'english': 'Summary only; complete practical narrative in Korean'}
    if data9:
        report['case009_units'] = {
            'timing': 'three paired server rounds per condition; concurrent requests share a scheduler',
            'quality': 'task-specific paired items; MMLU subject clusters; WikiText document denominators',
            'source_identity': data9['source_identity'],
            'source_files': data9['sources'],
        }
    if data10:
        report['case010_units'] = {
            'stages': 'separate v1 512-input and v2 1024-input evaluations of the same three checkpoints',
            'storage': 'serialized bytes; N128 capacity is separate from 1024 evaluation inputs',
            'source_files': data10['sources'],
        }
    if data11:
        report['case011_units'] = {**data11['units'], 'source_files': data11['sources'],
                                  'items_source': {C11+'/publication/posthoc/data/items.json': sha(root/C11/'publication/posthoc/data/items.json')}}
    if data12:
        report['case012_units'] = {**data12['units'], 'source_files': data12['sources'],
                                  'human_input_sha256': human12['image_set_sha256'],
                                  'image_assets': {name:data12['sources'][path] for name,path in data12['images'].items()},
                                  'human_evaluation': data12['human_evaluation']}
    if data13:
        report['case013_units'] = {**data13['units'], 'source_files': data13['sources'],
                                  'scope':data13['scope'], 'packed_gate':data13['packed']}
    if data14:
        report['case014_units'] = {'source_files':data14['sources'],'model_revision':data14['model_revision'],
             'stages':['P0','P1','P1b','P2'],'P2_documents':{'PG19':64,'ArXiv_content_groups':16},'primary':data14['p2']['primary'],
             'execution_source_sha256':data14['execution_source_sha256'],'new_publication_GPU_runs':0,'scope':data14['p2']['scope'],'download':data14['download']}
    git_head=subprocess.run(['git','rev-parse','HEAD'],cwd=root,text=True,capture_output=True)
    git_dirty=subprocess.run(['git','status','--porcelain'],cwd=root,text=True,capture_output=True)
    report['repository_build_revision']=git_head.stdout.strip() if git_head.returncode==0 else None
    report['repository_worktree_dirty']=bool(git_dirty.stdout.strip()) if git_dirty.returncode==0 else None
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
