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
from case015 import load_case015, render_case015, C15, SOURCE_COPIES as C15_SOURCES, ARCHIVE as C15_ARCHIVE, TITLES as C15_TITLES
from case016 import load_case016, render_case016, C16, SOURCE_COPIES as C16_SOURCES, FIGURES as C16_FIGURES, TITLES as C16_TITLES
from case017 import load_case017, render_case017, C17, SOURCE_COPIES as C17_SOURCES, FIGURES as C17_FIGURES, TITLES as C17_TITLES, EVIDENCE_ASSET as C17_ARCHIVE, EVIDENCE_SOURCE as C17_ARCHIVE_SOURCE
from practical import load_practical, render as practical_render, PAGES as GUIDE_PAGES, TITLES as GUIDE_TITLES, COMPARE_EXAMPLES

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
    data15 = load_case015(root)
    data16 = load_case016(root)
    data17 = load_case017(root)
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
    for name, path in COMPARE_EXAMPLES.items():
        files[name] = (root/path).read_bytes()
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
    if data15:
        files['data/case015.json'] = (json_text(data15)+'\n').encode()
        files['data/case015-download.json'] = (root/'downloads'/C15_ARCHIVE.replace('.zip','.json')).read_bytes()
        for name in ('case015.css','case015.js'): files['assets/'+name] = (root/'presentation/assets'/name).read_bytes()
        for name,path in C15_SOURCES.items(): files['sources/'+name] = (root/path).read_bytes()
    if data16:
        files['data/case016.json'] = (json_text(data16)+'\n').encode()
        files['assets/case016.css'] = (root/'presentation/assets/case016.css').read_bytes()
        for name,path in C16_SOURCES.items(): files['sources/'+name] = (root/path).read_bytes()
        for name,path in C16_FIGURES.items(): files[name] = (root/path).read_bytes()
    if data17:
        files['data/case017.json'] = (json_text(data17)+'\n').encode()
        files[C17_ARCHIVE] = (root/C17_ARCHIVE_SOURCE).read_bytes()
        for name in ('case017.css','case017.js'): files['assets/'+name] = (root/'presentation/assets'/name).read_bytes()
        for name,path in C17_SOURCES.items(): files['sources/'+name] = (root/path).read_bytes()
        for name,path in C17_FIGURES.items(): files[name] = (root/path).read_bytes()
    def versioned_case012(name):
        return '../'+name+'?v='+hashlib.sha256(files[name]).hexdigest()[:16]
    files['index.html'] = language_index().encode()
    case_paths = sorted(p.parent.relative_to(root).as_posix() for p in (root/'cases').glob('*/README.md'))
    names = {'en':['Check execution in the target environment','Compare document extraction after changing number precision','Approximate probability calculations and measure error','Measure the full cost of connecting input tokens','Compare calculation precision, speed and memory use','Track changed answers and equal top scores','Remove internal channels to shrink a model block','Save and reload a model; repair a separate smaller model'],
             'ko':['목표 환경에서 모델이 실행되는지 확인','숫자 정밀도를 바꾼 뒤 문서 추출 비교','확률 계산을 간단히 했을 때의 오차 확인','입력 간 연결 계산 전체의 처리 비용 측정','계산 정밀도·속도·메모리 사용량 비교','바뀐 답변과 최고점이 같은 답 추적','내부 계산 채널을 골라 모델 블록 축소','모델 저장·재실행과 별도 소형 모델의 출력 복구']}
    # IDs, translated archive labels, and verified projections are explicit.
    ids = [int(Path(path).name.split('-')[0]) for path in case_paths]
    require(ids == list(range(1, len(ids)+1)) and 8 <= len(ids) <= 17,
            'Unexpected case identity/count; extend the explicit display registry')
    registry = {
        9: (data9, 'Process real requests and compare answer quality', '실제 요청 처리 비용과 답변 품질 비교'),
        10: (data10, 'Save model memory, resume it and test consecutive correct answers', '모델의 중간 기억을 저장·재시작하고 연속 정답 확인'),
        11: (data11, 'Compare rules for turning saved model memory into answers', '저장된 기억을 답으로 바꾸는 규칙 비교'),
        # Current AI assessment and blank human form retain the historical scope.
        12: (data12, 'Image-generation budget: loops vs steps', '이미지 생성 시간: 반복과 단계'),
        13: (data13, 'Compare compressed model values, real answers and tool actions', '모델 압축 오차와 실제 답변·도구 작업 비교'),
        14: (data14, 'Test quality correction at the same 4-bit weight-storage budget', '같은 4비트 가중치 저장량에서 품질 보정 시험'),
        15: (data15, 'Same tools, different interfaces: can an AI finish after a rename?', '같은 도구, 다른 이름: 이름이 바뀌어도 AI가 일을 끝낼까?'),
    }
    registry[16] = (data16, 'Verify native NVFP4 KV storage, capacity and serving costs', '실제 NVFP4 KV 저장·수용량·서빙 비용 검증')
    registry[17] = (data17, 'Agent output contracts, workflow completion and measured costs', '도구 호출 규칙·실제 업무 완주·비용 비교')
    for case_id in ids[8:]:
        data, en, ko = registry[case_id]
        require(data is not None, f'Case{case_id:03} README requires verified display records')
        names['en'].append(en); names['ko'].append(ko)
    for lang,t in languages.items():
        t['methodLabels']=method_labels(lang)
        files[f'assets/{lang}.js'] = ('window.TEXT = '+script_json(t)+';\n').encode()
        other = 'ko' if lang == 'en' else 'en'
        for page in ('index','case008','case007','case006','guide')+GUIDE_PAGES+(('case009',) if data9 else ())+(('case010',) if data10 else ())+(('case011',) if data11 else ())+(('case012','case012-annotate') if data12 else ())+(('case013',) if data13 else ())+(('case014',) if data14 else ())+(('case015',) if data15 else ())+(('case016',) if data16 else ())+(('case017',) if data17 else ()):
            scripts = ''
            if page == 'index':
                body = home(t, lang, payloads, case_paths, names[lang], rev, data8, data9, data10, data11, data12, data13, practical_render(root, practical, lang, 'featured'), data14=data14, data15=data15, data16=data16, data17=data17)
            elif page in GUIDE_PAGES:
                body = practical_render(root, practical, lang, page)
            elif page == 'case017':
                body = render_case017(data17, lang)
                scripts = '<script defer src="../assets/case017.js"></script>'
            elif page == 'case016':
                body = render_case016(data16, lang)
            elif page == 'case015':
                body = render_case015(data15, lang)
                scripts = '<script defer src="../assets/case015.js"></script>'
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
            source_notes = {
                'case013': ('Saved quality tests using reconstructed compressed values; source checksums in JSON', '압축 값을 일반 정밀도로 복원해 계산한 품질 기록. JSON에 원자료 내용 대조값 기록'),
                'case012': ('Saved original images; AI assessments and human checks are separate; source checksums in JSON', '저장된 원본 이미지와 구분된 AI 평가·사람의 체크 기록. JSON에 원자료 내용 대조값 기록'),
                'case011': ('Original saved tests and later output-rule analysis; source checksums in JSON', '원래 저장한 실험과 나중에 수행한 출력 규칙 분석. JSON에 원자료 내용 대조값 기록'),
                'case010': ('Saved results from separate v1 and v2 tests; source checksums in JSON', '서로 나누어 수행한 v1·v2 시험의 저장된 결과. JSON에 원자료 내용 대조값 기록'),
                'case009': ('Reviewed saved request and quality tests; source checksums in JSON', '검토한 요청 처리·품질 시험의 저장된 기록. JSON에 원자료 내용 대조값 기록')
            }
            note_key = 'case012' if page.startswith('case012') else page
            page_rev = source_notes[note_key][lang=='ko'] if note_key in source_notes else C8_REVISION if page == 'case008' else rev
            if page=='case014':page_rev='Saved numbers from Case014 stages P0/P1/P1b/P2; experiment code and page-build versions are separate' if lang=='en' else 'Case014의 P0·P1·P1b·P2 단계에 저장된 수치. 실험 코드 버전과 이 화면을 만든 버전은 구분'
            if page=='case017':page_rev='Case017 stored P1/P2 measurements; disclosed expansion-gate deviation; Fresh is posthoc; no new publication GPU runs'
            if page=='case016':page_rev='Native NVFP4 KV technical P1-T; preserved agent P0/P1 and independent CPU audit'
            if page=='case015':page_rev='Case015 original P1 plus measured A1; raw scalar/fixture replay and distinct execution source hashes'
            if page in GUIDE_PAGES:
                page_rev = ('Recorded public evidence ' if lang=='en' else '공개된 실험 기록의 버전 ')+practical['evidence_revision']
            if page == 'compare':
                page_rev = 'Synthetic CPU example; no model execution or historical GPU measurement' if lang=='en' else '합성 CPU 예제: 모델 실행이나 과거 GPU 실측 자료가 아님'
            elif page == 'index':
                page_rev = ('Source checksums for each case are in the page-build record; guide evidence version ' if lang=='en' else '사례별 원자료 내용 대조값은 화면 빌드 기록에 수록. 사용 안내의 근거 버전 ')+practical['evidence_revision']
            body+=related_cases(lang,page)
            source_help = 'Use these versions and file checksums (hashes) to trace the records behind this page and compare file contents.' if lang=='en' else '이 화면이 어떤 기록과 코드 버전을 참고했는지 확인할 수 있습니다. 파일 내용 대조값(해시)은 같은 파일인지 비교하는 데 씁니다.'
            body+=f'<details class="provenance"><summary>{t["evidence"]}</summary><p class="small">{source_help}</p><p class="small">{t["evidence"]}: <code>{page_rev}</code> · '+link('../build_manifest.json',t['presentation'])+'</p></details>'
            navigation = site_navigation(lang,page)
            notice = '../sources/case013-notice.md.txt' if page=='case013' else '../sources/case012-notice.md.txt' if page.startswith('case012') else '../sources/case011-notice.md.txt' if page=='case011' else '../sources/case010-notice.md.txt' if page=='case010' else '../sources/case009-notice.md.txt' if page=='case009' else source_url(C8_REVISION,C8_PATH+'/NOTICE.md') if page=='case008' else source_url(rev,C7+'/NOTICE.md')
            if page=='case017':notice='../sources/case017-notice.md.txt'
            if page=='case016':notice='../sources/case016-notice.md.txt'
            if page=='case015':notice='../sources/case015-notice.md.txt'
            if page=='case014':notice='../sources/case014-notice.md.txt'
            if page in GUIDE_PAGES:
                notice = 'qwen3-4b.html#implementation'
            if page == 'compare':
                notice = 'https://github.com/Munsik-Kim/inference-lab/blob/main/packages/diova-compare/README'+('.ko' if lang=='ko' else '')+'.md'
            language_label = ('English summary' if lang=='ko' else '한국어 상세') if page in GUIDE_PAGES and page != 'compare' else t['language']
            titles = {'index':t['homeHeroTitle'], 'guide':t['guide'], 'case006':t['case006'], 'case007':t['case007'], 'case008':t['homeTitle8'],
                      'case009':'Case 009 · Serving and quality' if lang=='en' else 'Case 009 · 요청 비용과 품질',
                      'case010':'Case 010 · Save model memory and test correct-answer length' if lang=='en' else 'Case 010 · 모델의 중간 기억 저장과 연속 정답',
                      'case011':'Case 011 · Turn saved model memory into answers' if lang=='en' else 'Case 011 · 저장된 기억을 답으로 바꾸기',
                      'case013':'Case 013 · Weight error and real answers' if lang=='en' else 'Case 013 · 가중치 오차와 실제 답변',
                      'case014':'Case 014 · '+C14_TITLES[lang],
                      'case015':'Case 015 · '+C15_TITLES[lang],
                      'case016':'Case 016 · '+C16_TITLES[lang],
                      'case017':'Case 017 · '+C17_TITLES[lang],
                      'case012':'Case 012 · Loops vs steps' if lang=='en' else 'Case 012 · 이미지 생성 시간 배분',
                      'case012-annotate':'Case 012 · Human image annotation' if lang=='en' else 'Case 012 · 직접 정답 체크하기',
                      **dict(zip(GUIDE_PAGES,GUIDE_TITLES[lang]))}
            page_title = titles[page]
            description = t['pageDescriptions'].get(page,t['pageDescriptions']['case'])
            html=template.substitute(lang=lang,title=escape(page_title),description=escape(description,quote=True),brand=escape(t['brand']),brand_expansion=escape(t['brandExpansion']),navigation=navigation,navlabel=t['navigation'],asset_prefix='..',case=page,skip='Skip to content' if lang=='en' else '본문으로 이동',other=other,page=page+'.html',language=language_label,body=body,footer=t['foot'],license=source_url(rev,'LICENSE'),license_label='Code license (Apache-2.0)' if lang=='en' else '코드 이용 조건 (Apache-2.0)',notice=notice,notice_label='Sources and terms (NOTICE)' if lang=='en' else '출처·이용 조건 (NOTICE)',scripts=scripts)
            if page in ('index', *GUIDE_PAGES):
                html=html.replace('</head>', '<link rel="stylesheet" href="../assets/practical-guide.css"></head>')
            if page in ('case006','case007','case010','case011','case013','case014','case015','case016','case017'):
                html=html.replace('</head>', '<link rel="stylesheet" href="../assets/study.css"></head>')
            if page in ('index','case014') and data14:
                html=html.replace('</head>', '<link rel="stylesheet" href="../assets/case014.css"></head>')
            if page in ('index','case017') and data17:
                html=html.replace('</head>', '<link rel="stylesheet" href="../assets/case017.css"></head>')
            if page in ('index','case016') and data16:
                html=html.replace('</head>', '<link rel="stylesheet" href="../assets/case016.css"></head>')
            if page in ('index','case015') and data15:
                html=html.replace('</head>', '<link rel="stylesheet" href="../assets/case015.css"></head>')
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
    if data15:
        report['case015_units'] = {'source_files':data15['sources'],'model_revision':data15['model_revision'],
            'stages':['P0','P1','A1'],'P1_episodes':96,'A1_episodes':96,'A1_endpoint':data15['A1']['endpoint'],
            'execution_source_sha256':data15['execution_source_sha256'],'publication_additional_model_forwards':0,
            'scope':data15['execution_boundary'],'download':data15['download']}
        if data15.get('P2'):
            report['case015_units']['stages'].append('P2')
            report['case015_units']['P2'] = data15['P2']
    if data16:
        report['case016_units'] = {'source_files':data16['display_source_files'], 'execution_source_sha256':data16['execution_source_sha256'], 'model_revision':data16['model_revision'], 'status':data16['status'], 'mode':data16['mode'], 'stages':['original-agent-P0-P1','baseline-audit','technical-P1-T']}
    if data17:
        report['case017_units'] = {'source_files':data17['display_source_files'], 'execution_source_sha256':data17['execution_source_sha256'], 'model_revision':data17['model_revision'], 'stages':['P0','P1-DEV64','P2-exploratory-HOLDOUT192'], 'gate':data17['gate'], 'new_publication_model_requests':0, 'P3':'NOT_RUN'}
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
