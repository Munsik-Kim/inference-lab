"""Presentation-only layouts. Selection diagrams use retained group IDs."""
from html import escape


def anchor(url: str, label: str, cls: str = '') -> str:
    return f'<a class="{cls}" href="{escape(url, quote=True)}">{escape(label)}</a>'


def home(t: dict, lang: str, payloads: dict, case_paths: list, names: list,
         revision: str) -> str:
    docs = f'https://github.com/Munsik-Kim/inference-lab/blob/main/docs/{lang}/'
    source = f'https://github.com/Munsik-Kim/inference-lab/blob/{revision}/'
    text = lambda key: escape(t[key])
    # A schematic of the recorded 25% selections, not an activation or timing plot.
    rows = [r for r in payloads['case007']['records'] if r['budget'] == '4']
    selections = {}
    for method in ('INDEPENDENT', 'PAIRWISE'):
        sets = {tuple(r['selected_groups']) for r in rows if r['arm'] == method}
        if len(sets) != 1:
            raise ValueError('Inconsistent recorded selection for diagram')
        selections[method] = sets.pop()
    diagram = '<div class="group-axis" aria-hidden="true">'+''.join(f'<span>{i}</span>' for i in range(16))+'</div>'
    for name, removed in [('B · BF16', ()), *selections.items()]:
        diagram += f'<div class="group-row"><div class="group-label"><span>{escape(name)}</span><span>{16-len(removed)} / 16</span></div>'
        diagram += f'<div class="group-strip" role="img" aria-label="{escape(name)}: {text("removedGroups")} {escape(str(list(removed)))}">'
        diagram += ''.join(f'<span class="group-cell {"removed" if i in removed else "retained"}" aria-hidden="true"></span>' for i in range(16))+'</div></div>'
    diagram += '<div class="diagram-key"><span><i class="key-retained"></i>'+text('retained')+'</span><span><i class="key-removed"></i>'+text('removedGroups')+'</span></div>'
    out = '<section class="hero"><div class="hero-copy"><p class="eyebrow">LLM INFERENCE ENGINEERING</p>'
    out += '<h1>'+text('heroLine1')+'<span>'+text('heroLine2')+'</span></h1><p class="lede">'+text('intro')+'</p>'
    out += '<div class="actions hero-actions">'+anchor('#studies',t['explore'],'button')+anchor(docs+'START_HERE.md',t['beginner'],'text-link')+'</div>'
    out += '<p class="hero-meta"><span>RTX 5080</span><span>'+text('savedData')+'</span><span>EN / KO</span></p></div>'
    out += '<aside class="diagram"><div class="diagram-top"><span>CASE 007</span><span>4 / 16 · 25%</span></div><h2>'+text('diagramTitle')+'</h2><p>'+text('diagramIntro')+'</p>'+diagram
    out += '<p class="diagram-caption">'+text('diagramCaption')+'</p>'+anchor(source+'cases/007-interaction-aware-mlp-pruning/results/raw/selection.json',t['recordedSelection'],'text-link')+'</aside></section>'
    out += '<section id="studies" class="studies"><div class="section-heading"><div><p class="eyebrow">SELECTED STUDIES</p><h2>'+text('selectedStudies')+'</h2></div><p>'+text('studiesIntro')+'</p></div><div class="cards">'
    s7 = payloads['case007']['summary']['transfer']['4']
    s6 = payloads['case006']['summary']['groups']['standard/L4096']['arms']
    for case in ('case007', 'case006'):
        is7 = case == 'case007'; n = '7' if is7 else '6'
        out += '<article class="card study-card"><div class="card-heading"><span class="case-number">'+case[-3:]+'</span><span class="category">'+text('category'+n)+'</span></div>'
        out += '<h3>'+anchor(case+'.html',t['homeTitle'+n])+'</h3><p class="card-intro">'+text('homeIntro'+n)+'</p><div class="result-box">'
        if is7:
            delta = s7['mean_paired_relative_error_delta']*100
            out += '<p class="result-label">'+text('localChange')+'</p><p class="result-number">'+f'{delta:+.4f}'+' <span>'+text('pp')+'</span></p>'
            out += '<p class="result-caption">'+text('localScope')+'</p><p class="interval">95% CI ['+f'{s7["ci95"][0]*100:+.4f}, {s7["ci95"][1]*100:+.4f}] '+text('pp')+'</p>'
        else:
            out += '<p class="result-label">'+text('choiceChanges')+'</p><div class="paired-counts">'
            for arm in ('A_PUBLIC','V4'):
                flips = s6[arm]['outcomes']['flips']['numerator']
                out += '<div><span class="arm-label">'+arm+'</span><p class="result-number">'+str(flips)+'<span> / 192</span></p></div>'
            out += '</div><p class="result-caption">'+text('choiceScope')+'</p>'
        out += '</div><p class="card-limit">'+text('homeLimit'+n)+'</p><div class="card-bottom">'+anchor(case+'.html',t['open']+' →','text-link')+anchor(docs+'CASEBOOK.md#case-00'+n,t['fullStudy'],'quiet-link')+'</div></article>'
    out += '</div></section>'
    out += '<section class="reading-strip"><div><p class="eyebrow">START HERE</p><h2>'+text('readingTitle')+'</h2><p>'+text('readingIntro')+'</p></div><div class="reading-links">'
    for url, key in [(docs+'START_HERE.md','beginner'),(docs+'GLOSSARY.md','glossary'),('guide.html','guide')]:
        out += anchor(url,t[key]+' ↗','reading-link')
    out += '</div></section><section class="archive"><div class="section-heading"><div><p class="eyebrow">CASE ARCHIVE</p><h2>'+text('cases')+'</h2></div>'+anchor(docs+'CASEBOOK.md',t['casebook'],'text-link')+'</div><ol class="case-list">'
    for i,(path,name) in enumerate(zip(case_paths,names),1):
        out += '<li>'+anchor(docs+f'CASEBOOK.md#case-{i:03}',name,'case-name')+f'<span class="archive-number">{i:03}</span><span class="archive-arrow" aria-hidden="true">↗</span></li>'
    out += '</ol></section><section class="contact-strip"><h2>'+text('collaboration')+'</h2><p>'+text('collaborationIntro')+'</p><div class="actions">'+anchor(docs+'PORTFOLIO.md',t['code'],'text-link')+anchor('https://github.com/Munsik-Kim/inference-lab/issues',t['contact'],'quiet-link')+'</div></section>'
    return out


def language_index() -> str:
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'self' file:; style-src 'self' file:; object-src 'none'; base-uri 'none'"><link rel="icon" href="assets/icon.svg"><title>Inference Lab</title><link rel="stylesheet" href="assets/style.css"></head><body class="language-page"><main class="language-card"><span class="brand-mark" aria-hidden="true">i</span><p class="eyebrow">LLM INFERENCE ENGINEERING</p><h1>Inference Lab</h1><p>Model changes. Measured differences.</p><p lang="ko">모델을 바꾸고, 차이를 확인합니다.</p><div class="language-options"><a href="ko/index.html" lang="ko"><span>한국어</span><span aria-hidden="true">→</span></a><a href="en/index.html"><span>English</span><span aria-hidden="true">→</span></a></div><p class="small">Recorded experiments · 저장된 실험 결과</p></main></body></html>'''
