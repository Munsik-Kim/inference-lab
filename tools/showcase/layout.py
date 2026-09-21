"""Presentation-only layouts. Selection diagrams use retained group IDs."""
from html import escape

C8_PATH = 'cases/008-build-reconstruct-reload'
C8_REVISION = '9bc8b8fced8dfd5147f9bfdc67564eda04670810'


def report8_url(lang: str) -> str:
    if lang not in ('en', 'ko'):
        raise ValueError('Unsupported report language')
    name = 'REPORT.ko.md' if lang == 'ko' else 'REPORT.md'
    return f'https://github.com/Munsik-Kim/inference-lab/blob/{C8_REVISION}/{C8_PATH}/{name}'


def anchor(url: str, label: str, cls: str = '') -> str:
    return f'<a class="{cls}" href="{escape(url, quote=True)}">{escape(label)}</a>'


def output_metrics(data8: dict) -> dict[str, str]:
    """Format the existing source-checked projection; GB here is decimal."""
    q = data8['tracks']['Q']
    recovery = [100*v['recovery']['pooled_recovery']
                for v in data8['tracks']['R']['structures'].values()]
    return {
        'reduction': f"{100*q['file_reduction_fraction']:.1f}",
        'weight_files': f"{q['weight_bytes']['Q-BF16']/10**9:.3f} GB → {q['weight_bytes']['Q-W4']/10**9:.3f} GB",
        'recovery': f"{min(recovery):.1f}–{max(recovery):.1f}%",
    }


def home(t: dict, lang: str, payloads: dict, case_paths: list, names: list,
         revision: str, data8: dict | None = None) -> str:
    if len(case_paths) != len(names):
        raise ValueError('Every case needs a translated archive title')
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
    out = '<section class="cap-hero"><p class="eyebrow">DIOVA</p>'
    out += '<p class="brand-definition" lang="en">Deep-learning Inference Optimization, Validation &amp; Analysis</p>'
    out += '<h1>'+text('homeHeroTitle')+'</h1><p class="hero-subtitle">'+text('homeHeroSubtitle')+'</p><p class="lede">'+text('homeOutcomeIntro')+'</p>'
    out += '<p class="core-tech">Python · PyTorch · Transformers · LLM Compressor · safetensors · vLLM · NumPy</p><div class="actions home-jumps">'
    for url, key in [(docs+'PORTFOLIO.md#code-tour','homeBuildAction'),(docs+'GETTING_STARTED.md#tiny-model-demo','homeCPUAction'),('#projects','navProjects')]:
        out += anchor(url,t[key],'text-link')
    out += '</div></section><section id="capabilities" class="cap-section"><h2>'+text('capHeading')+'</h2><div class="cap-grid">'
    evidence = [
        ('model-structure', 'tools/modelpack/artifact.py', 'cases/008-build-reconstruct-reload/tests/test_core.py'),
        ('gpu-comparison', 'cases/002-bf16-fp8-document-extraction/scripts/run.py', 'cases/004-low-precision-attention-break-even/src/timing.py'),
        ('result-delivery', 'tools/showcase/replay_selection.py', 'tests/showcase/test_showcase.py'),
    ]
    # Case code is pinned to the unchanged evidence revision. Display code links to
    # the repository implementation; the build manifest identifies this local edit.
    for i,(cid,impl,test) in enumerate(evidence,1):
        prefix = (f'https://github.com/Munsik-Kim/inference-lab/blob/{C8_REVISION}/' if i == 1 else source) if i < 3 else 'https://github.com/Munsik-Kim/inference-lab/blob/main/'
        out += '<article class="cap-card" id="'+cid+'"><span class="case-number" aria-hidden="true">0'+str(i)+'</span><h3>'+text(f'cap{i}Title')+'</h3>'
        out += '<p>'+text(f'cap{i}Body')+'</p><p class="cap-tech">'+text(f'cap{i}Tech')+'</p><div class="cap-links">'
        out += anchor(prefix+impl,t[f'cap{i}Link1'])+anchor(prefix+test,t[f'cap{i}Link2'])+'</div></article>'
    out += '</div></section><section id="tech-stack" class="stack-section"><h2>'+text('stackHeading')+'</h2><dl class="stack-list">'
    for i in range(1,6):
        out += '<div><dt>'+text(f'stack{i}Title')+'</dt><dd><strong>'+text(f'stack{i}Tech')+'</strong><p>'+text(f'stack{i}Body')+'</p></dd></div>'
    out += '</dl>'+anchor(docs+'PORTFOLIO.md#code-tour',t['stackEvidence'],'text-link')+'</section>'
    out += '<span id="studies" class="home-anchor" aria-hidden="true"></span><section id="projects" class="studies"><div class="section-heading"><div><p class="eyebrow">SELECTED PROJECTS</p><h2>'+text('navProjects')+'</h2></div><p>'+text('studiesIntro')+'</p></div><div class="cards">'
    if data8 is None:
        raise ValueError('Missing Case008 display data')
    metrics = output_metrics(data8)
    modelpack = f'https://github.com/Munsik-Kim/inference-lab/blob/{C8_REVISION}/tools/modelpack/'
    out += '<article id="case008-report" class="card study-card featured-project"><div class="card-heading"><span class="case-number">008</span><span class="category">'+text('category8')+'</span></div>'
    out += '<h3>'+anchor('case008.html',t['outputTitle8'])+'</h3><p class="card-intro">'+text('outputIntro8')+'</p><div class="project-outputs">'
    for track in ('Q','R'):
        isq = track == 'Q'; key = 'output'+track
        title = t[key+'Title'].format(reduction=metrics['reduction'])
        out += '<section class="project-output" id="output-'+track.lower()+'"><p class="eyebrow">TRACK '+track+'</p><h4>'+escape(title)+'</h4><p class="output-implementation">'+text(key+'Body')+'</p>'
        out += '<div class="output-metric"><p class="metric-label">'+text(key+'Metric')+'</p><p class="output-value">'+escape(metrics['weight_files' if isq else 'recovery'])+'</p></div>'
        tech = 'LLM Compressor · compressed-tensors · vLLM' if isq else 'PyTorch · NumPy · safetensors'
        out += '<p class="project-tech">'+tech+'</p><p class="project-scope">'+text(key+'Scope')+'</p><div class="card-bottom">'
        out += anchor(modelpack+('quantized.py' if isq else 'artifact.py'),t[key+'Code'],'text-link')
        out += anchor('case008.html#track-'+track.lower(),t[key+'Process'],'text-link')
        out += anchor('case008.html#'+track.lower()+'-evaluation',t['outputEvaluation'],'quiet-link')+'</div></section>'
    out += '</div><div class="card-bottom project-evidence">'+anchor(report8_url(lang),t['outputMethods'],'text-link')+anchor(docs+'GETTING_STARTED.md#tiny-model-demo',t['homeCPUAction'],'text-link')+'</div></article>'
    for case in ('case007', 'case006'):
        is7 = case == 'case007'; n = '7' if is7 else '6'
        out += '<article id="project-'+case[-3:]+'" class="card study-card"><div class="card-heading"><span class="case-number">'+case[-3:]+'</span><span class="category">'+text('category'+n)+'</span></div>'
        out += '<h3>'+anchor(case+'.html',t['outputTitle'+n])+'</h3><p class="card-intro">'+text('outputBody'+n)+'</p>'
        out += '<ol class="output-stages">'+''.join('<li>'+escape(stage)+'</li>' for stage in t['outputStages'+n].split('|'))+'</ol>'
        out += '<p class="project-tech">'+text('projectTech'+n)+'</p><p class="project-scope">'+text('outputScope'+n)+'</p>'
        implementation = 'cases/007-interaction-aware-mlp-pruning/src/surgery.py' if is7 else 'cases/006-attention-decision-stability/src/intervention.py'
        action = docs+'GETTING_STARTED.md#cpu-selector' if is7 else case+'.html#results'
        out += '<div class="card-bottom">'+anchor(action,t['outputAction'+n],'text-link')+anchor(source+implementation,t['outputCode'+n],'quiet-link')+anchor(case+'.html#design',t['outputMethods'],'quiet-link')+'</div></article>'
    out += '</div><aside class="diagram project-diagram"><div class="diagram-description"><p class="eyebrow">CASE 007 · 4 / 16 · 25%</p><h3>'+text('diagramSection')+'</h3><p>'+text('diagramCaption')+'</p>'+anchor(source+'cases/007-interaction-aware-mlp-pruning/results/raw/selection.json',t['recordedSelection'],'text-link')+'</div><div class="diagram-data">'+diagram+'</div></aside></section>'
    reading = '<section class="reading-strip"><div><p class="eyebrow">START HERE</p><h2>'+text('readingTitle')+'</h2><p>'+text('readingIntro')+'</p></div><div class="reading-links">'
    for url, key in [(docs+'START_HERE.md','beginner'),(docs+'GLOSSARY.md','glossary'),('guide.html','guide')]:
        reading += anchor(url,t[key]+' ↗','reading-link')
    reading += '</div></section>'
    out += '<section class="archive"><div class="section-heading"><div><p class="eyebrow">CASE ARCHIVE</p><h2>'+text('cases')+'</h2></div>'+anchor(docs+'CASEBOOK.md',t['casebook'],'text-link')+'</div><ol class="case-list">'
    for i,(path,name) in enumerate(zip(case_paths,names),1):
        out += '<li>'+anchor(docs+f'CASEBOOK.md#case-{i:03}',name,'case-name')+f'<span class="archive-number">{i:03}</span><span class="archive-arrow" aria-hidden="true">↗</span></li>'
    out += '</ol></section>'+reading+'<section class="contact-strip"><h2>'+text('collaboration')+'</h2><p>'+text('collaborationIntro')+'</p><div class="actions">'+anchor(docs+'PORTFOLIO.md',t['code'],'text-link')+anchor('https://github.com/Munsik-Kim/inference-lab/issues',t['contact'],'quiet-link')+'</div></section>'
    return out


def language_index() -> str:
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'self' file:; style-src 'self' file:; object-src 'none'; base-uri 'none'"><link rel="icon" href="assets/icon.svg"><title>DIOVA</title><link rel="stylesheet" href="assets/style.css"></head><body class="language-page"><main class="language-card"><span class="brand-mark" aria-hidden="true">D</span><p class="eyebrow">DEEP LEARNING · INFERENCE · OPTIMIZATION</p><h1>DIOVA</h1><p>Deep-learning Inference Optimization, Validation &amp; Analysis</p><p lang="ko">딥러닝 추론 최적화 · 검증 · 분석</p><div class="language-options"><a href="ko/index.html" lang="ko"><span>한국어</span><span aria-hidden="true">→</span></a><a href="en/index.html"><span>English</span><span aria-hidden="true">→</span></a></div><p class="small">Recorded experiments · 저장된 실험 결과</p></main></body></html>'''
