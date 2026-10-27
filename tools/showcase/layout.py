"""Presentation-only layouts. Selection diagrams use retained group IDs."""
from html import escape
from case010 import home_case010
from case011 import home_case011
from case012_pages import home_case012
from case013 import home_case013
from case014 import home_case014
from case015 import home_case015
from case016 import home_case016
from case017 import home_case017
from case018 import home_case018
from case019 import home_case019
from case020 import home_case020
from case021 import home_case021

C8_PATH = 'cases/008-build-reconstruct-reload'
C8_REVISION = '9bc8b8fced8dfd5147f9bfdc67564eda04670810'


def report8_url(lang: str) -> str:
    if lang not in ('en', 'ko'):
        raise ValueError('Unsupported report language')
    name = 'REPORT.ko.md' if lang == 'ko' else 'REPORT.md'
    return f'https://github.com/Munsik-Kim/inference-lab/blob/{C8_REVISION}/{C8_PATH}/{name}'


def anchor(url: str, label: str, cls: str = '') -> str:
    return f'<a class="{cls}" href="{escape(url, quote=True)}">{escape(label)}</a>'


def site_navigation(lang: str, page: str) -> str:
    labels = ('홈', '모델별 안내', '전체 사례', '재현 방법') if lang == 'ko' else ('Home', 'Model guides', 'All cases', 'Reproduce')
    targets = ('index.html', 'llm-guide.html', 'index.html#cases', 'guide.html')
    return ''.join('<a'+(' aria-current="page"' if target == page+'.html' else '')
                   +' href="'+target+'">'+escape(label)+'</a>' for target,label in zip(targets, labels))


def related_cases(lang: str, page: str) -> str:
    pairs = {
        'case006': [('case007.html', '행렬을 실제로 줄인 비교', 'Physically smaller matrices'), ('case008.html#q-evaluation', '압축 저장본의 답변 비교', 'Answers after checkpoint compression')],
        'case007': [('case006.html', '같은 질문의 답변 비교', 'Answers on matched questions'), ('case008.html#track-r', 'Qwen3-0.6B의 고정 구조 복원', 'Fixed-structure Qwen3-0.6B reconstruction')],
        'case008': [('case009.html', '압축한 모델의 실제 요청 처리 비용 비교', 'Request-processing costs of the saved compressed model'), ('qwen3-4b.html', 'Qwen3-4B 모델별 안내', 'Qwen3-4B model guide')],
        'case009': [('case008.html#track-q', '비교에 사용한 저장본 제작·재실행', 'Building and reloading the compared artifact'), ('qwen3-4b.html#quality', '과제별 품질과 남은 문제', 'Task quality and remaining problems')],
        'case010': [('case011.html', '같은 상태에서 판독기만 바꾼 후속', 'Follow-up with a different readout')],
        'case011': [('case010.html', '앞선 상태 저장·재시작 실험', 'Earlier state storage and restart study')],
        'case012': [('case012-annotate.html', '저장된 이미지 직접 확인', 'Check the recorded images')],
        'case012-annotate': [('case012.html#human-results', '사람 평가 결과와 실험 조건', 'Human assessment and study conditions')],
        'case013': [('qwen3-4b.html#mitigation', '이 품질 확인용 구현의 도구 작업 실패를 줄인 후속', 'Follow-up reducing tool failures in this quality-check implementation'), ('case009.html#quality', '별도 GPTQ 실행 경로의 품질 기록', 'Quality in the separate GPTQ execution path'), ('case014.html#overview', '별도 0.6B 모델 전체의 가중치 보정 연구', 'Separate 0.6B whole-model weight-compensation study')],
        'case014': [('case013.html#overview', '다른 4비트 품질 확인용 구현의 답변 진단', 'Answer diagnostics for a different 4-bit quality-check implementation'), ('case008.html#track-r', '한 MLP의 별도 구조 복원', 'Separate one-MLP structure reconstruction')],
        'case018': [('case017.html#overview', '앞선 업무 완주·관측 계약 검사', 'Earlier workflow/observation contract evaluation'), ('case016.html#storage', 'NVFP4 저장·서빙의 별도 검증', 'Separate NVFP4 storage/serving verification')],
        'case017': [('case015.html#p2-recovery', '앞선 선택 실패 복구 진단', 'Earlier selected-failure recovery diagnosis'), ('case016.html#storage', '실제 NVFP4 KV 저장·서빙 검증', 'Native NVFP4 KV storage and serving verification')],
        'case016': [('case017.html#overview', '후속 합성 업무의 호출 규칙·완주 비용', 'Follow-up action contracts and workflow costs'), ('case009.html#serving', '별도 가중치 압축 모델의 서빙 비교', 'Serving a separate weight-compressed model'), ('case015.html#overview', '별도 도구 인터페이스 회귀시험', 'Separate tool-interface regression')],
        'case015': [('case013.html#overview', '앞선 INT4 도구 품질 진단', 'Earlier INT4 tool quality diagnostics'), ('qwen3-4b.html#mitigation', '별도 도구 계약 완화 연구', 'Separate tool-contract mitigation study')],
    }
    if page not in pairs:
        return ''
    title = '관련 사례와 다음 단계' if lang == 'ko' else 'Related cases and next steps'
    links = [(url,ko if lang == 'ko' else en) for url,ko,en in pairs[page]]
    links.append(('index.html#cases', '전체 사례' if lang == 'ko' else 'All cases'))
    return '<nav class="related-pages" aria-label="'+title+'"><strong>'+title+'</strong>'+''.join(anchor(url,label) for url,label in links)+'</nav>'


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
         revision: str, data8: dict | None = None, data9: dict | None = None, data10: dict | None = None, data11: dict | None = None, data12: dict | None = None, data13: dict | None = None,
         practical_stories: str = '', data14: dict | None = None, data15: dict | None = None, data16: dict | None = None, data17: dict | None = None, data18: dict | None = None, data19: dict | None = None, data20: dict | None = None, data21: dict | None = None) -> str:
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
    out = '<section class="cap-hero"><p class="eyebrow">'+text('homeAuthor')+'</p>'
    out += '<p class="brand-definition" lang="en">Deep-learning Inference Optimization, Validation &amp; Analysis</p>'
    out += '<h1>'+text('homeHeroTitle')+'</h1><p class="hero-subtitle">'+text('homeHeroSubtitle')+'</p><p class="lede">'+text('homeOutcomeIntro')+'</p>'
    out += '<div class="actions home-jumps">'
    for url, label in [('llm-guide.html','모델별 안내' if lang=='ko' else 'Model guides'),('#featured-stories','대표 구현 · 직접 사용하기' if lang=='ko' else 'Featured work · try it')]:
        out += anchor(url,label,'reading-link')
    out += '</div></section>'+practical_stories
    out += '<section id="collaboration" class="model-entry home-model-guides"><h2>'+text('collaboration')+'</h2><p>'+text('collaborationIntro')+'</p><ul>'
    work = ('내 그래픽카드에서 쓸 모델·실행 설정을 비교하고 느린 구간 찾기', '모델을 바꾼 뒤 전에 되던 답변·도구 작업이 계속 되는지 확인', '흩어진 실행 결과를 같은 질문끼리 비교하는 도구와 재실행 안내로 정리') if lang=='ko' else ('Compare models and execution settings for your graphics card; find slow operations', 'Check whether previously working answers and tool tasks still work after a model change', 'Turn scattered execution results into matched-question comparison tools and rerun instructions')
    out += ''.join('<li>'+escape(item)+'</li>' for item in work)+'</ul>'+anchor('https://github.com/Munsik-Kim/inference-lab/issues',t['contact'])+'</section>'
    out += '<section id="contributions" class="cap-section"><h2>'+text('contributionTitle')+'</h2><details class="implementation-details"><summary>'+('구현별 코드·테스트 펼쳐 보기' if lang=='ko' else 'Open implementation code and tests')+'</summary><div id="capabilities"><div id="tech-stack" class="contribution-table" role="region" tabindex="0" aria-label="'+text('contributionTitle')+'"><table><thead><tr>'
    out += ''.join('<th scope="col">'+escape(x)+'</th>' for x in t['contributionHeaders'])+'</tr></thead><tbody>'
    contribution_sources = [('tools/modelpack/artifact.py','cases/008-build-reconstruct-reload/tests/test_core.py'),('tools/modelpack/quantized.py','cases/008-build-reconstruct-reload/tests/test_boundaries.py'),('tools/modelpack/numerics.py','cases/008-build-reconstruct-reload/tests/test_core.py'),('packages/diova-compare/src/diova_compare/core.py','packages/diova-compare/tests/test_compare.py')]
    for i,(row,(code,test)) in enumerate(zip(t['contributionRows'],contribution_sources)):
        prefix='https://github.com/Munsik-Kim/inference-lab/blob/'+(C8_REVISION if code.startswith('tools/modelpack/') else 'main')+'/'
        code_url='../sources/diova-compare-core.py.txt' if data9 and code.startswith('packages/') else prefix+code
        test_url='../sources/diova-compare-tests.py.txt' if data9 and code.startswith('packages/') else prefix+test
        row_id = ' id="model-structure"' if i == 0 else ''
        out += '<tr'+row_id+'><td>'+escape(row[0])+'</td><td>'+escape(row[1])+'</td><td>'+anchor(code_url,'Code')+' · '+anchor(test_url,'Tests')+'</td></tr>'
    for cid, number, impl, test in (
        ('gpu-comparison', 2, 'cases/002-bf16-fp8-document-extraction/scripts/run.py', 'cases/004-low-precision-attention-break-even/src/timing.py'),
        ('result-delivery', 3, 'tools/showcase/replay_selection.py', 'tests/showcase/test_showcase.py'),
    ):
        prefix = source if number == 2 else 'https://github.com/Munsik-Kim/inference-lab/blob/main/'
        out += '<tr id="'+cid+'"><td>'+text(f'cap{number}Title')+'</td><td>'+text(f'cap{number}Tech')+'</td><td>'+anchor(prefix+impl,'Code')+' · '+anchor(prefix+test,'Tests')+'</td></tr>'
    if data14:
        out += '<tr><td>'+('앞선 층을 압축하면서 달라진 중간 계산값을 다음 층에서 수집해 가중치 보정·4비트 저장/재로딩·모델 전체 품질 비교' if lang=='ko' else 'Collect intermediate values changed by earlier-layer compression, fit the next layer’s weights, store/reload at 4 bits and compare whole-model quality')+'</td><td>Guo compensation · PyTorch · Transformers · NumPy</td><td>'+anchor('../sources/case014-runner.py.txt','Code')+' · '+anchor('../sources/case014-tests.py.txt','Tests')+'</td></tr>'
    if data17:
        out += '<tr><td>'+('도구 호출 규칙·생성 전 관측·실제 업무 완주와 비용 감사' if lang=='ko' else 'Action contracts, pre-generation observations, workflow completion and cost audits')+'</td><td>Python stdlib · vLLM · FlashInfer · xgrammar</td><td>'+anchor('../sources/case017-runner.py.txt','Code')+' · '+anchor('../sources/case017-tests.py.txt','Tests')+'</td></tr>'
    if data16:
        out += '<tr><td>'+('실제 NVFP4 KV 저장·읽기 검사와 고정 pool 동시 요청 비교' if lang=='ko' else 'Native NVFP4 KV storage/read checks and fixed-pool concurrent requests')+'</td><td>vLLM · FlashInfer · Python stdlib</td><td>'+anchor('../sources/case016-runner.py.txt','Code')+' · '+anchor('../sources/case016-tests.py.txt','Tests')+'</td></tr>'
    if data15:
        out += '<tr><td>'+('API 이름 변경·실제 상태·생성 전 관측 회귀시험' if lang=='ko' else 'API rename, actual state and pre-generation observation regression')+'</td><td>Qwen · PyTorch · Transformers · Python stdlib</td><td>'+anchor('../sources/case015-a1-runner.py.txt','Code')+' · '+anchor('../sources/case015-a1-tests.py.txt','Tests')+'</td></tr>'
    out += '</tbody></table></div></div><p>'+anchor(docs+'PORTFOLIO.md#code-tour',t['stackEvidence'],'text-link')+'</p></details></section>'
    out += home_case018(data18, lang) if data18 else ''
    out += home_case019(data19, lang) if data19 else ''
    out += home_case020(data20, lang) if data20 else ''
    out += home_case021(data21, lang) if data21 else ''
    out += home_case017(data17, lang) if data17 else ''
    out += home_case016(data16, lang) if data16 else ''
    out += home_case015(data15, lang) if data15 else ''
    out += home_case014(data14, lang) if data14 else ''
    out += '<details class="research-records"><summary>'+('다른 구현과 연구 기록 펼쳐 읽기' if lang=='ko' else 'Explore other implementations and research records')+'</summary>'
    out += '<span id="studies" class="home-anchor" aria-hidden="true"></span><section id="projects" class="studies"><div class="section-heading"><div><p class="eyebrow">IMPLEMENTATION & RESEARCH RECORDS</p><h2>'+text('navProjects')+'</h2></div><p>'+text('studiesIntro')+'</p></div><div class="cards">'
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
        if isq:
            out += '<p class="project-scope">'+text(key+'Metric')+' · '+metrics['reduction']+'% · '+escape(metrics['weight_files'])+'</p>'
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
    if data9:
        out += '<section class="panel" id="serving-study"><p class="eyebrow">CASE 009 · SERVING & QUALITY</p><h2>'+('길게 답하거나 여러 요청을 함께 처리할 때의 비용' if lang=='ko' else 'Costs of longer answers and concurrent requests')+'</h2><p>'+('원래 Qwen 4B와 가중치를 4비트로 줄인 모델의 요청 측정 코드를 구현했습니다. 출력 256토큰, 동시 요청 1/4/16/32에서 토큰당 시간(TPOT)과 초당 생성량을 비교하고, 별도로 실행한 공식 품질 평가도 보여 줍니다.' if lang=='ko' else 'Implemented a request-measurement runner for original Qwen 4B and its 4-bit-weight version. Compare time per output token (TPOT) and tokens per second at 256 output tokens and concurrency 1/4/16/32, alongside separate official quality tasks.')+'</p><figure><img class="serving-curve" src="../assets/serving-L128.svg" alt="'+('입력 128·출력 256의 TPOT·처리량 전체 곡선' if lang=='ko' else 'Complete TPOT and throughput curves, input128/output256')+'"><figcaption>'+('서버 3라운드 · 막대는 라운드 범위 · 영어 축 라벨. 품질 계산의 프로세스 종료 실패는 상세 표에 표시합니다.' if lang=='ko' else 'Three server rounds; whiskers show round range. Quality-process exit failures are marked in the detailed table.')+'</figcaption></figure>'+anchor('case009.html#serving','전체 요청 비용 곡선' if lang=='ko' else 'Complete serving curves','text-link')+' · '+anchor('case009.html#quality','공식 품질과 실행 상태' if lang=='ko' else 'Official quality and execution status','text-link')+'</section>'
    if data10:
        out += home_case010(data10, lang)
    if data11:
        out += home_case011(data11, lang)
    if data12:
        out += home_case012(data12, lang)
    if data13:
        out += home_case013(data13, lang)
    out += '</details>'
    reading = '<section class="reading-strip"><div><p class="eyebrow">START HERE</p><h2>'+text('readingTitle')+'</h2><p>'+text('readingIntro')+'</p></div><div class="reading-links">'
    for url, key in [('llm-guide.html','navModels'),('measurements.html','navMeasurements'),('guide.html','guide'),(docs+'START_HERE.md','beginner'),(docs+'GLOSSARY.md','glossary')]:
        reading += anchor(url,t[key]+' ↗','reading-link')
    reading += '</div></section>'
    out += '<section class="archive" id="cases"><div class="section-heading"><div><p class="eyebrow">CASE ARCHIVE</p><h2>'+text('cases')+'</h2></div>'+anchor(docs+'CASEBOOK.md',t['casebook'],'text-link')+'</div><ol class="case-list">'
    out += '</ol><details><summary>'+text('archiveEarlier')+'</summary><ol class="case-list">'
    for i,(path,name) in enumerate(zip(case_paths,names),1):
        if i == 6: out += '</ol></details><ol class="case-list">'
        url=f'case{i:03}.html' if i>=9 else docs+f'CASEBOOK.md#case-{i:03}'
        out += '<li>'+anchor(url,name,'case-name')+f'<span class="archive-number">{i:03}</span><span class="archive-arrow" aria-hidden="true">↗</span></li>'
    out += '</ol></section>'+reading
    return out


def language_index() -> str:
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'self' file:; style-src 'self' file:; object-src 'none'; base-uri 'none'"><link rel="icon" href="assets/icon.svg"><title>DIOVA</title><link rel="stylesheet" href="assets/style.css"></head><body class="language-page"><main class="language-card"><span class="brand-mark" aria-hidden="true">D</span><p class="eyebrow">DEEP LEARNING · INFERENCE · OPTIMIZATION</p><h1>DIOVA</h1><p>Deep-learning Inference Optimization, Validation &amp; Analysis</p><p lang="ko">딥러닝 추론 최적화 · 검증 · 분석</p><div class="language-options"><a href="ko/index.html" lang="ko"><span>한국어</span><span aria-hidden="true">→</span></a><a href="en/index.html"><span>English</span><span aria-hidden="true">→</span></a></div><p class="small">Recorded experiments · 저장된 실험 결과</p></main></body></html>'''
