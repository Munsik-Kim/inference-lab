"""Reader routes and presentation only; numerical claims use verified loaders."""
from html import escape

TOPICS = {
    'execution': ((1, 2, 4, 5, 8, 9, 16, 20), '실행 비용·메모리', 'Execution & memory'),
    'reliability': ((6, 13, 15, 17, 18, 21), '답변·작업 신뢰성', 'Answers & reliability'),
    'research': ((3, 7, 10, 11, 12, 14, 19), '수치·연구 구현', 'Numerical research'),
}

def label(ko, en, lang):
    return ko if lang == 'ko' else en

def a(url, text, cls=''):
    return f'<a class="{cls}" href="{escape(url, quote=True)}">{escape(text)}</a>'

def hero(t, lang, count):
    h='<section class="cap-hero"><div class="portfolio-intro"><p class="eyebrow">'+escape(t['homeAuthor'])+' · INFERENCE ENGINEERING</p>'
    h+='<h1>'+escape(t['homeHeroTitle'])+'</h1><p class="hero-subtitle">'+escape(t['homeHeroSubtitle'])+'</p>'
    h+='<p class="lede">'+escape(t['homeOutcomeIntro'])+'</p><div class="actions home-jumps">'
    h+=a('#featured-stories',label('대표 작업부터 보기','Explore selected work',lang),'reading-link')
    h+=a('llm-guide.html',label('모델 실행 안내','Model execution guide',lang),'reading-link')+'</div></div>'
    h+='<aside class="portfolio-map" aria-labelledby="portfolio-map-title"><p class="eyebrow">HOW I WORK</p><h2 id="portfolio-map-title">'+label('변경에서 근거까지','From a change to evidence',lang)+'</h2><ol>'
    for n,title,desc in [('01',label('구현','Implement',lang),label('저장·실행 경로와 비교 도구를 만듭니다.','Build execution paths and comparison tools.',lang)),('02',label('비교','Compare',lang),label('같은 입력으로 비용과 결과를 함께 봅니다.','Compare costs and outcomes on matched inputs.',lang)),('03',label('검산','Verify',lang),label('실패와 한계를 남기고 재현 경로를 제공합니다.','Keep failures, limits and a path to reproduction.',lang))]:
        h+='<li><span class="step-number" aria-hidden="true">'+n+'</span><div><strong>'+title+'</strong><p>'+desc+'</p></div></li>'
    h+='</ol><p class="portfolio-index">'+a('#cases',label(f'{count}개 사례의 코드와 기록',f'Code and records across {count} cases',lang))+'</p></aside></section>'
    return h

def latest(data20, data21, lang):
    if data20 is None or data21 is None:
        raise ValueError('Latest portfolio entries require verified Case020/021')
    f20, f21 = data20['facts'], data21['facts']
    h='<section class="portfolio-latest" id="latest-work" aria-labelledby="latest-title"><div class="section-heading"><div><p class="eyebrow">RECENT INVESTIGATIONS</p><h2 id="latest-title">'+label('최근에는 이 질문을 더 확인했습니다','Recent questions, measured carefully',lang)+'</h2></div>'+a('#cases',label('전체 사례 보기','Browse all cases',lang),'text-link')+'</div><div class="latest-grid">'
    h+='<article><p class="eyebrow">CASE 020 · MEMORY / COST</p><h3>'+a('case020.html#overview',label('작게 보관하면 실행도 빨라질까?','Does smaller storage also run faster?',lang))+'</h3><p>'+label('전체를 미리 복원하는 방법과 필요한 항목만 복원하는 방법의 메모리·호출·재사용 비용을 비교했습니다.','Compared memory, call costs and reuse when reconstructing a whole table or only requested entries.',lang)+'</p>'
    h+='<p class="finding"><strong>'+str(f20['P4_C_faster_blocks'])+'/'+str(f20['P4_C_B_block_comparisons'])+'</strong> '+label('후속 host 비교에서 중복 제거 C가 B보다 빨랐던 process block','follow-up host blocks where deduplication C was faster than B',lang)+'</p><p class="scope">'+label('합성 OPTIC operator 비교 · 모델 전체 속도와 별개','Synthetic OPTIC operator study · separate from whole-model speed',lang)+'</p>'+a('case020.html#reuse',label('준비 비용과 반복 사용의 차이 읽기','Read setup and reuse costs',lang),'text-link')+'</article>'
    h+='<article><p class="eyebrow">CASE 021 · NUMERICAL AGREEMENT</p><h3>'+a('case021.html#overview',label('같은 INT4인데 값은 왜 달라질까?','Why can the same INT4 artifact differ?',lang))+'</h3><p>'+label('검사한 단일항은 dense/Marlin에서 같았지만, 실제 값을 여러 개 결합하면 작은 차이가 남았습니다.','Tested single terms matched across dense and Marlin; combining actual values left small differences.',lang)+'</p>'
    h+='<p class="finding verdict"><strong>'+escape(f21['original_primary_verdict'])+'</strong> '+label(f"{f21['primary_documents']}문서에서 중간 정밀도 증가가 평균 KL을 줄였는지는 미확정",f"No established mean KL reduction from higher intermediate precision across {f21['primary_documents']} documents",lang)+'</p><p class="scope">'+label('Qwen3-0.6B · 사후 진단은 원 주 판정과 구분','Qwen3-0.6B · post-hoc probes remain separate from the primary verdict',lang)+'</p>'+a('case021.html#single-term',label('반올림 후보와 실제 관측 읽기','Read rounding candidates and observations',lang),'text-link')+'</article></div></section>'
    return h

def catalog(case_paths, names, lang):
    if len(case_paths)!=len(names): raise ValueError('Case title coverage')
    ids=[int(p.split('/')[-1].split('-')[0]) for p in case_paths]
    if ids!=list(range(1,22)): raise ValueError('Explicit portfolio topic mapping covers Cases001–021')
    assignments={n:key for key,(nums,_,_) in TOPICS.items() for n in nums}
    if set(assignments)!=set(ids) or sum(len(v[0]) for v in TOPICS.values())!=len(ids):raise ValueError('Topic coverage/duplicates')
    h='<section class="archive portfolio-catalog" id="cases" aria-labelledby="case-catalog-title"><div class="section-heading"><div><p class="eyebrow">RESEARCH INDEX</p><h2 id="case-catalog-title">'+label('관심 있는 문제에서 시작하기','Start with the problem you care about',lang)+'</h2></div>'+a('https://github.com/Munsik-Kim/inference-lab/blob/main/docs/'+lang+'/CASEBOOK.md',label('전체 연구 기록','Full casebook',lang),'text-link')+'</div>'
    h+='<p class="catalog-intro">'+label('21개 사례는 서로 다른 질문과 측정 범위의 기록입니다. 번호순 순위가 아니며, 초기 5개 사례는 GitHub 실험 노트로 연결됩니다.','These 21 cases ask different questions with different measurement scopes. They are not a performance ranking; the first five link to GitHub experiment notes.',lang)+'</p>'
    h+='<div id="case-controls" class="catalog-controls" hidden><label for="case-search">'+label('사례·키워드 검색','Search cases or keywords',lang)+'<input id="case-search" type="search" autocomplete="off" placeholder="'+label('예: 021, 메모리, INT4','e.g. 021, memory, INT4',lang)+'"></label><label for="case-topic">'+label('분야','Topic',lang)+'<select id="case-topic"><option value="all">'+label('전체 분야','All topics',lang)+'</option>'
    for key,(_,ko,en) in TOPICS.items(): h+='<option value="'+key+'">'+label(ko,en,lang)+'</option>'
    h+='</select></label><button class="secondary" id="case-reset" type="button">'+label('초기화','Reset',lang)+'</button></div>'
    h+='<p id="case-count" class="catalog-count" role="status" aria-live="polite" data-total="21" data-lang="'+lang+'">'+label('전체 21개 사례','All 21 cases',lang)+'</p><ol id="case-catalog" class="case-list">'
    for n,name in reversed(list(zip(ids,names))):
        key=assignments[n];_,ko,en=TOPICS[key]
        url=f'case{n:03}.html' if n>=6 else f'https://github.com/Munsik-Kim/inference-lab/blob/main/docs/{lang}/CASEBOOK.md#case-{n:03}'
        h+=f'<li data-case-id="{n:03}" data-topic="{key}"><span class="archive-number">{n:03}</span>'+a(url,name,'case-name')+'<span class="case-topic">'+label(ko,en,lang)+'</span><span class="archive-arrow" aria-hidden="true">↗</span></li>'
    h+='</ol><p id="case-empty" hidden>'+label('일치하는 사례가 없습니다. 검색어나 분야를 바꿔 주세요.','No matching case. Try another keyword or topic.',lang)+'</p></section>'
    return h

def reader_note(lang):
    return '<section class="portfolio-boundary" aria-labelledby="reader-note-title"><h2 id="reader-note-title">'+label('무엇을 직접 만들었나요?','What did I build?',lang)+'</h2><p>'+label('직접 만든 부분은 실험 fixture, 실행 adapter, 계측·비교·검산 도구와 재현 문서입니다. 기반 모델, 논문의 방법, Marlin·PyTorch·vLLM은 출처를 밝히고 재사용했습니다. 성공한 결과와 실패·미실행 범위를 함께 공개합니다.','The work includes fixtures, execution adapters, measurement and comparison tools, audits and reproduction guides. Base models, paper methods, Marlin, PyTorch and vLLM are attributed and reused. Successful results stay alongside failures and unexecuted work.',lang)+'</p><div class="actions">'+a('#contributions',label('구현별 코드·테스트','Code and tests by contribution',lang))+a('https://github.com/Munsik-Kim/inference-lab/blob/main/docs/'+lang+'/PORTFOLIO.md',label('역할·기술 판단·협업 범위','Role, engineering decisions and scope',lang))+a('compare.html#run',label('모델 없이 도구 사용하기','Try a tool without a model',lang))+'</div></section>'
