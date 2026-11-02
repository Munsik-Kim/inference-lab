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
    h+='<div class="actions home-jumps">'
    h+=a('#featured-stories',label('대표작 보기','See selected work',lang),'reading-link')
    h+=a('compare.html#run',label('모델 없이 예제 실행','Run a model-free example',lang),'reading-link')+'</div></div>'
    h+='<aside class="portfolio-map" aria-labelledby="portfolio-map-title"><p class="eyebrow">START WITH YOUR QUESTION</p><h2 id="portfolio-map-title">'+label('무엇을 확인하러 오셨나요?','What would you like to check?',lang)+'</h2><ol>'
    for n,url,title,desc in [('01','#contributions',label('기술 역량 확인','Review the engineering',lang),label('설계 판단 → 코드와 테스트','Decisions → code and tests',lang)),('02','#tools',label('도구 사용','Use a tool',lang),label('입력 예시 → CPU 실행 → 내 기록','Example inputs → CPU run → your records',lang)),('03','#research',label('연구 검토','Review the research',lang),label('질문과 관측 → 대조·원자료','Question and findings → controls and records',lang))]:
        h+='<li><span class="step-number" aria-hidden="true">'+n+'</span><div><strong>'+a(url,title)+'</strong><p>'+desc+'</p></div></li>'
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
    h+='<p class="catalog-intro">'+label(f'{len(ids)}개 사례를 번호순으로 모았습니다. 질문이나 주제로 좁혀 보세요. 초기 5개는 GitHub 원문으로 연결됩니다.',f'{len(ids)} cases, in number order. Narrow the list by question or topic. The first five open the original GitHub notes.',lang)+'</p>'
    h+='<div id="case-controls" class="catalog-controls" hidden><label for="case-search">'+label('사례·키워드 검색','Search cases or keywords',lang)+'<input id="case-search" type="search" autocomplete="off" placeholder="'+label('예: 021, 메모리, INT4','e.g. 021, memory, INT4',lang)+'"></label><label for="case-topic">'+label('분야','Topic',lang)+'<select id="case-topic"><option value="all">'+label('전체 분야','All topics',lang)+'</option>'
    for key,(_,ko,en) in TOPICS.items(): h+='<option value="'+key+'">'+label(ko,en,lang)+'</option>'
    h+='</select></label><button class="secondary" id="case-reset" type="button">'+label('초기화','Reset',lang)+'</button></div>'
    h+='<p id="case-count" class="catalog-count" role="status" aria-live="polite" data-total="21" data-lang="'+lang+'">'+label('전체 21개 사례','All 21 cases',lang)+'</p><ol id="case-catalog" class="case-list">'
    for n,name in zip(ids,names):
        key=assignments[n];_,ko,en=TOPICS[key]
        url=f'case{n:03}.html' if n>=6 else f'https://github.com/Munsik-Kim/inference-lab/blob/main/docs/{lang}/CASEBOOK.md#case-{n:03}'
        observation=OBSERVATIONS[n-1][0 if lang=='ko' else 1]
        route=label(' · GitHub 원문',' · GitHub notes',lang) if n<6 else ''
        h+=f'<li data-case-id="{n:03}" data-topic="{key}"><span class="archive-number">{n:03}</span>'+a(url,name,'case-name')+'<span class="case-observation">'+escape(observation)+'</span><span class="case-topic">'+label(ko,en,lang)+route+'</span><span class="archive-arrow" aria-hidden="true">↗</span></li>'
    h+='</ol><p id="case-empty" hidden>'+label('일치하는 사례가 없습니다. 검색어나 분야를 바꿔 주세요.','No matching case. Try another keyword or topic.',lang)+'</p></section>'
    return h

def reader_note(lang):
    return '<section id="about" class="portfolio-boundary" aria-labelledby="reader-note-title"><p class="eyebrow">MUNSIK KIM · DIOVA</p><h2 id="reader-note-title">'+label('무엇을 직접 만들었나요?','What did I build?',lang)+'</h2><p>'+label('직접 만든 부분은 실험 입력, 실행 연결 코드, 계측·비교·검산 도구와 재현 문서입니다. 기반 모델, 논문의 방법, Marlin·PyTorch·vLLM은 출처를 밝히고 재사용했습니다. 평균 점수만으로 놓치는 변화와 실제로 실행된 경로를 확인하는 데 초점을 둡니다. Codex가 구현·분석·문서화를 지원했습니다.','I build experiment inputs, execution adapters, measurement and comparison tools, audits and reproduction guides. Base models, paper methods, Marlin, PyTorch and vLLM are attributed and reused. I focus on changes that averages miss and on checking which execution path actually ran. Codex assisted implementation, analysis and documentation.',lang)+'</p><div class="actions">'+a('#contributions',label('구현별 코드·테스트','Code and tests by contribution',lang))+a('https://github.com/Munsik-Kim/inference-lab/blob/main/docs/'+lang+'/PORTFOLIO.md',label('역할·기술 판단·협업 범위','Role, engineering decisions and scope',lang))+'</div><details class="prototype-links"><summary>'+label('초기 앱 구현도 살펴보기','Browse earlier app prototypes',lang)+'</summary><p>'+a('https://github.com/Munsik-Kim/Local-LLM-jarvis','Local-LLM-jarvis')+' · '+a('https://github.com/Munsik-Kim/ai_game','ai_game')+' · '+a('https://github.com/Munsik-Kim/Physics-Problem-Solver','Physics-Problem-Solver')+'</p><p>'+label('현재 지원 범위와 외부 서비스 사용은 각 저장소 안내를 확인하세요. 주력 측정 사례와 별도의 앱 구현 기록입니다.','Check each repository for its current scope and external services. These app prototypes are separate from the measured studies above.',lang)+'</p></details></section>'

# Editorial summaries; numbers and verdicts stay in the source-checked case loaders.
# Number order corresponds to the full CASEBOOK, not a new scientific classification.
OBSERVATIONS = (
 ('GPU 실행 경로 수정 전후를 비교했습니다.', 'Compared a GPU execution path before and after a fix.'),
 ('자원 사용은 줄었지만 문서 추출 정답률도 낮았습니다.', 'Resource use was lower; document-extraction accuracy remained low.'),
 ('비교 기준의 scale을 나누어 수치 오차를 검사했습니다.', 'Checked numerical error against separate reference scales.'),
 ('일부 긴 입력은 빨랐지만 출력 오차 기준을 넘었습니다.', 'Some long inputs ran faster but missed the output-error gate.'),
 ('속도와 오차를 함께 만족한 설정을 찾지 못했습니다.', 'No tested setting met both speed and error criteria.'),
 ('같은 질문의 답변 변화와 최고점 동률을 추적합니다.', 'Tracks changed answers and top-score ties on matched questions.'),
 ('그룹 선택에서 실제 행렬 축소·별도 입력 평가까지 연결했습니다.', 'Connects group selection, smaller matrices and separate-input evaluation.'),
 ('저장·새 프로세스 재로딩을 확인했고 정답 변화는 혼재했습니다.', 'Checked storage and fresh-process reload; answer changes were mixed.'),
 ('요청 비용과 품질을 분리하고 비정상 종료도 보존했습니다.', 'Separates request costs from quality and retains abnormal exits.'),
 ('작은 상태 저장량과 재시작 뒤 기억 수명을 따로 검사했습니다.', 'Checks smaller saved state separately from memory lifetime after restart.'),
 ('같은 상태에서 판독기 보정과 첫 오류 길이를 비교했습니다.', 'Compares readout correction and first-error length on the same state.'),
 ('같은 초기 잡음에서 생성 단계·내부 반복과 요청 조건을 비교합니다.', 'Compares steps, internal loops and prompt constraints from matched initial noise.'),
 ('INT4 저장·재로딩을 검사했지만 packed 전체 수치 gate는 미통과입니다.', 'Checks INT4 storage and reload; the packed full-model numeric gate was missed.'),
 ('텍스트 예측 품질 변화가 데이터별로 달랐습니다.', 'Text-prediction quality changes differed across datasets.'),
 ('도구 이름 변경 뒤 호출뿐 아니라 저장·확인 결과도 검사합니다.', 'Checks saved state and verification as well as calls after tool renaming.'),
 ('고정 KV pool의 수용량과 동시 요청 비용을 구분합니다.', 'Separates fixed-KV-pool capacity from concurrent request costs.'),
 ('업무 완주와 추가 비용을 함께 기록한 탐색적 결과입니다.', 'Exploratory results retain both workflow completion and added costs.'),
 ('저장 상태가 맞아도 최종 보고까지 맞는 것은 아니었습니다.', 'Correct stored state did not ensure a correct final report.'),
 ('방향 정렬·크기·손실과 유한 실험·극한 이론을 구분합니다.', 'Separates alignment, scale and loss, and finite experiments from limits.'),
 ('상주량 절감, 호출 비용, 오래 재사용한 이득은 서로 다릅니다.', 'Lower residency, call costs and gains from long reuse are different outcomes.'),
 ('단일항은 같고 결합에서 차이; 전체 개선 여부는 INCONCLUSIVE입니다.', 'Single terms matched; combinations differed. Overall improvement: INCONCLUSIVE.'),
)

def tools_section(lang):
    h='<section id="tools" class="portfolio-tools"><p class="eyebrow">USE THE TOOLS</p><h2>'+label('저장된 결과로 먼저 시작하세요','Start with results you already have',lang)+'</h2><p>'+label('답변 비교만 필요하면 diova-compare, 지원 과제의 평가·기록 관리까지 필요하면 Kit에서 시작합니다.','Use diova-compare for paired answers, or the Kit for supported-task evaluation and record handling.',lang)+'</p><div class="tool-grid">'
    entries=[('diova-compare','compare.html',label('채점한 두 JSONL → 질문별 변화 보고서','Two graded JSONL files → changes by question',lang),label('독립 CPU 도구. 입력 식별자·정답 규칙을 확인하고 report.md, report.json, pairs.csv를 만듭니다. 모델 실행과 자유 답변 채점은 사용자가 준비합니다.','A standalone CPU tool. Checks identifiers and grading rules, then writes report.md, report.json and pairs.csv. You supply model runs and grading for free-form answers.',lang),[('compare.html#run',label('작은 예제 실행','Run the small example',lang)),('compare.html#own-records',label('내 자료 준비','Prepare your records',lang))]),('inference-engineering-kit','https://github.com/Munsik-Kim/inference-engineering-kit',label('지원 과제·외부 정답 → 평가·검산 기록','Supported tasks and external gold → evaluation and audit records',lang),label('v0.2.1 배포판의 기본 wheel은 Python 표준 라이브러리 CPU 경로입니다. 필드 추출·근거 답변·선택형 과제를 지원하고 비교 기능에 diova-compare를 재사용합니다. 실제 모델/HTTP 실행 의존성은 별도로 준비합니다.','The v0.2.1 base wheel has a Python-standard-library CPU path. It supports field extraction, evidence-backed answers and multiple choice, reusing diova-compare for comparisons. Live model/HTTP execution needs separate dependencies.',lang),[('https://github.com/Munsik-Kim/inference-engineering-kit/blob/main/QUICKSTART_CPU_KO.md',label('CPU 시작 안내','CPU quickstart (Korean)',lang)),('https://github.com/Munsik-Kim/inference-engineering-kit/releases/tag/v0.2.1',label('공개 배포판 v0.2.1','Public release v0.2.1',lang))])]
    for name,url,io,desc,links in entries:
        h+='<article><h3>'+a(url,name)+'</h3><p class="tool-io">'+io+'</p><p>'+desc+'</p><div class="actions">'+''.join(a(u,t) for u,t in links)+'</div></article>'
    return h+'</div><p class="scope">'+label('설치 위치와 명령은 각 도구 안내에 있습니다. 지원하지 않는 모델·과제나 정답 자동 생성을 약속하지 않습니다.','Follow each tool’s installation instructions. Neither promises arbitrary model/task support or automatic ground-truth creation.',lang)+'</p></section>'

def research_map(lang):
    rows=[('Teacher Error Geometry','teacher-error-geometry-lab',label('부분 재현','Partial reproduction',lang),label('교사와 비슷한 것과 정답에 가까운 것을 나누어, 기존 선형 이론을 독립 구현·부분 수치 재현했습니다.','Independently implements and partially reproduces existing linear theory, separating teacher agreement from truth risk.',lang)),('RTPA','rtpa-research',label('상태 저장 연구','State-storage research',lang),label('반복 계산의 중간 상태를 작게 저장할 때의 품질·저장량·비용을 구분합니다. 할당 탐색은 종료된 범위를 포함합니다.','Separates quality, storage and costs of compact recurrent state; includes a closed allocation-search track.',lang)),('Coupling Matters','coupling-matters-dllm',label('프리프린트·CPU 검산','Preprint and CPU audit',lang),label('명시한 가정 아래 경로의 첫 분기와 이후 결과를 분석합니다. 외부 심사를 받은 정리라는 뜻은 아닙니다.','Studies first divergence and later outcomes under explicit assumptions. Preprint status is not peer review.',lang)),('FDCA','fdca-visual-token',label('시각 토큰 실험','Visual-token experiments',lang),label('첫 분기에서 최종 이미지 차이까지 이어지는 분석입니다. 사용한 이미지 지표와 사람 판단을 구분합니다.','Connects first divergence to final-image differences, keeping image metrics separate from human judgment.',lang)),('Direction–Scale','direction-scale-archive',label('종료된 연구 기록','Closed research record',lang),label('제한된 scale 이론과 재현되지 않은 사전학습 이득을 함께 남겼습니다.','Retains restricted scale theory alongside pretrained gains that did not repeat.',lang)),('RCFS-DQ','rcfs-dq',label('제한된 기전 진단','Bounded mechanism study',lang),label('국소 기전·실패·반대 근거를 보존합니다. 자연 오류에서의 개선이나 범용 효과는 확립되지 않았습니다.','Preserves local mechanisms, failures and counterevidence. Natural-error gains and general effects remain unestablished.',lang))]
    h='<section id="research" class="portfolio-research"><p class="eyebrow">RESEARCH MAP</p><h2>'+label('질문에서 대조와 원자료로','From a question to controls and records',lang)+'</h2><p>'+label('바로 쓰는 도구와 연구 기록을 구분했습니다. 아래는 공개 정본이며, 종료된 연구도 그 한계와 함께 읽을 수 있습니다.','These public research records are separate from ready-to-use tools. Closed studies remain accessible with their limits.',lang)+'</p><ul class="research-list">'
    for name,repo,kind,desc in rows:
        h+='<li><div><h3>'+a('https://github.com/Munsik-Kim/'+repo,name)+'</h3><span class="research-kind">'+kind+'</span></div><p>'+desc+'</p></li>'
    return h+'</ul></section>'

def recovery_finding(data,lang):
    from case018 import display_facts
    f=display_facts(data); g=f['cells']['K0','R0']; n=f['tasks']
    return '<p class="story-metric"><span>Case018 · HOLDOUT · BF16 KV</span><strong>Q '+str(g['Q'])+'/'+str(n)+' · W '+str(g['W'])+'/'+str(n)+'</strong><span>'+label('Q: 상태·후속 절차 적합 / W: 정확한 최종 보고까지','Q: state/procedure fit / W: also requires an accurate final report',lang)+'</span></p><p class="scope">'+label('두 판정은 같은 성공률이 아닙니다. Case017의 기준 누락과 추가 비용도 원기록에 보존했습니다.','These are different endpoints. Case017’s omitted expansion criterion and added costs remain in its records.',lang)+'</p>'


def comparison_finding(root, lang):
    """Describe the shipped synthetic example, without running or grading a model."""
    import json
    directory = root/'packages/diova-compare/examples/compare'
    runs = [[json.loads(line) for line in (directory/name).read_text().splitlines() if line] for name in ('baseline.jsonl','candidate.jsonl')]
    before, after = ({r['sample_id']: r for r in rows} for rows in runs)
    if len(before)!=len(runs[0]) or len(after)!=len(runs[1]) or set(before)!=set(after) or not before:
        raise ValueError('Synthetic comparison IDs differ')
    for key in before:
        a0,a1=before[key],after[key]
        if any(a0[k]!=a1[k] for k in ('task','task_version','prompt_hash','gold','gold_definition')) or any(r['evidence_kind']!='synthetic_test' or type(r['correct']) is not bool for r in (a0,a1)):
            raise ValueError('Synthetic comparison contract differs')
    n=len(before); b=sum(r['correct'] for r in before.values()); c=sum(r['correct'] for r in after.values())
    loss=sum(before[k]['correct'] and not after[k]['correct'] for k in before)
    gain=sum(not before[k]['correct'] and after[k]['correct'] for k in before)
    return '<p class="story-metric"><span>'+label('공개 합성 예제 · 모델 성능 실측 아님','Public synthetic example · not model performance',lang)+'</span><strong>'+f'{b}/{n} → {c}/{n}'+'</strong><span>'+label(f'잃은 정답 {loss}개 · 새로 맞힌 정답 {gain}개',f'{loss} lost correct · {gain} newly correct',lang)+'</span></p>'
