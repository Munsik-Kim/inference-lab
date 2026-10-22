"""Case018 display: separate original recovery endpoints and posthoc observations."""
from html import escape
import importlib.util
import functools
from common import read, require, sha

C18='cases/018-agent-state-recovery'
TITLES={'ko':'상태 복구와 최종 보고는 함께 맞을까?','en':'State Recovery and Final-Report Consistency'}
SECTIONS=('overview','method','results','posthoc','cost','episodes','limits','reproduce','references')
EVIDENCE_ASSET='data/case018-original-evidence.tar.xz'
EVIDENCE_SOURCE=C18+'/publication/data/original-evidence.tar.xz'
SOURCE_COPIES={
    **{'case018-'+kind+'-'+lang+'.md.txt':C18+'/'+name+('.ko.md' if lang=='ko' else '.md')
       for kind,name in [('guide','README'),('report','REPORT'),('reproduce','REPRODUCE')] for lang in ('ko','en')},
    'case018-notice.md.txt':C18+'/NOTICE.md',
    'case018-quantity.py.txt':C18+'/publication/analysis/quantity.py',
    'case018-quantity-definition.json.txt':C18+'/publication/analysis/DEFINITION.json',
    'case018-quantity-ledger.json.txt':C18+'/publication/summary/quantity_ledger.json',
    'case018-episode-ledger.csv.txt':C18+'/publication/summary/episode_ledger.csv',
    'case018-public-checks.py.txt':C18+'/publication/run_checks.py',
    'case018-audit.py.txt':C18+'/src/agent_state_recovery/audit.py',
    'case018-mapping.json.txt':C18+'/publication/provenance/original_mapping.json',
    **{'case018-'+name+'.json.txt':C18+'/publication/summary/selected/'+name+'.json' for name in (
        'HOLDOUT-A00_K0_R0','HOLDOUT-A00_K1_R0','HOLDOUT-A06_K1_R0','HOLDOUT-A06_K1_R1','HOLDOUT-C01_K1_R0','HOLDOUT-C01_K1_R1')}
}
FIGURES={'assets/case018-'+n+'.svg':C18+'/publication/summary/'+n+'.svg' for n in ('quality','observation')}

def verifier(root):
    spec=importlib.util.spec_from_file_location('case018_public_verify',root/C18/'publication/verify.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def check_evidence_download(root,blob):
    # Repeat builds in one test process must not repeatedly inflate the same
    # large immutable evidence. Both archive bytes and mapping hash key the cache.
    return _check_evidence(root,blob,sha(root/C18/'publication/provenance/original_mapping.json'))

@functools.lru_cache(maxsize=2)
def _check_evidence(root,blob,mapping_sha):
    return verifier(root).check_archive(root/C18,blob)

def load_case018(root):
    case=root/C18
    if not (case/'README.md').exists():return None
    verifier(root).verify(case)
    data=read(case/'publication/summary/project.json')
    require(data['posthoc'] is True and [p['episodes'] for p in data['phases']]==[24,96],'Case018 phase/diagnostic contract')
    data['costs']=read(case/'publication/summary/costs.json')
    data['archive']=read(case/'publication/provenance/original_mapping.json')['archive']
    data['display_source_files']={p:sha(root/p) for p in [*SOURCE_COPIES.values(),*FIGURES.values(),EVIDENCE_SOURCE,
        C18+'/publication/summary/project.json',C18+'/publication/summary/costs.json',C18+'/publication/PUBLIC_MANIFEST.json']}
    return data

def a(url,label):return '<a href="'+escape(url,quote=True)+'">'+escape(label)+'</a>'
def table(headers,rows,label):
    return '<div class="scroll" tabindex="0" role="region" aria-label="'+escape(label,quote=True)+'"><table><thead><tr>'+''.join('<th scope="col">'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'

def home_case018(data,lang):
    ko=lang=='ko'
    text=('잘못 바꾼 재고·중복 이동·부분 차감을 복구한 뒤 저장 상태와 최종 보고를 함께 검사했습니다. 신규 합성24과제에서 BF16 KV는 Q24/W0, NVFP4 KV는 Q19/W5였습니다. 최신 검증 가드는 부적격 완료 수용을 막았지만 Q/W 개선은 없었습니다.' if ko else 'We checked restored inventory and final reports after wrong-target, duplicate and partial changes. On24 new synthetic tasks, BF16 KV scored Q24/W0 and NVFP4 KV Q19/W5. The freshness guard blocked ineligible finals without improving Q/W.')
    return '<section class="panel case018-entry" id="case018-entry"><h2>Case018 · '+escape(TITLES[lang])+'</h2><p>'+escape(text)+'</p><p>'+('Q=업무 값·보호·두 고정 CPU 후속 절차, W=정확한 완료 보고·최신 검증까지 충족. 수량 재관측 분석은 사후 관측 진단이며 인과효과가 아닙니다.' if ko else 'Q=state, protection and two fixed CPU continuations; W also requires a correct final and current observed validation. Quantity re-observation is a posthoc association, not a causal effect.')+'</p><div class="actions">'+a('case018.html#results','결과와 실제 사건' if ko else 'Results and recorded episodes')+' · '+a('case018.html#posthoc','사후 수량 관측' if ko else 'Posthoc quantity observations')+' · '+a('case018.html#reproduce','모델 없는 CPU 재생' if ko else 'Model-free CPU replay')+'</div></section>'

def render_case018(data,lang):
    ko=lang=='ko';t=lambda k,e:k if ko else e
    groups=data['phases'][1]['groups']
    names=t(['한눈에 보기','업무·비교 규칙','실제 복구와 완료 보고','사후 수량 관측','실행 비용','실제 사건','범위와 실패','직접 재생','출처'],
            ['Overview','Tasks and comparison','Recovery and final reports','Posthoc observations','Execution cost','Recorded episodes','Limits and failures','Reproduce','Sources'])
    out='<div class="case018-layout"><nav class="case018-toc" aria-label="'+t('이 페이지 목차','On this page')+'">'+''.join(a('#'+k,v) for k,v in zip(SECTIONS,names))+'</nav><article class="case018"><h1>Case018 — '+escape(TITLES[lang])+'</h1>'
    def section(i,body):return '<section class="panel" id="'+SECTIONS[i]+'"><h2>'+escape(names[i])+'</h2>'+body+'</section>'
    out+=section(0,'<p>'+t('잘못된 창고·상품을 바꾼 기록, 두 번 처리한 이동, 일부만 차감한 재고를 복구하는 작은 도구 환경을 구현했습니다. 저장 상태가 복구됐어도 최종 답의 수량이 틀릴 수 있는지 검사합니다.',
        'We built a small tool environment to recover wrong-target inventory changes, duplicate transfers and partial debits. We ask whether restored state and final reported quantities agree.')+'</p><p>'+t('Q는 업무 값·보호·평가기의 두 고정 CPU 후속 절차, W는 여기에 정확한 accepted final과 최신 성공 검증 관측을 더한 지표입니다. 후속 절차는 모델의 자율 행동이나 모든 미래 상태 동등성 검사가 아닙니다.',
        'Q checks business state, protection and two fixed evaluator CPU continuations. W additionally requires an accepted correct final and observed current successful validation. These programs are neither autonomous model follow-up work nor a proof of all future state equivalence.')+'</p><p class="note">'+t('신규 합성24task×4조건=96episode의 탐색 측정입니다. 두 정책 모두 BF16 KV Q24/W0, native NVFP4 KV Q19/W5. 가드는 부적격 final15시도/5episode를 막았으나 Q/W 회복은0입니다.',
        'Exploratory measurement:24 new synthetic tasks ×four conditions=96 episodes. Both policies:BF16 KV Q24/W0, native NVFP4 KV Q19/W5. The guard blocked15 ineligible final attempts in five episodes but recovered zero Q/W successes.')+'</p>')
    out+=section(1,'<p>'+t('A: 잘못 바꾼 대상을 되돌린 뒤 원래 작업은 후속 절차에서 수행. B: 정상1회는 보존하고 중복1회만 보상. C: 일부 차감만 되돌림. 수량 보상과 활성 처리 해제는 별도 API이며, 정답 checkpoint나 자동 reset은 모델에 제공하지 않습니다.',
        'A reverses the wrong target and defers intended work. B preserves one valid transfer and compensates only the duplicate. C reverses a partial debit. Quantity compensation and active-operation release are separate APIs; no gold checkpoint or automatic reset is given to the model.')+'</p>'+table([t('구성','Configuration'),t('의미','Meaning')],[['K0','BF16 KV'],['K1','Native NVFP4 KV'],['R0',t('형식에 맞는 final 수용·동일 사후 채점','Accept well-formed final; score afterward')],['R1',t('현재 검증의 범위·digest·version·실제 사전 관측 가드','Guard current validation scope/digest/version and prior observation')]],'Common policies')+'<p>'+t('양쪽 Qwen3-4B-Instruct-2507 가중치는 BF16 고정입니다. 모든 조건은 같은 공개 JSON 문법·도구·strict parser·최초 입력·예산을 사용합니다. R1은 final 수량 정답이나 hidden gold를 검사하지 않습니다. 공개 system은 두 정책 모두 마지막 변경 뒤 현재 validation을 요구했습니다.',
        'Both use BF16 Qwen3-4B-Instruct-2507 weights and the same public JSON grammar, tools, strict parser, initial input and budgets. R1 does not check final quantities or hidden gold. The public system already required current validation after the last change in both policies.')+'</p>')
    out+=section(2,table(['KV','Policy','D','B','P','Fresh','C1','C2','Q /24','W /24'],[[('BF16' if g['kv']=='K0' else 'NVFP4'),g['policy'],*[g[k] for k in ['D','B','P','Fresh','C1','C2','Q','W']]] for g in groups],'HOLDOUT each cell24 paired tasks')+'<p>'+t('D는 recovered:true 선언이며 거절된 final도 포함합니다. accepted final 없이도 종료 상태 Q는 평가하고 W는0입니다. 두 정책 간 Q/W 차이는 두 KV에서0pp [0,0]이나 동등성 증명이 아닙니다. KV 간 Q는 NVFP4가 −20.83pp, W는 +20.83pp로 지표별 순위가 달랐습니다.',
        'D includes recovered:true declarations in rejected finals. Terminal Q is evaluated even without accepted final; W is0. R1−R0 Q/W is0pp [0,0] in both KV settings, not equivalence evidence. NVFP4−BF16 Q is−20.83pp while W is+20.83pp; metric rankings differ.')+'</p><img src="../assets/case018-quality.svg" width="760" height="350" alt="HOLDOUT Q and W per24 synthetic tasks"><details><summary>'+t('DEV와 짝 비교 범위','DEV and paired-analysis scope')+'</summary>'+table(['KV','Policy','Q /6','W /6'],[[g['kv'],g['policy'],g['Q'],g['W']] for g in data['phases'][0]['groups']],'Separate DEV each cell6')+'<p>'+t('P1 DEV6은 별도 분모입니다. 등록 family-stratified paired bootstrap10000/seed61022018은 task의4조건을 함께 재표집합니다. pointwise95% 구간은 Q KV차이[−33.33,−8.33]pp, W[8.33,37.50]pp입니다. B/C1/C2 성공 집합이 실제 두 phase에서 같아, 후속 절차의 추가 판별력은 이번 모델 표본에서 관측되지 않았습니다.',
        'P1 DEV6 is a separate denominator. Registered family-stratified paired bootstrap10000/seed61022018 jointly resamples all four task conditions. Pointwise95% KV differences:Q[−33.33,−8.33]pp, W[8.33,37.50]pp. B/C1/C2 success sets are identical in both measured phases; added continuation discrimination was not observed in these model samples.')+'</p></details>')
    out+=section(3,'<p class="note">'+t('사후 관측 분석입니다. 이미 본 Q1/W0 76개와 A00가 동기이며, 새 전체 집계 전에 정의·코드·입력 hash를 저장했습니다. 재조회를 강제한 실험은 하지 않았습니다.',
        'Posthoc observational analysis, motivated by the already-seen76 Q1/W0 cases and A00. Definitions, code and input hashes were saved before the new full aggregate. No forced-reread experiment was run.')+'</p><p>'+t('현재 수량 관측=final 전 실제 받은 성공 read_records들이 필요한 모든 record를 포함하고, 그 뒤 각 수량이 변하지 않음. 수량이 바뀌었다 원복돼도 이전 조회는 만료됩니다. 메타데이터만 바뀐 revision은 수량을 만료시키지 않습니다. final 없는5개는 마지막 생성 전 관측 진단이며 수량 비교 NA입니다.',
        'Current quantity observation requires successful read_records results actually delivered before final, covering every required record without a later quantity change. Change-and-restore expires an old read; a metadata-only revision does not. Five no-final cases use last-generation observation diagnostics and have NA final comparison.')+'</p>'+table(['KV','Policy',t('수용/없음','Accepted/none'),'Read0 wrong','Read0 right','Read1 wrong','Read1 right'],[[g['kv'],g['policy'],str(g['accepted_final'])+'/'+str(g['no_accepted_final']),*[g['cross_table'][k] for k in ['read0_final0','read0_final1','read1_final0','read1_final1']]] for g in groups],'Posthoc HOLDOUT accepted finals; no-final separate')+'<p>'+t('accepted final91개 중 관측1인13개는 종료 수량 map과 일치했고 관측0인78개는 달랐습니다. 다만13개 중3개는 잘못 복구한 상태를 정확히 보고했으므로 Q/W 성공이 아닙니다. Q1/W0 76개 모두 최초 관측 수량과 일치했습니다. Fresh validation은 수량 재관측과 별도 지표입니다.',
        'Among91 accepted finals,13 observed-current cases match terminal quantities;78 without such observation differ. Three of those13 accurately report incorrectly recovered states and are not Q/W successes. All76 Q1/W0 cases match the first observed quantities. Fresh validation and quantity observation are distinct metrics.')+'</p><p>'+t('task·KV·정책·상태·경로·예산이 함께 달라지는 궤적의 관련성입니다. 재조회가 정답을 일으켰거나 NVFP4가 기억을 개선했다는 증거로 쓰지 않습니다.',
        'Task, KV, policy, state, trajectory and budgets differ jointly. This association does not establish that rereading caused correctness or NVFP4 improved memory.')+'</p><img src="../assets/case018-observation.svg" width="760" height="300" alt="Posthoc quantity observation cross table">'+a('../sources/case018-quantity-definition.json.txt',t('정확한 정의','Exact definition'))+' · '+a('../sources/case018-quantity-ledger.json.txt','120-row detailed ledger'))
    out+=section(4,table(['KV','Policy','Requests','Input tokens','Output tokens','Guard rejections','Episode s'],[[g['kv'],g['policy'],g['requests'],g['input_tokens'],g['output_tokens'],g['guard_rejections'],f"{g['episode_wall_seconds']:.3f}"] for g in groups],'All24 costs including failures')+'<p>'+t('NVFP4 R0→R1은 같은 Q19/W5에서 입력+18.91%, 출력+18.69%, episode wall+18.86%였습니다. 성공 episode만의 latency나 돈·전력 비용이 아닙니다. 누적917 attempts/915완전 응답, 관측 출력 하한57987 tokens이며 과거 취소2건의 실제 엔진 token은 미관측입니다.',
        'NVFP4 R0→R1 costs increase18.91% input,18.69% output and18.86% episode wall with unchanged Q19/W5. These are not successful-only latency, money or energy. Cumulative917 attempts/915 complete; observed output lower bound57987 tokens; engine output for two prior cancellations remains unknown.')+'</p><p>'+t('P1/P2 worker1294.761626764초는 load/compile·실패·종료를 포함하고 episode wall과 겹칩니다. 약4GiB KV pool의 슬롯29120→103552(약3.556배)은 저장 수용량이며 전체 VRAM·single100K context·속도 우위가 아닙니다. 새 CPU 검산 비용은 별도입니다.',
        'P1/P2 worker1294.761626764s includes load/compile/failures/shutdown and overlaps episode wall. Approximately4GiB KV pool slots29120→103552(3.556×) measure storage capacity, not whole VRAM, single100K context or speed superiority. New CPU verification costs are separate.')+'</p>')
    explorer='<p>'+t('6개 실제 사건의 원문과 설명을 구분합니다. 선택·요약은 사후이며 전체120개 원문은 archive에 있습니다.','These six measured excerpts distinguish raw text and explanations. Selection and summaries are posthoc; all120 full originals are archived.')+'</p><div id="case018-controls" hidden><label for="case018-episode">'+t('사건 선택','Choose episode')+'</label><select id="case018-episode"><option value="">'+t('전체 보기','Show all')+'</option>'+''.join('<option value="'+escape(e['episode_id'],quote=True)+'">'+escape(e['episode_id'])+'</option>' for e in data['selected_episodes'])+'</select><button type="button" id="case018-reset">'+t('전체 사건','All episodes')+'</button><span id="case018-count" role="status" aria-live="polite"></span></div><noscript><p>'+t('JavaScript 없이도 아래 모든 사건과 원문을 열 수 있습니다.','All episodes and raw text remain available below without JavaScript.')+'</p></noscript>'
    for e in data['selected_episodes']:
        eid=e['episode_id'];r=e['diagnostic']
        note=t('A00: 같은 최초 입력 이후 수량 재조회와 final 보고가 달라졌습니다.','A00: quantity re-observation and final reports diverge after identical initial inputs.') if 'A00' in eid else t('A06: 보상40/30이 총량+10을 만들어 보호·복구에 실패했습니다.','A06:40/30 compensation increases total stock by10, failing protection and recovery.') if 'A06' in eid else t('C01: 부분 차감의 도착 수량도 빼서 잘못 복구했습니다. 가드는 상태를 수리하지 않습니다.','C01: subtracting the destination during partial-debit repair leaves wrong stock. The guard does not repair state.')
        import json
        dump=lambda x:escape(json.dumps(x,ensure_ascii=False,indent=2))
        explorer+='<details class="case018-episode" id="episode-'+escape(eid,quote=True)+'"><summary>'+escape(eid)+' · Q/W/Fresh/read='+str(r['Q'])+'/'+str(r['W'])+'/'+str(r['Fresh'])+'/'+str(r['current_quantity_read_observed'])+'</summary><p>'+note+'</p><h3>'+t('공개 요청','Public request')+'</h3><pre>'+dump(e['request'])+'</pre><h3>'+t('수량과 판정: 설명용 파생 표','Quantities and verdicts: derived explanation')+'</h3><pre>'+dump({k:e[k] for k in ['initial_quantities','terminal_quantities','endpoint']})+'</pre>'
        for turn in e['turns']:
            explorer+='<section class="case018-turn"><h4>'+t('생성 ','Generation ')+str(turn['generation'])+' · '+escape(turn['decision'])+'</h4><p>'+t('생성 전 실제 제공된 결과 ID: ','Result IDs available before generation: ')+escape(str(turn['available_result_event_ids']))+'</p><details><summary>'+t('모델 원문 열기','Open model raw text')+'</summary><pre class="case018-raw">'+escape(turn['raw_text'])+'</pre></details><details><summary>'+t('실제 도구·가드 결과','Actual tool/guard result')+'</summary><pre>'+dump({'guard':turn['guard'],'event':turn['event']})+'</pre></details></section>'
        explorer+='<p>'+a('../sources/case018-'+eid+'.json.txt',t('원문 발췌·전체 포인터','Raw-text excerpt and full source pointer'))+'</p><p class="small">'+escape(r['source_path'])+' · SHA256 <code>'+escape(r['source_sha256'])+'</code></p></details>'
    out+=section(5,explorer)
    out+=section(6,'<p>'+t('R1은 최신 성공 검증의 관측을 요구할 뿐 최종 수량·복구 정답을 보증하지 않습니다. P0의 operational-state 반례와 실제 모델 표본의 B/C1/C2 집합 일치는 다른 결과입니다. 제어가 완료 선언을 막는 것과 실제 복구·정확한 답을 늘리는 것은 별도입니다. 작은 합성24과제의 탐색 관측이며 논문 전체·일반 업무 재현이 아닙니다.',
        'R1 enforces observation of current successful validation, not correct final quantities or recovery. P0 operational-state counterexamples and identical B/C1/C2 success sets in measured samples are distinct results. Blocking a completion claim differs from improving repair or reporting. These are exploratory24-task synthetic observations, not full-paper or general-workflow reproduction.')+'</p>')
    out+=section(7,'<p>'+t('일반 clone의 Python 표준 라이브러리만으로 새 외부 폴더에 복원·검산합니다.120 model endpoint와P0 26 synthetic trace, raw SSE/IDs·상태·관측·비용·paired·새 진단·SVG를 재생합니다. 모델·tokenizer·Torch/vLLM import와 네트워크 없이 가능합니다.',
        'A normal clone and Python standard library restore and check120 model endpoints and26 P0 synthetic traces, SSE/IDs, state, observations, costs, pairing, new diagnostics and SVG in a fresh external folder. No model/tokenizer/Torch/vLLM import or network is needed.')+'</p><pre>python3 -B cases/018-agent-state-recovery/publication/run_checks.py --output /tmp/case018-public-check-new --tests</pre><p>'+t('과거 tokenizer 재인코딩 receipt는 재사용이며 이번 새 tokenization/GPU 재현이 아닙니다. 경로를 포함한82개 부수 공개 사본의 치환은 old/new hash로 명시했습니다. 모든 모델 episode와 src Python 원형은 유지했습니다.',
        'Prior tokenizer re-encoding receipts are reused, not new tokenization or GPU reproduction.82 ancillary path-bearing public copies have explicit old/new hashes for prefix redactions. All model episode and src Python bytes remain original.')+'</p><div class="actions">'+a('../'+EVIDENCE_ASSET,t('전체 원자료 archive','Full evidence archive'))+' · '+a('../sources/case018-mapping.json.txt',t('복원 mapping','Restoration mapping'))+' · '+a('../sources/case018-episode-ledger.csv.txt','120-row CSV')+' · '+a('../sources/case018-reproduce-'+lang+'.md.txt',t('재현 범위','Reproduction scope'))+'</div>')
    out+=section(8,'<p>'+t('Qwen·vLLM·FlashInfer·xgrammar는 upstream입니다. 직접 구현한 부분은 도구 환경·parser·가드·관측 추적·별도 상태/SSE 감사와 공개 도구입니다. ReTrace의 복구 뒤 행동 평가, ACE의 개별/전체 제약 구분을 작은 도구 환경에 독립 응용했습니다. Codex 지원을 받았으며 사람 심사·다른 머신 GPU 재현이라고 하지 않습니다.',
        'Qwen, vLLM, FlashInfer and xgrammar are upstream. Direct work comprises the tool environment, parser, guard, observation tracking, separate state/SSE audit and publication tooling. We independently adapt ReTrace post-recovery evaluation and ACE component/joint constraints to small tools. Codex assisted; this is not human peer review or another-machine GPU reproduction.')+'</p>'+a('../sources/case018-report-'+lang+'.md.txt',t('정식 보고서','Full report'))+' · '+a('../sources/case018-audit.py.txt','Independent audit')+' · '+a('../sources/case018-quantity.py.txt','Posthoc code')+' · '+a('../sources/case018-notice.md.txt','NOTICE'))
    return out+'</article></div>'
