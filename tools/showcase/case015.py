"""Case015 verified P1 history plus measured order-API regression, kept separate."""
from html import escape
from pathlib import Path
import importlib.util
import subprocess
import hashlib
from common import read,require,sha

C15='cases/015-tool-interface-sensitivity'
A1=C15+'/followups/order-api-regression-v1'
SECTIONS=('overview','workflow','interfaces','results','episodes','reproduce','technical','references')
TITLES={'ko':'같은 도구, 다른 이름','en':'Same Tools, Different Interfaces'}
ARCHIVE='case015_tool_interface_order_workflow_publication_v1.zip'
SOURCE_COPIES={
    'case015-report-ko.md.txt':C15+'/REPORT.ko.md','case015-report-en.md.txt':C15+'/REPORT.md',
    'case015-reproduce-ko.md.txt':C15+'/REPRODUCE.ko.md','case015-reproduce-en.md.txt':C15+'/REPRODUCE.md',
    'case015-notice.md.txt':C15+'/NOTICE.md','case015-p1-report-ko.md.txt':C15+'/publication/originals/p1_review/REPORT.ko.md',
    'case015-p1-report-en.md.txt':C15+'/publication/originals/p1_review/REPORT.md',
    'case015-a1-runner.py.txt':A1+'/src/order_regression/runner.py','case015-a1-fixture.py.txt':A1+'/src/order_regression/fixture.py',
    'case015-a1-scorer.py.txt':A1+'/src/order_regression/scoring.py','case015-a1-audit.py.txt':A1+'/src/order_regression/audit.py',
    'case015-a1-tests.py.txt':A1+'/tests/test_order_contracts.py','case015-p1-model-adapter.py.txt':C15+'/src/interface_sensitivity/model_adapter.py',
    'case015-a1-protocol.json.txt':A1+'/specs/protocol.json','case015-a1-raw.json.txt':A1+'/results/a1_r1/episodes.jsonl',
    'case015-p1-raw.json.txt':C15+'/results/p1_r1/episodes.jsonl','case015-cpu-check.py.txt':C15+'/publication/scripts/run_checks.py',
}


def load_case015(root):
    case=root/C15
    if not (case/'README.md').exists():return None
    spec=importlib.util.spec_from_file_location('case015_raw_display',case/'publication/scripts/derive.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    value=m.derive(case);require(value==read(case/'publication/summary/project.json'),'Case015 display differs from raw')
    manifest=read(case/'publication/UNIFIED_PUBLIC_MANIFEST.json')
    for name,e in manifest['files'].items():require(sha(case/name)==e['sha256'],'Case015 source changed: '+name)
    download=read(root/'downloads'/ARCHIVE.replace('.zip','.json'))
    require(download['filename']==ARCHIVE and download['bytes']==(root/'downloads'/ARCHIVE).stat().st_size and download['sha256']==sha(root/'downloads'/ARCHIVE),'Case015 archive mismatch')
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True).stdout.strip()
    blob=subprocess.run(['git','show',revision+':downloads/'+ARCHIVE],cwd=root,capture_output=True)
    pinned=revision if blob.returncode==0 and hashlib.sha256(blob.stdout).hexdigest()==download['sha256'] else None
    ref=pinned or 'refs/heads/case015/publication-order-workflow'
    rows=lambda p:[__import__('json').loads(s) for s in p.read_text().splitlines() if s.strip()]
    return {**value,'A1_episodes':rows(root/A1/'results/a1_r1/episodes.jsonl'),
        'P1_episodes':rows(case/'results/p1_r1/episodes.jsonl'),'scenarios':read(root/A1/'data/scenarios.json'),
        'mappings':read(root/A1/'data/api_mappings.json'),'download':download,
        'download_url':'https://github.com/Munsik-Kim/inference-lab/raw/'+ref+'/downloads/'+ARCHIVE,
        'download_git_revision':pinned,'sources':{p:sha(root/p) for p in set(SOURCE_COPIES.values())|{C15+'/publication/summary/project.json',C15+'/publication/UNIFIED_PUBLIC_MANIFEST.json'}}}


def a(url,label):return '<a href="'+escape(url,quote=True)+'">'+escape(label)+'</a>'
def table(headers,rows,label):
    return '<div class="scroll" role="region" tabindex="0" aria-label="'+escape(label,quote=True)+'"><table><thead><tr>'+''.join('<th scope="col">'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def pre(value):return '<pre>'+escape(__import__('json').dumps(value,ensure_ascii=False,indent=2) if not isinstance(value,str) else value)+'</pre>'


def home_case015(data,lang):
    txt=lambda k,e:k if lang=='ko' else e
    counts=' · '.join(m+': '+', '.join(api.replace('API_','')+' '+str(next(r for r in data['A1']['summary']['success_table'] if r['method']==m and r['api_version']==api)['verified_success'])+'/16' for api in ['API_V1','API_V2']) for m in ['BF16','RTN','D16'])
    return '<section class="panel interface-latest" id="interface-study"><p class="eyebrow">CASE 015 · API REGRESSION TESTING</p><h2>'+a('case015.html#overview',TITLES[lang])+'</h2><p>'+txt('API 이름 변경 전후의 조회·계산·기록 수정·검증을 자동 비교하는 도구를 구현했습니다. 생성 전에 받은 결과와 실제 파일 상태를 따로 추적합니다.','Built tools to compare lookup, calculation, update and validation across an API rename, tracking received results separately from executed file state.')+'</p><p class="case015-counts">'+escape(counts)+'</p><p class="scope">'+txt('한 합성 주문 업무 · 4개 지시문 표현 · 16개 fixture · A1 96개 실측. 기존 P1 96개는 별도 분모입니다. Saved INT4 → dense BF16 품질 평가.','One synthetic order workflow · four instruction templates · 16 fixtures · 96 measured A1 episodes. Original96 P1 episodes retain a separate denominator. Saved INT4 → dense BF16 quality reference.')+'</p><div class="actions">'+a('case015.html#results',txt('API 회귀 결과','API regression results'))+' · '+a('case015.html#episodes',txt('주문 사건 탐색','Inspect order episodes'))+' · '+a('case015.html#reproduce',txt('모델 없는 CPU 예제','Model-free CPU example'))+'</div></section>'


def render_case015(data,lang):
    ko=lang=='ko';txt=lambda k,e:k if ko else e
    labels=txt(['한눈에 보기','주문 업무 흐름','이름만 다른 두 API','실제 결과','사건 탐색','직접 실행','기존 P1·통계·실행 계약','출처와 구현 기여'],['Overview','Order workflow','Two APIs, same behavior','Measured results','Episode explorer','Run it yourself','Original P1, statistics and contracts','Sources and contributions'])
    out='<header class="study-intro"><p class="eyebrow">DIOVA · CASE 015</p><h1>'+TITLES[lang]+'</h1><p class="lede">'+txt('API 이름이 바뀌어도 주문 정산을 정확히 끝낼까?','Will an order workflow still finish correctly after an API rename?')+'</p><p>'+txt('API 이름 변경 전후의 조회·계산·기록 수정·검증을 자동 비교하는 도구를 구현했습니다. 실제 도구 실행과 생성 전에 모델이 받은 정보를 재생해 판정을 검산합니다.','Implemented regression tools for lookup, calculation, record updates and validation across an API rename. Audits replay actual tool execution and information received before each generation.')+'</p></header><div class="study-layout"><aside class="study-toc"><p>'+txt('페이지 목차','ON THIS PAGE')+'</p><nav aria-label="'+txt('사례 목차','Case contents')+'"><ol>'+''.join('<li>'+a('#'+sid,label)+'</li>' for sid,label in zip(SECTIONS,labels))+'</ol></nav></aside><article class="study-article">'
    def section(i,body):return '<section class="study-section" id="'+SECTIONS[i]+'"><h2>'+labels[i]+'</h2>'+body+'</section>'
    results=data['A1']['summary'];pairs=results['paired_regression']
    facts=''.join('<li>'+escape(r['method'])+': '+str(r['both_success'])+'/16 '+txt('두 API 모두 성공','successes on both APIs')+' · '+str(r['loss'])+' '+txt('손실','losses')+' / '+str(r['gain'])+' '+txt('회복','gains')+'</li>' for r in pairs)
    out+=section(0,'<ul class="case015-facts">'+facts+'</ul><p>'+txt('A1은 하나의 합성 주문 업무와 4개 표현 template에 속한 16개 fixture를 시험했습니다. 모든 조건이 성공해도 API 변경에 대한 일반적 동등성이나 배포 승인을 뜻하지 않습니다.','A1 tests 16 fixtures nested in four instruction templates of one synthetic order workflow. Even universal success here does not establish general API equivalence or deployment approval.')+'</p><div class="actions">'+a('#episodes',txt('실제 주문 하나 살펴보기','Inspect one measured order'))+' · '+a('#reproduce',txt('CPU에서 기록 재실행','Replay the records on CPU'))+'</div>')
    steps=txt(['주문 조회 결과 받기','수량 × 단가 도구 결과 받기','해당 주문 total_minor 저장','저장 상태 검증','검증 결과 수신 후 최종 JSON'],['Receive order lookup','Receive quantity × price product','Write total_minor for that order','Validate saved state','Receive validation, then final JSON'])
    out+=section(1,'<ol class="case015-flow">'+''.join('<li>'+escape(x)+'</li>' for x in steps)+'</ol><p>'+txt('금액은 정수 최소단위입니다. 모델은 수량과 단가를 초기 요청에서 보지 않습니다. 잘못된 정수도 도구가 실제 계산·저장하며, oracle이 값을 고쳐주지 않습니다. 다른 주문과 불변 필드는 전 과정에서 보존해야 합니다.','Amounts use integer minor units. Initial requests hide quantity and price. Schema-valid wrong integers are actually computed or saved; the oracle does not repair arguments. Other orders and immutable fields must remain unchanged throughout.')+'</p><p class="method-key">'+txt('같은 응답에 생성된 호출은 모두 동일한 이전 관측 목록을 갖습니다. 조회와 곱셈을 한 번에 생성하면 조회 결과를 보고 계산한 것으로 세지 않습니다. 올바른 write → validate는 한 응답에서 실행해도 허용합니다.','All calls generated in one response share the same prior observation list. Generating read and multiply together does not count as using the read result. A correct write → validate pair in one response is allowed.')+'</p>')
    mapping=data['mappings'];out+=section(2,table(['Semantic ID','API_V1','API_V2'],[[s,mapping['API_V1'][s],mapping['API_V2'][s]] for s in mapping['API_V1']],'API names')+'<p>'+txt('설명·인자명·타입·구현·순서는 같습니다. API_V2에서는 옛 이름을 거절합니다. 이름 자동 교정·ENUM·추가 오류 힌트는 모두 꺼져 있습니다.','Descriptions, argument keys/types, implementation and order are unchanged. API_V2 rejects old names. No alias repair, ENUM or enhanced error hints are applied.')+'</p>')
    rows=[[r['method'],r['api_version'],f"{r['verified_success']}/{r['observed']}",r['STATE_ONLY_SUCCESS'],r['observation_violation_episodes'],r['UNKNOWN_TOOL'],r['model_calls']] for r in results['success_table']]
    body=table([txt('모델','Model'),'API',txt('검증된 workflow','Verified workflow'),txt('최종 상태 정답','Correct state'),txt('관측 위반 episode','Observation violations'),'UNKNOWN_TOOL',txt('모델 요청','Model requests')],rows,'A1 measured success and diagnostics')
    body+=table(['Model',txt('유지 성공','Both success'),txt('손실 V1→V2','Loss V1→V2'),txt('회복','Gain'),txt('양쪽 실패','Both fail')],[[r['method'],r['both_success'],r['loss'],r['gain'],r['both_fail']] for r in pairs],'API paired regression')
    body+='<p>'+txt('각 성공수의 분모는 16 scenario입니다. 기존 P1의 8 task와 합산하지 않습니다. 상태 정답만으로는 조회·계산 결과를 실제로 받은 뒤 행동했는지 알 수 없어 두 점수를 나눴습니다.','Each success denominator is 16 scenarios, separate from the eight original P1 tasks. Correct state alone cannot prove that the model acted after receiving lookup/calculation results, so the endpoints remain separate.')+'</p>'
    body+='<details><summary>'+txt('4개 지시문 표현별 전체 결과','All results by the four instruction templates')+'</summary>'+table(['Template','Model','API','Success /4'],[[t,r['method'],r['api_version'],r['success']] for t,rr in results['template_results'].items() for r in rr],'Template results')+'</details>'
    body+='<p>'+txt('실제 회복 사례: BF16 / A1-06은 V1에서 올바른 금액을 저장했지만 검증 ID 끝자리를 잘못 생성했고, V2에서는 같은 주문을 올바르게 검증했습니다. D16은 A1-00/06/08/14에서 두 API 모두 저장 호출 JSON parse에 실패했습니다. 관측 순서 위반과 미등록 이름은 0건이었습니다.','Observed recovery: BF16 / A1-06 saved the correct total on V1 but generated a wrong validation ID; V2 validated the correct order. D16 failed to parse write-call JSON on both APIs for A1-00/06/08/14. There were no observation-order violations or unknown names.')+'</p><p class="scope">'+txt('기록 보완: 조기 parse 실패 8건의 원 terminal_reason은 BUDGET_EXHAUSTED로 남았습니다. 실제 예산 소진이 아니며 별도 사후 주석에서 PARSE_ERROR로 설명합니다. 잘못된 주문 ID의 값 오류 1건도 주석에 포함했습니다. 원 응답·주 판정은 변경하지 않았습니다.','Record correction:8 early parse failures retained BUDGET_EXHAUSTED as the raw terminal_reason. These did not exhaust budgets and are annotated separately as PARSE_ERROR. One wrong-order-ID value error is also annotated. Original responses and primary verdicts are unchanged.')+'</p>'
    prior=data['P1']['summary']['contrasts']['D16-minus-BF16']
    body+='<p>'+txt('별도 기존 P1의 주 interaction: ','Separate original P1 primary interaction: ')+f"{100*prior['estimate']:+.2f} pp; 95% CI [{100*prior['CI95'][0]:+.2f}, {100*prior['CI95'][1]:+.2f}] pp."+' '+txt('불확정 결과와 모든 실패를 유지합니다.','The inconclusive result and all failures are retained.')+' '+a('#p1',txt('P1 전체 표와 사건','P1 complete table and episodes'))+'</p>'
    out+=section(3,body)
    controls='<div class="filters case015-filters"><label>'+txt('모델','Model')+'<select id="a1-method">'+''.join('<option>'+x+'</option>' for x in ['BF16','RTN','D16'])+'</select></label><label>API<select id="a1-api"><option>API_V1</option><option>API_V2</option></select></label><label>Scenario<select id="a1-scenario">'+''.join('<option>'+s['scenario_id']+'</option>' for s in data['scenarios'] if s['split']=='primary')+'</select></label></div><p id="a1-filter-status" role="status" aria-live="polite"></p><noscript>'+txt('JavaScript가 꺼져 있어 모든 episode를 아래에서 펼쳐 볼 수 있습니다.','JavaScript is off; all episodes remain available below.')+'</noscript>'
    tasks={s['scenario_id']:s for s in data['scenarios']};body=controls
    for r in data['A1_episodes']:
        task=tasks[r['scenario_id']];eid=r['episode_id'];body+='<details class="a1-episode" id="'+escape(eid,quote=True)+'" data-method="'+r['method']+'" data-api="'+r['api_version']+'" data-scenario="'+r['scenario_id']+'"><summary>'+escape(eid)+' · '+txt('워크플로','Workflow')+' '+str(r['VERIFIED_WORKFLOW_SUCCESS'])+' · '+txt('상태','State')+' '+str(r['STATE_ONLY_SUCCESS'])+'</summary>'
        body+='<h3>'+txt('요청과 선언된 이름','Request and declared names')+'</h3>'+pre(task['messages'])+pre(data['mappings'][r['api_version']])+'<details><summary>'+txt('초기 JSON','Initial JSON')+'</summary>'+pre(task['orders'])+'</details>'
        for response in r['responses']:
            turn=response['generated_in_turn'];body+='<section class="case015-turn"><h4>'+txt('응답 묶음','Response group')+' '+str(turn)+'</h4><p>'+txt('생성 전에 받은 결과 ID','Results received before generation')+': <code>'+escape(str(response['available_result_event_ids']))+'</code></p><p class="small">History SHA256: <code>'+response['pre_generation_history_hash']+'</code></p>'+pre(response['raw_text'])
            for event in r['tool_events']:
                if event['generated_in_turn']!=turn:continue
                body+='<details><summary>'+escape(str(event['raw_name']))+' → '+escape(str(event['semantic_tool_id']))+' / '+event['execution_status']+'</summary>'+pre({k:event[k] for k in ['raw_arguments','generated_after_event_ids','tool_result','state_before_execution','state_after_execution','state_version_before','state_version_after']})+'</details>'
            body+='</section>'
        body+='<h4>'+txt('최종 답·실제 상태·판정','Final answer, actual state and verdict')+'</h4>'+pre({k:r[k] for k in ['final_answer','final_fixture','VERIFIED_WORKFLOW_SUCCESS','STATE_ONLY_SUCCESS','REQUIRED_ACTIONS_DONE','OBSERVATION_ORDER_VALID','observation_violations','terminal_reason']})+'</details>'
    out+=section(4,body)
    cmd='OUT=$(mktemp -d)\npython3 -B cases/015-tool-interface-sensitivity/publication/scripts/run_checks.py --output "$OUT/case015" --tests'
    body='<p>'+txt('공개 API 정의 읽기 → 두 API의 합성 정답 경로 동등성 실행 → 저장된 모델 응답을 새 fixture에 재실행 → 회귀표 재계산 순서입니다. 모델·설치·GPU 없이 Python 표준 라이브러리로 실행합니다.','Read the public API definitions → execute equivalent synthetic oracle traces → replay recorded model responses into fresh fixtures → recompute regression tables. The CPU path uses the Python standard library, without a model, installation or GPU.')+'</p>'+pre(cmd)+'<div class="actions">'+a('../sources/case015-reproduce-'+lang+'.md.txt',txt('전체 CPU 명령과 별도 GPU 범위','Full CPU commands and separate GPU scope'))+' · '+a('../sources/case015-a1-runner.py.txt',txt('실행기','Runner'))+' · '+a('../sources/case015-a1-tests.py.txt',txt('계약 검사','Contract tests'))+' · '+a(data['download_url'],txt('공개 ZIP','Public ZIP'))+'</div>'
    out+=section(5,body)
    p1=data['P1']['summary'];p=p1['contrasts']['D16-minus-BF16'];rows=[]
    for method in ['BF16','RTN','D16']:rows.append([method,*[str(next(r['success'] for r in p1['success_table'] if r['method']==method and r['interface_id']==i))+'/8' for i in ['I0_ORIGINAL','I1_ORDER_ONLY','I2_STYLE_RENAME','I3_OPAQUE_RENAME']]])
    body='<details id="p1"><summary>'+txt('기존 P0/P1: 96개 long-context 도구 episode와 불확정 주 비교','Original P0/P1:96 long-context tool episodes and the inconclusive primary contrast')+'</summary>'+table(['Model','I0 original','I1 order','I2 camel','I3 opaque'],rows,'Original P1 results')+'<p>'+txt('기존 P1의 D16−BF16 signed 변화 interaction은 ','The original P1 D16−BF16 signed-change interaction is ')+f"{100*p['estimate']:+.2f} pp; 95% CI [{100*p['CI95'][0]:+.2f}, {100*p['CI95'][1]:+.2f}] pp."+' '+txt('0을 포함합니다. 손실과 회복이 상쇄될 수 있으며 8개 DEV task의 탐색 결과입니다.','The interval includes zero. Losses and gains may cancel; this is an exploratory eight-DEV-task result.')+'</p><p class="method-key">'+txt('P1의 같은 실행 직전 fixture와 생성 전 관측 이력 대응은 별개입니다. 한 응답의 뒤 호출은 앞 호출 결과를 아직 보지 못했을 수 있습니다. 원 판정은 유지하고 A1의 더 명시적인 관측 계약을 소급 적용하지 않습니다.','Matching P1 pre-execution fixtures and matching pre-generation observations are separate. Later calls in a response may have been generated before earlier results were received. Original verdicts remain unchanged; the A1 observation contract is not retroactively applied.')+'</p>'+a('../sources/case015-p1-report-'+lang+'.md.txt',txt('보존된 원 P1 보고서','Preserved original P1 report'))
    for r in data['P1_episodes']:
        body+='<details><summary>'+escape(r['episode_id'])+' · '+str(r['functional_success'])+'</summary><p>'+txt('표시 범위: 실행 상태 기록 / 생성 전 정책 입력 동일성은 별도 미확인. API가 다른 prompt hash를 동일 입력으로 취급하지 않습니다.','Display scope: recorded execution state / pre-generation policy input equivalence is not separately established. Different API prompt hashes are not treated as identical inputs.')+'</p>'+pre({'tool_events':r['tool_events'],'responses':[x['raw_text'] for x in r['responses']],'final_answer':r.get('final_answer'),'functional_success':r['functional_success'],'legacy_success':r['legacy_success']})+'</details>'
    body+='</details><details><summary>'+txt('측정·통계·모델 경계','Measurement, statistics and model boundaries')+'</summary>'+pre({'A1':results['auxiliary'],'runtime':data['A1']['runtime'],'model_revision':data['model_revision'],'execution_source_sha256':data['execution_source_sha256']})+'<p>'+txt('A1의 부가 구간은 4개 template 묶음을 보존한 10,000회 탐색 재표집입니다. 폭0인 구간도 동등성 증거가 아닙니다. Saved INT4→dense BF16, D16 beta 역배율 경로, greedy 192token·8요청·6도구예산을 유지했습니다.','Auxiliary A1 intervals use 10,000 exploratory resamples of four template groups. A zero-width interval is not equivalence. Saved INT4→dense BF16, D16 inverse-beta transport, greedy 192-token generation and 8-request/6-tool budgets are retained.')+'</p></details>'
    body+=table([txt('범위','Scope'),txt('상태','Status')],[[txt('P1·A1 원형/실측 기록 감사','P1/A1 preservation and recorded-response audit'),txt('각 단계 별도 완료·분모','Completed separately, distinct denominators')],['CIPR / training / P2 / fresh64 / D4 / packed serving',txt('미실행','Not run')],[txt('외부 업무·결제 서비스','External business/payment systems'),txt('연결하지 않은 합성 fixture','Synthetic fixture, no connection')]],'Study scope')
    out+=section(6,body)
    out+=section(7,'<p>'+a('../sources/case015-report-'+lang+'.md.txt',txt('정식 통합 보고서','Unified formal report'))+' · '+a('../sources/case015-notice.md.txt','NOTICE')+' · '+a('../data/case015.json',txt('원자료 연결과 hash','Raw source identities and hashes'))+'</p><p>'+txt('DIOVA의 registry·동등성 검사·상태/관측 기반 scorer·감사와 화면을 구현했습니다. Qwen·PyTorch·Transformers와 DualQuant 공개 수식 기반 reference를 재사용했습니다. CIPR 학습은 실행하지 않았습니다. Codex/ChatGPT가 구현과 별도 CPU 재계산을 보조했습니다.','DIOVA implements registry conversion, equivalence checks, state/observation scoring, audits and displays. It reuses Qwen, PyTorch, Transformers and an independent reference based on public DualQuant equations. No CIPR training was run. Codex/ChatGPT assisted implementation and separate CPU recomputation.')+'</p>')
    return out+'</article></div>'
