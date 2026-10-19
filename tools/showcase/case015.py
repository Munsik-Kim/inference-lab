"""Case015 verified P1 history plus measured order-API regression, kept separate."""
from html import escape
from pathlib import Path
import importlib.util
import subprocess
import hashlib
from common import read,require,sha
from case015_p2 import P2, SOURCES as P2_SOURCES, load_p2, render_p2, home_p2

C15='cases/015-tool-interface-sensitivity'
A1=C15+'/followups/order-api-regression-v1'
SECTIONS=('overview','workflow','interfaces','results','episodes','reproduce','technical','references')
TITLES={'ko':'같은 도구, 다른 이름','en':'Same Tools, Different Interfaces'}
ARCHIVE='case015_tool_interface_order_workflow_publication_v2.zip'
SOURCE_COPIES={
    'case015-guide-ko.md.txt':C15+'/README.ko.md','case015-guide-en.md.txt':C15+'/README.md',
    'case015-report-ko.md.txt':C15+'/REPORT.ko.md','case015-report-en.md.txt':C15+'/REPORT.md',
    'case015-reproduce-ko.md.txt':C15+'/REPRODUCE.ko.md','case015-reproduce-en.md.txt':C15+'/REPRODUCE.md',
    'case015-notice.md.txt':C15+'/NOTICE.md','case015-p1-report-ko.md.txt':C15+'/publication/originals/p1_review/REPORT.ko.md',
    'case015-p1-report-en.md.txt':C15+'/publication/originals/p1_review/REPORT.md',
    'case015-a1-runner.py.txt':A1+'/src/order_regression/runner.py','case015-a1-fixture.py.txt':A1+'/src/order_regression/fixture.py',
    'case015-a1-scorer.py.txt':A1+'/src/order_regression/scoring.py','case015-a1-audit.py.txt':A1+'/src/order_regression/audit.py',
    'case015-a1-tests.py.txt':A1+'/tests/test_order_contracts.py','case015-p1-model-adapter.py.txt':C15+'/src/interface_sensitivity/model_adapter.py',
    'case015-a1-protocol.json.txt':A1+'/specs/protocol.json','case015-a1-raw.json.txt':A1+'/results/a1_r1/episodes.jsonl',
    'case015-a1-annotations.json.txt':A1+'/results/a1_r1/diagnostic_annotations.json',
    'case015-p1-raw.json.txt':C15+'/results/p1_r1/episodes.jsonl','case015-cpu-check.py.txt':C15+'/publication/scripts/run_checks.py',
}

SOURCE_COPIES.update(P2_SOURCES)


def load_case015(root):
    case=root/C15
    if not (case/'README.md').exists():return None
    spec=importlib.util.spec_from_file_location('case015_raw_display',case/'publication/scripts/derive.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    value=m.derive(case);require(value==read(case/'publication/summary/project.json'),'Case015 display differs from raw')
    manifest=read(case/'publication/UNIFIED_PUBLIC_MANIFEST_V3.json')
    followup=case/'followups/json-recovery-p2-v1/publication/verify.py'
    retained=lambda name:case/name
    if followup.exists():
        preserved=importlib.util.spec_from_file_location('case015_p2_public_identity',followup);v=importlib.util.module_from_spec(preserved);preserved.loader.exec_module(v)
        v.verify(case)
        retained=lambda name:v.retained(case,name)
    for name,e in manifest['files'].items():require(sha(retained(name))==e['sha256'],'Case015 source changed: '+name)
    download=read(root/'downloads'/ARCHIVE.replace('.zip','.json'))
    require(download['filename']==ARCHIVE and download['bytes']==(root/'downloads'/ARCHIVE).stat().st_size and download['sha256']==sha(root/'downloads'/ARCHIVE),'Case015 archive mismatch')
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True).stdout.strip()
    blob=subprocess.run(['git','show',revision+':downloads/'+ARCHIVE],cwd=root,capture_output=True)
    pinned=revision if blob.returncode==0 and hashlib.sha256(blob.stdout).hexdigest()==download['sha256'] else None
    ref=pinned or 'refs/heads/case015/publication-order-workflow'
    rows=lambda p:[__import__('json').loads(s) for s in p.read_text().splitlines() if s.strip()]
    return {**value,'P2':load_p2(root),'A1_episodes':rows(root/A1/'results/a1_r1/episodes.jsonl'),
        'P1_episodes':rows(case/'results/p1_r1/episodes.jsonl'),'scenarios':read(root/A1/'data/scenarios.json'),
        'mappings':read(root/A1/'data/api_mappings.json'),'download':download,
        'download_url':'https://github.com/Munsik-Kim/inference-lab/raw/'+ref+'/downloads/'+ARCHIVE,
        'download_git_revision':pinned,'sources':{p:sha(root/p) for p in set(SOURCE_COPIES.values())|{C15+'/publication/summary/project.json',C15+'/publication/UNIFIED_PUBLIC_MANIFEST_V3.json'}}}


def a(url,label):return '<a href="'+escape(url,quote=True)+'">'+escape(label)+'</a>'
def table(headers,rows,label):
    return '<div class="scroll" role="region" tabindex="0" aria-label="'+escape(label,quote=True)+'"><table><thead><tr>'+''.join('<th scope="col">'+escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def pre(value):return '<pre>'+escape(__import__('json').dumps(value,ensure_ascii=False,indent=2) if not isinstance(value,str) else value)+'</pre>'


def model_key(lang):
    if lang == 'ko':
        return '<p class="method-key"><strong>비교 모델 읽기:</strong> 같은 Qwen3-4B에서 BF16은 원래 16비트 모델, RTN은 단순 반올림으로 가중치를 4비트 저장한 기준선, D16은 DualQuant 공개 수식 기반으로 가중치를 4비트 저장하고 중간 계산값을 16비트로 유지한 조건입니다. 압축 저장본은 BF16으로 복원해 작업 수행 품질을 비교했습니다.</p>'
    return '<p class="method-key"><strong>Model key:</strong> all conditions use Qwen3-4B. BF16 is the original 16-bit model; RTN stores weights at four bits using nearest rounding; D16 is an independent reference based on public DualQuant equations, with four-bit stored weights and 16-bit activations. Quantized storage is restored to BF16 for task-quality evaluation.</p>'


def task_terms(lang):
    terms = [('주문 / order','구매 내역 한 건입니다. order_id는 그 내역을 구별하는 ID입니다.'),
             ('수량 / 단가','구매한 개수 / 물건 하나의 가격입니다.'),
             ('JSON','항목 이름과 값을 짝지어 저장하는 텍스트 형식입니다. total_minor는 정수 단위의 총금액입니다.'),
             ('시나리오 / fixture','시험 문제 한 개 / 그 문제에서 사용하는 초기 실험용 파일입니다.'),
             ('episode','AI가 초기 파일에서 작업을 시작해 답을 마칠 때까지의 실행 한 번입니다.'),
             ('관측 이력','AI가 다음 응답을 만들기 전에 이미 받은 도구 결과입니다. 실제 파일 상태와 따로 기록합니다.')] if lang == 'ko' else [
             ('Order / order ID','One purchase record / the ID identifying it.'),
             ('Quantity / unit price','Item count / the price of one item.'),
             ('JSON','A text format for named fields and values. total_minor is the integer total amount.'),
             ('Scenario / fixture','One test problem / the initial synthetic file used for it.'),
             ('Episode','One run from the initial file until the AI stops answering.'),
             ('Received-result history','Tool results already received before the next AI response, recorded separately from executed file state.')]
    return '<div class="case015-terms" id="task-terms"><h3>'+('기록의 용어를 읽는 법' if lang == 'ko' else 'Terms in the records')+'</h3><dl>'+''.join('<dt>'+escape(k)+'</dt><dd>'+escape(v)+'</dd>' for k,v in terms)+'</dl></div>'


def home_case015(data,lang):
    txt=lambda k,e:k if lang=='ko' else e
    counts=' · '.join(m+': '+', '.join(api.replace('API_','')+' '+str(next(r for r in data['A1']['summary']['success_table'] if r['method']==m and r['api_version']==api)['verified_success'])+'/16' for api in ['API_V1','API_V2']) for m in ['BF16','RTN','D16'])
    return '<section class="panel interface-latest" id="interface-study"><p class="eyebrow">CASE 015 · TOOL NAME CHANGES</p><h2>'+a('case015.html#overview',TITLES[lang])+'</h2><p>'+txt('도구 이름만 바뀌어도 AI가 같은 일을 끝낼 수 있을까요? 구매한 개수와 개당 가격을 읽어 총금액을 계산·저장·확인하는 예제로 비교합니다. 기능과 입력 형식은 유지하고 도구를 부르는 이름만 바꾸었습니다.','Can an AI finish the same task after a tool rename? The example reads item count and price per item, calculates the purchase total, saves it and validates the record. Functions and input formats stay fixed; only their names change.')+'</p><p>'+txt('원래 이름을 V1, 바꾼 이름을 V2로 표시합니다. 아래는 같은 16개 문제에서 성공한 개수입니다. BF16은 원형 모델, RTN과 D16은 서로 다른 방식으로 4비트 저장한 비교 모델입니다.','V1 uses original names and V2 renamed names. Counts below are successes on the same 16 problems. BF16 is the original model; RTN and D16 are two conditions with four-bit stored weights.')+'</p><p class="case015-counts">'+escape(counts)+'</p><p class="scope">'+txt('실험용 구매 기록만 사용했습니다. 4개 영어 지시문 표현 × 4개 파일 × 2개 이름 표기 × 3개 모델 = 새 응용시험 A1의 96회입니다. 앞선 이름·순서 시험 P1의 96회와 따로 집계합니다. 압축 저장본을 BF16으로 복원한 품질 평가입니다.','Only synthetic purchase records are used. Four English instruction templates × four files × two naming versions × three models = 96 runs in the new A1 application. The earlier 96-run P1 name/order study is counted separately. Quality uses BF16 reload of quantized storage.')+'</p><div class="actions">'+a('case015.html#workflow',txt('예제 작업과 용어','Task and terms'))+' · '+a('case015.html#results',txt('이름 변경 전후 결과','Before/after results'))+' · '+a('case015.html#reproduce',txt('모델 없이 확인하기','Check without a model'))+'</div>'+home_p2(data,lang,a,'case015.html')+'</section>'


def render_case015(data,lang):
    ko=lang=='ko';txt=lambda k,e:k if ko else e
    labels=txt(['한눈에 보기','AI에게 시킨 일과 용어','도구 이름은 어떻게 바꿨나요?','실제 결과','실행 기록 살펴보기','직접 실행','기존 P1·통계·실행 계약','출처와 구현 기여'],['Overview','The task and its terms','What changed in the names?','Measured results','Explore recorded runs','Run it yourself','Original P1, statistics and contracts','Sources and contributions'])
    out='<header class="study-intro"><p class="eyebrow">DIOVA · CASE 015</p><h1>'+TITLES[lang]+'</h1><p class="lede">'+txt('도구 이름만 바뀌어도 AI가 같은 일을 끝낼 수 있을까요?','Can an AI finish the same task after a tool rename?')+'</p><p>'+txt('AI는 조회·계산·저장 같은 기능을 이름과 입력값으로 요청할 수 있습니다. 이를 도구 호출이라고 하며, 이런 기능을 제공하는 연결 방식을 API라고 부릅니다. 이 연구는 기능은 그대로 두고 이름 표기만 바꾸어 실제 작업 결과를 비교합니다.','An AI can request functions such as lookup, calculation and saving by supplying a name and input values. This is a tool call; an API is the interface providing those functions. This study keeps behavior fixed and compares task completion after changing only the names.')+'</p><p>'+txt('예제는 구매 내역의 총금액을 계산해 기록하는 일입니다. 아래에서 작업 순서와 용어를 먼저 설명하고, 이름 변경 전후의 결과와 실제 실행 기록으로 이어집니다.','The example calculates and saves a purchase total. The page first explains the task and terms, then shows before/after results and actual run records.')+'</p></header><div class="study-layout"><aside class="study-toc"><p>'+txt('페이지 목차','ON THIS PAGE')+'</p><nav aria-label="'+txt('사례 목차','Case contents')+'"><ol>'+''.join('<li>'+a('#'+sid,label)+'</li>' for sid,label in zip(SECTIONS,labels))+'</ol></nav></aside><article class="study-article">'
    def section(i,body):return '<section class="study-section" id="'+SECTIONS[i]+'"><h2>'+labels[i]+'</h2>'+body+'</section>'
    results=data['A1']['summary'];pairs=results['paired_regression']
    facts=''.join('<li>'+escape(r['method'])+': '+str(r['both_success'])+'/16 '+txt('두 이름 표기 모두 성공','successes with both naming versions')+' · '+str(r['loss'])+' '+txt('성공→실패','success→failure')+' / '+str(r['gain'])+' '+txt('실패→성공','failure→success')+'</li>' for r in pairs)
    out+=section(0,'<p>'+txt('구매 정보 조회→총금액 계산→저장→확인까지 실행하는 비교 도구를 구현했습니다. 새 응용시험을 A1이라고 부릅니다. 같은 작업을 요청하는 영어 표현 4개와 표현별 파일 4개로 16개 문제를 만들고, 두 이름 표기와 세 모델에서 총 96회 실행했습니다.','We built a comparison tool for reading purchase details, calculating a total, saving it and validating the record. This new application is called A1. Four English instruction templates with four files each make 16 problems, tested across two naming versions and three models for 96 runs.')+'</p>'+model_key(lang)+'<ul class="case015-facts">'+facts+'</ul><p>'+txt('여기서 성공은 금액이 맞는 것과 함께, 필요한 결과를 받은 뒤 실제 저장·확인·완료 답변을 끝낸 경우입니다. 실제 쇼핑몰이나 결제 서비스에는 연결하지 않았습니다.','Success requires the correct amount plus actual saving, validation and a final answer after receiving the necessary results. No shop or payment service is connected.')+'</p><div class="actions">'+a('#workflow',txt('어떤 일을 시켰는지 읽기','Read the task'))+' · '+a('#episodes',txt('실측 기록 하나 살펴보기','Inspect one measured run'))+'</div>')
    steps=txt(['지정된 구매 내역에서 개수와 개당 가격을 읽습니다.','조회 결과를 받은 다음, 곱셈 도구로 총금액을 계산합니다.','계산 결과를 받은 다음, 해당 구매 내역에 총금액을 저장합니다.','올바르게 저장됐는지 확인하는 도구를 실행합니다. 다른 구매 내역은 바꾸면 안 됩니다.','확인 결과를 받은 뒤 구매 ID·총금액·확인 여부를 정해진 형식으로 답합니다.'],['Read the item count and price per item for the requested purchase.','After receiving those details, use the multiplication tool to calculate the total.','After receiving the product, save it to that purchase record.','Run validation on the saved amount. Other purchase records must remain unchanged.','After receiving validation, return the purchase ID, total and validation flag in the required format.'])
    body='<ol class="case015-flow">'+''.join('<li>'+escape(x)+'</li>' for x in steps)+'</ol><p>'+txt('초기 요청에는 개수와 가격을 보여주지 않았으므로 AI는 조회 도구의 결과를 받아야 합니다. 금액은 소수점 없는 정수 단위입니다. 실제 구매 대신 실험용 파일을 사용합니다.','The initial request omits item count and price, so the AI must receive the lookup result. Amounts use integer units. Synthetic files stand in for real purchases.')+'</p>'+task_terms(lang)
    body+='<p class="method-key">'+txt('왜 결과를 받은 시점을 보나요? 조회와 계산을 같은 응답에서 한꺼번에 요청하면, 계산 명령을 만들 때 조회 결과를 아직 보지 못했기 때문입니다. 반면 저장→확인은 같은 응답에 있어도 차례대로 실행하면 허용합니다. 마지막 완료 답변은 실제 확인 결과를 받은 뒤에 해야 합니다.','Why record when results arrived? If lookup and calculation are requested in one response, the calculation was generated before the lookup result was received. Saving → validation may share a response if executed in order. The completion answer must follow actual receipt of validation.')+'</p>'
    out+=section(1,body)
    mapping=data['mappings'];names=[('구매 정보 읽기','Read purchase details','order.read'),('두 숫자 곱하기','Multiply two numbers','amount.multiply'),('총금액 저장','Save the total','order.write'),('저장 결과 확인','Validate the record','order.validate')]
    out+=section(2,'<p>'+txt('원래 도구 이름 묶음을 API_V1, 이름 표기를 바꾼 묶음을 API_V2라고 부릅니다. 예를 들어 read_order를 readOrder로 바꿉니다. 수행하는 기능은 같습니다.','API_V1 is the original set of tool names; API_V2 uses the renamed set. For example, read_order becomes readOrder, while the function stays the same.')+'</p>'+table([txt('하는 일','Function'),txt('원래 이름 V1','Original V1'),txt('바꾼 이름 V2','Renamed V2')],[[txt(k,e),mapping['API_V1'][s],mapping['API_V2'][s]] for k,e,s in names],'API names')+'<p>'+txt('설명·입력 항목 이름과 타입·기능·목록 순서는 유지했습니다. 바뀐 버전에서 옛 이름을 부르면 실행하지 않습니다. 이름 자동 교정이나 추가 힌트는 사용하지 않았습니다.','Descriptions, input-field names/types, behavior and tool order are fixed. Old names are rejected in the renamed version. There is no automatic name repair or extra hint.')+'</p>')
    rows=[[r['method'],r['api_version'],f"{r['verified_success']}/{r['observed']}",r['STATE_ONLY_SUCCESS'],r['observation_violation_episodes'],r['UNKNOWN_TOOL'],r['model_calls']] for r in results['success_table']]
    body='<p>'+txt('각 성공수는 같은 16개 문제 중 작업을 끝낸 개수입니다. ‘전체 작업 성공’은 위 다섯 단계를 충족한 경우이고, ‘마지막 파일 정답’은 최종 금액과 바꾸면 안 되는 항목이 모두 맞는지 검사합니다. 결과를 받은 순서와 중간 과정은 ‘전체 작업 성공’에서 추가로 확인합니다. 두 점수를 구분해 완료했다고 말만 한 답변을 성공으로 세지 않습니다.','Each success count is out of the same 16 problems. “Complete task” requires all five steps above; “Final file correct” checks the total and unchanged protected fields at the end. “Complete task” additionally checks received results and the intermediate process. This separates actual completion from a claim of completion.')+'</p>'
    body+=table([txt('모델','Model'),'API',txt('전체 작업 성공 /16','Complete task /16'),txt('마지막 파일 정답 /16','Final file correct /16'),txt('결과 수신 순서 위반','Receipt-order violations'),txt('미등록 이름 호출','Unknown tool names'),txt('모델 응답 요청','Model requests')],rows,'A1 measured success and diagnostics')
    body+=table([txt('모델','Model'),txt('둘 다 성공','Both success'),txt('성공→실패','Success→failure'),txt('실패→성공','Failure→success'),txt('둘 다 실패','Both fail')],[[r['method'],r['both_success'],r['loss'],r['gain'],r['both_fail']] for r in pairs],'API paired regression')
    body+='<p>'+txt('두 번째 표는 같은 문제의 V1과 V2를 짝지은 결과입니다. ‘성공→실패’는 이름 변경 후 잃은 성공, ‘실패→성공’은 새로 성공한 경우입니다. 한 가지 가상 작업의 결과이며, 앞선 P1의 8개 문제와 합산하지 않습니다.','The second table pairs V1 and V2 for each problem. Success→failure is a lost success after renaming; failure→success is a gain. This covers one synthetic workflow and is not pooled with the eight earlier P1 tasks.')+'</p>'
    body+='<details><summary>'+txt('영어 지시문 표현 4개별 전체 결과','All results by the four English instruction templates')+'</summary>'+table(['Template','Model','API','Success /4'],[[t,r['method'],r['api_version'],r['success']] for t,rr in results['template_results'].items() for r in rr],'Template results')+'</details>'
    body+='<p>'+txt('실제 사례: BF16의 A1-06 문제는 V1에서 금액을 맞게 저장했지만 확인 명령에 다른 구매 ID를 적었습니다. V2에서는 올바른 ID를 확인해 성공했습니다. D16은 A1-00/06/08/14의 저장 명령을 JSON 형식으로 완성하지 못했고, 두 이름 표기에서 같은 네 문제가 실패했습니다. 결과 수신 순서 위반과 미등록 도구 이름은 0건이었습니다.','Observed case: BF16 saved the correct amount for A1-06 on V1 but supplied a different purchase ID to validation. V2 validated the right ID and succeeded. D16 did not finish the write-call JSON on A1-00/06/08/14, failing on the same four problems with both naming versions. Receipt-order violations and unknown tool names were zero.')+'</p><details><summary>'+txt('원기록의 종료 사유 표기 보완','Corrections to diagnostic labels in the original records')+'</summary><p>'+txt('형식 오류로 일찍 끝난 8건의 원 terminal_reason은 BUDGET_EXHAUSTED로 남았습니다. 실제 예산 소진이 아니며 별도 사후 주석에서 PARSE_ERROR로 설명합니다. 잘못된 구매 ID의 값 오류 1건도 주석에 포함했습니다. 원 응답과 성공 판정은 바꾸지 않았습니다.','Eight early format failures retained BUDGET_EXHAUSTED as the raw terminal_reason. Budgets were not exhausted; separate post-hoc annotations explain PARSE_ERROR. They also count one wrong-purchase-ID value error. Original responses and success verdicts are unchanged.')+'</p></details>'
    prior=data['P1']['summary']['contrasts']['D16-minus-BF16']
    body+='<p>'+txt('앞선 P1은 기존 문제에서 도구 이름과 순서를 바꾼 별도 시험입니다. D16과 BF16의 성공 변화 차이는 ','The earlier P1 study separately changed tool names/order on existing tasks. Its D16 versus BF16 difference in success changes was ')+f"{100*prior['estimate']:+.2f} pp; 95% CI [{100*prior['CI95'][0]:+.2f}, {100*prior['CI95'][1]:+.2f}] pp."+' '+txt('pp는 퍼센트포인트, CI는 불확실성을 나타내는 구간입니다. 0을 포함해 방향이 불확정입니다.','pp means percentage points; CI is an uncertainty interval. Including zero makes the direction inconclusive.')+' '+a('#p1',txt('P1의 조건·전체 결과','P1 conditions and complete results'))+'</p>'
    body+=render_p2(data,lang,a,table)
    out+=section(3,body)
    controls='<div class="filters case015-filters"><label>'+txt('모델','Model')+'<select id="a1-method">'+''.join('<option>'+x+'</option>' for x in ['BF16','RTN','D16'])+'</select></label><label>API<select id="a1-api"><option>API_V1</option><option>API_V2</option></select></label><label>'+txt('시험 문제','Test problem')+'<select id="a1-scenario">'+''.join('<option>'+s['scenario_id']+'</option>' for s in data['scenarios'] if s['split']=='primary')+'</select></label></div><p id="a1-filter-status" role="status" aria-live="polite"></p><noscript>'+txt('JavaScript가 꺼져 있어 모든 episode를 아래에서 펼쳐 볼 수 있습니다.','JavaScript is off; all episodes remain available below.')+'</noscript>'
    tasks={s['scenario_id']:s for s in data['scenarios']};body='<p>'+txt('아래에서는 실행 한 번(episode)을 골라 봅니다. 요청과 호출 원문은 실험에 사용한 영어 그대로입니다. order는 구매 내역, quantity는 개수, unit_price_minor는 개당 가격, total_minor는 총금액입니다. 응답 묶음마다 먼저 받은 결과와 이후 실제로 바뀐 파일을 구분해 표시합니다.','Choose one recorded run (episode). Requests and calls remain in their original experimental English. order means a purchase record; quantity is item count, unit_price_minor is price per item, and total_minor is the total. Each response group separates already received results from subsequent file changes.')+'</p>'+controls
    annotations={r['episode_id']:r for r in data['A1']['diagnostic_annotations']['episodes']}
    for r in data['A1_episodes']:
        task=tasks[r['scenario_id']];eid=r['episode_id'];body+='<details class="a1-episode" id="'+escape(eid,quote=True)+'" data-method="'+r['method']+'" data-api="'+r['api_version']+'" data-scenario="'+r['scenario_id']+'"><summary>'+escape(eid)+' · '+txt('전체 작업 성공','Complete task')+' '+str(r['VERIFIED_WORKFLOW_SUCCESS'])+' · '+txt('마지막 파일 정답','Final file correct')+' '+str(r['STATE_ONLY_SUCCESS'])+'</summary>'
        body+='<h3>'+txt('요청과 선언된 이름','Request and declared names')+'</h3>'+pre(task['messages'])+pre(data['mappings'][r['api_version']])+'<details><summary>'+txt('초기 JSON','Initial JSON')+'</summary>'+pre(task['orders'])+'</details>'
        for response in r['responses']:
            turn=response['generated_in_turn'];body+='<section class="case015-turn"><h4>'+txt('응답 묶음','Response group')+' '+str(turn)+'</h4><p>'+txt('생성 전에 받은 결과 ID','Results received before generation')+': <code>'+escape(str(response['available_result_event_ids']))+'</code></p><p class="small">History SHA256: <code>'+response['pre_generation_history_hash']+'</code></p>'+pre(response['raw_text'])
            for event in r['tool_events']:
                if event['generated_in_turn']!=turn:continue
                body+='<details><summary>'+escape(str(event['raw_name']))+' → '+escape(str(event['semantic_tool_id']))+' / '+event['execution_status']+'</summary>'+pre({k:event[k] for k in ['raw_arguments','generated_after_event_ids','tool_result','state_before_execution','state_after_execution','state_version_before','state_version_after']})+'</details>'
            body+='</section>'
        annotation=annotations[r['episode_id']]
        body+='<h4>'+txt('최종 답·실제 상태·판정','Final answer, actual state and verdict')+'</h4>'+pre({**{k:r[k] for k in ['final_answer','final_fixture','VERIFIED_WORKFLOW_SUCCESS','STATE_ONLY_SUCCESS','REQUIRED_ACTIONS_DONE','OBSERVATION_ORDER_VALID','observation_violations']},
            'raw_terminal_reason':r['terminal_reason'],'derived_terminal_reason':annotation['derived_terminal_reason'],
            'posthoc_metadata_note':annotation['terminal_annotation'],'raw_ARGUMENT_VALUE_WRONG':r['ARGUMENT_VALUE_WRONG'],
            'derived_ARGUMENT_VALUE_WRONG_including_unknown_order_id':annotation['ARGUMENT_VALUE_WRONG_including_rejected_unknown_order_id']})+'</details>'
    out+=section(4,body)
    checker=P2+'/publication/run_checks.py' if data.get('P2') else C15+'/publication/scripts/run_checks.py'
    cmd='OUT=$(mktemp -d)\npython3 -B '+checker+' --output "$OUT/case015" --tests'
    body='<p>'+txt('공개 API 정의 읽기 → 두 API의 합성 정답 경로 동등성 실행 → 저장된 모델 응답을 새 fixture에 재실행 → 회귀표 재계산 순서입니다. 모델·설치·GPU 없이 Python 표준 라이브러리로 실행합니다.','Read the public API definitions → execute equivalent synthetic oracle traces → replay recorded model responses into fresh fixtures → recompute regression tables. The CPU path uses the Python standard library, without a model, installation or GPU.')+'</p>'+pre(cmd)+'<div class="actions">'+a('../sources/case015-reproduce-'+lang+'.md.txt',txt('전체 CPU 명령과 별도 GPU 범위','Full CPU commands and separate GPU scope'))+' · '+a('../sources/case015-a1-runner.py.txt',txt('실행기','Runner'))+' · '+a('../sources/case015-a1-tests.py.txt',txt('계약 검사','Contract tests'))+' · '+a(data['download_url'],txt('측정 당시 검토 ZIP v2','Historical review ZIP v2'))+'</div>'
    body+='<p>'+txt('다운로드 v2는 원 P1/A1 검토본이며 P2는 포함하지 않습니다. P2의 별도 보고서·재현 링크를 이용하세요. 원 P1/A1 측정값은 같습니다.','The v2 download preserves P1/A1 and does not contain P2. Use the separate P2 report/reproduction links. Historical P1/A1 measurements are unchanged.')+'</p>'
    out+=section(5,body)
    p1=data['P1']['summary'];p=p1['contrasts']['D16-minus-BF16'];rows=[]
    for method in ['BF16','RTN','D16']:rows.append([method,*[str(next(r['success'] for r in p1['success_table'] if r['method']==method and r['interface_id']==i))+'/8' for i in ['I0_ORIGINAL','I1_ORDER_ONLY','I2_STYLE_RENAME','I3_OPAQUE_RENAME']]])
    body='<details id="p1"><summary>'+txt('기존 P0/P1: 긴 입력에서 이름·순서를 바꾼 별도 시험','Original P0/P1: a separate name/order test with long inputs')+'</summary><p>'+txt('P0는 모델 없이 변환·실행 규칙을 검사한 단계입니다. P1은 기존 8개 문제에 긴 입력(약 4천 또는 8천 토큰)을 주고, 원형(I0)·목록 순서 변경(I1)·이름 표기 변경(I2)·의미를 드러내지 않는 이름(I3)을 세 모델에서 비교한 96회입니다. 토큰은 모델이 읽는 텍스트 조각입니다.','P0 checks transformation/execution rules without a model. P1 gives eight existing tasks long inputs (about 4,000 or 8,000 tokens) and compares original names/order (I0), reordered tools (I1), renamed style (I2) and opaque names (I3) across three models for 96 runs. A token is a piece of text read by the model.')+'</p>'+table([txt('작업','Task'),txt('요구한 일','Required action')],[[txt('문서 읽기','Read a document'),txt('장소 이름과 담당자 정보 읽기','Read site and owner fields')],[txt('조회 후 계산','Read then calculate'),txt('개수와 개당 가격을 읽고 곱셈 도구 실행','Read count and price, then use multiplication')],[txt('최신 상태 찾기','Find latest status'),txt('이전·현재 표에서 최신 정보 찾기','Read the current value from an old/current table')],[txt('설정 수정·확인','Update and validate settings'),txt('파일의 두 값을 바꾸고 확인 도구 실행','Change two config fields and validate')]],'Original P1 tasks')+table(['Model','I0 original','I1 order','I2 camel','I3 opaque'],rows,'Original P1 results')+'<p>'+txt('기존 P1의 D16−BF16 signed 변화 interaction은 ','The original P1 D16−BF16 signed-change interaction is ')+f"{100*p['estimate']:+.2f} pp; 95% CI [{100*p['CI95'][0]:+.2f}, {100*p['CI95'][1]:+.2f}] pp."+' '+txt('0을 포함합니다. 손실과 회복이 상쇄될 수 있으며 8개 DEV task의 탐색 결과입니다.','The interval includes zero. Losses and gains may cancel; this is an exploratory eight-DEV-task result.')+'</p><p class="method-key">'+txt('P1의 같은 실행 직전 fixture와 생성 전 관측 이력 대응은 별개입니다. 한 응답의 뒤 호출은 앞 호출 결과를 아직 보지 못했을 수 있습니다. 원 판정은 유지하고 A1의 더 명시적인 관측 계약을 소급 적용하지 않습니다.','Matching P1 pre-execution fixtures and matching pre-generation observations are separate. Later calls in a response may have been generated before earlier results were received. Original verdicts remain unchanged; the A1 observation contract is not retroactively applied.')+'</p>'+a('../sources/case015-p1-report-'+lang+'.md.txt',txt('보존된 원 P1 보고서','Preserved original P1 report'))
    for r in data['P1_episodes']:
        body+='<details><summary>'+escape(r['episode_id'])+' · '+str(r['functional_success'])+'</summary><p>'+txt('표시 범위: 실행 상태 기록 / 생성 전 정책 입력 동일성은 별도 미확인. API가 다른 prompt hash를 동일 입력으로 취급하지 않습니다.','Display scope: recorded execution state / pre-generation policy input equivalence is not separately established. Different API prompt hashes are not treated as identical inputs.')+'</p>'+pre({'tool_events':r['tool_events'],'responses':[x['raw_text'] for x in r['responses']],'final_answer':r.get('final_answer'),'functional_success':r['functional_success'],'legacy_success':r['legacy_success']})+'</details>'
    body+='</details><details><summary>'+txt('측정·통계·모델 경계','Measurement, statistics and model boundaries')+'</summary>'+pre({'A1':results['auxiliary'],'runtime':data['A1']['runtime'],'model_revision':data['model_revision'],'execution_source_sha256':data['execution_source_sha256']})+'<p>'+txt('A1의 부가 구간은 4개 template 묶음을 보존한 10,000회 탐색 재표집입니다. 폭0인 구간도 동등성 증거가 아닙니다. Saved INT4→dense BF16, D16 beta 역배율 경로, greedy 192token·8요청·6도구예산을 유지했습니다.','Auxiliary A1 intervals use 10,000 exploratory resamples of four template groups. A zero-width interval is not equivalence. Saved INT4→dense BF16, D16 inverse-beta transport, greedy 192-token generation and 8-request/6-tool budgets are retained.')+'</p></details>'
    body+=table([txt('범위','Scope'),txt('상태','Status')],[[txt('P2 같은 관측 JSON 복구','P2 same-observation JSON recovery'),txt('선택된 지점의 실측 완료; 위 별도 결과','Measured on selected checkpoints; separate results above')],[txt('P1·A1 원형/실측 기록 감사','P1/A1 preservation and recorded-response audit'),txt('각 단계 별도 완료·분모','Completed separately, distinct denominators')],['CIPR / training / full replay / fresh64 / D4 / packed serving',txt('미실행','Not run')],[txt('외부 업무·결제 서비스','External business/payment systems'),txt('연결하지 않은 합성 fixture','Synthetic fixture, no connection')]],'Study scope')
    out+=section(6,body)
    out+=section(7,'<p>'+a('../sources/case015-guide-'+lang+'.md.txt',txt('작업·용어 안내','Task and terminology guide'))+' · '+a('../sources/case015-report-'+lang+'.md.txt',txt('정식 통합 보고서','Unified formal report'))+' · '+a('../sources/case015-notice.md.txt','NOTICE')+' · '+a('../data/case015.json',txt('원자료 연결과 hash','Raw source identities and hashes'))+'</p><p>'+txt('DIOVA의 registry·동등성 검사·상태/관측 기반 scorer·감사와 화면을 구현했습니다. Qwen·PyTorch·Transformers와 DualQuant 공개 수식 기반 reference를 재사용했습니다. CIPR 학습은 실행하지 않았습니다. Codex/ChatGPT가 구현과 별도 CPU 재계산을 보조했습니다.','DIOVA implements registry conversion, equivalence checks, state/observation scoring, audits and displays. It reuses Qwen, PyTorch, Transformers and an independent reference based on public DualQuant equations. No CIPR training was run. Codex/ChatGPT assisted implementation and separate CPU recomputation.')+'</p>')
    return out+'</article></div>'
