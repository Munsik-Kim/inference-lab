"""Case018 display: separate original recovery endpoints and posthoc observations."""
from html import escape
import importlib.util
import functools
from common import read, require, sha

C18='cases/018-agent-state-recovery'
TITLES={'ko':'재고 복구와 완료 보고를 따로 검사했습니다','en':'Checking inventory recovery and completion reports'}
SECTIONS=('overview','results','method','posthoc','cost','episodes','limits','reproduce','references')
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

def display_facts(data):
    """Read-only presentation of the existing, verified summary; no rescoring."""
    groups = data['phases'][1]['groups']
    cells = {(g['kv'], g['policy']): g for g in groups}
    cross = {k: sum(g['cross_table'][k] for g in groups) for k in groups[0]['cross_table']}
    correct_Q1 = sum(g['conditional_Q1']['cross_table']['read1_final1'] for g in groups)
    base, guard = cells['K1', 'R0'], cells['K1', 'R1']
    return dict(groups=groups, cells=cells, cross=cross,
                tasks=groups[0]['episodes'], accepted=sum(g['accepted_final'] for g in groups),
                no_final=sum(g['no_accepted_final'] for g in groups),
                wrong_recovery_reports=cross['read1_final1']-correct_Q1,
                guard_episodes=guard['no_accepted_final'], guards=guard['guard_rejections'],
                output_increase=100*(guard['output_tokens']/base['output_tokens']-1),
                input_increase=100*(guard['input_tokens']/base['input_tokens']-1),
                wall_increase=100*(guard['episode_wall_seconds']/base['episode_wall_seconds']-1))


def quantity_pair(values, suffixes):
    return ' / '.join(str(next(v for k, v in values.items() if k.endswith('-'+s))) for s in suffixes)


def opening_quantities(episode):
    history = next(t['event']['result']['history'] for t in episode['turns']
                   if t['event'] and t['event']['tool'] == 'read_history')
    return {k: r['quantity'] for k, r in history[0]['records'].items()}


def home_case018(data, lang):
    ko = lang == 'ko'; f = display_facts(data)
    b, n = f['cells']['K0', 'R0'], f['cells']['K1', 'R0']
    intro = ('잘못 옮긴 재고, 중복 이동, 일부 차감 기록을 AI가 되돌리게 하고, 복구된 상태와 마지막 보고를 따로 검사하는 평가기를 만들었습니다.' if ko else
             'We built an inventory environment and evaluator to check recovery from wrong-target transfers, duplicate processing and partial debits, separately from the agent’s completion report.')
    result = (f"같은 {f['tasks']}개 합성 과제에서 두 정책 모두 상태·후속 검사 통과는 BF16 KV {b['Q']}/{f['tasks']}, NVFP4 KV {n['Q']}/{f['tasks']}였습니다. 검증·완료 보고까지 충족한 수는 각각 {b['W']}/{f['tasks']}, {n['W']}/{f['tasks']}였습니다. 양쪽 모델 가중치는 BF16으로 고정했습니다." if ko else
              f"On the same {f['tasks']} synthetic tasks, both policies passed state and follow-up checks in {b['Q']}/{f['tasks']} tasks with BF16 KV and {n['Q']}/{f['tasks']} with NVFP4 KV. Including validation and a correct completion report, the counts were {b['W']}/{f['tasks']} and {n['W']}/{f['tasks']}. Model weights were BF16 in both configurations.")
    return ('<section class="panel case018-entry" id="case018-entry"><h2>Case018 · '+escape(TITLES[lang])+
            '</h2><p>'+escape(intro)+'</p><p>'+escape(result)+'</p><div class="actions">'+
            a('case018.html#episodes', '실제 사건 보기' if ko else 'Recorded episodes')+' · '+
            a('case018.html#results', '결과와 방법' if ko else 'Results and method')+' · '+
            a('case018.html#reproduce', '모델 없이 검산하기' if ko else 'CPU record replay')+'</div></section>')


def render_case018(data, lang):
    import json
    ko = lang == 'ko'; t = lambda k, e: k if ko else e
    f = display_facts(data); groups = f['groups']; cells = f['cells']
    selected = {e['episode_id']: e for e in data['selected_episodes']}
    b = selected['HOLDOUT-A00_K0_R0']; n = selected['HOLDOUT-A00_K1_R0']
    c = selected['HOLDOUT-C01_K1_R0']; damaged = selected['HOLDOUT-A06_K1_R0']
    dump = lambda x: escape(json.dumps(x, ensure_ascii=False, indent=2))
    names = dict(zip(SECTIONS, t(
        ['한눈에 보기', '결과가 갈린 실제 장면', '업무와 검사 방법', '수량 관측과 보고', '실행 비용과 KV 저장', '실제 사건과 원문', '결과를 읽는 범위', '모델 없이 검산하기', '구현과 출처'],
        ['Overview', 'What happened', 'Tasks and checks', 'Quantity observations and reports', 'Execution cost and KV storage', 'Recorded episodes and raw text', 'Reading these results', 'CPU record replay', 'Implementation and sources'])))
    def section(key, body):
        return '<section class="panel" id="'+key+'"><h2>'+escape(names[key])+'</h2>'+body+'</section>'
    def policy(p):
        return t('기본 수용 (R0)', 'Basic acceptance (R0)') if p == 'R0' else t('최신 검증 가드 (R1)', 'Current-validation guard (R1)')
    def kv(k): return 'BF16' if k == 'K0' else 'NVFP4'
    def fraction(g, key): return str(g[key])+' / '+str(g['episodes'])
    quality = table(t(['KV 저장', '완료 응답 정책', '상태·후속 검사 통과 (Q)', '검증·완료 보고까지 충족 (W)'],
                      ['KV storage', 'Completion policy', 'State and follow-up checks (Q)', 'Validation and correct report (W)']),
                    [[kv(g['kv']), policy(g['policy']), fraction(g, 'Q'), fraction(g, 'W')] for g in groups],
                    t('같은 합성 과제의 네 조건별 결과', 'Four conditions on the same synthetic tasks'))
    quality = '<div class="case018-quality">'+quality+'</div>'
    out = ('<div class="case018-layout"><nav class="case018-toc" aria-label="'+t('이 페이지 목차', 'On this page')+'">'+
           ''.join(a('#'+key, names[key]) for key in SECTIONS)+'</nav><article class="case018"><h1>Case018 — '+escape(TITLES[lang])+'</h1>')
    out += section('overview', '<p class="case018-lead">'+t(
        '잘못 옮긴 재고, 두 번 처리한 이동, 일부만 차감한 기록을 AI가 되돌리게 했습니다. 재고가 올바르게 복구됐는지와 마지막 보고까지 맞는지를 따로 검사했습니다.',
        'We asked an AI agent to undo wrong-target inventory changes, duplicate transfers and partial debits. We checked the recovered state separately from its completion report.')+'</p><p>'+t(
        '재고·처리 원장, 도구 API와 판정기를 구현해 실제 저장 값·관측·완료 응답을 재생 검사했습니다.',
        'We built the inventory/operation ledger, tool APIs and evaluator to replay saved values, observed results and completion responses.')+'</p>'+quality+'<dl class="case018-definitions"><div><dt>'+t('상태·후속 검사 통과 (Q)', 'State and follow-up checks (Q)')+'</dt><dd>'+t(
        '업무 값 복원, 전 과정 보호, 평가기의 두 고정 CPU 후속 절차를 모두 통과한 경우입니다.',
        'Business values are restored, protection holds throughout recovery, and both fixed CPU follow-up programs pass.')+'</dd></div><div><dt>'+t('검증·완료 보고까지 충족 (W)', 'Validation and correct report (W)')+'</dt><dd>'+t(
        'Q에 더해 최신 성공 검증을 실제 받고, 수용된 완료 응답의 내용까지 정확한 경우입니다.',
        'Q passes, current successful validation is received, and the accepted completion response is correct.')+'</dd></div></dl><p>'+t(
        f"각 행은 같은 {f['tasks']}개 합성 과제입니다. 양쪽 Qwen 모델 가중치는 BF16으로 고정하고, 앞선 토큰의 attention 계산에 쓰는 key/value 저장(KV) 구성과 완료 응답 정책을 비교했습니다.",
        f"Each row uses the same {f['tasks']} synthetic tasks. Qwen model weights stay BF16; we compare key/value storage for attention over previous tokens (KV) and completion-response policies.")+'</p><p>'+t(
        f"최신 검증 가드는 완료 응답 수용을 차단했지만 복구·정확한 완료 건수는 늘리지 못했습니다. NVFP4에서 출력 토큰은 {f['output_increase']:.2f}% 늘었습니다.",
        f"The validation guard blocked ineligible completion responses without increasing recovered or fully verified completions. NVFP4 output tokens increased by {f['output_increase']:.2f}%.")+'</p><div class="actions">'+
        a('#episodes', t('실제 사건 보기', 'Recorded episodes'))+' · '+a('#results', t('결과와 방법', 'Results and method'))+' · '+a('#reproduce', t('모델 없이 검산하기', 'CPU record replay'))+'</div>')

    story_rows = [[t('처음 조회', 'First read'), quantity_pair(b['diagnostic']['first_read_quantities'], 'xy'), quantity_pair(n['diagnostic']['first_read_quantities'], 'xy')],
                  [t('복구 후 저장', 'Saved after repair'), quantity_pair(b['terminal_quantities'], 'xy'), quantity_pair(n['terminal_quantities'], 'xy')],
                  [t('완료 응답에 적은 수량', 'Reported quantities'), quantity_pair(b['diagnostic']['final_quantities'], 'xy'), quantity_pair(n['diagnostic']['final_quantities'], 'xy')]]
    story = '<h3>'+t('A00: 재고는 돌아왔지만 보고에는 처음 값이 남았습니다', 'A00: the inventory was restored, but the report kept the first values')+'</h3><p>'+t(
        f"BF16 사례의 재고는 {quantity_pair(b['terminal_quantities'], 'xy')}로 돌아왔지만, 완료 응답에는 {quantity_pair(b['diagnostic']['final_quantities'], 'xy')}가 남았습니다. 아래는 잘못 변경된 두 기록의 수량입니다.",
        f"In the BF16 episode, inventory returned to {quantity_pair(b['terminal_quantities'], 'xy')}, but the completion report still said {quantity_pair(b['diagnostic']['final_quantities'], 'xy')}. These are the two records changed by the injected error.")+'</p>'+table(
        t(['시점', 'BF16 KV · 기본 수용', 'NVFP4 KV · 기본 수용'], ['Stage', 'BF16 KV · basic acceptance', 'NVFP4 KV · basic acceptance']), story_rows,
        t('A00의 처음 조회, 복구 후 저장, 최종 보고 수량', 'A00 first read, restored inventory and reported quantities'))+'<p>'+t(
        'BF16 경로는 보상과 활성 처리 해제 뒤 공개 검증의 valid:true를 받았습니다. NVFP4 경로는 같은 복구 뒤 현재 수량을 다시 읽고 검증한 다음 그 수량을 보고했습니다.',
        'The BF16 trajectory received valid:true from public validation after compensation and release of the active operation. The NVFP4 trajectory also reread the restored quantities before validation and reported those values.')+'</p><p>'+t(
        '공개 검증은 비음수 수량·보호 필드·수량 보존·원장 참조를 확인합니다. 응답에는 전체 현재 수량이 들어 있지 않습니다. 별도 판정기가 완료 응답의 수량을 저장된 수량과 대조하므로, 최신 검증 관측과 현재 수량 재관측은 구별됩니다.',
        'Public validation checks nonnegative inventory, protected fields, conservation and ledger references. Its response does not contain all current quantities. A separate evaluator compares reported quantities with saved quantities, so observing validation differs from rereading inventory.')+'</p><details><summary>'+t('A00 관측 시점과 원문 근거', 'A00 observation timing and raw evidence')+'</summary>'
    for e in (b, n):
        events = [turn['event'] for turn in e['turns'] if turn['event']]
        order = ' → '.join(event['event_id']+' '+event['tool'] for event in events)
        d = e['diagnostic']
        story += '<p><code>'+escape(e['episode_id'])+'</code>: '+escape(order)+'</p><p>'+escape(f"Q={d['Q']}, Fresh={d['Fresh']}, W={d['W']}")+' · '+a('#episode-'+e['episode_id'], t('모델 원문 열기', 'Open recorded model text'))+'</p>'
    story += '</details><h3 id="guard">'+t('완료 응답 가드는 무엇을 바꿨나?', 'What did the completion guard change?')+'</h3><dl class="case018-guard"><dt>'+t('확인한 것', 'What it checked')+'</dt><dd>'+t(
        '모델이 현재 상태의 범위·digest·version에 맞는 성공 검증 결과를 실제 받은 뒤 완료 응답을 냈는지 확인했습니다.',
        'Whether the agent actually received successful validation matching the current scope, digest and version before generating its completion response.')+'</dd><dt>'+t('차단한 것', 'What it blocked')+'</dt><dd>'+t(
        f"NVFP4 R1의 {f['guard_episodes']}개 episode에서 부적격 완료 응답 {f['guards']}시도를 수용하지 않았습니다.",
        f"It refused {f['guards']} ineligible completion attempts across {f['guard_episodes']} NVFP4 R1 episodes.")+'</dd><dt>'+t('그대로 남은 것', 'What remained')+'</dt><dd>'+t(
        '가드는 상태 수리와 최종 수량 비교를 수행하지 않습니다. 해당 궤적의 실패 상태는 회복되지 않았고, 두 KV 구성 모두 R0와 R1의 Q/W 통과 수가 같았습니다.',
        'The guard neither repairs inventory nor compares reported quantities. Those failed states remained unrecovered; R0 and R1 had the same Q/W counts in both KV configurations.')+'</dd></dl><h3>'+t('현재 값을 맞게 보고해도 복구 목표는 틀릴 수 있습니다', 'A correct report of current values can still describe a wrong recovery')+'</h3><p>'+t(
        f"C01의 복구 목표는 {quantity_pair(opening_quantities(c), 'sd')}였지만 NVFP4 경로는 {quantity_pair(c['terminal_quantities'], 'sd')}로 저장했습니다. R0는 현재 수량을 그대로 보고했으나 상태 복구에 실패했습니다. R1은 같은 잘못된 상태에서 완료 응답 수용을 차단했습니다.",
        f"C01’s recovery target was {quantity_pair(opening_quantities(c), 'sd')}, but the NVFP4 trajectory saved {quantity_pair(c['terminal_quantities'], 'sd')}. R0 accurately reported the current inventory but failed recovery. R1 blocked completion acceptance while that wrong state remained.")+'</p><p>'+t(
        f"A06은 목표 {quantity_pair(opening_quantities(damaged), 'xy')} 대신 {quantity_pair(damaged['terminal_quantities'], 'xy')}를 남기고 총수량 보존도 깨뜨렸습니다. 이 과정의 보호 실패는 종료 수량을 정확히 적어도 남습니다.",
        f"A06 left {quantity_pair(damaged['terminal_quantities'], 'xy')} instead of the target {quantity_pair(opening_quantities(damaged), 'xy')} and broke total-stock conservation. Accurately reporting the ending inventory does not erase that protection failure.")+'</p><p>'+a('#episode-HOLDOUT-C01_K1_R0', 'C01 R0')+' · '+a('#episode-HOLDOUT-C01_K1_R1', 'C01 R1')+' · '+a('#episode-HOLDOUT-A06_K1_R0', 'A06 R0')+'</p>'
    out += section('results', story)

    details = '<p>'+t(
        f"잘못된 대상·중복 처리·부분 차감의 세 유형에서 각 {f['tasks']//3}개씩, 총 {f['tasks']}개 합성 과제를 만들었습니다. 같은 과제를 네 조건으로 실행한 {data['phases'][1]['episodes']}개 episode를 비교합니다.",
        f"We generated {f['tasks']//3} synthetic tasks for each of three error types: wrong target, duplicate processing and partial debit. The same {f['tasks']} tasks were run under four conditions, producing {data['phases'][1]['episodes']} episodes.")+'</p>'+table(t(['주입 오류', '복구 목표'], ['Injected error', 'Recovery target']), [
        [t('잘못된 대상 (A)', 'Wrong target (A)'), t('잘못 바꾼 상품 기록을 되돌리고 활성 처리 해제', 'Undo the wrong-SKU change and release its active operation')],
        [t('중복 처리 (B)', 'Duplicate processing (B)'), t('정상 1회는 보존하고 추가 1회만 보상', 'Preserve the first valid transfer and compensate only the extra transfer')],
        [t('부분 차감 (C)', 'Partial debit (C)'), t('실행된 출발 수량 차감만 되돌리고 활성 처리 해제', 'Undo only the source debit and release the active operation')]], t('세 오류 유형의 복구 목표', 'Recovery targets for the three error types'))+'<p>'+t(
        '모델은 기록·이력·처리 원장을 조회하고, 보상 수량을 직접 지정하며, 처리 해제·검증·완료 도구를 호출합니다. 복구 목표 checkpoint와 정답 script는 별도 평가기에서 보관합니다.',
        'The agent reads records, history and operations, chooses compensation quantities, and calls release, validation and final tools. The target checkpoint and repair script stay with the separate evaluator.')+'</p><h3>'+t('두 고정 후속 절차', 'Two fixed follow-up programs')+'</h3><ol><li>'+t(
        '첫 절차는 A/C에서 원래 남은 정상 이동을, B에서 다음 1개 이동을 실행하고 전체 수량을 조회·검증합니다.',
        'The first program performs the intended pending transfer for A/C, or the next one-unit transfer for B, then reads and validates the inventory.')+'</li><li>'+t(
        '둘째 절차는 등록된 원래 이동을 다시 전달한 뒤 다음 1개 이동을 실행합니다. 이미 완료한 B의 원래 이동은 중복 방지로 한 번 더 적용되지 않아야 합니다.',
        'The second program redelivers the original registered transfer, then performs the next one-unit transfer. For B, duplicate prevention must keep the already completed transfer from being applied again.')+'</li></ol><p>'+t(
        '각 절차는 종료 상태의 독립 복사본에서 시작하며 변경 직전 공개 조회의 revision을 씁니다. 정상 checkpoint에서도 같은 인자·순서로 실행해 저장 값, 원장, 중복 처리 결과와 보호 계약을 비교합니다. 이번 모델 표본에서는 업무 값 복원과 두 절차의 성공 집합이 같았습니다. 활성 원장이 남은 P0 반례는 평가기의 별도 검사로 보존했습니다.',
        'Each program starts from an independent copy of the terminal state and reads its current revision before mutation. It runs the same arguments and steps from the clean checkpoint to compare business values, ledger state, duplicate handling and protection. In these model samples, business-restoration and continuation success sets coincided. P0 counterexamples with an unreleased operation test the evaluator separately.')+'</p><details><summary>'+t('정확한 지표·DEV 결과·짝 비교', 'Exact metrics, DEV results and paired comparisons')+'</summary><p>'+t(
        'B=업무 값 복원, P=전 과정 보호, C1/C2=두 고정 후속 절차, Q=B∧P∧C1∧C2입니다. Fresh는 현재 성공 검증의 사전 관측이며 W=Q∧수용된 정확한 final∧Fresh입니다. final 내용 판정은 task/recovered·전체 수량 map·검증 event ID를 확인합니다. D는 recovered:true 선언으로 거절된 final도 포함합니다. 수용된 final이 없어도 종료 상태의 Q를 계산하고 W=0으로 둡니다. 이때 Fresh는 마지막 실제 생성 직전 관측의 진단입니다.',
        'B=business restoration, P=protection throughout, C1/C2=the two fixed continuations, and Q=B∧P∧C1∧C2. Fresh is prior observation of successful current validation; W=Q∧accepted correct final∧Fresh. Final-content checks cover task/recovered, the full quantity map and validation event ID. D counts recovered:true declarations, including rejected finals. Without an accepted final, Q is still assessed at termination and W=0; Fresh then describes observations before the last actual generation.')+'</p><figure><img src="../assets/case018-quality.svg" width="760" height="350" alt="State/follow-up Q and verified-report W counts per 24 tasks"><figcaption>'+t('같은 24개 과제에서 Q와 W가 드러낸 차이', 'Q and W highlight different outcomes on the same 24 tasks')+'</figcaption></figure>'+table(['KV', 'Policy', 'D', 'B', 'P', 'Fresh', 'C1', 'C2', 'Q', 'W'],
        [[kv(g['kv']), g['policy'], *[g[k] for k in ['D', 'B', 'P', 'Fresh', 'C1', 'C2', 'Q', 'W']]] for g in groups], t('24 과제 기준 세부 지표', 'Detailed metrics per 24 tasks'))+table(['KV', 'Policy', 'Q / 6', 'W / 6'],
        [[kv(g['kv']), g['policy'], g['Q'], g['W']] for g in data['phases'][0]['groups']], t('별도 DEV 6개 과제', 'Separate six-task DEV sample'))+table(
        t(['짝 비교', '지표', '차이 (pp)', '탐색적 95% 구간 (pp)'], ['Paired comparison', 'Metric', 'Difference (pp)', 'Exploratory 95% interval (pp)']),
        [[p['comparison'], p['endpoint'], f"{p['difference_pp']:.2f}", ' / '.join(f'{v:.2f}' for v in p['exploratory_95_pp'])] for p in data['paired_P2']], 'Registered paired task comparisons')+'<p>'+t(
        'DEV 6개와 HOLDOUT 24개는 별도 분모입니다. 등록된 family-stratified paired bootstrap은 10,000회, seed 61022018로 같은 task의 네 조건을 함께 재표집합니다. 구간은 이 합성 과제 집합의 pointwise 탐색 구간입니다. R1−R0의 관측 차이와 구간이 0이어도 다른 과제에서 효과가 같다는 보장은 주지 않습니다.',
        'DEV uses six tasks and HOLDOUT uses 24, with separate denominators. The registered family-stratified paired bootstrap jointly resamples each task’s four conditions over 10,000 draws, seed 61022018. Intervals are exploratory and pointwise for this synthetic task set. A zero R1−R0 difference and zero-width interval here leaves effects on other tasks unresolved.')+'</p><p>'+t(
        '공통 공개 JSON 문법·엄격한 parser·8개 API·greedy를 사용합니다. 요청 16회, 요청당 출력 512 tokens, episode 출력 8,192 tokens, 문맥 24,576 tokens, 요청/episode 제한 180/900초입니다. R1 거절과 재생성도 같은 예산에 포함합니다. 원 계획의 모델 성공수 확대 임계값은 null입니다.',
        'All conditions share public JSON grammar, a strict parser, eight APIs and greedy decoding: 16 requests, 512 output tokens per request, 8,192 per episode, 24,576 context tokens and 180/900-second request/episode deadlines. R1 rejections and subsequent generations use that same budget. The original protocol sets the model-success expansion threshold to null.')+'</p></details>'
    out += section('method', details)

    cross = f['cross']
    posthoc = '<p>'+t(
        '기존 실행 기록에서 뒤늦게 조사한 사후 분석입니다. 앞서 본 A00 사례와 상태는 통과했지만 완료 보고는 실패한 기록을 동기로, 전체 집계 전에 정의·코드·입력 hash를 저장했습니다.',
        'This posthoc analysis examines existing execution records. A00 and the previously observed state-pass/report-fail cases motivated the definition, code and input hashes saved before the new full aggregate.')+'</p><p>'+t(
        f"시스템이 수용한 완료 응답 {f['accepted']}개에서, 현재 수량 재관측과 종료 수량 보고의 관계를 비교했습니다.",
        f"We compared observed current inventory with reported terminal quantities across {f['accepted']} completion responses accepted by the system.")+'</p>'+table(
        t(['현재 수량 재관측', '종료 수량과 보고 일치', '불일치'], ['Current quantities reread', 'Report matches terminal quantities', 'Mismatch']),
        [[t('있음', 'Yes'), cross['read1_final1'], cross['read1_final0']], [t('없음', 'No'), cross['read0_final1'], cross['read0_final0']]],
        t('수용된 완료 응답의 수량 관측과 보고 일치', 'Quantity observations and report consistency in accepted finals'))+'<p class="case018-callout">'+t(
        f"보고가 일치한 {cross['read1_final1']}건 중 {f['wrong_recovery_reports']}건은 잘못 복구한 상태를 정확히 보고한 경우입니다. 완료 응답이 수용되지 않은 {f['no_final']}건의 수량 비교는 NA로 따로 둡니다.",
        f"Of the {cross['read1_final1']} matching reports, {f['wrong_recovery_reports']} accurately described incorrectly recovered states. The {f['no_final']} episodes without an accepted final have NA comparisons and are counted separately.")+'</p><p>'+t(
        f"상태·후속 검사는 통과했지만 완료 보고까지 충족하지 못한 {data['phases'][1]['Q1_W0']['count']}개 episode의 최종 수량은 모두 처음 관측한 수량과 같았습니다. 각 조건이 같은 과제를 반복하므로, 이 건수는 고유 과제 수와 다릅니다.",
        f"All {data['phases'][1]['Q1_W0']['count']} episodes that passed Q but failed W reported their first observed quantities. Conditions repeat the same tasks, so these episode counts are not counts of distinct tasks.")+'</p><p>'+t(
        '현재 수량 재관측은 필요한 모든 기록을 실제 전달된 성공 read_records 결과로 확인하고, 이후 그 수량이 변하지 않은 경우입니다. 최신 성공 validation을 받았는지는 별도 지표로 유지합니다.',
        'Current-quantity re-observation requires actually delivered successful read_records results covering every required record, with no later quantity change. Receiving successful current validation remains a separate measure.')+'</p><p>'+t(
        '기존 실행 기록을 나눠 보니 최신 수량 관측과 보고 정확성이 함께 나타났습니다. 행동 경로와 KV 구성도 함께 달랐으므로, 재조회를 강제했을 때 정확성이 얼마나 바뀌는지는 별도 비교 실험에서 확인해야 합니다.',
        'Current-quantity observations and accurate reports occurred together in these saved trajectories. Actions and KV configurations also differed, so measuring the effect of requiring a reread would need a separate comparison experiment.')+'</p><details><summary>'+t('관측 정의와 조건별 상세 표', 'Observation definition and tables by condition')+'</summary><p>'+t(
        '실패한 조회·미전달 결과·일부 기록만 본 경우는 전체 관측으로 세지 않습니다. 수량이 바뀌었다가 원복돼도 이전 조회는 만료됩니다. 메타데이터만 바뀐 global revision은 수량 관측을 만료시키지 않습니다. 모든 상태 변경 뒤 조회한 보조 flag도 별도로 저장했으며 이번 실제 표본에서는 두 flag가 같습니다. 수용된 final의 생성 직전 available event ID를 기준으로 하고, final이 없으면 마지막 실제 생성 직전을 진단 anchor로 씁니다.',
        'Failed reads, undelivered results and partial record coverage do not count as full observation. A quantity change followed by restoration expires the earlier read; a metadata-only global revision does not. A separate after-all-state-mutations read flag is stored; the flags coincide in this sample. The anchor is the available event IDs before the accepted final’s generation, or before the last actual generation when no final was accepted.')+'</p>'+table(
        ['KV', 'Policy', t('수용 / 없음', 'Accepted / none'), 'No read · mismatch', 'No read · match', 'Read · mismatch', 'Read · match'],
        [[kv(g['kv']), g['policy'], f"{g['accepted_final']} / {g['no_accepted_final']}", *[g['cross_table'][k] for k in ['read0_final0', 'read0_final1', 'read1_final0', 'read1_final1']]] for g in groups], 'HOLDOUT accepted-final comparison by condition')+'<p>'+t(
        'DEV 분석과 Q=1 하위 집계는 보고서·전체 ledger에서 확인할 수 있습니다. Q=1 집계는 실행 결과로 선택된 보조 집계입니다.',
        'The report and full ledger retain the separate DEV analysis and Q=1 subgroup. The Q=1 summary is an outcome-selected auxiliary analysis.')+'</p><figure><img src="../assets/case018-observation.svg" width="760" height="300" alt="Posthoc current-quantity observations and terminal-report consistency"></figure>'+a('../sources/case018-quantity-definition.json.txt', t('정확한 사후 분석 정의', 'Exact posthoc definition'))+' · '+a('../sources/case018-quantity-ledger.json.txt', t('전체 120행 근거', 'All 120 diagnostic rows'))+'</details>'
    out += section('posthoc', posthoc)

    costs = data['costs']
    out += section('cost', table(['KV', 'Policy', 'Requests', 'Input tokens', 'Output tokens', t('수용 차단', 'Rejections'), 'Episode s'],
        [[kv(g['kv']), g['policy'], g['requests'], g['input_tokens'], g['output_tokens'], g['guard_rejections'], f"{g['episode_wall_seconds']:.3f}"] for g in groups],
        t('실패를 포함한 조건별 24회 실행 비용', 'Costs of all 24 episodes per condition, including failures'))+'<p>'+t(
        f"NVFP4의 R0→R1에서 입력 토큰은 {f['input_increase']:.2f}%, 출력은 {f['output_increase']:.2f}%, episode wall은 {f['wall_increase']:.2f}% 늘었습니다. 표는 성공·실패 24회 전체의 요청과 시간을 합산합니다. 입력과 출력은 다른 계산 비용이므로 따로 표시합니다.",
        f"NVFP4 R0→R1 increased input tokens by {f['input_increase']:.2f}%, output by {f['output_increase']:.2f}% and episode wall by {f['wall_increase']:.2f}%. The table totals all 24 successes and failures. Input and output tokens are reported separately because their computation costs differ.")+'</p><p>'+t(
        f"P1/P2 누적 요청 시도는 {costs['attempts']:,}회, 완전 응답은 {costs['complete']:,}개입니다. 관측 출력 하한은 {costs['observed_output_lower_bound']:,} tokens이며 과거 취소 {costs['unknown_engine_output_attempts']}건의 실제 엔진 생성량은 미관측으로 남습니다. worker wall {costs['worker_wall_seconds']:.9f}초는 적재·컴파일·실패·종료를 포함하고 episode wall과 겹칩니다. 새 CPU 검산 비용은 별도 receipt에 기록합니다.",
        f"P1/P2 total {costs['attempts']:,} request attempts and {costs['complete']:,} complete responses. Observed output is a lower bound of {costs['observed_output_lower_bound']:,} tokens; engine output for {costs['unknown_engine_output_attempts']} earlier cancellations remains unknown. Worker wall of {costs['worker_wall_seconds']:.9f} seconds includes loading, compilation, failures and shutdown and overlaps episode wall. New CPU replay cost is recorded separately.")+'</p><h3>'+t('같은 KV 공간에 저장한 토큰 슬롯', 'Token slots in the same KV memory budget')+'</h3><p>'+t(
        '약 4 GiB의 KV 저장 공간에 담을 수 있는 토큰 슬롯은 BF16 29,120개에서 NVFP4 103,552개로 늘었습니다. 실제 pool은 각각 4,293,918,720 / 4,294,508,544 bytes이고 토큰당 147,456 / 41,472 bytes입니다. 모델 가중치와 실행 임시 메모리는 이 슬롯 계산에 포함되지 않습니다.',
        'An approximately 4 GiB KV pool holds 29,120 BF16 token slots or 103,552 NVFP4 slots. Actual pools are 4,293,918,720 / 4,294,508,544 bytes, at 147,456 / 41,472 bytes per token. Model weights and temporary execution memory are outside this slot calculation.')+'</p><p>'+t(
        '기술 receipt는 packed uint8 데이터·FP8 block scale, 36층의 저장·읽기와 유한한 attention 출력을 기록합니다. 실제 구성은 vLLM 0.32.0·FlashInfer 0.7.0.post1·xgrammar 0.2.8이며 두 KV의 커널 경로도 포함한 비교입니다. 적재 실패·취소·종료 경고는 원기록에 남겨 두었습니다.',
        'Technical receipts record packed uint8 data, FP8 block scales, storage and reads across 36 layers, and finite attention outputs. The configurations use vLLM 0.32.0, FlashInfer 0.7.0.post1 and xgrammar 0.2.8; their kernel paths also differ. Loading failures, cancellations and shutdown warnings remain in the original records.')+'</p>')

    explorer = '<p>'+t(
        'A00의 두 구성, A06의 보호 실패, C01의 기본 수용과 가드 경로를 실제 원문으로 비교합니다. 설명용 요약 아래에서 모델 출력과 당시 제공된 도구 결과를 펼칠 수 있습니다.',
        'Compare the two A00 configurations, A06’s protection failure and C01’s acceptance/guard paths. Each explanation links to the recorded model output and tool results available at that point.')+'</p><div id="case018-controls" hidden><label for="case018-episode">'+t('사건 선택', 'Choose episode')+'</label><select id="case018-episode"><option value="">'+t('전체 보기', 'Show all')+'</option>'+''.join('<option value="'+escape(e['episode_id'], quote=True)+'">'+escape(e['episode_id'])+'</option>' for e in data['selected_episodes'])+'</select><button type="button" id="case018-reset">'+t('전체 사건', 'All episodes')+'</button><span id="case018-count" role="status" aria-live="polite"></span></div><noscript><p>'+t('아래 모든 사건과 원문은 JavaScript 없이도 열 수 있습니다.', 'All recorded episodes and raw text can be opened below without JavaScript.')+'</p></noscript>'
    for e in data['selected_episodes']:
        eid = e['episode_id']; r = e['diagnostic']
        note = (t('복구 후 저장 수량과 마지막 보고를 비교하는 A00 경로입니다.', 'A00 compares restored inventory with the completion report.') if 'A00' in eid else
                t('잘못된 보상으로 총수량 보존을 깨뜨린 A06 경로입니다.', 'A06 breaks total-stock conservation through incorrect compensation.') if 'A06' in eid else
                t('부분 차감 복구 중 도착 수량까지 뺀 C01 경로입니다.', 'C01 subtracts destination inventory while undoing a source-only debit.'))
        explorer += '<details class="case018-episode" id="episode-'+escape(eid, quote=True)+'"><summary>'+escape(eid)+' · '+t('상태·후속 검사 ', 'State/follow-up ')+str(r['Q'])+' · '+t('검증·완료 보고 ', 'Verified report ')+str(r['W'])+'</summary><p>'+note+'</p><p>'+t('기술 대응: ', 'Metric keys: ')+escape(f"Q={r['Q']}, W={r['W']}, Fresh={r['Fresh']}, current read={r['current_quantity_read_observed']}")+'</p><h3>'+t('공개 복구 요청', 'Public recovery request')+'</h3><pre>'+dump(e['request'])+'</pre><details><summary>'+t('수량과 판정 — 설명용 파생 자료', 'Quantities and verdicts — derived explanation')+'</summary><pre>'+dump({k:e[k] for k in ['initial_quantities', 'terminal_quantities', 'endpoint']})+'</pre></details>'
        for turn in e['turns']:
            explorer += '<section class="case018-turn"><h4>'+t('생성 ', 'Generation ')+str(turn['generation'])+' · '+escape(turn['decision'])+'</h4><p>'+t('생성 전 실제 제공된 결과 ID: ', 'Result IDs available before generation: ')+escape(str(turn['available_result_event_ids']))+'</p><details><summary>'+t('모델 원문 열기', 'Open model raw text')+'</summary><pre class="case018-raw">'+escape(turn['raw_text'])+'</pre></details><details><summary>'+t('실제 도구·가드 결과', 'Actual tool/guard result')+'</summary><pre>'+dump({'guard':turn['guard'], 'event':turn['event']})+'</pre></details></section>'
        explorer += '<p>'+a('../sources/case018-'+eid+'.json.txt', t('원문 발췌·전체 포인터', 'Raw-text excerpt and full source pointer'))+'</p><p class="small">'+escape(r['source_path'])+' · SHA256 <code>'+escape(r['source_sha256'])+'</code></p></details>'
    out += section('episodes', explorer)
    out += section('limits', '<p>'+t(
        '이 결과는 세 유형의 합성 재고 환경에서 어떤 복구·보고 실패가 생기는지 보여줍니다. Q는 저장 상태와 정해진 후속 절차의 적합성, W는 검증을 관측하고 정확히 보고하는 workflow까지 확인하므로 서로 다른 결과를 드러냅니다. 두 후속 절차가 다루는 행동 경로와 평가 시점은 위 방법에 고정했습니다.',
        'These results show recovery and reporting failures in three synthetic inventory scenarios. Q tests the saved state and fixed follow-up programs; W also tests the workflow of observing validation and reporting correctly. The two measures therefore expose different failures. The follow-up programs cover the paths and evaluation points specified above.')+'</p><p>'+t(
        '수량 재조회는 모델이 선택한 행동이었습니다. 이번 자료의 연관 분석과, 앞으로 재조회를 요구했을 때의 효과를 비교하는 실험은 구분해서 읽어야 합니다.',
        'Rereading quantities was an action chosen by the agent. The association in these records and a future experiment that requires rereading answer different questions.')+'</p>')
    out += section('reproduce', '<p>'+t(
        'Python 3.12+ 표준 라이브러리로 공개 기록을 CPU에서 재생합니다. 존재하지 않는 새 output 경로를 지정하세요. 모델·tokenizer 다운로드나 Torch/vLLM 설치가 필요하지 않습니다.',
        'Replay the public records on CPU using Python 3.12+ and its standard library. Choose a new output path that does not exist. No model/tokenizer download or Torch/vLLM installation is required.')+'</p><pre>python3 -B cases/018-agent-state-recovery/publication/run_checks.py --output /tmp/case018-public-check-new --tests</pre><p>'+t(
        '공개 원자료를 복원해 120개 모델 episode의 상태 전이·관측·최종 판정을 다시 계산합니다. 별도 검증 대상인 P0 합성 trace 26개도 당시 동결 소스로 검사합니다. SSE/IDs·비용·짝 비교·사후 진단·그림 재생성을 포함하며, 원 tokenizer 재인코딩은 보존된 receipt를 사용합니다.',
        'The command restores public evidence and recalculates transitions, observations and verdicts for 120 model episodes. It separately checks 26 P0 synthetic traces with their frozen source. It also checks SSE/IDs, costs, paired comparisons, posthoc diagnostics and regenerated figures; original tokenizer re-encoding is documented by retained receipts.')+'</p><p>'+t(
        '로컬 공개 사본과 GitHub-hosted Ubuntu에서 CPU 기록 재생을 확인했습니다. 현재 main의 실행 상태와 receipt는 ',
        'CPU record replay has been checked in a local public-only copy and on GitHub-hosted Ubuntu. Current main runs and receipts are available in ') +
        a('https://github.com/Munsik-Kim/inference-lab/actions/workflows/case018-checks.yml?query=branch%3Amain', t('CPU 기록 재생 Actions', 'CPU record replay Actions'))+'</p><div class="actions">'+a('../'+EVIDENCE_ASSET, t('전체 원자료 archive', 'Full evidence archive'))+' · '+a('../sources/case018-mapping.json.txt', t('복원 mapping', 'Restoration mapping'))+' · '+a('../sources/case018-episode-ledger.csv.txt', '120-row CSV')+' · '+a('../sources/case018-reproduce-'+lang+'.md.txt', t('재현 범위와 원 identity', 'Reproduction scope and original identities'))+'</div>')
    out += section('references', '<p>'+t(
        '직접 구현한 부분은 재고·처리 원장 환경, 보상·처리 해제 API, 엄격한 parser, 완료 가드, 생성 전 관측 추적, 별도 상태·후속 절차/SSE 감사, 사후 수량 분석과 공개 도구입니다. Qwen 모델·vLLM·FlashInfer KV 커널·xgrammar는 upstream을 사용했습니다.',
        'Direct work comprises the inventory/operation-ledger environment, compensation and release APIs, strict parser, completion guard, pre-generation observation tracking, separate state/continuation/SSE audits, posthoc quantity analysis and public tools. The Qwen model, vLLM, FlashInfer KV kernels and xgrammar come from upstream.')+'</p><p>'+t(
        'ReTrace의 복구 뒤 행동 평가와 ACE의 개별·전체 제약 구분을 작은 도구 환경에 독립 응용했습니다. GUI/on-policy 원문과 이 합성 환경의 대응은 SOURCE_MAP에 기록했습니다. Codex의 구현·문서화 지원과 라이선스는 NOTICE에 정리했습니다.',
        'We independently adapt ReTrace’s post-recovery action evaluation and ACE’s component/joint-constraint view to a small tool environment. SOURCE_MAP explains how this differs from the original GUI/on-policy setting. NOTICE records Codex assistance and license provenance.')+'</p>'+a('../sources/case018-report-'+lang+'.md.txt', t('정식 보고서', 'Full report'))+' · '+a('../sources/case018-audit.py.txt', 'Independent audit')+' · '+a('../sources/case018-quantity.py.txt', 'Posthoc code')+' · '+a('../sources/case018-notice.md.txt', 'NOTICE'))
    return out+'</article></div>'
