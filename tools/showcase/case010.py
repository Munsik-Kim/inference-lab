"""Presentation of one Case010 project with two preserved research stages.

The display is generated from the unified scalar projection, which is recomputed
from the original snapshots before it is consumed. No model runs occur here.
"""
from html import escape
import importlib.util
import sys
from pathlib import Path

from common import json_text, read, require, sha

C10 = 'cases/010-ckda-finite-precision-memory-horizon'
FIGURE_SOURCE = C10 + '/demo/budget_horizon.png'
FIGURE_ASSET = 'assets/case010-budget-horizon.png'
SOURCE_COPIES = {
    'case010-report-en.md.txt': C10 + '/REPORT.md',
    'case010-report-ko.md.txt': C10 + '/REPORT.ko.md',
    'case010-reproduction-en.md.txt': C10 + '/REPRODUCTION.md',
    'case010-reproduction-ko.md.txt': C10 + '/REPRODUCTION.ko.md',
    'case010-notice.md.txt': C10 + '/NOTICE.md',
    'case010-online-v2.py.txt': C10 + '/versions/v2/source/online_v2.py',
    'case010-packed.py.txt': C10 + '/versions/v2/source/v1_reference/codec/packed.py',
    'case010-restart-tests.py.txt': C10 + '/versions/v2/tests/test_online_v2.py',
}


def load_case010(root: Path):
    case = root / C10
    if not (case / 'README.md').exists():
        return None
    source = case / 'scripts/derive_summary.py'
    require(source.is_file() and not source.is_symlink(), 'Missing Case010 scalar projection')
    spec = importlib.util.spec_from_file_location('showcase_case010_projection', source)
    module = importlib.util.module_from_spec(spec)
    identity_spec = importlib.util.spec_from_file_location('showcase_case010_identity', case/'scripts/verify_unified.py')
    identity_module = importlib.util.module_from_spec(identity_spec)
    identity_spec.loader.exec_module(identity_module)
    old_identity = sys.modules.get('verify_unified')
    sys.modules['verify_unified'] = identity_module
    try:
        spec.loader.exec_module(module)
    finally:
        if old_identity is None:
            del sys.modules['verify_unified']
        else:
            sys.modules['verify_unified'] = old_identity
    data = read(case / 'summary/project.json')
    require(data == module.derive(case), 'Case010 summary differs from recorded sources')
    paths = [C10 + '/summary/project.json', C10 + '/scripts/derive_summary.py',
             C10 + '/provenance/snapshot_manifest.json', FIGURE_SOURCE,
             *SOURCE_COPIES.values()]
    require(all((root/p).is_file() and not (root/p).is_symlink() for p in paths),
            'Missing/unsafe Case010 display source')
    return {'evidence_kind': 'UNIFIED_RECORDED_STAGES',
            'source_identity': 'Case010 v1/v2 archive identities and byte-preserved snapshots',
            'sources': {p: sha(root/p) for p in paths}, 'project': data}


def table(headers, rows, label):
    return ('<div class="scroll" tabindex="0" role="region" aria-label="' + escape(label, quote=True)
            + '"><table><thead><tr>'
            + ''.join('<th scope="col">' + escape(str(h)) + '</th>' for h in headers)
            + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join('<td>' + escape(str(v)) + '</td>' for v in row) + '</tr>'
                      for row in rows) + '</tbody></table></div>')


def a(url, label, download=None):
    attr = ' download="'+escape(download, quote=True)+'"' if download else ''
    return '<a href="' + escape(url, quote=True) + '"'+attr+'>' + escape(label) + '</a>'


def home_case010(data, lang):
    ko = lang == 'ko'
    h = data['project']['headline']
    title = '저비트 상태 저장과 실패를 보존하는 재시작' if ko else 'Packed recurrent state and failure-preserving restart'
    body = ('저비트 state codec와 새 프로세스 재시작을 구현했습니다. 같은 저장 한도에서 판독 수명과 계산 비용을 비교하며, 세 학습 CKDA의 새 입력 결과를 제공합니다.' if ko else
            'Built packed-state codecs and fresh-process restart. Compare readout horizon and computation cost under storage caps, with fresh-input results from three trained CKDA checkpoints.')
    scope = ('stream별 직렬화 상태 · ' if ko else 'Serialized state per stream · ')
    scope += f"Native {h['native_stream_bytes']:,} → INT8 {h['int8_stream_bytes']:,} bytes ({h['reduction_pct']:.1f}% " + ('감소)' if ko else 'smaller)')
    return ('<section class="panel" id="memory-study"><p class="eyebrow">CASE 010 · RECURRENT MEMORY</p><h2>'
            + a('case010.html', title) + '</h2><p>' + body + '</p><p class="project-scope">'
            + scope + '</p><div class="actions">'
            + a('case010.html#implementation', '구현 코드' if ko else 'Implementation')
            + a('case010.html#reproduce', 'CPU 재계산' if ko else 'CPU recalculation')
            + a('case010.html#results', '두 단계의 결과' if ko else 'Results from both stages')
            + '</div></section>')


def render_case010(data, lang):
    require(lang in ('en', 'ko'), 'Unsupported Case010 language')
    ko = lang == 'ko'
    d = data['project']
    tr = lambda en, kr: kr if ko else en
    title = tr('Finite-Precision Recurrent Memory: Storage, Restart, and Readout Horizon',
               '저정밀 recurrent state의 저장·재시작과 기억 수명')
    h = '<div class="study-intro"><p class="eyebrow">CASE 010</p><h1>' + title + '</h1><p class="lede">'
    h += tr('Built actual low-bit recurrent-state storage and fresh-process restart that preserves failure status and random-number state. The study compares correct-readout lengths under storage caps, alongside arithmetic precision, state evolution and CPU cost.',
            'Recurrent state를 실제 저비트로 저장하고, 실패 상태와 난수를 보존해 새 프로세스에서 이어 실행하는 도구를 구현했습니다. 같은 저장 한도에서 올바른 상태를 유지하는 길이를 비교하고, 정밀도·상태 변화·CPU 계산 비용을 함께 분석합니다.')
    h += '</p><p class="project-tech">Python · NumPy · PyTorch · packed bytes · CKDA</p></div>'
    sections = [('objective', tr('Goal and task', '목표와 데이터')), ('theory', tr('Method and definitions', '방법과 정의')),
                ('implementation', tr('Implementation', '구현 기능')), ('stage-a', tr('Stage A · v1', '단계 A · v1')),
                ('stage-b', tr('Stage B · v2', '단계 B · v2')), ('results', tr('Results', '실험 결과')),
                ('diagnostics', tr('Diagnostics and cost', '진단과 비용')), ('reproduce', tr('CPU checks and reports', 'CPU 검사와 보고서'))]
    h += '<nav class="actions" aria-label="' + tr('Contents', '목차') + '">' + ''.join(a('#'+i, text) for i, text in sections) + '</nav>'
    def section(i, body):
        return '<section class="study-section" id="'+i+'"><h2>'+dict(sections)[i]+'</h2>'+body+'</section>'
    h += section('objective', '<p>'+tr('A recurrent state is the model’s memory carried into the next token. S3 is the six permutations of three objects: each input rearranges the current order, and the model reads out the resulting permutation after every step.', 'Recurrent state는 모델이 다음 token으로 전달하는 기억 상태입니다. S3는 세 물체의 순열 여섯 개입니다. 입력마다 현재 순서를 바꾸고, 매 단계에서 그 누적 순열을 모델이 판독합니다.')+'</p><p>'+tr('The question is whether residual correction preserves correct symbolic-state readout longer under the same total serialized-storage cap.', '검증 질문은 같은 총 직렬화 저장 한도에서 잔차 보정이 symbolic state를 더 오래 올바르게 판독하게 하는가입니다.')+'</p><p>'+tr('The task is the left cumulative product of arbitrary S3 group elements. An integer multiplication table supplies the reference labels; candidate state updates receive only the current input and their stored state. Every group-token prefix is scored, including correct answers that return after an earlier mistake.',
        '과제는 S3군의 임의의 원소를 왼쪽 누적곱으로 추적하는 것입니다. 별도의 정수 곱셈표가 정답을 계산하며, 후보의 상태 갱신은 현재 입력과 저장한 상태만 받습니다. 모든 group-token prefix를 채점하고, 앞서 틀린 뒤 다시 맞힌 token도 기록합니다.')+'</p><p>'+tr('Both stages reuse the same three small trained checkpoints. Stage A and Stage B have different test inputs, sample counts and confidence families; they are separate evaluations of one project.',
        '두 단계는 같은 작은 학습 checkpoint 세 개를 사용합니다. 평가 입력·표본 수·신뢰구간의 동시 비교 범위가 달라 하나의 프로젝트 안에서 별도 실험으로 표시합니다.')+'</p>')
    h += '<p class="project-scope">'+escape(d['model']['name'])+'</p>'
    h += section('theory', '<p>'+tr('At each write-back, the transition acts on the decoded state plus its residual. The next state is packed; the new local write-back residual is either retained in full or projected onto a CAL-frozen basis. The previous residual is transported through the transition, rather than added in its old coordinates.',
        '매번 저장할 때 복원한 상태와 잔차를 합친 값에 전이를 적용합니다. 새 상태를 bit packing하고, 그 저장 오차를 전체 잔차 또는 CAL에서 고정한 기저의 계수로 저장합니다. 이전 잔차는 과거 좌표 그대로 더하지 않고 전이를 통과시킵니다.')+'</p>'+table(
        [tr('Quantity', '지표'), tr('Definition', '정의')], [
            ['τ',tr('First scored group token with a wrong or invalid prediction; no failure is right-censored at Tmax.', '오답 또는 invalid prediction이 처음 나온 채점 group token. 끝까지 미실패이면 Tmax에서 우측 검열.')],
            ['RMST0',tr('Mean consecutive correct tokens before the first failure: mean(min(τ−1,Tmax)).', '첫 실패 이전 평균 연속 정답 token 수: mean(min(τ−1,Tmax)).')],
            ['T0.05',tr('Largest observed length with empirical first-error risk ≤ 5%.', '경험적 최초 실패 비율이 5% 이하인 최대 관측 길이.')],
            [tr('Supported grid horizon', '신뢰지원 grid 하한'),tr('Largest predeclared grid length with a simultaneous upper-binomial risk bound ≤ 5%; not a between-method significance test.', '사전 grid에서 동시 이항 위험 상한이 5% 이하인 최대 길이. 방법 간 차이의 검정은 아님.')],
            ['Total(N)',tr('Shared bytes + N × persistent bytes per stream; N is storage capacity, not the evaluation sample count.', '공유 bytes + N × stream별 지속 저장 bytes. N은 저장 용량이며 평가 표본 수가 아님.')]], tr('Metric definitions','지표 정의')))
    h += section('implementation', '<p>'+tr('The current runtime is v2. Its envelope serializes cursor, terminal code and first terminal-write index. A numeric state failure preserves the last committed body and RNG; later inputs advance the cursor and return an invalid prediction. A finite wrong label does not terminate the stream. A nonfinite readout with finite state also continues.',
        '현재 실행 경로는 v2입니다. cursor·terminal code·최초 terminal write index를 직렬화합니다. 상태의 수치 실패가 발생하면 마지막 정상 body와 RNG를 유지하고, 이후 입력은 cursor를 진행시키며 invalid prediction을 반환합니다. 유한한 오답은 stream을 종료하지 않습니다. 상태가 유한한 비유한 readout도 다음 입력을 계속 처리합니다.')+'</p><div class="actions">'+a('../sources/case010-online-v2.py.txt',tr('Failure-aware runtime · Python','실패 보존 runtime · Python'))+a('../sources/case010-packed.py.txt',tr('Packing source · Python','Packing 소스 · Python'))+a('../sources/case010-restart-tests.py.txt',tr('Restart contracts · Python','재시작 계약 검사 · Python'))+'</div>')
    h += '<div class="study-table">'+table([tr('Stage','단계'),tr('Inputs / checkpoint','checkpoint별 입력'),tr('Length grid','길이 grid'),'Family'], [[tr('A / v1','A / v1'),d['stages']['v1']['N'],len(d['stages']['v1']['grid']),str(d['stages']['v1']['families'])],[tr('B / v2','B / v2'),d['stages']['v2']['N'],len(d['stages']['v2']['grid']),d['stages']['v2']['family']]],tr('Separate evaluation protocols','서로 다른 평가 protocol'))+'</div>'
    h += section('stage-a', '<p>'+tr('Stage A explored packed state, transported residuals and storage budgets, then strengthened the same-budget baselines. The original 26-arm analysis used a 546-comparison family; the 30-arm supplement used 630. Its 0/54 supported-horizon comparisons are specific to that menu, not a proof that all codecs fail.',
        '단계 A에서는 packed state·전달 잔차·저장 예산을 탐색한 뒤, 같은 예산의 기준선을 강화했습니다. 원래 26-arm 분석은 546-family, 30-arm 보충은 630-family입니다. 보충의 신뢰지원 하한 비교 0/54는 그 비교 목록의 결과이며 모든 codec의 불가능성을 뜻하지 않습니다.')+'</p><p>'+tr('Stage A used 512 test sequences per checkpoint and seven length-grid points. Its original implementation, measurements and reports remain byte-preserved in versions/v1.',
        'checkpoint별 TEST 512개와 길이 grid 7개를 사용했습니다. 원래 구현·측정·보고서는 versions/v1에 같은 바이트로 보존합니다.')+'</p>')
    h += section('stage-b', '<p>'+tr('Stage B addressed a restart boundary that lost numeric-failure state, then evaluated five fixed settings on 1,024 fresh sequences of 2,048 group tokens per checkpoint. It retained the original checkpoints and CAL basis, with one stronger mixed 5/6-bit comparator selected from CAL importance and actual byte cost.',
        '단계 B에서는 수치 실패 표시를 잃던 재시작 경계를 고치고, checkpoint별 fresh 입력 1,024개 × group token 2,048에서 다섯 고정 설정을 비교했습니다. 원래 checkpoint와 CAL 기저를 유지하며, CAL 중요도와 실제 저장량으로 고른 혼합 5/6-bit 기준선을 추가했습니다.')+'</p><p>'+tr('The simultaneous family is 5 arms × 3 checkpoints × 13 horizons = 195. These inputs are not pooled with v1. Zero terminal events means the numeric runtime stayed active; it does not mean every token was answered correctly.',
        '동시 비교 범위는 5 arms × 3 checkpoints × 13 horizons = 195입니다. v1 입력과 합산하지 않습니다. terminal 0은 수치 실행 상태가 유지됐다는 뜻이며 모든 token의 정답을 뜻하지 않습니다.')+'</p>')
    promotions = d['methods']['mixed_promoted_channels_by_seed']
    promoted = ', '.join(str(r['count'])+'/'+str(r['total_channels']) for r in promotions)
    methods = [
        ['NATIVE_FP32', tr('FP32 recurrent state with the same v2 failure envelope.', '같은 v2 실패 envelope를 쓰는 FP32 반복 상태.')],
        ['UNIFORM_8 / UNIFORM_5', tr('Pack every state element at 8 / 5 bits, with stored scales.', '모든 state 원소를 각각 8 / 5bit로 packing하며 scale도 저장.')],
        ['LOWRANK_4_8_R2', tr('4-bit state plus 8-bit rank-2 residual coefficients per head, using the v1 CAL-frozen basis.', '4bit state와 head별 rank2 잔차의 8bit 계수. v1 CAL에서 고정한 기저 사용.')],
        ['MIXED_5_6_BUDGET', tr('Start at 5 bits and promote CAL-ranked channels to 6 bits while real serialized N128 bytes stay within the rank2 cap.', '모든 채널을 5bit로 시작해 CAL 순서대로 6bit 승격. 실제 직렬화 N128 bytes가 rank2 한도 안에 남는 구성.')+' '+tr('Promoted channels, seed0/1/2: ', '6bit 승격 채널, seed0/1/2: ')+promoted],
    ]
    h += '<div class="study-table">'+table([tr('Setting', '설정'), tr('Stored representation', '저장 표현')], methods, tr('Five codec settings', '다섯 codec 설정'))+'</div>'
    # Tables are filled entirely from the checked unified scalar projection.
    h += section('results', result_tables(d, lang))
    h += section('diagnostics', diagnostic_tables(d, lang))
    command = ('CASE=cases/010-ckda-finite-precision-memory-horizon\n'
               'python3 -B "$CASE/scripts/verify_unified.py"\n')
    h += section('reproduce', '<p>'+tr('Inspect the byte-preserved sources, verify the integrated projection, or restore the historical workspace for the original scalar auditors. These CPU checks do not rerun trained-model inference.',
        '원형 소스를 읽고 통합 표시값을 검사하거나, 과거 경로를 외부 공간에 복원해 원래 scalar auditor를 실행할 수 있습니다. 이 CPU 검사는 학습 모델 추론을 다시 실행하지 않습니다.')+'</p><pre>'+escape(command)+'</pre><div class="actions">'+a('../sources/case010-reproduction-'+lang+'.md.txt', tr('CPU commands and research requirements','CPU 명령과 연구 재실행 조건'))+a('../sources/case010-report-'+lang+'.md.txt',tr('Read the full report','통합 보고서 읽기'))+a('../data/case010.json',tr('Download scalar data and source hashes','스칼라와 출처 hash 다운로드'),'case010.json')+'</div><p>'+tr('The website serves recorded results. Original research reruns require the pinned upstream environment and private checkpoints; this integration added no model inference, training or GPU run.',
        '웹 화면은 저장된 결과를 제공합니다. 원래 연구 재실행에는 고정 upstream 환경과 private checkpoint가 필요하며, 이번 통합은 모델 추론·학습·GPU 실행을 추가하지 않았습니다.')+'</p>')
    return h


def result_tables(d, lang):
    """Source-backed comparison tables. Schema-specific formatting is kept here."""
    ko = lang == 'ko'
    tr = lambda en, kr: kr if ko else en
    h = d['headline']
    out = '<h3>'+tr('Native and INT8: serialized state','Native와 INT8: 직렬화 상태')+'</h3>'
    out += table([tr('Setting','설정'), tr('Per-stream bytes','stream별 bytes')],
                 [['NATIVE_FP32', f"{h['native_stream_bytes']:,}"], ['UNIFORM_8', f"{h['int8_stream_bytes']:,}"]], tr('Serialized state bytes','직렬화 상태 bytes'))
    out += '<p>'+tr(f"INT8 uses {h['reduction_pct']:.1f}% fewer serialized state bytes. This is not a measurement of whole-process RAM or device VRAM.",
                    f"INT8의 직렬화 state 크기는 {h['reduction_pct']:.1f}% 작습니다. 프로세스 전체 RAM이나 장치 VRAM 측정값은 아닙니다.")+'</p>'
    out += fresh_tables(d, lang)
    out += '<figure><img class="serving-curve" src="../'+FIGURE_ASSET+'" alt="'+tr('V2 storage budget and readout horizon by checkpoint; empirical lengths and confidence-supported bounds shown separately', 'checkpoint별 v2 저장 예산과 판독 길이: 경험적 길이와 신뢰지원 하한을 구분')+'"><figcaption>'+tr('Stage B / v2 · English figure labels · serialized storage, not allocator memory. V1 and v2 are not a before/after performance curve.', '단계 B / v2 · 그림 라벨은 영어 · allocator 메모리가 아닌 직렬화 저장량. v1과 v2를 before/after 성능곡선으로 합치지 않습니다.')+'</figcaption></figure>'
    return out


def fresh_tables(d, lang):
    ko = lang == 'ko'
    tr = lambda en, kr: kr if ko else en
    rows = {(r['model_seed'], r['arm']): r for r in d['fresh']['rows']}
    budgets = {(r['model_seed'], r['arm']): r['v2'] for r in d['storage']['rows']}
    native_rows = []
    for pair in d['headline']['int8_minus_native_RMST0_by_seed']:
        seed = pair['model_seed']
        native_rows.append([seed, f"{rows[seed,'NATIVE_FP32']['RMST0']:.2f}",
                            f"{rows[seed,'UNIFORM_8']['RMST0']:.2f}", f"{pair['delta_tokens']:+.2f}"])
    h = table(['Checkpoint seed', 'Native RMST0', 'INT8 RMST0', 'INT8 − Native'], native_rows,
              tr('Native and INT8 mean consecutive correct tokens', 'Native와 INT8 평균 연속 정답 token'))
    h += '<p>'+tr('RMST0 and differences are in group tokens. All three point differences are below one token in absolute value; this is not an equivalence test.',
                  'RMST0와 차이의 단위는 group token입니다. 세 점추정의 절댓값이 모두 1token 미만이며, 품질 동등성 검정은 아닙니다.')+'</p>'
    h += '<h3>'+tr('Rank2 versus the CAL-selected mixed 5/6-bit baseline', 'Rank2와 CAL로 선택한 혼합 5/6-bit 기준선')+'</h3>'
    budget_rows = []
    for arm in ('UNIFORM_5','LOWRANK_4_8_R2','MIXED_5_6_BUDGET'):
        b = budgets[0,arm]
        budget_rows.append([arm, f"{b['per_stream_persistent_bytes']:,}", f"{b['shared_bytes']:,}", f"{b['total_bytes']['128']:,}"])
    h += table([tr('Setting', '설정'), tr('Stream bytes', 'stream bytes'), tr('Shared bytes', '공유 bytes'), 'Total bytes · N128'],
               budget_rows, tr('Actual storage budget', '실제 저장 예산'))
    h += '<p>'+tr('The N128 cap counts codec configuration, basis or mixed map, scales, cursor, failure header and the shared token table. Native and INT8 are capacity/fidelity references above this cap at N128. The evaluation still uses 1,024 independent input sequences per checkpoint.',
                  'N128 한도는 codec 설정·기저 또는 혼합 map·scale·cursor·실패 header·공유 token table을 포함합니다. Native와 INT8은 N128에서 이 한도를 넘는 용량·충실도 참고군입니다. 실제 평가는 checkpoint마다 독립 입력 1,024개를 사용합니다.')+'</p>'
    pairs = [r for r in d['fresh']['paired_comparisons'] if r['candidate']=='LOWRANK_4_8_R2' and r['baseline']=='MIXED_5_6_BUDGET']
    h += table(['Seed', tr('Rank2 − Mixed RMST0', 'Rank2 − Mixed 평균 길이'), tr('Pointwise 95% interval', '개별 95% 구간')],
               [[r['model_seed'], f"{r['RMST0_delta']:+.2f}", f"[{r['pointwise_95_CI'][0]:+.2f}, {r['pointwise_95_CI'][1]:+.2f}]"] for r in pairs],
               tr('Paired mean differences in group tokens', 'group token 단위 paired 평균 차이'))
    h += '<p>'+tr('Rank2 did not establish a consistent same-budget advantage across the three checkpoints. Paired RMST intervals are pointwise; differences in supported-grid bounds are not a test of a between-method horizon difference.',
                  'Rank2는 세 checkpoint에서 일관된 동일 예산 우위를 확보하지 못했습니다. Paired RMST 구간은 각 비교의 구간이며, 신뢰지원 grid 하한의 차이를 방법 간 horizon 차이 검정으로 읽지 않습니다.')+'</p>'
    for seed in sorted({r['model_seed'] for r in d['fresh']['rows']}):
        selected = [r for r in d['fresh']['rows'] if r['model_seed']==seed]
        h += '<h3>Checkpoint seed '+str(seed)+'</h3>'
        h += table([tr('Setting', '설정'), 'RMST0', tr('Empirical T0.05', '경험적 T0.05'), tr('Supported grid', '신뢰지원 grid'), 'Terminal / N'],
                   [[r['arm'], f"{r['RMST0']:.2f}", r['empirical_T05_all_tokens'], r['supported_T05_grid'], f"{r['terminal_count']}/{r['N']}"] for r in selected],
                   tr('Five fixed settings, seed ', '고정 설정 다섯 개, seed ')+str(seed))
    return h


def diagnostic_tables(d, lang):
    ko = lang == 'ko'
    tr = lambda en, kr: kr if ko else en
    counts = d['restart']['counts']
    h = '<h3>'+tr('Restart boundary: preserved historical checks', '재시작 경계: 보존한 과거 검사')+'</h3>'
    h += table([tr('Check', '검사'), tr('Recorded outcome', '원래 기록')], [
        [tr('v1 uninterrupted versus split prediction mismatch', 'v1 전체 실행과 분할 재시작 예측 불일치'), counts['v1_uninterrupted_split_prediction_mismatches']],
        [tr('v2 full/split/fresh-process prediction matches', 'v2 전체·분할·새 프로세스 예측 일치'), f"{counts['v2_full_split_fresh_prediction_matches']}/{counts['historical_cells']}"],
        [tr('v2 final payload matches', 'v2 최종 payload 일치'), f"{counts['v2_full_split_fresh_final_payload_hash_matches']}/{counts['historical_cells']}"],
        [tr('Selected actual failure boundaries', '고른 실제 실패 경계'), counts['technical_failure_boundary_checks']]], tr('Historical restart diagnostics', '과거 재시작 진단'))
    h += '<p>'+tr('These are retained v2 diagnostics, not new trained-model executions in this integration. Selected failure replays establish the execution contract, not a population failure rate.',
                  '이 표는 보존된 v2 진단 기록이며 이번 통합에서 학습 모델을 새로 실행한 결과가 아닙니다. 고른 실패 재실행은 실행 계약을 확인하며 모집단 실패율을 추정하지 않습니다.')+'</p>'
    h += '<h3>'+tr('Fixed-coefficient arithmetic precision', '고정 계수 산술 정밀도')+'</h3><p>'+tr('D00 uses FP32 recurrence/readout. D10 raises recurrence precision; D01 raises readout precision; D11 raises both. Every mode starts from the same already-represented FP32 coefficients and weights. Gates and projections were not recomputed from scratch in FP64.',
                  'D00은 recurrence와 readout 모두 FP32, D10은 recurrence, D01은 readout, D11은 둘 다 FP64로 승격합니다. 이미 FP32로 표현한 같은 계수와 가중치에서 시작하며, gate와 projection을 FP64로 처음부터 재계산하지 않았습니다.')+'</p>'
    precision_rows = []
    for seed in sorted({r['model_seed'] for r in d['precision']['endpoints']}):
        base = next(r for r in d['precision']['endpoints'] if r['model_seed']==seed and r['mode']=='D00')
        pairs = [r for r in d['precision']['pairs'] if r['model_seed']==seed]
        precision_rows.append([seed, f"{float(base['RMST0']):.3f}", base['empirical_T05_all_tokens'],
                               tr('Identical predictions / τ', '예측 / τ 동일') if all(int(r['scored_prediction_disagreements'])==0 and r['tau_equal_sequences']==r['N'] for r in pairs) else tr('See differing pairs', '서로 다른 pair 확인')])
    h += table(['Seed', 'D00 RMST0', 'D00 T0.05', 'D00 / D10 / D01 / D11'], precision_rows,
               tr('Precision diagnostic; 32 sequences × 2048 tokens', '정밀도 진단; 32개 입력 × 2048tokens'))
    h += '<p>'+tr('All four modes retained identical predictions and first-failure positions on this 32-sequence diagnostic cohort. This observation does not prove a memory-capacity limit.',
                  '32개 별도 진단 입력에서 네 mode의 예측과 최초 실패 위치가 같았습니다. 이 관측만으로 기억 용량의 한계를 증명하지 않습니다.')+'</p>'
    h += '<h3>'+tr('Late state growth', '후기 상태 크기 변화')+'</h3>'
    h += table(['Seed', tr('Rank2 maximum state norm', 'Rank2 최대 state norm'), tr('Write index', 'write 위치'), tr('First error τ', '최초 오답 τ')],
               [[r['model_seed'], f"{float(r['maximum']):.4e}",r['write_position'], r['first_failure_tau']] for r in d['trace']['rank2_state_norm_extrema']],
               tr('Rank2 diagnostic extrema', 'Rank2 진단 극값'))
    h += '<p>'+tr('These maxima occurred after earlier readout errors, on a separate fixed diagnostic cohort. The timing relationship alone does not identify a cause.',
                  '이 극값은 별도로 고정한 진단 입력에서 앞선 판독 오류 이후에 나타났습니다. 시점의 선후만으로 원인을 규명하지 않습니다.')+'</p>'
    h += '<h3>'+tr('Complete-call CPU cost', 'Complete-call CPU 비용')+'</h3>'
    arms = d['stages']['v2']['arms']
    h += table(['Seed', 'Native','INT8','INT5','Rank2','Mixed5/6'],
               [[seed]+[f"{next(r for r in d['fresh']['rows'] if r['model_seed']==seed and r['arm']==arm)['cpu_ms_per_group_token']:.4f}" for arm in arms] for seed in (0,1,2)],
               tr('CPU milliseconds per group token', 'group token당 CPU 밀리초'))
    h += '<p>'+tr('ms/group-token; CPU two threads, 16 sequences × 128 group tokens, median of three fixed repeats. The complete call includes packing, transition, projection, status handling and original readout. Loading, file I/O and evaluator diagnostics are outside the timer. This is prototype throughput cost, not single-request latency or GPU speed.',
                  '단위 ms/group-token. CPU 2 threads, 16개 입력 × group token 128개, 고정 반복 3회의 중앙값입니다. packing·전이·projection·상태 처리·원래 readout을 포함하고, 로딩·파일 I/O·평가 진단은 시간 밖입니다. prototype 처리 비용이며 단일 요청 latency나 GPU 속도가 아닙니다.')+'</p>'
    return h
