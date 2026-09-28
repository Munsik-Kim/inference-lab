"""Case011 presentation from frozen records and separately identified post-hoc tables.

No Torch, model loading, fitting, or inference is used by this display projection.
"""
from html import escape
from pathlib import Path
from common import read, require, sha, safe_name
from case010 import table, a

C11 = 'cases/011-frozen-state-readout-adaptation'
SECTIONS = ('overview', 'implementation', 'results', 'analysis', 'reproduce', 'report')
REPORT_SECTIONS = ('background', 'questions', 'theory', 'methods', 'experiments', 'results', 'analysis', 'conclusion', 'references')
SOURCE_COPIES = {
    'case011-report-en.md.txt': C11+'/REPORT.md',
    'case011-report-ko.md.txt': C11+'/REPORT.ko.md',
    'case011-reproduction.md.txt': C11+'/REPRODUCTION.md',
    'case011-notice.md.txt': C11+'/NOTICE.md',
    'case011-adapter.py.txt': C11+'/source/adapter.py',
    'case011-head-patch.py.txt': C11+'/source/head_patch.py',
    'case011-audit.py.txt': C11+'/analysis/audit.py',
    'case011-adapter-tests.py.txt': C11+'/tests/test_adapter.py',
}
FIGURE_SOURCES = {'assets/case011-primary.svg': C11+'/publication/posthoc/data/primary_rmst0.svg',
                  'assets/case011-ce.svg': C11+'/publication/posthoc/data/ce_by_band.svg'}
FIGURE_ASSETS = tuple(FIGURE_SOURCES)


def load_case011(root: Path):
    case = root/C11
    if not (case/'README.md').exists():
        return None
    identity = read(case/'publication/original_identity.json')
    require(identity['case'] == '011-frozen-state-readout-adaptation', 'Wrong Case011 identity')
    originals = identity['research_files']
    names = ['results/derived/aggregate.json', 'configs/protocol.json', 'configs/selected_heads.json',
             'results/head_diagnostics.json', 'results/byte_ledger.json', 'results/timing/timing.json',
             'inputs/manifest.json', 'provenance/source_manifest.json']
    for relative in names:
        require(relative in originals and sha(case/relative) == originals[relative]['sha256'],
                'Changed frozen Case011 display source: '+relative)
    d = read(case/names[0]); protocol = read(case/'configs/protocol.json')
    require(d['logical_arms'] == 18 and d['recurrent_rollout_cells'] == 6, 'Wrong Case011 evaluation units')
    require(protocol['cohorts']['fresh'][1:] == [1024, 2048], 'Wrong Case011 input denominator')
    require(len(d['arms']) == 18 and len(d['primary']) == 3, 'Missing Case011 results')
    for key, raw in d['source_cells'].items():
        for suffix, field in [('predictions.npz', 'predictions_sha256'), ('summary.json', 'summary_sha256')]:
            path = 'results/fresh/'+key+'/'+suffix
            require(sha(case/path) == raw[field] == originals[path]['sha256'], 'Case011 recorded source mismatch: '+path)
            names.append(path)
    head = read(case/'results/head_diagnostics.json')
    require(all(row['final_head_parameters'] == 1158 and row['additional_recurrent_state_bytes'] == 0
                for row in head['rows']), 'Unsupported Case011 implementation headline')
    byte_ledger = read(case/'results/byte_ledger.json')
    # Avoid shipping full step arrays or duplicated model traces as site assets.
    result = {'schema': 'case011-showcase-v1', 'evidence_kind': 'RECORDED_FROZEN_STATE_EXPERIMENT',
        'source_identity': {'case': identity['case'], 'original_archive_sha256': identity['archive_sha256'],
                            'identity_basis': 'Case011 file hashes; remote commit/deployment recorded separately'},
        'units': {'independent_base_sequences': 1024, 'max_group_tokens': 2048, 'checkpoints': 3,
                  'recurrent_rollouts': 6, 'logical_readout_conditions': 18, 'fit_max_group_token': 256,
                  'rmst0': 'mean correct group tokens before first error; higher is better',
                  'CE': 'mean gold-label cross entropy in nats; lower is better',
                  'storage': 'serialized persistent recurrent-state bytes, not whole RAM/VRAM',
                  'primary_interval': '5000 paired base-sequence percentile bootstraps; 98.333...% each, approximate family95 intent'},
        'primary': d['primary'], 'secondary': d['secondary'], 'arms': d['arms'], 'scores': d['scores'],
        'implementation': {'parameter_names': protocol['parameter_names'], 'parameter_shapes': protocol['parameter_shapes'],
                           'final_layer_values': protocol['trainable_parameters'], 'extra_recurrent_state_bytes': 0},
        'storage_bytes': {row['storage']: row['per_stream_bytes'] for row in byte_ledger['cells']},
        'timing': read(case/'results/timing/timing.json'),
        'solver': protocol['solver'], 'selected_heads': read(case/'configs/selected_heads.json')['heads']}
    posthoc_path = case/'publication/posthoc/data/summary.json'
    require(posthoc_path.is_file(), 'Case011 reviewed post-hoc analysis is required')
    result['posthoc'] = read(posthoc_path)
    names.extend(['publication/posthoc/data/summary.json', 'publication/posthoc/protocol.json', 'publication/original_identity.json',
                  'publication/posthoc/data/output_manifest.json', *[p[len(C11)+1:] for p in FIGURE_SOURCES.values()]])
    for entry in result['posthoc']['sources']:
        name = safe_name(entry['path'])
        require((case/name).is_file() and not (case/name).is_symlink() and entry['sha256'] == sha(case/name),
                'Case011 post-hoc source mismatch: '+name)
        if name in originals:
            require(entry['sha256'] == originals[name]['sha256'], 'Changed original Case011 post-hoc source: '+name)
        else:
            require(name.startswith('publication/'), 'Unidentified Case011 post-hoc source: '+name)
        names.append(name)
    for path in SOURCE_COPIES.values():
        require((root/path).is_file() and not (root/path).is_symlink(), 'Missing Case011 source copy')
    result['sources'] = {C11+'/'+p: sha(case/p) for p in sorted(set(names))}
    result['sources'].update({path: sha(root/path) for path in SOURCE_COPIES.values()})
    return result


def load_items(root: Path):
    path = root/C11/'publication/posthoc/data/items.json'
    require(path.is_file(), 'Case011 item-level projection is required')
    return read(path)


def home_case011(data, lang):
    ko = lang == 'ko'
    labels = ('마지막 층 1,158개 값', '추가 recurrent-state 저장 0 B', '새 프로세스 재적용') if ko else (
        '1,158 final-layer values', '0 extra recurrent-state bytes', 'Fresh-process patch reload')
    title = 'Case 011 — 같은 상태, 다른 판독기' if ko else 'Case 011 — Same State, Different Readout'
    body = ('모델이 저장한 상태는 같은 방식으로 계속 갱신하고, 답을 읽는 마지막 층만 교체하는 도구를 만들었습니다. 보정한 두 tensor를 patch로 저장하고 새 프로세스에서 다시 적용해 결과를 비교합니다.' if ko else
            "Built a tool that keeps the model’s evolving state unchanged across readout comparisons and replaces only the final layer used to read an answer. The adapted two-tensor patch can be saved and applied in a fresh process.")
    finding = ('정답 확률 점수는 개선됐지만, 처음부터 연속해서 맞히는 평균 길이는 짧아졌습니다. 같은 상태에서도 정답에 높은 점수를 주는 것과 첫 오류를 늦추는 것은 달랐습니다.' if ko else
               'Gold-label scores improved, but the average uninterrupted correct prefix became shorter. On the same state, better probability scores did not translate into a later first error.')
    scope = ('CKDA 기호 상태추적 · 기존 checkpoint 3개 · 마지막 선형층만 지도 보정' if ko else
             'CKDA symbolic state tracking · Three existing checkpoints · Final-layer supervised adaptation')
    return ('<section class="panel readout-home" id="readout-study"><p class="eyebrow">CASE 011 · READOUT ADAPTATION</p><h2>'+a('case011.html#overview', title)+'</h2><p>'+body+'</p><ul class="readout-features">'
            + ''.join('<li>'+escape(s)+'</li>' for s in labels)+'</ul><p class="readout-finding">'+finding+'</p><p class="project-scope">'+scope+'</p><div class="actions">'
            + a('case011.html#overview', '쉽게 살펴보기' if ko else 'Overview')
            + a('case011.html#implementation', '구현 코드' if ko else 'Implementation')
            + a('case011.html#results', '상세 결과' if ko else 'Detailed results')+'</div></section>')


def format_num(value, digits=2):
    return '—' if value is None else f'{value:,.{digits}f}'


def render_case011(data, lang):
    require(lang in ('en', 'ko'), 'Unsupported Case011 language')
    ko = lang == 'ko'
    tr = lambda en, kr: kr if ko else en
    names = [tr('Overview','한눈에 보기'), tr('What was built','무엇을 구현했나요?'),
             tr('Experiment and results','어떻게 실험했고 무엇이 달라졌나요?'), tr('Interpretation','결과를 어떻게 이해해야 하나요?'),
             tr('Run and inspect','직접 확인하기: 실행·코드'),tr('Full report and references','자세한 보고서·출처')]
    toc = '<nav class="readout-toc" id="contents" aria-label="'+tr('Contents','목차')+'"><ol>'+''.join('<li>'+a('#'+key,name)+'</li>' for key,name in zip(SECTIONS,names))+'</ol></nav>'
    def section(key, text):
        return '<section class="study-section" id="'+key+'"><h2>'+names[SECTIONS.index(key)]+'</h2>'+text+'<p class="back-toc">'+a('#contents',tr('Back to contents','목차로 돌아가기'))+'</p></section>'
    def p(text): return '<p>'+text+'</p>'
    def details(label, text, key=None):
        return '<details'+(' id="'+key+'"' if key else '')+'><summary>'+label+'</summary>'+text+'</details>'
    def actions(pairs): return '<div class="actions">'+''.join(a(url,label) for url,label in pairs)+'</div>'
    def tbl(headers, rows, label): return '<div class="study-table">'+table(headers,rows,label)+'</div>'
    arm = {(r['checkpoint_seed'],r['storage'],r['head']):r for r in data['arms']}
    title = tr('Case 011 — Frozen-State Readout Adaptation and Length Extrapolation','Case 011 — 상태를 고정한 판독층 보정과 길이 외삽')
    h = '<div class="study-intro"><p class="eyebrow">CASE 011 · FROZEN-STATE READOUT</p><h1>'+title+'</h1><p class="lede">'+tr(
        'Kept the way the model updates its memory, and refitted only the final layer that reads an answer. The tools compare probability scores and the first error on the same state, and save the adaptation for reuse in a fresh process.',
        '모델이 기억을 갱신하는 방식은 그대로 두고, 그 기억에서 답을 읽는 마지막 층만 다시 맞췄습니다. 같은 상태에서 정답 확률과 첫 오류 시점이 어떻게 달라지는지 비교하고, 보정 결과를 저장해 다른 프로세스에서 다시 적용할 수 있게 만들었습니다.')+'</p></div>'+toc
    h += section('overview', p(tr('Could changing only the readout delay the first wrong answer?', '같은 모델 상태에서 마지막 판독층만 바꾸면 첫 오류를 늦출 수 있을까요?'))
        +p(tr('The task tracks the accumulated order of three objects. Each input is one of six S3 permutations; after every input, the model predicts the accumulated permutation. A separate integer multiplication table supplies the correct labels.', '과제는 세 항목의 누적 순서를 추적하는 일입니다. 입력은 S3의 여섯 순열 중 하나이며, 매 입력 뒤 모델이 지금까지 누적한 순열을 맞힙니다. 정답은 별도의 정수 곱셈표로 계산합니다.'))
        +p(tr('A recurrent state is a set of numbers updated after each input. The readout uses those numbers to choose an answer. Here, one token is one symbolic S3 group element, not a word or a natural-language context window.',
              '모델 내부 상태(recurrent state)는 이전 입력을 요약하며 매 입력마다 갱신하는 숫자입니다. 판독층(readout)은 이 숫자에서 답을 고르는 마지막 계산층입니다. 이번 token은 S3 입력 원소 하나를 뜻하는 기호 단계이며 자연어 단어나 LLM 문맥 길이가 아닙니다.'))
        +'<div class="readout-finding">'+p(tr('The tools preserve the same cache across readout comparisons. Gold-label scores improved, including beyond the fitting range, while the mean correct prefix became shorter in all three checkpoints.',
        '같은 cache를 유지하며 판독기를 비교하는 도구를 구현했습니다. 보정에 사용한 길이를 넘어서도 정답 확률 점수는 개선됐지만, 첫 오답 이전의 평균 연속 정답 길이는 세 checkpoint 모두 짧아졌습니다.'))+'</div>'
        +p(tr('“Frozen” means that ORIGINAL, SHORT and MIXED read the same evolving state and features within each storage method. Their answers do not feed back into the next state. Native and INT8 states are separate representations, not bitwise-identical states.',
        '상태를 고정했다는 것은 시간 내내 같은 숫자를 쓴다는 뜻이 아닙니다. 같은 저장 방식 안에서 ORIGINAL·SHORT·MIXED가 매 시점 동일한 상태와 특징을 읽고, 답이 다음 상태를 바꾸지 않는다는 뜻입니다. Native와 INT8의 상태는 서로 다른 저장 표현입니다.'))
        +actions([('#implementation',tr('See the implementation','구현 살펴보기')),('#results',tr('Compare recorded results','측정 결과 비교'))]))
    flow = [tr('Same inputs','같은 입력'),tr('State update within one storage method','같은 저장 방식의 상태 갱신'),tr('Shared feature φ','같은 특징 φ'),tr('ORIGINAL / SHORT / MIXED final layer','ORIGINAL / SHORT / MIXED 마지막 층'),tr('Scores, answers and first error','점수·정답·최초 오류 비교')]
    imp = '<ol class="readout-flow">'+''.join('<li>'+escape(x)+'</li>' for x in flow)+'</ol>'
    imp += p(tr('Only mlp.2.weight (6×192) and mlp.2.bias (6) change: 1,158 final-layer values. The input feature is the existing GELU output. Embeddings, recurrence, normalization, hidden layer, codec and per-stream random state are unchanged.',
                '바꾼 부분은 mlp.2.weight(6×192)와 mlp.2.bias(6), 총 1,158개 값입니다. 입력 특징은 기존 GELU 출력입니다. embedding·전이·정규화·앞쪽 은닉층·codec·stream별 난수 상태는 유지합니다.'))
    imp += p(tr('Save a two-tensor patch → start a fresh process → load the original checkpoint plus patch. The loader checks the base hash, tensor names, shapes and dtypes. Original test receipts record matching cache bytes, cursor, terminal status and RNG.',
                '두 tensor patch 저장 → 새 프로세스 → 원래 checkpoint와 patch 로드 순서로 다시 실행합니다. loader는 기준 hash, tensor 이름·shape·dtype을 검사합니다. 원래 검사 기록에는 cache bytes·cursor·terminal·RNG의 일치가 남아 있습니다.'))
    imp += actions([('../sources/case011-adapter.py.txt',tr('Feature-boundary code','특징 경계 코드')),('../sources/case011-head-patch.py.txt',tr('Two-tensor patch loader','두 tensor patch loader')),('../sources/case011-adapter-tests.py.txt',tr('State contract tests','상태 계약 테스트'))])
    imp += tbl([tr('Implemented here','직접 구현'),tr('Upstream supplies','Upstream 제공')],[[tr('Frozen-state adapter, patch contract and paired first-error analysis','고정 상태 adapter·patch 계약·paired 최초 오류 분석'),'DIOVA / Python · NumPy'],[tr('Existing CKDA backbone, group task and original readout','기존 CKDA backbone·군 과제·원래 판독기'),'ComplexKDA'],[tr('Bounded final-layer fitting and FP32 export checks','예산을 고정한 마지막 층 적합·FP32 저장 검사'),'PyTorch Linear / L-BFGS']],tr('Implementation and upstream roles','구현과 upstream 역할'))
    imp += p(tr(f"The readout adds 0 persistent recurrent-state bytes. Native stays at {data['storage_bytes']['NATIVE_FP32']:,} B per stream and INT8 at {data['storage_bytes']['UNIFORM_8']:,} B. Patch files, fitting arrays and temporary features have separate costs; the checkpoint and private head weights are not downloads on this website.",
                f"추가 recurrent-state 저장량은 0 B입니다. stream별 Native {data['storage_bytes']['NATIVE_FP32']:,} B와 INT8 {data['storage_bytes']['UNIFORM_8']:,} B를 유지합니다. patch 파일·fitting 배열·임시 특징 메모리는 별도 비용이며, 이 웹사이트는 checkpoint나 private head weights를 배포하지 않습니다."))
    h += section('implementation',imp)
    intro = p(tr('Three existing checkpoints were evaluated on the same 1,024 sequences, each up to 2,048 group steps. A checkpoint is one saved model trained with its own seed. Both refits use native features; the selected heads transfer to INT8 without refitting.',
        '서로 다른 학습 seed로 얻은 기존 checkpoint 3개를 같은 입력 1,024개에서 각각 2,048단계까지 평가했습니다. 두 보정 판독기는 Native 특징으로 적합한 뒤 INT8에 그대로 적용했습니다.'))
    intro += tbl([tr('Readout','판독기'),tr('Fitting positions','보정 위치'),tr('Final deployed state','최종 저장 상태')],[['ORIGINAL','—',tr('Original final layer','원래 마지막 층')],['SHORT_REFIT','1–32',tr('Identical to ORIGINAL under the fixed stopping rule','고정 종료 기준 아래 ORIGINAL과 동일')],['MIXED_REFIT','1–256',tr('Same shape, two adapted tensors','같은 shape의 보정 tensor 2개')]],tr('Readout definitions','판독기 정의'))
    intro += p(tr('Mean uninterrupted correct length (RMST0) counts the steps from the start through the step before the first error. Higher is better. Gold-label cross entropy (CE) falls when more probability is assigned to the correct answer; lower is better. CE does not replace accuracy or first-error length.',
        '평균 연속 정답 길이(RMST0)는 처음부터 첫 오답 직전까지 맞힌 단계 수의 평균이며 클수록 좋습니다. 정답 확률 점수(교차엔트로피, CE)는 정답에 높은 확률을 줄수록 작아지는 손실입니다. CE는 정답률이나 최초 오류 길이를 대신하지 않습니다.'))
    rows=[]
    for r in data['primary']:
        seed=r['checkpoint_seed'];o=arm[(seed,'UNIFORM_8','ORIGINAL')];m=arm[(seed,'UNIFORM_8','MIXED_REFIT')]
        rows.append([f'seed{seed}',format_num(o['rmst0_tokens']),format_num(m['rmst0_tokens']),format_num(r['mean_delta_tokens'])])
    intro += '<h3>'+tr('INT8: the first error arrived earlier on average','INT8: 평균적으로 첫 오류가 더 빨라졌습니다')+'</h3>'+tbl(['Checkpoint','ORIGINAL = SHORT','MIXED',tr('Change (tokens)','변화 (token)')],rows,tr('Primary INT8 mean correct-prefix length in tokens','주 비교: INT8 평균 연속 정답 길이, token'))
    intro += p(tr('All three primary changes are negative. Gold CE beyond position 256 improved in all three INT8 checkpoints; all-token accuracy improved in two of three. Better probability scores did not yield a longer uninterrupted prefix.',
                  '주 비교의 변화는 세 checkpoint 모두 음수입니다. INT8에서 위치 256 이후 CE는 3/3에서, 전체 token 정답률은 2/3에서 개선됐습니다. 더 좋은 확률 점수가 더 긴 연속 정답으로 이어지지는 않았습니다.'))
    intro += '<figure><img class="serving-curve" src="../assets/case011-primary.svg" alt="'+tr('INT8 MIXED minus ORIGINAL mean correct-prefix length: negative in all three checkpoints; paired bootstrap intervals remain below zero','INT8 MIXED에서 ORIGINAL을 뺀 평균 연속 정답 길이: 세 checkpoint 모두 음수이며 bootstrap 구간도 0 아래')+'"><figcaption>'+tr('Mean-length change, tokens. Higher is better; zero means no change. Whiskers are the original 98.333…% paired bootstrap intervals, not exact coverage guarantees. Corresponding values are in the expandable table below.','평균 길이 변화, token. 클수록 좋고 0은 변화 없음입니다. 막대는 원래 98.333…% paired bootstrap 구간이며 exact coverage 보장이 아닙니다. 대응 수치는 아래 전체 조건 표에 있습니다.')+'</figcaption></figure>'
    expanded=[]
    for seed in range(3):
        for storage in ('NATIVE_FP32','UNIFORM_8'):
            o=arm[(seed,storage,'ORIGINAL')];m=arm[(seed,storage,'MIXED_REFIT')]
            pair=next(r for r in (data['primary'] if storage=='UNIFORM_8' else data['secondary']) if r['checkpoint_seed']==seed and (storage=='UNIFORM_8' or r['comparison']=='NATIVE_FP32/MIXED_REFIT - NATIVE_FP32/ORIGINAL'))
            expanded.append([f'seed{seed}',storage,format_num(o['rmst0_tokens']),format_num(m['rmst0_tokens']),format_num(pair['mean_delta_tokens']),f"[{format_num(pair['interval_tokens'][0])}, {format_num(pair['interval_tokens'][1])}]",'98.333…%' if storage=='UNIFORM_8' else '95% secondary',f"{o['empirical_t005_tokens']} → {m['empirical_t005_tokens']}",f"{100*o['token_accuracy']:.2f}% → {100*m['token_accuracy']:.2f}%"])
    intro += details(tr('Exact values, intervals and all conditions','正確한 수치·신뢰구간과 전체 조건 보기').replace('正確','정확'),tbl(['Checkpoint','Storage','Original RMST0','Mixed RMST0','Δ tokens',tr('Interval','구간'),tr('Level','수준'),'Empirical T0.05',tr('Token accuracy','Token 정답률')],expanded,tr('All storage comparisons','전체 저장 방식 비교'))
        +p(tr('The primary comparison uses 5,000 paired base-sequence bootstrap samples. Each 98.333…% interval has Bonferroni family-95% intent across three checkpoints; coverage is approximate. Native intervals are secondary pointwise 95%. Empirical T0.05 is descriptive, not a confidence-supported horizon.',
        '주 비교는 base sequence를 짝지어 5,000회 bootstrap합니다. 각 98.333…% 구간은 세 checkpoint의 Bonferroni family 95%를 의도한 근사 구간입니다. Native는 보조 pointwise 95%입니다. 경험적 T0.05는 기술통계이며 신뢰지원 horizon이 아닙니다.'))
        +p(tr('The same 1,024 base sequences are reused across conditions. There are six recurrent rollouts and eighteen logical readout conditions, not eighteen independent rollouts. Both sets of 16,384 supervised feature rows were extracted from a shared 256-step Native rollout; equal rows do not imply equal separately executed fitting compute. Lambda is selected on four-band DEV CE; the primary TEST metric is RMST0.',
        '같은 1,024개 base sequence를 조건별로 재사용했습니다. 실제 recurrent rollout은 6개이고 논리 판독 조건은 18개입니다. 두 조건의 각 16,384개 특징 행은 공유한 길이 256의 Native rollout에서 추출했습니다. 같은 행 수를 같은 독립 학습 계산량으로 해석하지 않습니다. lambda는 DEV의 네 구간 CE로 선택했고, 주 TEST 지표는 RMST0입니다.')),'all-conditions')
    h += section('results',intro)
    interpretation = p(tr('A higher probability on later correct answers can coexist with an earlier first mistake. These metrics ask different questions: probability assigned to gold, correct tokens overall, and the uninterrupted prefix before any error.',
        '뒤쪽 문제의 정답 확률이 좋아져도 앞쪽에서 먼저 한 번 틀리면 연속 정답 길이는 짧아질 수 있습니다. 정답에 준 확률, 전체 token 정답률, 첫 오류 이전의 연속 정답 길이는 서로 다른 질문입니다.'))
    interpretation += '<figure><img class="serving-curve" src="../assets/case011-ce.svg" alt="'+tr('Gold cross-entropy change for INT8 across six position bands, one line per checkpoint; lower is better','INT8의 여섯 위치 구간별 정답 교차엔트로피 변화, checkpoint별 한 선; 낮을수록 좋음')+'"><figcaption>'+tr('MIXED minus ORIGINAL gold CE (nats), equal weight per recorded token inside each band. Fitting uses positions through 256; the last three bands are longer positions. This score axis is separate from length in tokens.',
        'MIXED−ORIGINAL 정답 CE(nats). 각 구간 안의 기록된 token에 같은 가중치를 줍니다. 마지막 세 구간은 fitting 최대 위치 256을 넘는 구간입니다. 길이(token)와 다른 축입니다.')+'</figcaption></figure>'
    bandrows=[]
    for seed in range(3):
        score = {(r['first_group_token'],r['last_group_token'],r['head']):r for r in data['scores'] if r['checkpoint_seed']==seed and r['storage']=='UNIFORM_8' and r['position_scope']=='position_range'}
        for lo,hi in ((1,32),(33,128),(129,256),(257,512),(513,1024),(1025,2048)):
            o=score[(lo,hi,'ORIGINAL')];m=score[(lo,hi,'MIXED_REFIT')]
            bandrows.append([f'seed{seed}',f'{lo}–{hi}',f"{o['total_token_observations']:,}",format_num(o['gold_ce_nats'],5),format_num(m['gold_ce_nats'],5),format_num(m['gold_ce_nats']-o['gold_ce_nats'],5)])
    interpretation += details(tr('CE values and denominators by position','위치별 CE 수치와 분모'),tbl(['Checkpoint',tr('Positions','위치'),tr('Token denominator','Token 분모'),'Original CE','Mixed CE','Δ nats'],bandrows,tr('Position band cross entropy','위치 구간별 교차엔트로피')))
    interpretation += p(tr('SHORT is a no-op control under the fixed stopping rule: its final tensors match ORIGINAL. MIXED seed0 and seed1 reached the 200-iteration cap. These are the selected finite iterates under the recorded budget, not a certificate of a converged optimal readout.',
        'SHORT는 고정 종료 기준 아래 최종 tensor가 원형과 같은 대조군입니다. MIXED seed0·seed1은 200 iteration 상한에 도달했습니다. 기록된 예산 안에서 고른 유한한 해이며 최적 판독기로 충분히 수렴했다는 인증은 아닙니다.'))
    interpretation += p(tr('The input-only and same-position/current-token shuffle controls reach about one-sixth accuracy overall; actual INT8 MIXED reaches 39.72–51.22%. This supports useful state-dependent features relative to those controls. Overall chance-like accuracy does not imply chance at every position.',
        '현재 입력·위치만 보는 대조와 같은 시점·현재 token 안에서 특징을 섞은 대조의 전체 정답률은 약 1/6입니다. 실제 INT8 MIXED는 39.72–51.22%였습니다. 이 대조보다 유용한 상태 의존 특징이 있다는 진단이며, 전체 약 1/6이 모든 위치에서 chance라는 뜻은 아닙니다.'))
    interpretation += posthoc_html(data, lang)
    interpretation += '<details id="input-shifts"><summary>'+tr('Explore inputs with longer, shorter or unchanged prefixes','첫 오류가 이동한 입력 보기')+'</summary><p>'+tr('Post-hoc comparison of the same saved records. Defaults include every category. The table shows first-error positions, not new model answers.','같은 저장 기록의 사후 비교입니다. 기본값은 모든 범주이며, 표는 새 모델 답변 대신 최초 오류 위치를 보여줍니다.')+'</p><noscript><p>'+tr('JavaScript enables the item filters; the complete aggregate tables above and JSON download remain available.','JavaScript를 켜면 입력 필터를 사용할 수 있습니다. 위 집계표와 JSON 다운로드는 그대로 볼 수 있습니다.')+'</p></noscript><div id="case011-items" hidden></div>'+a('../data/case011-items.json',tr('Download every input comparison (JSON)','모든 입력 비교 다운로드(JSON)'), 'case011-items.json')+'</details>'
    h += section('analysis',interpretation)
    command = 'CASE=cases/011-frozen-state-readout-adaptation\nOUT=$(mktemp -d)\npython3 -B "$CASE/publication/run_checks.py" --output "$OUT/checks"'
    run = p(tr('Browse the saved results, inspect the feature adapter and patch loader, or recompute the public records on CPU. The website is a static record viewer. Original model reruns require the pinned environment and private checkpoints; no head weights are provided here.',
        '저장 결과를 읽고, 특징 adapter·patch loader를 살펴보거나 공개 기록을 CPU로 재계산할 수 있습니다. 이 사이트는 정적 결과 탐색 화면입니다. 원래 모델 재실행에는 고정 환경과 private checkpoint가 필요하며 head weights는 제공하지 않습니다.'))
    run += actions([('https://github.com/Munsik-Kim/inference-lab/raw/refs/heads/main/downloads/case011_readout_adaptation_reviewed_publication_v1.zip',tr('Code and recorded measurements (ZIP; no weights)','코드·측정 기록 ZIP — 가중치 제외'))])
    run += '<pre>'+escape(command)+'</pre>'+actions([('../sources/case011-reproduction.md.txt',tr('Verified CPU commands and requirements','검증한 CPU 명령과 실행 요건')),('../sources/case011-audit.py.txt',tr('Independent scalar auditor','독립 스칼라 검산기')),('../data/case011.json',tr('Download values and source hashes','표시값·출처 hash 다운로드'))])
    times=[r['ms_per_group_token'] for r in data['timing']['rows']]
    run += details(tr('Recorded CPU and storage costs','기록된 CPU·저장 비용'),p(tr(f"54 separate complete-call timings: two CPU threads, 16 sequences ×128 group tokens, three repeats per condition. The observed range is {min(times):.5f}–{max(times):.5f} ms/group-token. Loading and diagnostic calculations are outside the timer; BOS work is in the numerator. This is neither single-request latency nor GPU speed.",
        f"조건별 3회씩, CPU 2 threads·16입력×128 group token의 독립 complete-call 54회를 측정했습니다. 관측 범위는 {min(times):.5f}–{max(times):.5f} ms/group-token입니다. 로딩·진단은 timer 밖이며 BOS 계산은 분자에 포함합니다. 단일 요청 latency나 GPU 속도 수치가 아닙니다.")))
    h += section('reproduce',run)
    report_url = 'https://github.com/Munsik-Kim/inference-lab/blob/main/'+C11+('/REPORT.ko.md' if ko else '/REPORT.md')
    reportnames = [tr('Background','배경'),tr('Questions and hypotheses','가설과 검증 질문'),tr('Theory and metrics','이론과 지표'),tr('Methods','방법'),tr('Experiments','실험'),tr('Results','실험 결과'),tr('Analysis','결과 분석'),tr('Conclusion','결론'),tr('References and contributions','레퍼런스 및 기여')]
    report = '<ol class="report-contents">'+''.join('<li>'+a(report_url+'#'+key,title)+'</li>' for key,title in zip(REPORT_SECTIONS,reportnames))+'</ol>'
    report += actions([(report_url,tr('Read the full report on GitHub','GitHub 정식 보고서')),('../sources/case011-report-'+lang+'.md.txt',tr('Offline report source','오프라인 보고서 원문')),('../sources/case011-notice.md.txt','NOTICE')])
    report += p(tr('ComplexKDA supplies the existing model and symbolic task; PyTorch and NumPy supply numerical operations and the bounded L-BFGS solver. DIOVA implements the frozen-state intervention, patch contract and paired evaluation. Probe control-task literature informs the diagnostics. Detailed references and implementation credits are recorded in NOTICE.',
        'ComplexKDA의 기존 모델·기호 과제와 PyTorch·NumPy 수치 연산 및 L-BFGS를 사용했습니다. DIOVA는 고정 상태 개입, patch 계약, paired 평가를 구현했습니다. probe control-task 선행연구는 진단의 참고이며, 상세 인용과 구현 기여의 구분은 NOTICE에 기록합니다.'))
    h += section('report',report)
    return '<div class="readout-study">'+h+'</div>'


def posthoc_html(data, lang):
    """Separate descriptive same-record diagnostics from the original primary intervals."""
    ko = lang == 'ko'; tr = lambda en, kr: kr if ko else en
    post=data['posthoc']
    h='<details id="posthoc"><summary>'+tr('Same-record analysis: storage interaction, errors and solver','같은 기록의 사후 분석: 저장 방식·오류 이동·solver')+'</summary><p>'+tr('POST_HOC_SAME_RECORDS · Existing predictions and scalar sums, with no new fitting or model forward. The primary comparison and its interval remain unchanged.','POST_HOC_SAME_RECORDS · 기존 예측과 스칼라 합에서 계산했으며 새 fitting·모델 forward는 없습니다. 원래 주 비교와 구간은 유지합니다.')+'</p>'
    rows=[]
    for r in post['interaction']:
        rows.append(['seed'+str(r['checkpoint_seed']),format_num(r['native_delta_tokens']),format_num(r['int8_delta_tokens']),format_num(r['interaction_tokens']),f"[{format_num(r['interval_tokens'][0])}, {format_num(r['interval_tokens'][1])}]"])
    h+='<h3>'+tr('How does storage change the readout effect?','저장 방식에 따라 보정 효과가 달랐나요?')+'</h3><div class="study-table">'+table(['Checkpoint','Native Δ','INT8 Δ','INT8 Δ − Native Δ',tr('Post-hoc 95% interval','사후 95% 구간')],rows,tr('Storage interaction in group tokens','저장 방식 상호작용, group token'))+'</div><p>'+tr('The interaction uses the same base-sequence pairing in both storage methods. These descriptive pointwise intervals include zero and are separate from the original primary family.','두 저장 방식의 동일 base-sequence pairing을 유지한 차이입니다. 기술적 pointwise 구간은 모두 0을 포함하며 원래 주 비교의 family와 다릅니다.')+'</p>'
    rows=[]
    for r in post['error_shifts']:
        rows.append(['seed'+str(r['checkpoint_seed']),r['storage'],r['longer'],r['shorter'],r['same'],format_num(r['q25_tokens']),format_num(r['median_tokens']),format_num(r['q75_tokens'])])
    h+='<h3>'+tr('Inputs with shifted first errors','최초 오류가 이동한 입력')+'</h3><div class="study-table">'+table(['Checkpoint','Storage',tr('Longer','더 길게'),tr('Shorter','더 짧게'),tr('Same','같음'),'Q25','Median','Q75'],rows,tr('Per-input prefix movement','입력별 연속 정답 길이 변화'))+'</div><p>'+tr('Each row covers 1,024 sequences. Quartiles are changes in uninterrupted prefix length, in group tokens. These are paired movements, not independent samples from each readout.','행마다 입력 1,024개입니다. 사분위수는 연속 정답 길이의 변화(group token)이며, 판독기별 독립 표본이 아닌 같은 입력의 이동입니다.')+'</p>'
    rows=[]
    for r in post['bands']:
        if r['storage']!='UNIFORM_8':continue
        t=r['transitions'];o=r['original'];m=r['mixed']
        rows.append(['seed'+str(r['checkpoint_seed']),f"{r['first_group_token']}–{r['last_group_token']}",f"{r['denominator_tokens']:,}",t['original_only_correct'],t['mixed_only_correct'],t['wrong_to_wrong_different_answer'],f"{o['first_error_count']} → {m['first_error_count']}"])
    h+='<h3>'+tr('INT8: where answers and first errors move','INT8: 답변과 최초 오류의 위치')+'</h3><div class="study-table">'+table(['Checkpoint',tr('Positions','위치'),tr('Token count','Token 수'),tr('Original only correct','원래만 정답'),tr('Mixed only correct','보정만 정답'),tr('Different wrong answer','다른 오답'),tr('First errors O→M','최초 오류 O→M')],rows,tr('Band error movements','구간별 오류 이동'))+'</div><p>'+tr('Answer transitions count token observations. First errors count sequences whose first error occurs in that band, out of all 1,024 sequences; this is not conditional hazard. Both-wrong unchanged counts and Native rows are in JSON.','답변 전환은 token 관측 수입니다. 최초 오류는 전체 1,024개 중 그 구간에서 처음 틀린 입력 수이며 조건부 hazard가 아닙니다. 양쪽 같은 오답과 Native 행은 JSON에 있습니다.')+'</p>'
    rows=[]
    for r in post['solver']['candidates']:
        if not r['selected']:continue
        rows.append(['seed'+str(r['checkpoint_seed']),r['head'],r['lambda'],r['iterations'],r['objective_evaluations'],f"{r['initial_objective']:.6g} → {r['final_objective']:.6g}",f"{r['final_max_abs_gradient']:.3g}",r['status']])
    h+='<h3>'+tr('Selected solver outcomes','선택된 solver 상태')+'</h3><div class="study-table">'+table(['Checkpoint','Readout','λ','Iterations','Evaluations',tr('Objective','목적함수'),tr('Final gradient max','최종 gradient 최대값'),tr('Status','상태')],rows,tr('Selected finite-iteration fits','선택된 유한 iteration 적합'))+'</div><p>'+tr('Objective = mean CE + λ × squared displacement from the original weight and bias. Fitting uses FP64; selection and execution use the exported FP32 head. SHORT already satisfies the gradient tolerance; MIXED seed2 stops on small change without satisfying the gradient criterion.','목적함수는 평균 CE + λ × 원래 weight·bias로부터 변화량의 제곱합입니다. fitting은 FP64이고 선택·실행은 FP32로 저장한 head입니다. SHORT는 시작점에서 gradient 기준을 만족했고, MIXED seed2는 gradient 기준 미달 상태에서 작은 변화로 종료했습니다.')+'</p>'
    h+='<p>'+tr('Per-item margins or logits were not retained, so margins conditional on newly wrong answers are unavailable. The all-item margin sums cannot identify a selected-error mechanism.','입력별 margin·logits를 저장하지 않아 새 오답에 조건부인 margin은 계산할 수 없습니다. 전체 입력의 margin 합으로 선택된 오류의 원인을 판정하지 않습니다.')+'</p>'+a('../data/case011.json',tr('Complete tables, definitions and hashes (JSON)','전체 표·정의·hash(JSON)'))+'</details>'
    return h


def figures(root):
    """Byte-preserved reviewed SVGs, produced by the separate post-hoc analysis."""
    return {asset:(root/path).read_bytes() for asset,path in FIGURE_SOURCES.items()}
