"""Generate bilingual Case011 reports from completed retained scalar artifacts.

Refuses partial cells. Only documentation is written; source, predictions,
protocol, selections and aggregate artifacts remain read-only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from statistics import median

CASE = Path(__file__).resolve().parents[1]
HEADS = ('ORIGINAL', 'SHORT_REFIT', 'MIXED_REFIT')
STORAGES = ('NATIVE_FP32', 'UNIFORM_8')


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])


def fmt(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def write(case):
    a = read(case / 'results/derived/aggregate.json')
    if a['status'] != 'COMPLETE_SAVED_SCALAR_AGGREGATION' or len(a['arms']) != 18 or len(a['source_cells']) != 6:
        raise ValueError('All six fixed TEST cells must complete before reports are written')
    d = read(case / 'results/head_diagnostics.json')
    heads = {r['model_seed']: r for r in d['rows']}
    checkpoint_metadata = read(case.parent / '010-ckda-finite-precision-memory-horizon/versions/v2/provenance/v1_checkpoints.json')
    checkpoint_file_bytes = {r['seed']: r['file_bytes'] for r in checkpoint_metadata['seeds']}
    if set(checkpoint_file_bytes) != {0,1,2}:
        raise ValueError('Missing original checkpoint byte identity')
    arms = {(r['checkpoint_seed'], r['storage'], r['head']): r for r in a['arms']}
    primary = {r['checkpoint_seed']: r for r in a['primary']}
    receipts = {(seed, storage): read(case / f'results/fresh/seed{seed}/{storage}/receipt.json')
                for seed in range(3) for storage in STORAGES}
    if any(x['status'] != 'COMPLETE' or x['N'] != 1024 or x['T'] != 2048 for x in receipts.values()):
        raise ValueError('A TEST receipt is incomplete')
    if any(heads[s]['heads']['SHORT_REFIT']['parameter_changes'] != 0 for s in range(3)):
        raise ValueError('SHORT interpretation requires its recorded unchanged tensors')
    if any(arms[s, st, 'ORIGINAL']['rmst0_tokens'] != arms[s, st, 'SHORT_REFIT']['rmst0_tokens']
           for s in range(3) for st in STORAGES):
        raise ValueError('Original/SHORT RMST differs despite the claimed unchanged head')
    timing_path = case / 'results/timing/timing.json'
    timing = read(timing_path) if timing_path.exists() else None
    if timing is not None and timing['status'] != 'COMPLETE':
        timing = None
    score_rows = [r for r in a['scores'] if r['storage'] == 'UNIFORM_8' and r['position_scope'] == 'position_range']
    def score(seed, head, low, high):
        return next(r for r in score_rows if (r['checkpoint_seed'], r['head'], r['first_group_token'], r['last_group_token']) == (seed, head, low, high))
    def late_mean(seed, head, key):
        rows = [r for r in score_rows if r['checkpoint_seed'] == seed and r['head'] == head and r['first_group_token'] > 256]
        if key == 'ce':
            return sum(r['gold_ce_sum_nats'] for r in rows) / sum(r['valid_score_observations'] for r in rows)
        return sum(r['correct_tokens'] for r in rows) / sum(r['total_token_observations'] for r in rows)
    ce_better = sum(late_mean(s, 'MIXED_REFIT', 'ce') < late_mean(s, 'ORIGINAL', 'ce') for s in range(3))
    acc_better = sum(late_mean(s, 'MIXED_REFIT', 'acc') > late_mean(s, 'ORIGINAL', 'acc') for s in range(3))
    positive = sum(primary[s]['interval_tokens'][0] > 0 for s in range(3))
    negative = sum(primary[s]['interval_tokens'][1] < 0 for s in range(3))
    overlap = 3 - positive - negative
    delta_text = ' / '.join(f"{primary[s]['mean_delta_tokens']:+.2f}" for s in range(3))
    if negative != 3:
        raise ValueError('The outcome prose describes three negative primary intervals; review it if data changes')
    smoke = read(case / 'provenance/adapter_smoke.json')
    reload = read(case / 'provenance/head_reload.json')
    boundaries = read(case / 'provenance/execution_checkpoints.json')
    if not smoke['passed'] or len(smoke['cases']) != 6 or any(r['maximum_logit_difference'] != 0 for r in smoke['cases']):
        raise ValueError('Readout parity prose requires completed exact three-checkpoint/two-storage SMOKE')
    if reload['status'] != 'PASS' or reload['fresh_child_processes'] != 12 or any(r['status'] != 'PASS' for r in reload['rows']):
        raise ValueError('Fresh-process reload prose requires all 12 successful receipts')
    if boundaries['status'] != 'PASS' or len(boundaries['checkpoints']) != 30:
        raise ValueError('Execution-boundary prose requires all 30 completed checks')
    terminal = sum(r['terminal_count'] for r in receipts.values())
    nbytes = receipts[0, 'NATIVE_FP32']['ledger']['per_stream_persistent_bytes']
    qbytes = receipts[0, 'UNIFORM_8']['ledger']['per_stream_persistent_bytes']
    main_rows = []
    for s in range(3):
        o, sh, m = [arms[s, 'UNIFORM_8', h] for h in HEADS]
        p = primary[s]
        main_rows.append([s, fmt(o['rmst0_tokens']), fmt(sh['rmst0_tokens']), fmt(m['rmst0_tokens']),
                          f"{p['mean_delta_tokens']:+.2f}", '[' + ', '.join(fmt(x) for x in p['interval_tokens']) + ']'])
    horizon_rows = [[s, *(arms[s, 'UNIFORM_8', h]['empirical_t005_tokens'] for h in HEADS)] for s in range(3)]
    horizon_text = ' / '.join(f"seed{s}: {arms[s, 'UNIFORM_8', 'ORIGINAL']['empirical_t005_tokens']}→{arms[s, 'UNIFORM_8', 'MIXED_REFIT']['empirical_t005_tokens']}" for s in range(3))
    native_rows = []
    for s in range(3):
        r = next(x for x in a['secondary'] if x['checkpoint_seed'] == s and x['pair_index'] == 0)
        native_rows.append([s, fmt(arms[s, 'NATIVE_FP32', 'ORIGINAL']['rmst0_tokens']),
                           fmt(arms[s, 'NATIVE_FP32', 'MIXED_REFIT']['rmst0_tokens']),
                           f"{r['mean_delta_tokens']:+.2f}", '[' + ', '.join(fmt(x) for x in r['interval_tokens']) + ']'])
    ranges = [(1,32),(33,128),(129,256),(257,512),(513,1024),(1025,2048)]
    length_rows = []
    for s in range(3):
        for low, high in ranges:
            o, m = score(s, 'ORIGINAL', low, high), score(s, 'MIXED_REFIT', low, high)
            length_rows.append([s, f'{low}–{high}', fmt(o['token_accuracy']*100), fmt(m['token_accuracy']*100),
                                fmt(o['gold_ce_nats'], 4), fmt(m['gold_ce_nats'], 4)])
    control_rows = []
    for s in range(3):
        c = next(x for x in a['controls'] if x['checkpoint_seed'] == s and x['storage'] == 'UNIFORM_8' and x['head'] == 'MIXED_REFIT')
        control_rows.append([f'INT8 MIXED / seed{s}', fmt(arms[s, 'UNIFORM_8', 'MIXED_REFIT']['token_accuracy']*100),
                             fmt(c['token_accuracy']*100), fmt(c['rmst0'])])
    input_only = next(x for x in a['controls'] if x['head'] == 'INPUT_ONLY')
    fitting_rows = [[s, h, heads[s]['heads'][h]['lambda'], heads[s]['heads'][h]['parameter_changes'],
                     heads[s]['heads'][h]['solver_status']] for s in range(3) for h in HEADS[1:]]
    bytes_rows = [['Native state / stream', nbytes], ['INT8 state / stream', qbytes],
                  ['Model parameter values / before and after', heads[0]['model_tensor_bytes_before_and_after']],
                  ['Final Linear values / before and after', heads[0]['final_head_tensor_bytes_before_and_after']],
                  ['Original checkpoint file / each retained seed', checkpoint_file_bytes[0]],
                  ['Added recurrent-state bytes / refitted head', 0],
                  ['One FIT feature matrix: 16,384 × 192 × FP32', 16384*192*4]]
    patch_rows = [[s, h, heads[s]['heads'][h]['patch_payload_bytes'], heads[s]['heads'][h]['patch_manifest_bytes'],
                   heads[s]['heads'][h]['patch_total_file_bytes']] for s in range(3) for h in HEADS[1:]]
    timing_rows = []
    if timing:
        for s in range(3):
            for st in STORAGES:
                values = [[r['ms_per_group_token'] for r in timing['rows'] if (r['model_seed'],r['storage'],r['readout']) == (s,st,h)] for h in HEADS]
                if any(len(x) != 3 for x in values): raise ValueError('Incomplete individual-head timing')
                timing_rows.append([s, st, *(fmt(median(x),5) for x in values)])
    for ko in (False, True):
        language = 'ko' if ko else 'en'
        alt = 'README.md' if ko else 'README.ko.md'
        report_name = 'REPORT.ko.md' if ko else 'REPORT.md'
        title = ('Case 011 — 상태를 고정한 판독층 보정과 길이 외삽' if ko else 'Case 011 — Frozen-State Readout Adaptation and Length Extrapolation')
        intro = ('같은 recurrent state를 공유하는 세 판독기를 비교하고, 마지막 Linear의 두 tensor만 보정·저장·재로딩하는 도구를 구현했습니다. 기존 CKDA checkpoint 세 개에서 Native와 INT8 저장을 고정한 채 새 1,024개 입력을 길이 2,048까지 읽었습니다.' if ko else
                 'Built a shared-state comparison runner and a two-tensor save/reload path for adapting only the final Linear. The study reads 1,024 new sequences through 2,048 group tokens on each of three existing CKDA checkpoints, keeping Native and INT8 storage fixed.')
        finding = (f'INT8에서 MIXED 보정은 세 checkpoint 모두 평균 연속 정답 길이를 줄였습니다. seed0/1/2의 MIXED−ORIGINAL 변화는 {delta_text} token이며, 세 주 비교의 구간은 모두 0 아래입니다. 반면 위치 257–2,048의 평균 정답 CE는 {ce_better}/3에서 낮아졌고, token 정답률은 {acc_better}/3에서 높아졌습니다. 정답에 부여한 점수의 개선이 처음부터 정확히 읽는 길이의 증가로 이어지지는 않았습니다.' if ko else
                   f'Under INT8, the MIXED refit reduced uninterrupted correct lifetime on all three checkpoints. MIXED−ORIGINAL changes for seeds 0/1/2 were {delta_text} tokens, with all three primary intervals below zero. Over positions 257–2,048, mean gold CE nevertheless decreased in {ce_better}/3 and token accuracy increased in {acc_better}/3. Better gold scores did not translate into a longer interval of correctness from the start.')
        headnames = ['Checkpoint', 'ORIGINAL', 'SHORT', 'MIXED', 'Δ MIXED−ORIGINAL', '98.333% interval']
        primary_table = table(headnames, main_rows)
        readme = f'''# {title}

[{'English' if ko else '한국어'}]({alt}) · [DIOVA](../../{'README.ko.md' if ko else 'README.md'})

{intro}

{('## 구현한 기능' if ko else '## Working components')}

- [Frozen feature adapter](source/adapter.py): {('기존 GELU 출력에서 정확히 분리한 192차원 특징, backbone gradient 0.' if ko else 'Exact existing 192-dimensional GELU features; no backbone gradients.')}
- [Head fitting](source/fitting.py) · [patch loader](source/head_patch.py): {('최종 1,158개 parameter만 CPU FP64로 적합하고 FP32 두 tensor를 원래 자리에 복사합니다.' if ko else 'Fit 1,158 final parameters in CPU FP64, then copy two FP32 tensors into their original slots.')}
- [Paired runner](source/run_eval.py) · [independent audit](analysis/audit.py): {('같은 cache에서 판독기 세 개의 최초 오류·정답 점수·control을 비교합니다.' if ko else 'Compare first error, gold scores and controls for three readouts over one cache.')}
- [Fresh-process reload](source/reload_check.py): {('실제 작은 checkpoint에서 logits·features·cache bytes를 대조합니다.' if ko else 'Compare logits, features and cache bytes on the actual small checkpoints.')}

{('기술: Python · NumPy · PyTorch. 공개 스칼라 검산에는 모델과 GPU가 필요하지 않습니다.' if ko else 'Stack: Python · NumPy · PyTorch. Public scalar reanalysis needs neither a model nor a GPU.')}

{('## 확인한 결과' if ko else '## Recorded results')}

{finding}

{primary_table}

{('단위는 첫 오류 전 연속 정답 group token의 평균(RMST0)입니다. 각 행은 같은 1,024개 sequence의 paired 비교이며, 5,000회 bootstrap의 98.333% 구간은 세 주 비교에 대한 Bonferroni family 95%를 목표로 한 근사 구간입니다.' if ko else 'Values are RMST0: mean consecutive correct group tokens before the first error. Each row pairs the same 1,024 sequences. The 98.333% intervals use 5,000 bootstrap resamples with an approximate Bonferroni family target of 95% across three primary contrasts.')}

{('SHORT_REFIT의 FP32 tensor는 원형과 동일했습니다. MIXED_REFIT는 두 tensor를 바꿨으며 seed0/1은 사전 200 iteration 상한에 도달한 유한 해입니다. 각 solver 상태와 길이별 점수는 보고서에 함께 남겼습니다.' if ko else 'SHORT_REFIT retained the original FP32 tensors. MIXED_REFIT changed both tensors; seeds 0/1 reached the predeclared 200-iteration limit with finite accepted iterates. Solver status and scores by length remain visible in the report.')}

{('## CPU로 확인하기' if ko else '## Check the saved results on CPU')}

{('이 Case011 디렉터리에서 Python과 NumPy로 실행합니다. 출력은 아직 없는 새 디렉터리를 지정합니다.' if ko else 'From this Case011 directory, run with Python and NumPy and choose an output directory that does not already exist.')}

```bash
python -B analysis/aggregate.py --input-only-fit results/input_only_fit.json \\
  --output /tmp/case011-new-audit --no-figures
```

{('이 명령은 공개 prediction·gold·스칼라 합계의 재계산입니다. 실제 모델 fitting/재실행은 별도의 원래 checkpoint·upstream·head patch가 필요합니다. 원래 모델 weights와 fitted head tensor는 검토 ZIP에서 제외하고 정확한 manifest만 제공합니다.' if ko else 'This recalculates public predictions, gold and scalar sums. Model fitting/replay separately requires the original checkpoints, upstream and head patches. Original model weights and fitted head tensors are excluded from the review ZIP; exact manifests are included.')}

[{'결과표·분석' if ko else 'Results and analysis'}]({report_name}) · [Methods](METHODS.md) · [Reproduction](REPRODUCTION.md) · [Protocol](configs/protocol.json) · [{'로컬 결과 탐색기' if ko else 'Local result explorer'}](demo/{'ko.html' if ko else 'en.html'}) · [Attribution](NOTICE.md)

{('현재 상태: 로컬 연구·검토본. 원격 게시·PR·Pages 배포는 실행하지 않았습니다.' if ko else 'Status: local research and review candidate. No remote publication, PR or Pages deployment was performed.')}
'''
        sections = (['배경','가설','이론','방법','실험','결과','분석','결론','레퍼런스'] if ko else
                    ['Background','Hypotheses','Theory','Methods','Experiments','Results','Analysis','Conclusion','References'])
        toc = '\n'.join(f'{i}. [{name}](#s{i})' for i,name in enumerate(sections,1))
        def h(i): return f'<a id="s{i}"></a>\n## {i}. {sections[i-1]}\n'
        text = f'# {title}\n\n[한국어](REPORT.ko.md) · [English](REPORT.md) · [README]({"README.ko.md" if ko else "README.md"})\n\n{intro}\n\n{toc}\n\n'
        text += h(1) + ('Case010은 저비트 state 저장과 실패를 보존하는 재시작을 구현했고, 고정 FP32 계수의 FP64 산술 승격에서도 별도 진단 입력의 예측·최초 실패가 같았습니다. 그 사실만으로 오류의 원인을 판독기로 확정할 수는 없습니다. Case011은 같은 state에서 최종 판독 가중치를 실제로 바꾸는 별도의 개입입니다.\n\n' if ko else
                         'Case010 implemented low-bit state storage and failure-preserving restart. Its arithmetic diagnostic found unchanged predictions/first failures after promoting fixed FP32 coefficients to FP64 on a separate diagnostic cohort. That does not establish a readout cause. Case011 is a separate intervention that actually changes the final readout values over the same state.\n\n')
        text += h(2) + ('사전 주 질문은 INT8에서 MIXED_REFIT가 ORIGINAL보다 RMST0를 늘리는가입니다. SHORT는 위치 1–32의 보정 대조, MIXED는 위치 1–256의 보정이며 둘 다 Native 특징으로만 fitting합니다. 위치 512/1024/2048은 fitting 범위를 넘는 길이 외삽입니다. TEST 후 후보·lambda·seed를 추가하지 않았습니다.\n\n' if ko else
                         'The predeclared primary question is whether MIXED_REFIT improves INT8 RMST0 over ORIGINAL. SHORT is the position-1–32 refit control; MIXED uses positions through 256. Both fit Native features only. Positions 512/1024/2048 test length extrapolation. No candidates, lambdas or seeds were added after TEST.\n\n')
        text += h(3) + ('고정 recurrence와 codec가 만든 state를 원래 normalization·gating·projection·embedding residual·LayerNorm·hidden Linear·GELU로 판독한 결과가 `phi_t`입니다. 후보는 `Wc phi_t + bc`, 원형은 `W0 phi_t + b0`를 계산합니다. 두 계산은 다음 state에 영향을 주지 않습니다.\n\n`tau`는 첫 오답/INVALID의 1-based group-token 위치이고 `RMST0=mean(min(tau−1,2048))`입니다. 미실패는 2048 우측 검열로 기여하며, 이후 정답 복귀는 survival을 되돌리지 않습니다. 경험적 T0.05는 누적 최초 오류 비율이 5% 이하인 마지막 관측 위치이며 이번에는 신뢰지원 horizon 하한을 계산하지 않습니다.\n\n' if ko else
                         '`phi_t` is the original GELU feature after the fixed state read, normalization, gating, projection, embedding residual, LayerNorm and hidden Linear. Candidate and original logits are `Wc phi_t + bc` and `W0 phi_t + b0`; neither affects the next state.\n\n`tau` is the first wrong/INVALID 1-based group-token position. `RMST0=mean(min(tau−1,2048))`, with an unfailed sequence censored at 2048. Later recovery does not restore survival. Empirical T0.05 is the final observed position with first-error risk at most 5%; no confidence-supported horizon is calculated here.\n\n')
        text += h(4) + ('변경 허용 tensor는 `mlp.2.weight [6,192]`, `mlp.2.bias [6]`뿐입니다. 같은 세 기존 checkpoint를 유지하고 state·계수·hidden feature 생성은 FP32로 고정했습니다. 같은 storage의 세 head는 하나의 feature를 읽습니다. 정답은 독립 S3 정수 left-product 계산기에서 만들며 후보에 전달하지 않습니다. 단순 오답은 계속 실행, 유한 state의 invalid readout은 −1 후 계속 실행, state terminal은 기존 v2 흡수 실패로 보존합니다.\n\nFIT는 각 조건 16,384 supervised 행입니다. SHORT는 첫 32위치, MIXED는 네 band별 8위치를 각 512개 sequence에서 사용합니다. 실제 feature 추출은 둘이 공유한 전체 256길이 Native trajectory이며, 같은 label 행 수를 같은 독립 학습 계산량이라고 주장하지 않습니다. CPU FP64 L-BFGS는 `mean CE + λ(||W−W0||²+||b−b0||²)`를 최소화합니다. λ={1e−4,1e−2,1}, 최대 200 iterations/1000 objective evaluations, FP32 변환 뒤 네 DEV band의 동일 가중 CE, 정확한 동점은 큰 λ 우선입니다. [전체 방법](METHODS.md).\n\n' if ko else
                         'Only `mlp.2.weight [6,192]` and `mlp.2.bias [6]` may change. The same three existing checkpoints, FP32 states/coefficients and hidden-feature computation are retained. All three heads for a storage read one shared feature. An independent integer S3 left-product evaluator supplies gold only to evaluation. Wrong labels continue; invalid readout from finite state yields −1 and continues; numerical state failure retains the original v2 absorbing terminal policy.\n\nEach refit receives 16,384 supervised rows. SHORT uses the first 32 positions, while MIXED uses eight positions in each of four bands per 512 sequences. Extraction actually shares the full 256-position Native trajectory, so equal supervised rows are not claimed to be equal independently executed training compute. CPU FP64 L-BFGS minimizes `mean CE + λ(||W−W0||²+||b−b0||²)`, with λ={1e−4,1e−2,1}, at most 200 iterations/1000 evaluations. Selection uses equal-band DEV CE after FP32 conversion, with exact ties favoring larger λ. [Full methods](METHODS.md).\n\n')
        text += h(5) + table(['Role','Seed','Sequences','Group tokens'],[['SMOKE',61601,8,32],['FIT',61101,512,256],['DEV',61201,128,256],['FRESH_TEST',61301,1024,2048]]) + '\n\n'
        text += ('입력은 S3의 여섯 군 원소 모두에서 uniform 생성했습니다. BOS write0은 미채점이며 모든 group prefix를 채점했습니다. 위 각 코호트의 실제 bytes·ID·hash를 먼저 동결했습니다. 새 backbone 학습 0, 새 codec 0, fitting 후보 18개, 선택된 head 6개, primary rollout 6개/논리 arm 18개입니다. 원형 Case010의 512/1024입력을 이번 표본에 합치지 않았습니다.\n\n' if ko else
                 'Inputs sample all six S3 elements uniformly. BOS write0 is unscored and every group prefix is scored. Actual bytes, IDs and hashes were frozen before execution. There were zero new backbone training runs, zero new codecs, 18 fitting candidates, six selected heads and six primary rollouts/18 logical arms. Historical Case010 cohorts are not pooled into this sample.\n\n')
        text += table(['Seed','Head','λ','Changed FP32 values','Solver status'], fitting_rows) + '\n\n'
        text += ('SHORT의 초기 gradient가 종료 허용오차 안에 있어 FP32 parameter가 그대로였습니다. MIXED seed0/1은 `MAX_ITER_NOT_CONVERGED`, seed2는 작은 변화/방향에 의한 optimizer 종료입니다. 수렴 미달을 감추거나 TEST를 보고 iteration을 늘리지 않았습니다. 추가 head 진단은 이미 저장한 DEV 특징과 parameter만 읽었으며 선택을 바꾸지 않았습니다.\n\n' if ko else
                 'SHORT stopped within the initial gradient tolerance and left FP32 parameters unchanged. MIXED seeds 0/1 have `MAX_ITER_NOT_CONVERGED`; seed2 stopped for a small change/direction. The iteration cap was not extended after TEST. Supporting head diagnostics read already saved DEV features and parameters without changing selection.\n\n')
        text += h(6) + ('### 6.1 주 비교: INT8의 평균 연속 정답 길이\n\n' if ko else '### 6.1 Primary: consecutive correct lifetime under INT8\n\n') + primary_table + '\n\n' + finding + '\n\n'
        text += ('각 checkpoint의 n=1,024 paired sequence, 5,000 bootstrap, 각 98.333% 근사 구간입니다. 세 모델을 합쳐 n=3,072로 재표집하지 않습니다.\n\n' if ko else 'Each checkpoint uses 1,024 paired sequences, 5,000 bootstrap draws and an approximate 98.333% interval. Checkpoints are not pooled into 3,072 independent model observations.\n\n')
        text += table(['Checkpoint','ORIGINAL T0.05','SHORT T0.05','MIXED T0.05'],horizon_rows) + '\n\n'
        text += ('위 길이는 매-token 경험값이며 CI 하한이 아닙니다. 최초 실패 없이 2,048까지 통과한 입력도 관측 끝에서 검열됩니다.\n\n' if ko else 'These horizons are every-token empirical values, not confidence lower bounds. A sequence passing through 2,048 remains right-censored at the observation limit.\n\n')
        text += '![INT8 survival](results/derived/figures/int8_survival.png)\n\n![Primary paired RMST0](results/derived/figures/primary_rmst0.png)\n\n'
        text += ('### 6.2 Native 보조 비교\n\n' if ko else '### 6.2 Secondary Native comparison\n\n') + table(['Seed','ORIGINAL RMST0','MIXED RMST0','Δ tokens','Pointwise 95% interval'],native_rows) + '\n\n'
        text += ('주 비교와 다른 pointwise 95% 구간입니다. INT8 MIXED−SHORT는 SHORT tensor가 원형과 같으므로 점추정이 MIXED−ORIGINAL과 같지만 보조 구간의 confidence/재표집 seed는 별도입니다. 같은 readout의 Native↔INT8 비교 전체는 [secondary.csv](results/derived/secondary.csv)에 있습니다.\n\n' if ko else
                 'These secondary intervals are pointwise 95%. INT8 MIXED−SHORT has the same point estimate as MIXED−ORIGINAL because SHORT tensors are unchanged, while its secondary confidence level/resampling seed differ. All same-readout Native↔INT8 contrasts are in [secondary.csv](results/derived/secondary.csv).\n\n')
        text += ('### 6.3 INT8의 길이 구간별 정답 점수\n\n' if ko else '### 6.3 INT8 gold scores by position range\n\n') + table(['Seed','Positions','ORIGINAL accuracy %','MIXED accuracy %','ORIGINAL CE nats','MIXED CE nats'],length_rows) + '\n\n'
        text += ('정답 CE는 낮을수록 좋으며 유효 token 관측당 평균입니다. 정답률은 전체 token 분모를 사용합니다. 257 이상은 fitting 위치 범위 밖이며, 512/1024/2048 한 지점의 값과 구간 평균을 혼동하지 않습니다. 각 분모·invalid 수·margin 및 Native 값은 [scores.csv](results/derived/scores.csv)에 있습니다. 공개 재계산은 저장한 score 합계/분모를 사용하며 비공개 전체 logits를 재구성하지 않습니다.\n\n' if ko else
                 'Lower gold CE is better; it is averaged per valid token observation. Accuracy retains the full token denominator. Positions beyond 256 are outside fitting support. A range average is distinct from a single-position value at 512/1024/2048. [scores.csv](results/derived/scores.csv) preserves denominators, invalid counts, margins and Native values. Public reanalysis uses retained score sums/counts, not reconstructed private full logits.\n\n')
        text += '![Gold CE by position](results/derived/figures/gold_ce_by_position.png)\n\n'
        text += ('### 6.4 Control\n\n' if ko else '### 6.4 Controls\n\n') + table(['Condition','True feature accuracy %','Shuffled accuracy %','Shuffled RMST0'],control_rows) + '\n\n'
        text += (f'input-only control은 전체 token 정답률 {input_only["token_accuracy"]*100:.2f}%, RMST0 {input_only["rmst0"]:.2f} token이었습니다. 첫 group token의 gold는 current token 자체여서 그 위치는 state-dependent 정보가 필요하지 않습니다. 위치 t와 current token 안의 순열은 label을 쓰지 않고 singleton은 그대로 두어 수를 기록했습니다. Head 성과가 control과 다르다는 것은 이 feature 경계에서 state 관련 판독을 지지하지만 정확한 군 표현이나 모든 decoder에서의 복원 가능성을 증명하지 않습니다.\n\n' if ko else
                 f'The input-only control achieved {input_only["token_accuracy"]*100:.2f}% token accuracy and RMST0 {input_only["rmst0"]:.2f} tokens. At the first group token, gold equals the current token, so state-dependent information is unnecessary there. Within-position/current-token permutations use no labels; singleton rows are retained and counted. Separation from these controls supports state-dependent readout at this feature boundary, not a proof of an exact learned group representation or recovery by every decoder.\n\n')
        text += ('### 6.5 저장량·재로딩·비용\n\n' if ko else '### 6.5 Storage, reload and cost\n\n') + table(['Object / scope','Bytes'],bytes_rows) + '\n\n' + table(['Seed','Patch','Tensor values B','Manifest B','Total files B'],patch_rows) + '\n\n'
        text += (f'6개 fresh rollout의 terminal stream 수 합계는 {terminal}입니다. 이것은 정답 stream 수가 아니라 수치 state 실패 수입니다. Head별 parameter·cache·임시 feature·파일 장부를 분리했습니다. Cell의 shared ledger는 원래 v2 cache/table 계약이며 새 head 선택 manifest는 별도 파일 비용입니다. 전체 artifact의 디스크 크기가 같다는 뜻이 아닙니다. 원래 checkpoint 파일 크기는 [Case010의 파일 identity](../010-ckda-finite-precision-memory-horizon/versions/v2/provenance/v1_checkpoints.json)에서 읽었습니다. 원래 checkpoint와 fitted head tensor는 ZIP에 넣지 않으며 [정확한 patch manifest](provenance/head_patches/)와 loader를 제공합니다.\n\n' if ko else
                 f'The six fresh rollout cells recorded {terminal} terminal streams in total. This counts numerical state failures, not correct sequences. Model/head values, cache, temporary features and files are separate ledger entries. The cell shared ledger describes the base v2 cache/table contract; selected-head manifests are separate artifact costs. Total artifact disk sizes are not claimed identical. Original checkpoint file sizes come from [Case010 file identities](../010-ckda-finite-precision-memory-horizon/versions/v2/provenance/v1_checkpoints.json). The review ZIP excludes original checkpoints and fitted head tensors while providing [exact patch manifests](provenance/head_patches/) and the loader.\n\n')
        text += ('검사는 실제 실행 기록별로 나눴습니다. 고정 SMOKE의 3 checkpoint×2 storage에서 기존 readout과 feature+head logits가 bitwise 일치했고 최대 차이는 0입니다. [새 프로세스 12건](provenance/head_reload.json)은 logits·labels·features·cache·cursor·terminal/RNG 일치를 통과했습니다. [저장 경계 30건](provenance/execution_checkpoints.json)은 모델 실행 없이 cursor·prefix·inventory·hash를 따로 검증했습니다. 이번 고유 단위 테스트는 79개 PASS, skip 0이며, 여섯 공개 TEST cell과 19개 control 요약의 스칼라를 별도 구현으로 검산했습니다. 검사 횟수는 재실행을 더해 부풀리지 않았습니다. [검사 receipt](provenance/validation_receipt.json), [기존 3,708개 보호 파일의 변경 0 확인](provenance/protection.json).\n\n' if ko else
                 'Validation is separated by execution scope. Across three checkpoints × two storage modes on the fixed SMOKE cohort, original-readout and feature-plus-head logits were bitwise equal, with maximum difference zero. [Twelve fresh processes](provenance/head_reload.json) passed checks of logits, labels, features, cache, cursor and terminal/RNG state. [Thirty saved boundaries](provenance/execution_checkpoints.json) independently passed cursor, prefix, inventory and hash checks without model execution. The current unique unit suite passed 79 tests with no skips; a separate implementation audited six public TEST cells and 19 control summaries. Reruns are not added to the unique test count. [Validation receipt](provenance/validation_receipt.json); [zero changes across 3,708 protected files](provenance/protection.json).\n\n')
        if timing_rows:
            text += ('공유 table·config와 N=1/16/128 저장량, 테스트·적합의 임시 배열은 [전체 byte ledger](results/byte_ledger.json)에 분리했습니다. 프로세스 peak RAM은 미측정입니다.\n\n' if ko else
                     'The [complete byte ledger](results/byte_ledger.json) separates shared table/config costs, N=1/16/128 totals and temporary evaluation/fitting arrays. Process peak RAM was not measured.\n\n')
            text += table(['Seed','Storage','ORIGINAL ms/token','SHORT ms/token','MIXED ms/token'],timing_rows) + '\n\n'
            text += ('위 값은 독립 CPU complete-call 3회의 중앙값입니다. 2 threads, 16×128 group tokens, BOS는 실행시간에 포함하고 분모에서는 제외했습니다. 같은 모양의 Linear 하나를 실행하며 head를 동시에 읽는 연구 runner의 시간이 아닙니다. 속도비 유의성·GPU 속도·단일 요청 latency를 주장하지 않습니다. 원 반복값은 [timing.json](results/timing/timing.json)에 있습니다.\n\n' if ko else
                     'Values are medians of three individual-head CPU complete calls: two threads and 16×128 group tokens. BOS work is in the numerator but not the group-token denominator. Each call uses one same-shaped Linear, not the multi-head research runner. No speedup significance, GPU speed or individual-request latency claim is made. [timing.json](results/timing/timing.json) retains every repetition.\n\n')
        else:
            text += ('CPU timing: NOT_RUN 또는 아직 완료 receipt 없음. 완료되면 원 JSON에서 표를 생성하며 추정값을 쓰지 않습니다.\n\n' if ko else 'CPU timing: NOT_RUN or no completed receipt yet. The table is generated only from a completed timing JSON, never estimates.\n\n')
        text += h(7) + (f'**동일 state에서의 개입을 확인했고, 정답 점수와 첫 오류 수명이 다르게 움직였습니다.** CE는 정답 확률의 음의 로그를 token별로 평균하지만 RMST0는 한 번의 이른 오류에도 줄어듭니다. Fitting의 목적함수는 평균 CE와 가중치 변화 penalty이며 sequence의 최초 오류를 직접 최적화하지 않습니다. 이 실험은 두 목표의 불일치를 보여주지만, 그 차이의 기하학적 원인까지 규명하지는 않았습니다.\n\nINT8의 경험적 T0.05 변화는 {horizon_text} token입니다. seed0에서는 이 경험적 위치가 늘었지만 RMST0는 줄었습니다. 같은 seed에서도 위험 5%의 위치와 평균 연속 길이는 다른 분포 요약입니다. 이 경험값에 신뢰지원 하한의 의미를 붙이지 않습니다.\n\nSHORT가 바뀌지 않은 결과는 이 fitting 입력·gradient tolerance·FP32 경계에서 추가 보정이 없었다는 뜻입니다. MIXED는 이미 오류가 있는 더 긴 FIT 특징을 포함하고 두 tensor를 바꿨습니다. 두 차이는 fitting 위치 범위의 결과로 해석하되, iteration cap에 도달한 seed0/1에 대해 가능한 모든 선형 판독기의 최적 성능이라고 주장하지 않습니다.\n\nFP64 진단의 재실행, codec 변경, backbone 재학습으로 얻은 결과가 아닙니다. 같은 세 checkpoint에서 하나의 fresh TEST를 사용한 결과이며 다른 학습 seed 모집단이나 LM 문맥 길이 전체로 확대하지 않습니다. 기술적 all-terminal 경계 오류는 합성 검사에서 TEST 전 수정했고, 실제 결과의 부호에 따라 실행 규칙을 바꾸지 않았습니다.\n\n' if ko else
                 f'**The same-state intervention is verified: gold scores and first-error lifetime moved differently.** CE averages the negative log probability of gold across tokens; RMST0 can decrease after a single earlier error. Fitting minimizes average CE plus a weight-change penalty rather than directly optimizing the first error of a sequence. This study observes a mismatch between the objectives without identifying its geometric cause.\n\nINT8 empirical T0.05 changes were {horizon_text} tokens. For seed0 this empirical position rose while RMST0 fell. A 5%-risk position and mean uninterrupted lifetime summarize different parts of the same distribution. These empirical values are not confidence-supported lower bounds.\n\nAn unchanged SHORT head means no additional correction under these fitting inputs, gradient tolerance and FP32 boundary. MIXED includes longer FIT features with existing errors and changes both tensors. This is a contrast in fitting-position support; seeds 0/1 reaching their iteration cap prevents interpreting their result as the optimum of all linear decoders.\n\nThese are not repeated FP64 diagnostics, new codecs or backbone retraining results. They concern one fresh TEST on the same three checkpoints, not all training initializations or language-model context lengths. A technical all-terminal boundary error was fixed on synthetic checks before TEST; result direction did not change execution rules.\n\n')
        text += h(8) + (f'완성한 결과물은 같은 cache의 세 판독기 비교, 1,158개 parameter의 제한된 지도 보정, 두 tensor patch와 새 프로세스 재실행, 공개 prediction 기반 독립 검산입니다. INT8의 MIXED 보정 후 seed0/1/2의 평균 연속 정답 길이 변화는 각각 {delta_text} token이며, 세 주 비교 구간 모두 0 아래입니다. 위치 256 밖의 정답 CE 개선은 관측했지만 처음부터 정확히 읽는 길이는 늘리지 못했습니다. 이 결과는 현재 feature·최종 Linear·fitting 예산의 범위이며, 모든 decoder에서 정보 복원이 불가능하다는 증명이 아닙니다.\n\n현재는 로컬 검토 단계입니다. stage/commit/push/PR/merge/Pages/Drive 업로드는 이번 Case011에서 실행하지 않았습니다.\n\n' if ko else
                 f'The completed artifacts are a shared-cache three-readout comparison, bounded supervision of 1,158 parameters, two-tensor patches with fresh-process replay, and independent public-prediction audits. Under INT8, MIXED−ORIGINAL changes in uninterrupted correct lifetime were {delta_text} tokens for seeds 0/1/2, with all primary intervals below zero. Gold CE improved beyond position 256, but correctness from the start did not last longer. This result concerns the current feature, final Linear and fitting budget; it is not a proof that every decoder must fail to recover information.\n\nThis is a local review candidate. No Case011 staging, commit, push, PR, merge, Pages or Drive upload was performed.\n\n')
        text += h(9) + '- [Case010 version map](../010-ckda-finite-precision-memory-horizon/VERSION_MAP.md) · [fixed precision/readout source](../010-ckda-finite-precision-memory-horizon/versions/v2/source/precision.py).\n- OpenEuroLLM / ComplexKDA, pinned commit `ef9d108d1692387cae37f5b2d539a71826a127c1`; original model/task and coefficients.\n- John Hewitt and Percy Liang (2019), [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/). Control motivation; this study does not reproduce that paper’s experimental control task.\n- [Protocol and source freeze](configs/protocol.json) · [selected heads](configs/selected_heads.json) · [scalar aggregate](results/derived/aggregate.json) · [head diagnostics](results/head_diagnostics.json) · [NOTICE](NOTICE.md).\n\n'
        text += ('Munsik Kim의 연구·구현에 Codex가 코드·검사·분석·문서 작성을 지원했습니다. 기존 모델, recurrence/codec, 수치 라이브러리와 새로운 adapter·fitting·patch·검산 도구의 기여를 [NOTICE](NOTICE.md)에 구분했습니다.\n' if ko else
                 'Codex assisted Munsik Kim with implementation, checks, analysis and documentation. [NOTICE](NOTICE.md) distinguishes the original model, recurrence/codecs and numerical libraries from the new adapter, fitting, patch and audit integration.\n')
        (case / ('README.ko.md' if ko else 'README.md')).write_text(readme)
        (case / report_name).write_text(text)
    return {'written': ['README.md','README.ko.md','REPORT.md','REPORT.ko.md'], 'cells':6,
            'timing_included': timing is not None, 'positive_primary_intervals':positive, 'negative_primary_intervals':negative}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, default=CASE)
    args = p.parse_args()
    print(json.dumps(write(args.case)))
