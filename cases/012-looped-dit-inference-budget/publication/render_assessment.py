"""Add the supplemental AI assessment to bilingual documents and an offline view.

Display numbers come from assessment_v1/summary.json. Original scientific data,
HTML demos and pending-human summary remain the historical snapshot.
"""
import html
import json
from pathlib import Path

NAMES = ('A_time', 'B_time', 'C_time')
CATEGORIES = {
    'count': ('개수', 'Count'), 'color_binding': ('대상별 색', 'Object colors'),
    'left_right': ('좌우 관계', 'Left/right'), 'compound': ('복합 조건', 'Compound'),
}


def percent(x):
    return f'{100*x:.2f}%'


def result_table(summary, ko):
    q = summary['quality']
    header = ('| Loop / Step | 요청 시간 중앙값(초) | 모든 평가 조건 충족 — AI 판독 |'
              if ko else '| Loops / Steps | Median request time (s) | Every listed constraint met — AI assessment |')
    lines = [header, '|---|---:|---:|']
    for name in NAMES:
        t, s = summary['timing'][name], q['settings'][name]
        lines.append(f'| {t["loops"]} / {t["steps"]} | {t["main_complete_median_seconds"]:.3f} | {s["passed_images"]}/{s["images"]} ({percent(s["all_constraint_pass_rate"])}) |')
    return '\n'.join(lines)


def section(summary, ko):
    q = summary['quality']
    p, pair = q['primary'], q['paired']
    cat = ['| ' + ('범주' if ko else 'Category') + ' | L1/S89 | L2/S66 | L4/S50 |', '|---|---:|---:|---:|']
    for key, labels in CATEGORIES.items():
        cat.append('| ' + labels[0 if ko else 1] + ' | ' + ' | '.join(
            f'{round(q["categories"][key][s]["all_constraint_pass_rate"]*q["categories"][key][s]["n"])}/{q["categories"][key][s]["n"]}' for s in NAMES) + ' |')
    sensitivity = ', '.join(f'L{summary["timing"][s]["loops"]} {percent(q["settings"][s]["all_uncertain_as_pass_rate"])}' for s in NAMES)
    gains, losses = sum(q['constraint_gains'].values()), sum(q['constraint_losses'].values())
    if ko:
        return f'''### 저장된 MAIN 192장의 AI 평가

코드 에이전트가 원본 PNG 192장을 하나씩 판독했습니다. **AI 평가자 1개, 인간 평가자 0명**입니다. 설정·시간·seed 표시는 가렸으며, 이 세션에서 앞서 48장의 예시를 본 이력이 있어 독립적인 완전 블라인드 평가로 부르지 않습니다. 원래 사람 평가 대기 기록은 보존하고, 새 평가는 별도 [프로토콜](publication/assessment_v1/protocol.json)과 [개별 판독](publication/assessment_v1/annotations.json)으로 추가했습니다.

{result_table(summary, True)}

각 설정 64장, 동일한 64개 문장·초기 잡음 쌍입니다. 충족은 동결 checklist의 모든 항목이 `satisfied`인 경우입니다. 미적 외관이나 문장 속 모든 표현을 채점한 값은 아닙니다. 체크리스트는 개수 범주에서 개수만, 다른 범주에서 명시된 색·관계·복합 항목을 평가합니다.

**주 비교 L4−L1은 {100*p['estimate']:+.2f} percentage points**였습니다. 16개 문장 cluster를 5,000회 재표집한 95% bootstrap 구간은 [{100*p['interval'][0]:+.2f}, {100*p['interval'][1]:+.2f}] pp입니다. 0을 포함하며, 한 AI 판독자의 판단 오류는 이 구간에 포함되지 않습니다. 네 관련 template family의 작은 실험입니다.

같은 입력 64쌍에서 둘 다 충족 {pair['both_pass']}쌍, L1만 충족 {pair['A_only']}쌍, L4만 충족 {pair['C_only']}쌍, 둘 다 미충족 {pair['neither']}쌍입니다. 개별 조건에서는 획득 {gains}건·손실 {losses}건이었습니다. 이는 서로 다른 최종 이미지의 비교이며, 내부 반복이 한 이미지를 수정한 궤적은 아닙니다.

{chr(10).join(cat)}

범주별 분모는 설정당 16장입니다. `uncertain` {q['uncertain_constraint_count']}개 항목은 기본 점수에서 미충족입니다. 모두 충족으로 바꾸는 낙관적 민감도에서는 {sensitivity}가 됩니다. 개수·색·관계의 판독 근거와 모든 입력은 [전체 이미지](publication/IMAGES.ko.md)에서 확인할 수 있습니다.

가까운 시간 예산에서의 quality–time 관측입니다. DEV의 ±5% 시간 맞추기는 실패했고, MAIN의 시간도 같지 않습니다. L2는 L4와 같은 충족 장수를 더 짧은 중앙 시간에 얻었지만, 이 자료만으로 모든 문장의 품질 우위나 추천 preset을 확정하지 않습니다. [공통 결과 JSON](publication/assessment_v1/summary.json) · [독립 CPU 산술 검사](publication/assessment_v1/audit.py)
'''
    return f'''### AI assessment of all 192 saved MAIN images

A code agent inspected each original PNG individually: **one AI rater, zero human raters**. Settings, time and seed labels were masked. The session had previously seen 48 example images, so this is not independent fully blind evaluation. The original human-pending record is preserved; this supplement has its own [protocol](publication/assessment_v1/protocol.json) and [item judgments](publication/assessment_v1/annotations.json).

{result_table(summary, False)}

Each setting has 64 images from the same 64 prompt–noise pairs. A pass requires every frozen checklist item to be `satisfied`. This does not score aesthetics or every phrase in the prose prompt. Count prompts score count only; other categories score their explicit color, relation and compound items.

**Primary L4−L1: {100*p['estimate']:+.2f} percentage points**, with a 95% paired prompt-cluster bootstrap interval [{100*p['interval'][0]:+.2f}, {100*p['interval'][1]:+.2f}] pp. The interval includes zero. It resamples 16 prompts 5,000 times and does not include AI judgment error. There are four related template families in this small experiment.

Across 64 paired inputs: both pass {pair['both_pass']}, L1 only {pair['A_only']}, L4 only {pair['C_only']}, neither {pair['neither']}. Individual constraints show {gains} gains and {losses} losses. These compare final images; they are not observations of one image being corrected within a loop trajectory.

{chr(10).join(cat)}

Category denominators are 16 images per setting. The {q['uncertain_constraint_count']} `uncertain` constraint labels count as failures for the primary score. Treating all of them as passes gives the optimistic sensitivity: {sensitivity}. Inspect every image and its evidence in [the full comparison](publication/IMAGES.md).

These are quality–time observations near a budget. DEV missed the ±5% time target, and MAIN times differ too. L2 reached the same pass count as L4 at a shorter median time, without establishing general quality superiority or a quality-ranked preset. [Shared result JSON](publication/assessment_v1/summary.json) · [Independent CPU arithmetic audit](publication/assessment_v1/audit.py)
'''


def outcome(score, ko):
    values = score['values']
    if score['pass']:
        return '모든 항목 충족' if ko else 'Every item met'
    no = [k for k, v in values.items() if v == 'not_satisfied']
    unknown = [k for k, v in values.items() if v == 'uncertain']
    pieces = []
    if no:
        pieces.append(('미충족 ' if ko else 'Not met ') + ', '.join(no))
    if unknown:
        pieces.append(('판단 어려움 ' if ko else 'Uncertain ') + ', '.join(unknown))
    return '; '.join(pieces)


def apply_assessment(root):
    root = Path(root)
    pub = root / 'publication'
    summary = json.loads((pub / 'assessment_v1/summary.json').read_text())
    q = summary['quality']
    records = [r for r in json.loads((root / 'analysis/record_index.json').read_text())['rows'] if r['job']['phase'] == 'main']
    lookup = {(r['job']['prompt_id'], r['job']['seed_label'], r['job']['setting']['id']): r for r in records}
    prompts = [p for p in json.loads((root / 'configs/prompts.json').read_text())['prompts'] if p['split'] == 'MAIN']
    for ko in (False, True):
        name = 'README.ko.md' if ko else 'README.md'
        text = (root / name).read_text()
        if ko:
            start = text.index('| 내부 반복 Loop |')
            end = text.index('\n\nRTX 5080', start)
            text = text[:start] + result_table(summary, True) + text[end:]
            text = text.replace('**전체 품질 평가는 주석 대기입니다.** 아직 조건 충족률이나 품질 기반 추천 설정을 계산하지 않았습니다. 이미지가 만들어진 것과 요구를 맞힌 것은 다른 결과입니다.',
                f'**192장 전체의 AI 평가를 마쳤습니다.** 모든 명시 조건을 충족한 장수는 위 표와 같습니다. 같은 입력끼리 L4와 L1을 비교하면 {q["paired"]["C_only"]}쌍이 충족으로 바뀌고 {q["paired"]["A_only"]}쌍이 충족을 잃었습니다. 차이는 {100*q["primary"]["estimate"]:+.2f}pp이며, 불확실성 구간이 0을 포함해 더 깊은 반복의 품질 우위를 확정하지 않습니다.\n\nAI 판독자 1개 · 인간 판독 0명 · 애매한 항목은 미충족. [평가 결과와 방법](publication/assessment_v1/README.ko.md) · [입력별 판독 근거](publication/IMAGES.ko.md)')
            text = text.replace('공식 인간 주석이나 전체 정답률이 아닙니다.', '전체 AI 평가는 별도로 192장 모두 수행했으며, 이 사후 예시는 인간 주석이 아닙니다.')
            text = text.replace('시각 점검에서는 앞의 두 이미지에 정육면체 세 개, 마지막 이미지에 네 개가 보였습니다. 더 깊은 내부 반복과 더 적은 생성 단계의 조합이 이미 맞힌 개수 조건을 잃는 사례입니다.',
                '전체 checklist 판독에서는 첫 이미지가 세 정육면체 조건을 충족했습니다. 가운데 이미지는 노란 물체 세 개 중 원통 모양이 포함돼 미충족으로, 마지막 이미지는 정육면체 네 개로 미충족 판정했습니다. 앞선 예비 시각 설명에서 가운데 물체를 정육면체로 읽은 해석을 이번 항목별 판독으로 구체화했습니다.')
            text = text.replace('최종 비교는 실제 시간과 조건별 획득·손실을 함께 읽어야 합니다.', '전체 평가도 실제 시간과 조건별 획득·손실을 함께 읽어야 합니다.')
            text = text.replace('python publication/verify.py', 'python publication/verify.py\npython publication/assessment_v1/audit.py')
            text = text.replace('`demo/viewer.ko.html`', '`publication/assessment_v1/viewer.ko.html`')
        else:
            start = text.index('| Internal loops |')
            end = text.index('\n\nRTX 5080', start)
            text = text[:start] + result_table(summary, False) + text[end:]
            text = text.replace('**Quality annotations are pending.** Constraint pass rates and quality-ranked presets have not been calculated. A generated image is not automatically a correct image.',
                f'**AI assessment is complete for all 192 images.** The table shows images meeting every explicit constraint. In the same inputs, L4 gains {q["paired"]["C_only"]} passes and loses {q["paired"]["A_only"]} versus L1. The difference is {100*q["primary"]["estimate"]:+.2f}pp; its uncertainty interval includes zero, leaving deeper-loop quality superiority unestablished.\n\nOne AI rater · Zero human raters · Uncertain items count as failures. [Assessment and method](publication/assessment_v1/README.md) · [Item evidence](publication/IMAGES.md)')
            text = text.replace('not human annotation or an overall accuracy score.', 'not human annotation. The separate full AI assessment covers all 192 images.')
            text = text.replace('Visual inspection shows three cubes in the first two images and four in the last. The deeper-loop/fewer-step configuration loses a count condition met by the other two.',
                'Full checklist assessment marks the first image as meeting the three-cube condition. The middle image includes cylindrical shapes among three yellow bodies and is marked not satisfied; the last has four cubes. This refines the earlier preliminary description that treated the middle objects as cubes.')
            text = text.replace('Compare measured time with constraint gains and losses.', 'The full assessment also compares measured time with constraint gains and losses.')
            text = text.replace('python publication/verify.py', 'python publication/verify.py\npython publication/assessment_v1/audit.py')
            text = text.replace('`demo/viewer.en.html`', '`publication/assessment_v1/viewer.en.html`')
        (root / name).write_text(text)

        report_name = 'REPORT.ko.md' if ko else 'REPORT.md'
        report = (root / report_name).read_text()
        title = '### 품질 판독' if ko else '### Quality'
        a = report.index(title)
        b = report.index('<a id="s7"></a>', a)
        report = report[:a] + section(summary, ko) + '\n' + report[b:]
        if ko:
            report = report.replace('현재 모든 예정 이미지와 시간 기록을 확보했으며 품질 주석을 기다립니다.', '예정된 생성·측정을 완료하고, 저장된 MAIN 192장을 AI가 판독했습니다. L2/L4는 각각 53/64장, L1은 51/64장이 모든 명시 조건을 충족했습니다.')
            report = report.replace('품질 결과는 아직 계산하지 않았습니다.', '원래 사람 평가 대기 단계 이후, 사용자 요청에 따라 같은 checklist의 AI 판독을 별도 프로토콜로 추가했습니다. 새 생성이나 설정 선택은 없습니다.')
            report = report.replace('현재 판정 가능한 것은 생성 경로·동일 초기 잡음·측정·재현 계약입니다. 더 많은 loop가 개수·색·좌우 정답을 개선하거나 훼손했는지는 사람의 주석 후 확인합니다.', 'AI 판독에서는 L4−L1의 전체 충족률 점추정이 양수였으나 구간은 0을 포함했습니다. 개수 범주의 장수는 늘고 복합 범주의 전체 충족 장수는 같았으며, 동일 입력의 획득과 손실이 공존합니다.')
            report = report.replace('품질 기반 winner와 preset은 **ANNOTATION_PENDING**입니다.', '저장된 192장의 **AI 평가를 완료**했습니다. 주 차이는 불확실하며 품질 winner를 선언하지 않습니다. 사람 평가와 판독자 간 일치도는 미수행입니다.')
            report = report.replace('새 학습·추가 품질 sweep·GitHub/Pages 게시는 수행하지 않았습니다.', '새 학습·추가 이미지 생성은 0건입니다. 현재 공개 범위는 feature branch와 검토 PR이며 main 병합·Pages 배포는 별도입니다.')
            report = report.replace('### 저장된 기록의 추가 분석 — 이번 공개 정리', '### 첫 공개 정리의 기록 분석 — 전체 AI 평가 이전')
            report = report.replace('최종 품질 우위와 preset은 실제 blind 주석 후 원래 prompt-cluster 집계로 판단합니다.', '그 당시 품질 점수는 미계산이었습니다. 이번 전체 AI 점수는 위 별도 평가 절에서 제시하며, 원래 pending summary는 보존합니다.')
        else:
            report = report.replace('All planned MAIN and SMOKE images are generated; quality judgement awaits annotation.', 'Planned generation is complete and an AI has assessed all 192 saved MAIN images. L2/L4 each meet every listed constraint in 53/64 images, versus 51/64 at L1.')
            report = report.replace('Current evidence establishes generation, paired initial noise, measurement and reproduction contracts. Whether additional loops gain or lose requested conditions awaits human annotation.', 'AI scoring gives a positive L4−L1 point estimate, with an interval containing zero. Count-category passes increase and compound-category passes stay equal, while paired gains and losses coexist.')
            report = report.replace('Quality winners and quality-ranked presets remain **ANNOTATION_PENDING**.', '**AI assessment is complete** for all 192 saved images. The primary difference is uncertain and no quality winner is declared. Human assessment and inter-rater agreement remain unmeasured.')
            report = report.replace('No new training, extra quality sweep or GitHub/Pages publication was performed.', 'New training and image generation are zero. Current publication scope is a feature branch and review PR; main merge and Pages deployment are separate.')
            report = report.replace('### Additional analysis of saved records — publication preparation', '### Initial publication analysis — before full AI assessment')
            report = report.replace('Final quality rankings and presets await blind annotations and the original prompt-cluster aggregation.', 'Quality scores were unavailable at that stage. The separate full AI assessment is reported above, preserving the original pending summary.')
        (root / report_name).write_text(report)

        # A standalone supplement reuses the same shared table/text, with links
        # adjusted to its location. Historical reports remain in original_docs.
        head = '# ' + ('전체 이미지 AI 평가' if ko else 'Full-image AI assessment')
        counterpart = 'README.md' if ko else 'README.ko.md'
        supp = head + '\n\n' + f'[{"English" if ko else "한국어"}]({counterpart}) · [{"사례 소개" if ko else "Case overview"}](../../{name})\n\n'
        supp += section(summary, ko).replace('(publication/assessment_v1/', '(').replace('(publication/IMAGES', '(../IMAGES')
        supp += '\n```bash\npython publication/assessment_v1/audit.py\n```\n\n' + ('Case 012 폴더에서 실행합니다. 이 검사는 판독 점수의 산술·해시를 재계산하며 AI의 판단을 독립적으로 재판독하는 것은 아닙니다.' if ko else 'Run from the Case 012 directory. This checks scoring arithmetic and hashes; it does not independently re-rate the AI judgments.') + '\n'
        (pub / 'assessment_v1' / ('README.ko.md' if ko else 'README.md')).write_text(supp)

        im_name = 'IMAGES.ko.md' if ko else 'IMAGES.md'
        im = (pub / im_name).read_text()
        im = im.replace('품질 주석은 미완료입니다.', '전체 192장의 AI 판독을 완료했습니다. 인간 평가는 미수행입니다. 미충족과 판단 어려움을 그대로 표시합니다.') if ko else im.replace('Annotations are pending.', 'AI assessment covers all 192 images; human evaluation is not performed. Failed and uncertain items remain visible.')
        for p in prompts:
            sentence = p['korean_display'] if ko else p['english']
            checklist = ' · '.join(c['id'] + ': ' + c['ko' if ko else 'en'] for c in p['constraints'])
            im = im.replace(sentence + '\n', sentence + '\n\n' + ('평가 항목: ' if ko else 'Scored checklist: ') + checklist + '\n', 1)
        for r in records:
            old = f'{r["complete_seconds"]:.3f}s · ' + ('미평가' if ko else 'Not annotated')
            # Times alone can collide; use the unique original PNG URL as anchor.
            prefix = f'](../{r["relative_image_path"]}) · '
            im = im.replace(prefix + old, prefix + f'{r["complete_seconds"]:.3f}s · **{outcome(q["per_image"][r["job_id"]], ko)}**')
        im += '\n' + ('AI의 항목별 판독 근거: ' if ko else 'AI evidence for every judgment: ') + f'[annotations.json](assessment_v1/annotations.json) · [CPU {"재계산" if ko else "audit"}](assessment_v1/audit.py)\n'
        (pub / im_name).write_text(im)
        render_viewer(root, summary, lookup, prompts, ko)

    reproduction = (root / 'REPRODUCTION.md').read_text()
    reproduction = reproduction.replace('No new generation or quality labels were added.', 'The initial edition added no quality labels. A later, separately frozen AI supplement now scores every MAIN image; it changes no generation record or the original human-pending summary.')
    reproduction += '''
## 6. Completed supplemental AI assessment — no model or judge API

The AI judgments already exist in `publication/assessment_v1/annotations.json`.
These commands recalculate them; they do not ask an AI to judge new images.
One AI rater assessed all 192 MAIN images with setting labels masked, with
prior exposure to 48 examples disclosed. Human verification was not performed.
`uncertain` counts as failure in the primary score, with a separate sensitivity.

```bash
python publication/assessment_v1/audit.py
python publication/assessment_v1/analyze.py --output /your/new-scratch/ai-recalculated.json
```

Run from Case012 with the CPU environment activated. The output child must not
already exist. `analyze.py` reads the frozen original records and completed AI
labels without rewriting historical `analysis/summary.json`. `audit.py` uses a
separate scalar implementation for counts, pairing and the original bootstrap.
Arithmetic agreement is not independent visual or human verification.

Open `publication/assessment_v1/viewer.ko.html` or `viewer.en.html` locally for
the completed AI results, or read `publication/IMAGES.*.md` directly on GitHub.
The original `demo/viewer.*.html` remains the historical pre-annotation snapshot.
The annotation UI supports a future human export in a restored scratch copy;
do not overwrite the AI supplement or publish a synthetic UI test as labels.
'''
    (root / 'REPRODUCTION.md').write_text(reproduction)

    # Repository introductions keep their existing structure and design. Only
    # the already-added Case012 block and archive row change.
    repo = root.parents[1]
    if (repo / 'README.md').exists() and (repo / 'docs/en/CASEBOOK.md').exists():
        pa, pb, pc = [q['settings'][s]['passed_images'] for s in NAMES]
        delta = 100*q['primary']['estimate']
        bounds = [100*x for x in q['primary']['interval']]
        for ko in (False, True):
            name = 'README.ko.md' if ko else 'README.md'
            text = (repo / name).read_text()
            old = ('MAIN 192장의 생성·측정을 완료했습니다. Loop2/Step66이 가장 짧은 중앙 시간을 보였고, 시각 점검에서는 더 깊은 반복·더 적은 단계 조합이 맞힌 개수 조건을 잃는 사례도 보였습니다. 전체 조건 충족률은 사람의 주석을 기다립니다.' if ko else
                   'Generation and measurement are complete for 192 MAIN images. Loops 2 / Steps 66 had the shortest median; visual inspection also found a deeper-loop/fewer-step example that lost a previously met count condition. Overall constraint pass rates await human annotation.')
            new = (f'MAIN 192장의 생성·측정과 AI 판독을 마쳤습니다. 모든 평가 조건을 충족한 장수는 Loop1이 {pa}/64장, Loop2와4가 각각 {pb}/64장입니다. Loop2/Step66이 가장 짧은 중앙 시간이었고, Loop4와1의 충족률 차이는 불확실했습니다. 같은 입력에서 얻은 조건과 잃은 조건을 함께 공개합니다. AI 판독자 1개, 인간 평가 0명입니다.' if ko else
                   f'Generation, measurement and AI assessment are complete for 192 MAIN images. Every listed constraint is met in {pa}/64 images at Loops 1 and {pb}/64 each at Loops 2 and 4. Loops 2 / Steps 66 had the shortest median; the Loops 4 vs 1 pass-rate difference was uncertain. Paired gains and losses are both reported. One AI rater; zero human raters.')
            if old in text:
                text = text.replace(old, new)
            text = text.replace('품질 주석은 대기 중.', '전체192장 AI 평가와 조건별 획득·손실.') if ko else text.replace('Quality annotations are pending.', 'AI assessment of all 192 images with paired constraint gains and losses.')
            (repo / name).write_text(text)
            book = repo / ('docs/ko/CASEBOOK.md' if ko else 'docs/en/CASEBOOK.md')
            text = book.read_text()
            old = ('**해석:** 첫 고정 seed의48장에 대한 AI 시각 점검에는 더 깊은 loop·더 적은 step 조합이 맞힌 개수 조건을 잃는 사례가 있었습니다. 이는 전체 인간 평가의 정답률이 아닙니다. 조건 충족률·paired 획득/손실은 주석 대기이며, 동일 시간을 쓴 최종 품질 우위는 아직 판단하지 않습니다.' if ko else
                   '**Interpretation:** AI visual inspection of 48 images from the first frozen seed found a deeper-loop/fewer-step example that lost a previously met count condition. This is not an overall human accuracy score. Constraint pass rates and paired gains/losses await annotation; equal-time final quality superiority remains unassessed.')
            new = (f'**AI 평가:** MAIN 192장을 같은 checklist로 판독했습니다. L1 {pa}/64장, L2 {pb}/64장, L4 {pc}/64장이 모든 항목을 충족했습니다. L4−L1은 {delta:+.2f}pp, 95% prompt-cluster bootstrap 구간 [{bounds[0]:+.2f}, {bounds[1]:+.2f}]pp로 0을 포함합니다. 같은 입력에서 L4만 충족 {q["paired"]["C_only"]}쌍, L1만 충족 {q["paired"]["A_only"]}쌍입니다. AI 판독자 1개·인간 0명이며, {q["uncertain_constraint_count"]}개 애매한 항목은 기본 점수에서 미충족입니다. 이전 예시 노출과 시간 불일치를 포함한 평가 범위는 별도 [평가 문서](../../cases/012-looped-dit-inference-budget/publication/assessment_v1/README.ko.md)에 있습니다.' if ko else
                   f'**AI assessment:** all 192 MAIN images were judged with the same checklist. Every item is met in L1 {pa}/64, L2 {pb}/64 and L4 {pc}/64 images. L4−L1 is {delta:+.2f}pp, with a 95% prompt-cluster bootstrap interval [{bounds[0]:+.2f}, {bounds[1]:+.2f}]pp including zero. L4 alone passes {q["paired"]["C_only"]} pairs and L1 alone passes {q["paired"]["A_only"]}. One AI rater, zero humans; {q["uncertain_constraint_count"]} uncertain items count as failures. Prior example exposure and time mismatch are disclosed in the [assessment supplement](../../cases/012-looped-dit-inference-budget/publication/assessment_v1/README.md).')
            if old in text:
                text = text.replace(old, new)
            book.write_text(text)


def render_viewer(root, summary, lookup, prompts, ko):
    e = html.escape
    folder = root / 'publication/assessment_v1'
    lang, other = ('ko', 'en') if ko else ('en', 'ko')
    q = summary['quality']
    title = '생성 시간 배분과 전체 이미지 평가' if ko else 'Generation budget and every image assessment'
    rows = []
    for s in NAMES:
        t, v = summary['timing'][s], q['settings'][s]
        rows.append(f'<tr><th scope="row">L{t["loops"]} / S{t["steps"]}</th><td>{t["main_complete_median_seconds"]:.3f}s</td><td>{v["passed_images"]}/{v["images"]} ({percent(v["all_constraint_pass_rate"])})</td></tr>')
    articles = []
    for p in prompts:
        checklist = ''.join(f'<li>{e(c["id"])}: {e(c["ko" if ko else "en"])}</li>' for c in p['constraints'])
        seeds = []
        for seed in (72301, 72302, 72303, 72304):
            cards = []
            for s in NAMES:
                r = lookup[p['prompt_id'], seed, s]
                score = q['per_image'][r['job_id']]
                cards.append(f'<article class="card"><h3>L{r["job"]["setting"]["loops"]} / S{r["job"]["setting"]["steps"]}</h3>'
                    f'<a href="../../{e(r["relative_image_path"])}"><img loading="lazy" width="256" height="256" src="../../demo/thumbs/{e(r["job_id"])}.jpg" alt="{e(p["prompt_id"])}; {seed}; {e(s)}"></a>'
                    f'<p><strong>{e(outcome(score, ko))}</strong><br>{r["complete_seconds"]:.3f}s</p>'
                    f'<details><summary>{"AI 판독 근거" if ko else "AI judgment evidence"}</summary><p>{e(score["visible_evidence"])}</p></details>'
                    f'<a href="../../{e(r["relative_image_path"])}">{"512px 원본 PNG" if ko else "Original 512px PNG"}</a></article>')
            seeds.append(f'<h3>Seed {seed}</h3><div class="grid">{"".join(cards)}</div>')
        articles.append(f'<section class="pair" id="{e(p["prompt_id"])}"><h2>{e(p["prompt_id"])}</h2><p>{e(p["korean_display"] if ko else p["english"])}</p><details><summary>{"동결 평가 항목" if ko else "Frozen checklist"}</summary><ul>{checklist}</ul></details>{"".join(seeds)}</section>')
    options = ''.join(f'<option value="{e(p["prompt_id"])}">{e(p["prompt_id"])}</option>' for p in prompts)
    passes = [q['settings'][s]['passed_images'] for s in NAMES]
    finding = (f'AI가 192장 모두를 판독했습니다. 모든 평가 항목을 맞힌 장수는 L1 {passes[0]}장, L2 {passes[1]}장, L4 {passes[2]}장입니다. 더 깊은 반복이 언제나 요구를 더 잘 맞추지는 않았습니다.' if ko else f'An AI assessed all 192 images. Every listed constraint is met in {passes[0]} images at L1, {passes[1]} at L2 and {passes[2]} at L4. Deeper repetition did not always improve requested composition.')
    # No model or annotation mutation runs in this viewer. All results are HTML
    # before JavaScript; JS only offers an optional prompt filter.
    text = f'''<!doctype html>
<html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)} — DIOVA</title><link rel="stylesheet" href="../../demo/style.css"></head>
<body><a class="skip" href="#content">{"본문으로 이동" if ko else "Skip to content"}</a><header><div><a href="../../{'README.ko.md' if ko else 'README.md'}">DIOVA · Case 012</a><a id="language" href="viewer.{other}.html">{"English" if ko else "한국어"}</a></div></header>
<main id="content"><p class="eyebrow">{"저장된 결과 · AI 1개 판독자" if ko else "Saved results · One AI rater"}</p><h1>{e(title)}</h1><p>{finding}</p>
<nav><a href="#results">{"핵심 결과" if ko else "Results"}</a><a href="#images">{"모든 이미지" if ko else "Every image"}</a><a href="#method">{"평가 방법" if ko else "Assessment method"}</a><a href="../../{'REPORT.ko.md' if ko else 'REPORT.md'}">{"정식 보고서" if ko else "Full report"}</a></nav>
<section id="results"><h2>{"시간과 조건 충족" if ko else "Time and constraint passes"}</h2><div class="table-wrap"><table><thead><tr><th>Loop / Step</th><th>{"요청 중앙 시간" if ko else "Median request"}</th><th>{"모든 항목 충족" if ko else "Every item met"}</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p>{"같은 64개 문장·초기 잡음 쌍. L4−L1은 +3.13pp, 95% prompt-cluster 구간 −3.13~+10.94pp입니다. 실제 시간은 같지 않으며 인간 판독은 0명입니다." if ko else "Same 64 prompt–noise pairs. L4−L1 is +3.13pp, with a 95% prompt-cluster interval −3.13 to +10.94pp. Actual times differ; there are zero human raters."}</p></section>
<section id="images"><h2>{"모든 입력 비교" if ko else "Compare every input"}</h2><p>{"설정당 64장과 항목별 미충족·판단 어려움을 모두 표시합니다. 썸네일을 누르면 원본 PNG가 열립니다." if ko else "All 64 images per setting, including failed and uncertain constraints. Click a thumbnail for its original PNG."}</p><label for="prompt-filter">{"문장 선택" if ko else "Prompt filter"}</label><select id="prompt-filter"><option value="all">{"전체 16문장" if ko else "All 16 prompts"}</option>{options}</select><noscript><p>{"JavaScript 없이도 아래 전체 결과를 읽을 수 있습니다." if ko else "All results below are readable without JavaScript."}</p></noscript></section>
{"".join(articles)}
<section id="method"><h2>{"평가 방법과 직접 검산" if ko else "Method and CPU audit"}</h2><p>{"동결 checklist의 개수·색·좌우 항목만 채점했습니다. 판단 어려움 7개는 기본 점수에서 미충족입니다. 설정 표시는 가렸지만 이전 48장 예시 노출이 있어 독립 완전 블라인드 평가가 아닙니다. 인간 검증은 미수행입니다." if ko else "Only frozen count, color and relation checklist items are scored. Seven uncertain items count as failures. Settings were masked, but prior exposure to 48 examples means this is not independent fully blind evaluation. Human verification is not performed."}</p><p><a href="{'README.ko.md' if ko else 'README.md'}">{"민감도·분모·구간" if ko else "Sensitivity, denominators and intervals"}</a> · <a href="annotations.json">{"항목별 판독 기록" if ko else "Item judgments"}</a> · <a href="summary.json">{"공통 결과 JSON" if ko else "Result JSON"}</a> · <a href="audit.py">{"독립 CPU 산술 검사" if ko else "Independent CPU arithmetic audit"}</a></p><pre>python publication/assessment_v1/audit.py</pre></section><a href="#content">{"맨 위로" if ko else "Back to top"}</a></main><script src="viewer.js"></script></body></html>
'''
    # Use the common data even for the short HTML interpretation sentence.
    text = text.replace('+3.13pp', f'{100*q["primary"]["estimate"]:+.2f}pp').replace('−3.13~+10.94pp', f'{100*q["primary"]["interval"][0]:+.2f}~{100*q["primary"]["interval"][1]:+.2f}pp').replace('−3.13 to +10.94pp', f'{100*q["primary"]["interval"][0]:+.2f} to {100*q["primary"]["interval"][1]:+.2f}pp')
    (folder / f'viewer.{lang}.html').write_text(text)


if __name__ == '__main__':
    apply_assessment(Path(__file__).resolve().parents[1])
