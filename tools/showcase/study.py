"""Source-checked study explanations and tables, separate from item filters."""
from __future__ import annotations
from html import escape
from pathlib import Path
from statistics import mean, median
from common import C6, C7, SUP, read, require, safe_name, sha
from data import source_url

SECTION_IDS = ('objective','data','theory','design','validation','results','interpretation','conclusion')

def method_labels(lang: str) -> dict[str,str]:
    if lang=='ko':
        return {'B':'B · 원래 기준선','INDEPENDENT':'INDEPENDENT · 개별 점수 선택',
                'PAIRWISE':'PAIRWISE · 상호작용 포함 선택','A_PUBLIC':'A_PUBLIC · FP8 값 결합',
                'V4':'V4 · FP16 값 결합'}
    return {'B':'B · original baseline','INDEPENDENT':'INDEPENDENT · individual scores',
            'PAIRWISE':'PAIRWISE · interactions included','A_PUBLIC':'A_PUBLIC · FP8 value combination',
            'V4':'V4 · FP16 value combination'}


def method_key(case: str, lang: str) -> str:
    keys=('B','INDEPENDENT','PAIRWISE') if case=='007' else ('B','A_PUBLIC','V4')
    labels=method_labels(lang)
    text=' / '.join(labels[k] for k in keys)
    return '<p class="method-key">'+escape(text)+' · <a href="#objective">'+('이 방법은 무엇인가?' if lang=='ko' else 'What do these methods mean?')+'</a></p>'

def content(root: Path, lang: str, case: str) -> tuple[dict, dict]:
    manifest = read(root/'presentation/studies/sources.json')
    for path, digest in manifest['files'].items():
        safe_name(path)
        f = root/path
        require(f.is_file() and not f.is_symlink() and sha(f) == digest, 'Changed study source: '+path)
    versions = {v:read(root/f'presentation/studies/{v}.json') for v in ('en','ko')}
    require(set(versions['en']) == set(versions['ko']) == {'case006','case007'}, 'Missing study language/case')
    for version in versions.values():
        for c in version.values():
            require(tuple(s['id'] for s in c['sections']) == SECTION_IDS, 'Study section order/coverage')
            require(all(s['paragraphs'] and s['title'] for s in c['sections']), 'Empty study section')
            require(all(s['path'] in manifest['files'] for s in c['sources']), 'Unverified study source')
    return versions[lang][case], manifest

def paired_counts(rows: list[dict]) -> dict:
    """Display counts; unique prompts within exactly one set/arm/readout/budget."""
    require(bool(rows), 'Missing result comparison')
    require(len({(r['set'],r['arm'],r['readout'],r['budget']) for r in rows}) == 1,
            'Do not pool sets, arms, readouts or budgets')
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate scenario in study table')
    n = len(rows); bc = sum(r['B']['correct'] for r in rows)
    result = {'n':n,'B_correct':bc,'candidate_correct':sum(r['candidate']['correct'] for r in rows),
              'flips':sum(r['flip'] for r in rows), 'wrong_to_wrong':sum(r['wrong_to_wrong'] for r in rows),
              'B_ties':sum(len(r['B']['top'])>1 for r in rows),
              'candidate_ties':sum(len(r['candidate']['top'])>1 for r in rows),
              'delta_nll':mean(r['delta_nll'] for r in rows),
              'full_kl':mean(r['full_kl'] for r in rows) if all(r['full_kl'] is not None for r in rows) else None}
    for cell in ('both_correct','regression','gain','both_wrong'):
        result[cell] = sum(r['cell']==cell for r in rows)
    require(sum(result[c] for c in ('both_correct','regression','gain','both_wrong'))==n,'Missing correctness cell')
    require(result['flips']==result['regression']+result['gain']+result['wrong_to_wrong'],'Incorrect flip partition')
    require(result['B_correct']==result['both_correct']+result['regression'],'Incorrect B denominator')
    return result

def tables(data: dict, lang: str) -> list[dict]:
    """Use unrounded retained scalars/option records; never copy headline strings."""
    ko = lang=='ko'; s=data['summary']; records=data['records']
    def label(en, kr):return kr if ko else en
    def t(key,title,headers,rows,note='',source=''):
        return dict(key=key,title=title,headers=headers,rows=rows,note=note,source=source)
    def rows_for(arm,setname='HELD_OUT',readout='H_NATIVE',budget='4'):
        return [r for r in records if r['arm']==arm and r['set']==setname and r['readout']==readout and (data['case']!='007' or r['budget']==budget)]
    def fmt(x,n=6):return f'{x:.{n}f}'
    def ratio(a,b):return f'{a}/{b}' if b else label('Undefined (denominator 0)','정의 불가 (분모 0)')
    def interval(x,factor=1):return '['+', '.join(fmt(v*factor) for v in x)+']'
    out=[]
    if data['case']=='007':
        local=s['local']['HELD_OUT']
        out.append(t('local',label('Local reconstruction · lower is closer to B','국소 재구성 · 낮을수록 B에 가까움'),[label('Measure','지표'),'INDEPENDENT','PAIRWISE'],[
            [f'{int(b)*100//16}% · '+label('mean error','평균 오차'),*[fmt(local[m+'_'+b]['relative_error']['mean']*100,4)+'%' for m in ('INDEPENDENT','PAIRWISE')]] for b in ('4','8')],source=C7+'/results/derived/summary.json'))
        out.append(t('transfer',label('Paired change · PAIRWISE minus INDEPENDENT','쌍별 변화 · PAIRWISE−INDEPENDENT'),[label('Budget','삭제 예산'),label('Mean change (percentage points)','평균 변화 (%p)'),label('95% interval (%p)','95% 구간 (%p)')],[
            [f'{int(b)*100//16}%',fmt(s['transfer'][b]['mean_paired_relative_error_delta']*100),interval(s['transfer'][b]['ci95'],100)] for b in ('4','8')],
            label('Paired prompt intervals; the 50% structures and outputs are identical.','입력 단위 쌍별 구간입니다. 50%는 구조와 출력이 동일합니다.'),C7+'/results/derived/summary.json'))
        counts=[paired_counts(rows_for(m)) for m in ('INDEPENDENT','PAIRWISE')]
        specs=[(label('Correct choices','정답 수'),lambda x:ratio(x['candidate_correct'],x['n'])),
               (label('Choice flips vs B','B 대비 선택 변경'),lambda x:ratio(x['flips'],x['n'])),
               (label('Regressions / B-correct','정답 손실 / B 정답'),lambda x:ratio(x['regression'],x['B_correct'])),
               (label('Gains / B-wrong','정답 획득 / B 오답'),lambda x:ratio(x['gain'],x['n']-x['B_correct'])),
               (label('Different wrong choices / all','다른 오답 변경 / 전체'),lambda x:ratio(x['wrong_to_wrong'],x['n'])),
               ('KL(B ∥ candidate) · '+label('mean','평균'),lambda x:fmt(x['full_kl'])),
               ('Δ'+label('gold-choice NLL','정답 선택지 NLL')+' · nats',lambda x:fmt(x['delta_nll']))]
        out.append(t('quality',label('25% deletion · baseline fidelity and gold scores','25% 삭제 · 기준선 충실도와 정답 점수'),[label('Measure','지표'),'INDEPENDENT','PAIRWISE'],[[name,*[fn(x) for x in counts]] for name,fn in specs],
            label('B: 94/192 correct, 98/192 wrong. ΔNLL=candidate−B; negative is better. KL measures closeness to B and was computed from full vectors; public option scores cannot reconstruct full KL.','B는 정답 94/192, 오답 98/192입니다. ΔNLL=후보−B이며 음수가 더 좋습니다. KL은 B와의 분포 차이입니다. 전체 벡터로 계산한 저장값이며 공개 네 선택지 점수로 전체 KL을 복원할 수 없습니다.'),C7+'/results/derived/pairs.json'))
        rand=[]
        for b in ('4','8'):
            values=[v['relative_error']['mean']*100 for k,v in local.items() if k.startswith('RANDOM_'+b+'_')]
            require(len(values)==20,'Incomplete frozen random reference')
            rand.append([f'{int(b)*100//16}%',str(len(values)), ' / '.join(fmt(x,4)+'%' for x in (min(values),median(values),max(values)))])
        out.append(t('random',label('Frozen random sets · local reference only','고정 무작위 집합 · 국소 비교만 측정'),[label('Budget','삭제 예산'),label('Sets','집합 수'),label('Set-mean error: min / median / max','집합별 평균 오차: 최소 / 중앙 / 최대')],rand,
            label('First average each set over the same 192 inputs, then summarize the 20 means. These are not 3,840 independent inputs.','같은 192개 입력에서 집합별 오차를 먼저 평균한 뒤 20개 평균을 요약했습니다. 독립 입력 3,840개가 아닙니다.'),C7+'/results/derived/summary.json'))
        out.append(t('cost-summary',label('Measured cost · fixed six-input subset','측정 비용 · 고정 입력 6개'),[label('Configuration','설정'),label('One MLP: ratio [95% CI]','한 층 MLP: 속도비 [95% 구간]'),label('Full prefill: ratio [95% CI]','전체 prefill: 속도비 [95% 구간]')],[[m,*[fmt(s['timing'][scope][m]['speedup_median'],3)+'× '+ '['+', '.join(fmt(v,3) for v in s['timing'][scope][m]['ci95'])+']' for scope in ('MLP','MODEL_PREFILL')]] for m in ('INDEPENDENT_4','PAIRWISE_4','INDEPENDENT_8','PAIRWISE_8')],
            label('Ratio=B time / candidate time; above 1 is faster. These are configuration-level timings, not times for the item selected below. Every full-prefill interval includes 1.','속도비=B 시간/후보 시간이며 1보다 크면 빠릅니다. 설정별 측정으로, 아래에서 고른 개별 항목의 시간이 아닙니다. 전체 prefill 구간은 모두 1을 포함합니다.'),C7+'/results/derived/summary.json'))
    else:
        for setname,title in [('standard',label('Original readout · standard 192','원래 출력 계산 · 표준 192개')),('boundary_pool',label('Original readout · selected stress 46','원래 출력 계산 · 선정 스트레스 46개'))]:
            counts=[paired_counts(rows_for(m,setname)) for m in ('A_PUBLIC','V4')]
            specs=[(label('Correct choices','정답 수'),lambda x:ratio(x['candidate_correct'],x['n'])),(label('Choice flips vs B','B 대비 선택 변경'),lambda x:ratio(x['flips'],x['n'])),(label('Regressions / B-correct','정답 손실 / B 정답'),lambda x:ratio(x['regression'],x['B_correct'])),(label('Gains / B-wrong','정답 획득 / B 오답'),lambda x:ratio(x['gain'],x['n']-x['B_correct'])),(label('Different wrong choices / all','다른 오답 변경 / 전체'),lambda x:ratio(x['wrong_to_wrong'],x['n']))]
            out.append(t(setname,title,[label('Measure','지표'),'A_PUBLIC','V4'],[[name,*[fn(x) for x in counts]] for name,fn in specs],
                ('B '+label('correct','정답')+': '+ratio(counts[0]['B_correct'],counts[0]['n'])+'. '+label('Stress is baseline-conditioned and is not pooled with standard.','스트레스는 B 점수에 조건화한 집합으로 표준 집합과 합산하지 않습니다.')),C6+'/results/study/summary.json'))
        values=[s['groups']['standard/L4096']['arms'][a]['task_balanced']['delta_choice_nll'] for a in ('A_PUBLIC','V4')]
        out.append(t('nll',label('Standard · task-balanced paired gold-choice NLL change','표준 · 과제별 동일 가중치의 정답 선택지 NLL 쌍별 변화'),[label('Setting','설정'),label('Mean ΔNLL (nats)','평균 ΔNLL (nats)'),label('95% interval','95% 구간')],[[a,fmt(v['mean']),interval(v['ci95'])] for a,v in zip(('A_PUBLIC','V4'),values)],
            label('Candidate minus B; positive is worse. Both intervals include zero; this does not establish equivalence.','후보−B이며 양수는 더 나쁩니다. 두 구간 모두 0을 포함하지만 동등성을 입증하지 않습니다.'),C6+'/results/study/summary.json'))
        for setname,title in [('standard',label('Post-hoc FP32 readout · same standard 192','사후 FP32 출력 계산 · 같은 표준 192개')),('boundary_pool',label('Post-hoc FP32 readout · same selected stress 46','사후 FP32 출력 계산 · 같은 선정 스트레스 46개'))]:
            counts=[paired_counts(rows_for(m,setname,'H_FP32')) for m in ('A_PUBLIC','V4')]
            specs=[(label('Choice flips vs FP32 B','FP32 B 대비 선택 변경'),lambda x:ratio(x['flips'],x['n'])),(label('Regressions / FP32 B-correct','정답 손실 / FP32 B 정답'),lambda x:ratio(x['regression'],x['B_correct'])),(label('Gains / FP32 B-wrong','정답 획득 / FP32 B 오답'),lambda x:ratio(x['gain'],x['n']-x['B_correct'])),(label('Current B top ties','현재 B 최고점 동률'),lambda x:ratio(x['B_ties'],x['n'])),(label('Current candidate top ties','현재 후보 최고점 동률'),lambda x:ratio(x['candidate_ties'],x['n']))]
            out.append(t('shadow-'+setname,title,[label('Measure','지표'),'A_PUBLIC','V4'],[[name,*[fn(x) for x in counts]] for name,fn in specs],
                label('POST-HOC · SAME INPUTS. Candidate and B both use FP32 readout here. These views add no independent scenarios.','사후 진단 · 같은 입력. 이 표의 후보와 B는 모두 FP32 출력 계산을 사용합니다. 독립 시나리오가 추가된 것이 아닙니다.'),SUP+'/results/derived/summary.json'))
        out.append(t('cost-summary',label('Original model prefill · 12 fixed prompts','원래 모델 prefill · 고정 입력 12개'),[label('Setting','설정'),label('B / candidate time','B / 후보 시간'),label('95% interval','95% 구간')],[[a,fmt(s['model_timing'][a]['paired_ratio_median'],4)+'×',interval(s['model_timing'][a]['ci95'])] for a in ('A_PUBLIC','V4')],
            label('Three process rounds; model prefill, not operator-only cost or shadow-readout timing. Strict structured generation: 0/24 for every arm.','3개 프로세스 라운드의 모델 prefill입니다. 단독 attention 호출이나 사후 출력 head의 시간이 아닙니다. 엄격한 구조화 생성은 모든 설정에서 0/24였습니다.'),C6+'/results/study/summary.json'))
    labels=method_labels(lang)
    for table in out:table['headers']=[labels.get(h,h) for h in table['headers']]
    return out

def table_html(table: dict, revision: str, lang: str) -> str:
    title=table.get('title',''); key=table.get('key','dataset')
    h=f'<div class="study-table" data-table="{escape(key)}">'
    if title:h+='<h3>'+escape(title)+'</h3>'
    if key=='method-definitions':
        h+='<div class="method-cards">'
        for row in table['rows']:
            require(len(row)==len(table['headers']), 'Method column mismatch')
            h+='<div class="method-card"><h4>'+escape(row[0])+'</h4><dl>'
            for title,value in zip(table['headers'][1:],row[1:]):
                h+='<div><dt>'+escape(title)+'</dt><dd>'+escape(value)+'</dd></div>'
            h+='</dl></div>'
        h+='</div>'
    else:
        h+='<div class="scroll" tabindex="0" role="region" aria-label="'+escape(title or ('Dataset' if lang=='en' else '데이터셋'))+'"><table><thead><tr>'
        h+=''.join('<th scope="col">'+escape(str(x))+'</th>' for x in table['headers'])+'</tr></thead><tbody>'
        for row in table['rows']:
            require(len(row)==len(table['headers']),'Table column mismatch')
            h+='<tr>'+''.join(('<th scope="row">' if i==0 else '<td>')+escape(str(x))+('</th>' if i==0 else '</td>') for i,x in enumerate(row))+'</tr>'
        h+='</tbody></table></div>'
    if table.get('note'):h+='<p class="table-note">'+escape(table['note'])+'</p>'
    if table.get('source'):h+='<a class="study-source" href="'+escape(source_url(revision,table['source']),quote=True)+'">'+(('방법 구현·원문' if lang=='ko' else 'Method source') if key=='method-definitions' else ('수치 원자료' if lang=='ko' else 'Source values'))+' ↗</a>'
    return h+'</div>'

def result_html(data: dict, lang: str) -> str:
    return method_key(data['case'],lang)+''.join(table_html(t,data['revision'],lang) for t in tables(data,lang))

def render(root: Path, data: dict, lang: str) -> str:
    c,manifest=content(root,lang,'case'+data['case']);ko=lang=='ko'
    h='<div class="page-intro study-intro"><p class="eyebrow">CASE '+data['case']+' · QWEN3-0.6B · RTX 5080</p><h1>'+escape(c['title'])+'</h1><p class="lede">'+escape(c['intro'])+'</p><p class="study-see">'+escape(c['see'])+'</p><div class="actions"><a class="button" href="#objective">'+('실험 순서대로 읽기' if ko else 'Read the study')+'</a><a href="#explorer">'+('입력별 결과 바로 보기' if ko else 'Jump to individual results')+'</a></div></div>'
    h+='<div class="study-layout"><aside class="study-toc"><p>'+('이 실험의 목차' if ko else 'In this study')+'</p><nav aria-label="'+('실험 목차' if ko else 'Study contents')+'"><ol>'
    for i,s in enumerate(c['sections'],1):h+='<li><a href="#'+s['id']+'"><span>'+f'{i:02}'+'</span> '+escape(s['title'])+'</a></li>'
    h+='<li><a href="#explorer"><span>09</span> '+('입력별 결과 보기' if ko else 'Inspect an input')+'</a></li></ol></nav></aside><article class="study-article">'
    for i,s in enumerate(c['sections'],1):
        h+='<section class="study-section" id="'+s['id']+'"><h2><span>'+f'{i:02}'+'</span> '+escape(s['title'])+'</h2>'
        for p in s['paragraphs']:h+='<p>'+escape(p)+'</p>'
        if 'table' in s:h+=table_html(s['table'],manifest['source_revision'],lang)
        if s.get('tables'):h+=result_html(data,lang)
        h+='</section>'
    h+='<div class="study-sources"><h3>'+('원문과 검산 경로' if ko else 'Original evidence and checks')+'</h3><ul>'
    for s in c['sources']:h+='<li><a href="'+escape(source_url(manifest['source_revision'],s['path']),quote=True)+'">'+escape(s[lang])+'</a></li>'
    h+='</ul></div></article></div>'
    h+='<section class="explorer-intro" id="explorer"><p class="eyebrow">09 · '+('입력별 결과 탐색' if ko else 'ITEM EXPLORER')+'</p><h2>'+('이제 같은 입력의 결과를 직접 비교하세요' if ko else 'Compare the same input across settings')+'</h2><ol>'
    for p in (['평가 집합·과제·삭제 예산 또는 출력 계산을 고릅니다. 아래 숫자는 현재 필터의 분모를 사용합니다.','입력 ID를 선택한 뒤 프롬프트와 별도로 계산한 정답을 확인합니다. 길이 조절용 무관한 채움 내용은 화면에서 생략되며 원문 링크로 확인할 수 있습니다.','같은 입력의 설정별 선택·정답 확률·국소 오차를 비교합니다. JSON 다운로드에는 원래 정밀도의 값과 출처가 들어 있습니다.'] if ko else ['Choose a set, task and pruning budget or readout. Counts below use the current filter’s denominator.','Select an input ID and read its prompt and independently computed gold answer. Irrelevant length filler is omitted from the display; the source link retains the full prompt.','Compare choices, gold probabilities and local errors for that same input. Download JSON for retained precision and source identifiers.']):h+='<li>'+escape(p)+'</li>'
    return h+'</ol>'+method_key(data['case'],lang)+'</section>'
