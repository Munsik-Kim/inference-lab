"""Display-only Korean corrections and compact browsing of preserved image pairs."""
import re
from html import escape
from common import read, require
from case012 import C12

COPY_PATH = 'presentation/content/case012-ko.json'


def review_copy(root, data):
    copy = read(root/COPY_PATH)
    require(copy['schema'] == 'case012-korean-display-v1', 'Case012 translation schema')
    source = read(root/C12/'configs/prompts.json')
    # The source is a keyed split dictionary in the frozen study.
    def collect(value):
        if isinstance(value, list):
            return [p for v in value for p in collect(v)]
        if isinstance(value, dict):
            if value.get('split') == 'MAIN' and 'constraints' in value: return [value]
            return [p for v in value.values() for p in collect(v)]
        return []
    prompts = collect(source)
    require(len(prompts) == 16 and set(copy['prompts']) == {p['prompt_id'] for p in prompts}, 'Case012 translation coverage')
    groups = []
    for p in prompts:
        text = copy['prompts'][p['prompt_id']]
        require(set(text['constraints']) == {c['id'] for c in p['constraints']}, 'Case012 translated constraint IDs')
        groups.append({'prompt_id':p['prompt_id'], 'english':p['english'], 'ko':text,
                       'constraints':[{**c, 'ko':text['constraints'][c['id']]} for c in p['constraints']]})
    items = {r['image_id']:r['prompt_id'] for r in data['pairs']}
    require(len(items) == 192 and all(sum(v==g['prompt_id'] for v in items.values())==12 for g in groups), 'Case012 grouped image coverage')
    return {'schema':'case012-review-copy-v1', 'groups':groups, 'items':items,
            'purpose':'Display-only wording and grouping. Original annotation image-set identity/order stays unchanged.'}


def polish_viewer(root, data, lang, html):
    """Keep every source-backed judgment and image; collapse repetitive comparisons."""
    ko = lang == 'ko'
    copy = review_copy(root,data)
    original = read(root/C12/'analysis/annotation_items.json')['items']
    by_english = {i['english']:i for i in original if i['split']=='MAIN'}
    for ordinal,g in enumerate(copy['groups'],1):
        old = by_english[g['english']]
        title = f'{ordinal:02d}. '+(g['ko']['title'] if ko else g['english'].split('. ')[0])
        pid = g['prompt_id']
        html = html.replace(f'<option value="{pid}">{pid}</option>',f'<option value="{pid}">{escape(title)}</option>')
        # Repeated prompt descriptions occur once per request, not per generated image.
        pattern = rf'<section class="pair" id="{pid}"><h2>{pid}</h2>(.*?)</section>'
        match = re.search(pattern,html,re.S)
        require(match is not None, 'Case012 viewer prompt markup changed')
        body = match.group(1); number = [0]
        if ko:
            body = body.replace(escape(old['korean_display']),escape(g['ko']['request']))
            for c in old['constraints']:
                body = body.replace(escape(c['ko']),escape(g['ko']['constraints'][c['id']]))
        def comparison(m):
            number[0] += 1; n = number[0]
            label = f'비교 {n}/4 · 같은 초기 잡음의 세 설정' if ko else f'Comparison {n}/4 · Three settings, one starting noise'
            return f'<details class="comparison" id="{pid}-{m.group(1)}"><summary>{label}</summary>{m.group(2)}</details>'
        body = re.sub(r'<h3>Seed (\d+)</h3>(<div class="grid">.*?</div>)',comparison,body,flags=re.S)
        require(number[0]==4, 'Case012 comparison count mismatch')
        replacement = f'<details class="pair" id="{pid}"><summary>{escape(title)}</summary>'+body+'</details>'
        html = html[:match.start()]+replacement+html[match.end():]
    if ko:
        replacements = {
            '생성 시간 배분과 전체 이미지 평가':'같은 요청으로 만든 이미지 비교',
            '모든 입력 비교':'요청별 이미지 비교', '문장 선택':'보고 싶은 요청', '전체 16문장':'전체 요청 목록 (16개)',
            '요청 중앙 시간':'요청 시간 중앙값', '모든 항목 충족':'평가 항목 모두 충족',
            '동결 평가 항목':'이 요청의 평가 항목', '저장된 결과 · AI 1개 판독자':'저장된 이미지 · 한 AI의 평가', 'AI 판독 근거':'AI 평가 메모 (영문 원문)',
            '판단 어려움':'판단 불확실',
            '설정당 64장과 항목별 미충족·판단 어려움을 모두 표시합니다. 썸네일을 누르면 원본 PNG가 열립니다.':'요청 하나를 펼치고 비교 1–4 중 하나를 고르세요. 같은 문장과 초기 잡음으로 만든 세 이미지를 나란히 보여 줍니다. 모든 64개 비교를 확인할 수 있으며, 이미지를 누르면 원본이 열립니다.',
            'AI가 192장 모두를 판독했습니다.':'AI가 서로 다른 원본 이미지 192장을 평가했습니다.',
            'JavaScript 없이도 아래 전체 결과를 읽을 수 있습니다.':'JavaScript를 꺼도 아래 요청과 비교 항목을 펼쳐 전체 결과를 볼 수 있습니다.'}
        for before,after in sorted(replacements.items(),key=lambda pair:-len(pair[0])):html=html.replace(before,after)
    else:
        html=html.replace('All 16 prompts','All requests (16)').replace('All paired inputs','Browse by request')
    if ko:
        html=re.sub(r'<strong>(.*?)</strong>',lambda m:'<strong>'+re.sub(r'\bc([1-4])\b',r'\1번 항목',m.group(1))+'</strong>',html,flags=re.S)
        html=re.sub(r'<li>c([1-4]):',r'<li>\1.',html)
        primary=data['quality']['primary']
        caption=(f'같은 요청과 초기 잡음으로 만든 64쌍을 비교했습니다. L4−L1의 조건 충족 비율 차이는 {100*primary["estimate"]:+.3f}%p이며, 95% 불확실성 구간은 [{100*primary["interval"][0]:+.3f}, {100*primary["interval"][1]:+.4f}]%p입니다. 구간에 0이 포함되어 더 깊은 반복이 낫다고 확정하기 어렵습니다. 실제 요청 시간도 완전히 같지는 않습니다. 사람의 평가와는 별도로 집계한 결과입니다.')
        html=re.sub(r'<p>같은 64개 문장·초기 잡음 쌍\..*?</p>','<p>'+caption+'</p>',html,count=1,flags=re.S)
    # Exactly one comparison is initially visible, including without JavaScript.
    html=html.replace('<details class="pair"', '<details open class="pair"',1)
    html=html.replace('<details class="comparison"', '<details open class="comparison"',1)
    style='<style>.pair,.comparison{border:1px solid #d8d3cb;border-radius:8px;padding:16px;margin:14px 0;background:#fff}.pair>summary,.comparison>summary{cursor:pointer;font-weight:600;line-height:1.6}.pair[open]>summary,.comparison[open]>summary{margin-bottom:18px}.pair[hidden]{display:none!important}</style>'
    return html.replace('</head>',style+'</head>')
