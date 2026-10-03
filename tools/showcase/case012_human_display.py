"""Static human-review summaries and per-image labels, separate from the AI record."""
import re
from html import escape
from common import require

SETTINGS=('A_time','B_time','C_time')


def human_results(data, lang):
    d=data['human_review'];t=lambda en,ko:ko if lang=='ko' else en
    out='<div id="human-results"><h3>'+t('Completed human checklist','완료한 사람 체크')+'</h3><p>'+t(
        f'All 192 images have answers. The user directly checked {d["label_provenance"]["direct_images"]} images and explicitly applied those answers to {d["label_provenance"]["reused_images"]} visually similar images of the same request.',
        f'전체 192장에 답이 있습니다. 사용자가 직접 체크한 {d["label_provenance"]["direct_images"]}장의 답을 같은 요청의 비슷한 사진 {d["label_provenance"]["reused_images"]}장에도 적용했습니다.')+'</p><div class="study-table"><table><thead><tr><th>Loop / Step</th><th>'+t('All conditions met · Human','모든 조건 충족 · 사람')+'</th><th>'+t('Median request time','요청 시간 중앙값')+'</th></tr></thead><tbody>'
    for s in SETTINGS:
        q=d['settings'][s];tm=data['timing'][s]
        out+=f'<tr><th scope="row">L{tm["loops"]} / S{tm["steps"]}</th><td>{q["passed_images"]}/{q["images"]}<br>{100*q["all_constraint_pass_rate"]:.2f}%</td><td>{d["timing"][s]["median_complete_seconds"]:.3f} s</td></tr>'
    p=d['paired'];out+='</tbody></table></div><p>'+t(
        f'L4 gained {p["C_only"]} passes and lost {p["A_only"]} versus L1: a net difference of one image ({100*d["primary"]["estimate"]:.4f} percentage points). L2 had the shortest median time. These counts do not establish a deeper-loop advantage.',
        f'L4는 L1과 비교해 {p["C_only"]}개를 얻고 {p["A_only"]}개를 잃어, 순차이는 1장({100*d["primary"]["estimate"]:.4f}%p)이었습니다. 요청 시간 중앙값은 L2가 가장 짧았습니다. 이 장수 차이만으로 깊은 반복의 우위를 확정하기는 어렵습니다.')+'</p><p>'+t(
        'Combined requests were hardest: all conditions passed in roughly half of their images. This checklist scores count, requested colors and spatial relations. Realism and the natural shape of identifiable objects are not separately scored.',
        '복합 조건이 가장 어려웠고, 모든 조건을 맞힌 이미지는 약 절반이었습니다. 이번 점수는 개수·요청한 색·공간 관계를 평가합니다. 알아볼 수 있는 물체의 형태가 자연스러운지나 사진의 사실성은 별도로 채점하지 않습니다.')+'</p><details><summary>'+t('Category counts, reused answers and uncertainty','요청 종류별 결과·묶음 답·불확실 항목 보기')+'</summary><div class="study-table"><table><thead><tr><th>'+t('Request','요청 종류')+'</th><th>L1</th><th>L2</th><th>L4</th></tr></thead><tbody>'
    names={'count':t('Count','개수'),'color_binding':t('Object-color binding','대상별 색'),'left_right':t('Left/right','좌우'),'compound':t('Combined','복합 조건')}
    for cat,name in names.items():
        out+='<tr><th scope="row">'+name+'</th>'+''.join(f'<td>{d["categories"][cat][s]["passed_images"]}/{d["categories"][cat][s]["images"]}</td>' for s in SETTINGS)+'</tr>'
    origins=d['label_provenance'];out+='</tbody></table></div><p>'+t(
        f'Of 480 answers, {origins["direct_constraints"]} were direct and {origins["reused_constraints"]} reused. Three uncertain answers count as unmet. Reused judgments are not independent: no confidence interval is attached to this descriptive human comparison. The reviewer could access the prior public results.',
        f'480개 항목 중 직접 체크는 {origins["direct_constraints"]}개, 묶음 적용은 {origins["reused_constraints"]}개입니다. 판단 불확실 3개는 미충족으로 계산했습니다. 적용한 답은 독립 판독이 아니므로 이번 사람 비교에는 신뢰구간을 붙이지 않았습니다. 평가자는 기존 공개 결과를 볼 수 있었습니다.')+'</p><p>'+t(
        f'Human and AI pass judgments differ on {d["AI_comparison"]["image_pass_disagreements"]}/192 images. Both assessments are retained separately.',
        f'사람과 AI의 전체 조건 통과 여부는 {d["AI_comparison"]["image_pass_disagreements"]}/192장에서 달랐습니다. 두 답안을 별도로 보존했습니다.')+'</p></details><div class="actions">'
    links=[('case012-human-readme-'+lang+'.md.txt',t('Human review and calculation','사람 평가와 계산 안내')),
           ('case012-human-annotations.json',t('Answers and reuse sources (JSON)','답과 묶음 출처 (JSON)'))]
    out+=''.join('<a href="../sources/'+path+'">'+label+'</a>' for path,label in links)
    return out+'</div></div>'


def add_image_judgments(data, lang, html):
    """Add all 192 assigned labels without changing a source AI judgment or PNG."""
    t=lambda en,ko:ko if lang=='ko' else en
    summary=human_results(data,lang)
    html=re.sub(r'<section id="results">(.*?)</section>',lambda m:'<section id="results"><h2>'+t('Human review and recorded time','사람 평가와 기록된 시간')+'</h2>'+summary+'<details id="ai-assessment"><summary>'+t('Original AI assessment and its interval','기존 AI 평가와 신뢰구간 보기')+'</summary>'+m.group(1)+'</details></section>',html,count=1,flags=re.S)
    require('id="human-results"' in html,'Human result section missing')
    html=re.sub(r'(<h1>.*?</h1>)<p>.*?</p>',lambda m:m.group(1)+'<p>'+t(
        'The completed human checklist is shown first. Each image retains its human answers and the original AI judgment separately.',
        '완료한 사람 체크를 먼저 보여 줍니다. 각 이미지에는 사람의 답과 기존 AI 판단을 구분해 표시했습니다.')+'</p>',html,count=1,flags=re.S)
    html=html.replace('Saved results · One AI rater','Saved images · Human and AI judgments').replace('저장된 이미지 · 한 AI의 평가','저장된 이미지 · 사람과 AI의 평가')
    html=html.replace('there are zero human raters','the original AI assessment had zero human raters').replace('인간 검증은 미수행입니다.','기존 AI 평가 단계에는 사람 판독이 없었고, 이번 사람 체크는 별도로 표시합니다.')
    html=html.replace('Method and CPU audit','Original AI method and CPU audit').replace('Human verification is not performed.','The original AI stage had no human labels; the completed user review is shown separately above.').replace('평가 방법과 직접 검산','기존 AI 평가 방법과 직접 검산')
    rows=data['human_review']['per_image'];seen=set()
    def card(match):
        body=match.group(1);ids=set(re.findall(r'assets/case012-images/([0-9a-f]{24})\.png',body))
        require(len(ids)==1,'Human viewer image mapping');image_id=ids.pop();require(image_id not in seen,'Duplicate human viewer image');seen.add(image_id)
        row=rows[image_id];origin=t('Direct human check','사람 직접 체크') if not row['reused_constraints'] else t('Human answers reused from a visual group','비슷한 사진의 사람 답 적용')
        values={'satisfied':t('Satisfied','충족'),'not_satisfied':t('Not satisfied','미충족'),'uncertain':t('Uncertain','판단 불확실')}
        labels=' · '.join(str(int(c[1:]))+t(': ', '번: ')+values[v] for c,v in sorted(row['values'].items()))
        body=body.replace('<p><strong>','<p><strong>AI: ',1)
        note='<div class="human-image-label" data-human-image="'+image_id+'"><p><strong>'+t('Human: ','사람: ')+t('All conditions met','모든 조건 충족')*row['passed']+t('Check unmet items','미충족 항목 있음')*(not row['passed'])+'</strong><br>'+escape(origin)+'</p><details><summary>'+t('Human checklist','사람 체크 항목')+'</summary><p>'+escape(labels)+'</p></details></div>'
        return '<article class="card">'+body+note+'</article>'
    html=re.sub(r'<article class="card">(.*?)</article>',card,html,flags=re.S)
    require(seen==set(rows),'Incomplete human image viewer coverage')
    html=html.replace('</head>','<style>.human-image-label{border-top:1px solid #d8d3cb;margin-top:12px;padding-top:8px}#results table{width:100%;border-collapse:collapse}#results th,#results td{padding:12px 8px;text-align:left}#results .study-table{overflow:auto}#results details{margin:16px 0}#results a{display:inline-block;margin:6px 12px 6px 0}</style></head>')
    return html
