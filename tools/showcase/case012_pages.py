"""Case012 Pages: recorded images, AI scores and a separate blank human form."""
import hashlib
import re
from html import escape
from pathlib import Path
from urllib.parse import urlsplit
from common import read, require, sha, json_text
from case012 import C12, load_case012 as load_recorded

SECTIONS = ('overview', 'loops', 'method', 'results', 'cost', 'reproduce', 'report')
SETTINGS = ('A_time', 'B_time', 'C_time')
SOURCE_COPIES = {
    'case012-report-en.md.txt': C12+'/REPORT.md',
    'case012-report-ko.md.txt': C12+'/REPORT.ko.md',
    'case012-notice.md.txt': C12+'/NOTICE.md',
    'case012-adapter.py.txt': C12+'/source/adapter.py',
    'case012-reproduction.md.txt': C12+'/REPRODUCTION.md',
    'case012-ai-audit.py.txt': C12+'/publication/assessment_v1/audit.py',
    'case012-ai-readme-en.md.txt': C12+'/publication/assessment_v1/README.md',
    'case012-ai-readme-ko.md.txt': C12+'/publication/assessment_v1/README.ko.md',
    'case012-ai-annotations.json': C12+'/publication/assessment_v1/annotations.json',
    'case012-ai-summary.json': C12+'/publication/assessment_v1/summary.json',
    'case012-rubric.json.txt': C12+'/configs/rubric.json',
    'case012-human-score.py.txt': 'tools/showcase/score_case012.py',
}


def load_case012(root):
    recorded = load_recorded(root)
    if recorded is None:
        return None
    case = root/C12
    ai = read(case/'publication/assessment_v1/summary.json')
    require(ai['status'] == 'MODEL_ASSISTED_EVALUATION_COMPLETE' and ai['evaluator']['type'] == 'model-assisted'
            and ai['evaluator']['human_raters'] == 0, 'Case012 AI/human scope mismatch')
    for path, digest in ai['source_hashes'].items():
        require(not Path(path).is_absolute() and '..' not in Path(path).parts and sha(case/path) == digest,
                'Case012 assessment source mismatch: '+path)
    inv_path = case/'publication/original_inventory.json'
    require(sha(inv_path) == '3cf346b6ce9303a7100e39ca48d67dd3388cf1d38a9e83e0dd7375d51a110278',
            'Case012 original identity mismatch')
    inventory = read(inv_path)['files']
    rows = [r for r in read(case/'analysis/record_index.json')['rows'] if r['job']['phase'] == 'main']
    require(len(rows) == 192 and len({r['job_id'] for r in rows}) == 192, 'Case012 MAIN ID coverage')
    images, sources, compact = {}, {C12+'/'+p: h for p,h in ai['source_hashes'].items()}, []
    for row in rows:
        path = row['relative_image_path']; record = path.replace('/image.png', '/record.json')
        require(row['status'] == 'SUCCESS' and row['finite'], 'Case012 invalid recorded MAIN image')
        require(sha(case/path) == row['image_sha256'] == inventory[path]['sha256'], 'Case012 image identity')
        require(sha(case/record) == inventory[record]['sha256'], 'Case012 record identity')
        asset = 'assets/case012-images/'+row['job_id']+'.png'
        images[asset] = C12+'/'+path
        sources[C12+'/'+path] = row['image_sha256']; sources[C12+'/'+record] = sha(case/record)
        compact.append({'image_id': row['job_id'], 'prompt_id': row['job']['prompt_id'],
                        'seed_label': row['job']['seed_label'], 'setting': row['job']['setting']['id'],
                        'image_sha256': row['image_sha256'], 'asset': asset,
                        'complete_seconds': row['complete_seconds']})
    for path in SOURCE_COPIES.values(): sources[path] = sha(root/path)
    for lang in ('en', 'ko'):
        path = C12+f'/publication/assessment_v1/viewer.{lang}.html'; sources[path] = sha(root/path)
    for path in ('publication/assessment_v1/viewer.js', 'demo/style.css'):
        sources[C12+'/'+path] = sha(case/path)
    return {'schema': 'case012-showcase-v1', 'recorded_quality_status': 'ANNOTATION_PENDING',
            'quality': ai['quality'], 'evaluator': ai['evaluator'], 'limitations': ai['limitations'],
            'human_evaluation': 'USER_ANNOTATION_AVAILABLE_NOT_YET_COLLECTED',
            'timing': ai['timing'], 'units': {'images': 192, 'prompt_noise_pairs': 64,
                'prompt_clusters': 16, 'seeds_per_prompt': 4, 'checkpoint_count': 1,
                'timing': 'warmed complete request seconds', 'quality': 'all frozen constraints met'},
            'sources': sources, 'images': images, 'pairs': compact}


def annotation_data(root, data):
    original = read(root/C12/'analysis/annotation_items.json')
    items = sorted((i for i in original['items'] if i['split'] == 'MAIN'),
                   key=lambda i: hashlib.sha256(('case012-human-main-v1:'+i['image_id']).encode()).digest())
    require(len(items) == 192, 'Case012 human annotation coverage')
    blind = [{**{k: i[k] for k in ('image_id', 'image_sha256', 'english', 'korean_display', 'constraints')},
              'path': '../assets/case012-images/'+i['image_id']+'.png'} for i in items]
    result = {'schema': 'case012-human-input-v1', 'scope': 'MAIN_ONLY',
              'rubric_sha256': original['rubric_sha256'], 'items': blind}
    result['image_set_sha256'] = hashlib.sha256(json_text(result).encode()).hexdigest()
    require({i['image_id'] for i in blind} == {r['image_id'] for r in data['pairs']}, 'Case012 blind pairing')
    return result


def a(url, label, cls=''):
    return f'<a class="{cls}" href="{escape(url, quote=True)}">{escape(label)}</a>'


def home_case012(data, lang):
    ko = lang == 'ko'
    text = ('같은 모델에서 내부 반복과 생성 단계에 시간을 나눠 쓰는 도구를 구현했습니다. 같은 초기 잡음의 이미지 192장을 비교하고, 이제 직접 요구 조건을 체크할 수 있습니다.' if ko else
            'Built a runner that allocates time between internal loops and generation steps in one model. Compare 192 images from paired starting noise and check their requested constraints yourself.')
    q = data['quality']['settings']
    finding = (f'AI 판독에서는 L1 {q["A_time"]["passed_images"]}/64장, L2·L4 각각 {q["C_time"]["passed_images"]}/64장이 모든 조건을 충족했습니다. 더 깊은 반복의 우위는 불확실했으며, 사람의 평가는 별도로 기록합니다.' if ko else
               f'One AI rater marked all constraints met in {q["A_time"]["passed_images"]}/64 images at L1 and {q["C_time"]["passed_images"]}/64 at L2 and L4. The deeper-loop advantage was uncertain; human judgments are recorded separately.')
    return ('<section class="panel" id="image-budget-study"><p class="eyebrow">CASE 012 · IMAGE GENERATION</p><h2>'
            +a('case012.html#overview', '같은 생성 시간, 어디에 계산을 더 쓸까?' if ko else 'Where should an image-generation budget go?')
            +'</h2><p>'+text+'</p><p>'+finding+'</p><p class="project-scope">Looped-DiT B/32 · RTX 5080 · 512×512 · '+('AI 1개 판독자 · 사람 평가 미수집' if ko else 'One AI rater · Human labels not yet collected')+'</p><div class="actions">'
            +a('case012.html#results', '실제 결과' if ko else 'Measured results')
            +a('case012-annotate.html', '직접 정답 체크하기' if ko else 'Check the images yourself')+'</div></section>')


def render_case012(data, lang):
    ko = lang == 'ko'
    headings = ('한눈에 보기', 'Loop와 Step은 무엇인가?', '어떻게 비교했나요?', '이미지와 결과', '시간·메모리', '직접 확인하기', '보고서·출처') if ko else (
        'Overview', 'What are loops and steps?', 'How were settings compared?', 'Images and results', 'Time and memory', 'Check and reproduce', 'Report and sources')
    title = '같은 생성 시간, 어디에 계산을 더 쓸까?' if ko else 'Allocating an Image-Generation Budget: Loops vs Steps'
    out = '<article class="case012-study"><div class="study-intro"><p class="eyebrow">CASE 012 · LOOPED-DiT B/32</p><h1>'+title+'</h1><p class="lede">'+('같은 모델에서 계산을 내부 반복에 더 쓸지, 이미지를 다듬는 단계에 더 쓸지 비교했습니다. 실행·시간 기록·이미지 판독 도구를 만들었고, 저장된 이미지에 직접 정답을 표시할 수 있습니다.' if ko else 'Compared spending computation on internal repetition versus additional image-refinement steps in the same model. The runner records time and images; you can now label the saved images yourself.')+'</p></div><nav class="case012-toc" aria-label="'+('목차' if ko else 'Contents')+'">'+''.join(a('#'+s,h) for s,h in zip(SECTIONS,headings))+'</nav>'
    def section(i, body):
        return '<section class="study-section" id="'+SECTIONS[i]+'"><h2>'+headings[i]+'</h2>'+body+'</section>'
    out += section(0, '<p>'+('질문은 한정된 생성 시간에서 어떤 계산 배분이 개수·색·좌우 조건을 더 잘 맞추는가입니다. 내부 반복이 많아도 이미 맞힌 조건을 깨뜨리는 이미지가 있었습니다.' if ko else 'The question is which allocation better meets count, color and left/right requests within a measured time budget. More internal repetition also broke some previously satisfied requests.')+'</p><div class="actions">'+a('case012-images.html', '192장과 AI 판독 근거 보기' if ko else 'All 192 images and AI evidence')+a('case012-annotate.html', '직접 정답 체크하기' if ko else 'Annotate the images yourself')+'</div>')
    out += section(1, '<p>'+('Step은 잡음에서 이미지를 점차 다듬는 생성 단계입니다. Loop는 각 단계 안에서 같은 핵심 계산 블록을 여러 번 사용하는 내부 반복입니다. B/32의 32는 이미지 조각 크기이며 반복 횟수가 아닙니다.' if ko else 'A step progressively refines an image from noise. A loop reuses the core calculation blocks inside each step. The 32 in B/32 is the patch size, not the number of loops.')+'</p>')
    out += section(2, '<p>'+('한 저장본·같은 영어 문장과 초기 잡음을 사용했습니다. 16개 문장마다 잡음 4개를 만들어 64쌍을 세 설정으로 생성했습니다. 별도의 시간 측정으로 단계 수를 골랐지만 목표 오차를 벗어나 실제 시간은 완전히 같지 않습니다.' if ko else 'One checkpoint, identical English prompts and matched starting noise: 16 prompts × four noise seeds make 64 pairs under three settings. Steps were selected using separate timing measurements; the matching target was missed, so actual request times differ.')+'</p><p class="scope">RTX 5080 · 512×512 · Euler · CFG 6 · EMA · eager</p>')
    rows = []
    for setting in SETTINGS:
        t,q = data['timing'][setting],data['quality']['settings'][setting]
        rows.append(f'<tr><th scope="row">L{t["loops"]} / S{t["steps"]}</th><td>{t["main_complete_median_seconds"]:.3f} s</td><td>{q["passed_images"]}/64<br>{100*q["all_constraint_pass_rate"]:.2f}%</td></tr>')
    p = data['quality']['primary']; pair = data['quality']['paired']
    out += section(3, '<p>'+('다음은 AI 한 개가 모든 이미지의 명시 조건을 판독한 결과입니다. 사람의 체크 결과와 구분합니다. 애매한 항목은 주 점수에서 미충족으로 계산했습니다.' if ko else 'These scores are from one AI assessor reading every image against the explicit checklist. They are separate from your human labels. Uncertain items count as failures in the primary score.')+'</p><div class="study-table"><table><thead><tr><th>Loop / Step</th><th>'+('요청 시간 중앙값' if ko else 'Median request time')+'</th><th>'+('모든 조건 충족 · AI' if ko else 'All constraints met · AI')+'</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div><p>'+('L4는 L1보다 '+str(pair['C_only'])+'쌍에서 충족을 얻고 '+str(pair['A_only'])+'쌍에서 잃었습니다. 더 깊게 반복하면 항상 좋아진다고 결론내릴 수 없습니다.' if ko else f'L4 gained passes in {pair["C_only"]} pairs and lost them in {pair["A_only"]}. Deeper repetition did not consistently improve requested composition.')+'</p><details><summary>'+('차이와 불확실성 보기' if ko else 'Difference and uncertainty')+f'</summary><p>L4−L1: {100*p["estimate"]:+.3f} pp · 95% [{100*p["interval"][0]:+.3f}, {100*p["interval"][1]:+.4f}] pp.</p><p>'+('문장 단위 paired bootstrap 5,000회입니다. 0을 포함하며 AI 판독 오류까지 포함한 구간은 아닙니다. 관련된 네 template 계열의 작은 표본입니다.' if ko else '5,000 paired prompt-cluster bootstrap repetitions. The interval includes zero and excludes AI judgment error. This small sample has four related template families.')+'</p></details>'+a('case012-images.html', '모든 이미지와 항목별 근거' if ko else 'Every image and constraint evidence'))
    out += section(4, '<p>'+('시간은 문자열 입력부터 T5 인코딩, 이미지 생성, CPU 이미지 변환이 끝날 때까지입니다. 모델 로딩·파일 저장·평가는 제외했습니다. 설정 모두에서 PyTorch allocated peak는 약 1.88 GiB였으며 장치 전체 사용량과는 다릅니다.' if ko else 'Request time runs from a string input through T5 encoding, sampling and completed CPU image conversion. Model loading, disk writes and judging are excluded. PyTorch allocated peak was about 1.88 GiB across settings; it is separate from total device usage.')+'</p><p>'+('CUDA event와 wall clock의 순서가 어긋난 45개 기록도 그대로 남겼습니다. 두 시간을 빼서 순수 오버헤드를 계산하지 않습니다.' if ko else 'The 45 event/wall-clock ordering discrepancies remain in the records. Subtracting the two is not used as a pure overhead measurement.')+'</p>')
    out += section(5, '<p>'+('정답 체크 화면은 192장을 한 장씩 보여줍니다. 설정과 AI 답안은 보이지 않습니다. 충족·미충족·판단 불확실을 선택하고 JSON을 내보내세요. 진행 상태는 이 브라우저에만 저장되며 사이트로 자동 전송되지 않습니다.' if ko else 'The annotation screen shows each of the 192 images without setting labels or AI answers. Choose satisfied, not satisfied or uncertain, then export JSON. Progress stays in this browser and is not automatically sent to the site.')+'</p><div class="actions">'+a('case012-annotate.html', '직접 정답 체크하기' if ko else 'Start annotating')+'</div><details><summary>'+('내 JSON을 모델 없이 검산하기' if ko else 'Recalculate your JSON without models')+'</summary><pre>python -B tools/showcase/score_case012.py --annotations my-labels.json --output ../case012-human-results.json</pre><p>'+('저장소 clone과 CPU 요구 패키지가 필요하며 결과 파일은 저장소 밖의 새 경로여야 합니다. 부분 주석은 대기 상태로 남깁니다.' if ko else 'Requires a repository clone and CPU requirements. Use a new output path outside the repository. Partial annotations stay pending.')+'</p>'+a('../sources/case012-human-score.py.txt', '검산 코드' if ko else 'Scoring code')+'</details>')
    out += section(6, '<div class="actions">'+a('../sources/case012-report-'+lang+'.md.txt', '정식 보고서' if ko else 'Full report')+a('../sources/case012-adapter.py.txt', '실행 adapter 코드' if ko else 'Adapter source')+a('../sources/case012-rubric.json.txt', '고정 평가 규칙' if ko else 'Frozen rubric')+'</div><p>'+('Looped-DiT 모델은 OpenSenseNova의 구현입니다. DIOVA는 실행·시간 예산 선택·평가·탐색 도구를 구현했습니다. OpenAI Codex가 구현·분석·작성을 지원했고, 공개 AI 판독 결과는 사람 평가와 구분합니다.' if ko else 'Looped-DiT is the upstream model by OpenSenseNova. DIOVA implemented the runner, measured-budget selection, assessment and inspection tools. OpenAI Codex assisted implementation, analysis and writing; public AI ratings are distinct from human evaluation.')+'</p>'+a('https://github.com/OpenSenseNova/Looped-DiT', 'Looped-DiT source'))
    return out+'</article>'


def render_annotation(lang):
    ko = lang == 'ko'
    t = lambda en, ko_text: ko_text if ko else en
    return ('<article class="case012-study"><div class="study-intro"><p class="eyebrow">CASE 012 · '+t('YOUR HUMAN LABELS', '직접 이미지 판독')+'</p><h1>'+t('Does the image meet the requested conditions?', '요청한 조건을 지켰나요?')+'</h1><p>'+t('Judge all 192 MAIN images one at a time. Settings, timing and AI answers are hidden. Your choices stay separate from the published AI scores.', 'MAIN 192장을 한 장씩 판단합니다. 설정·시간·AI 답안은 가려져 있으며, 직접 체크한 결과는 공개 AI 점수와 별도로 저장됩니다.')+'</p></div><details><summary>'+t('How to judge', '평가 규칙')+'</summary><p>'+t('Count identifiable requested objects, including cropped bodies and background instances. Shadows and reflections are not extra objects. Judge the main-body color; left/right uses the centers in image coordinates. Use uncertain when identity or counts cannot be determined. Only the listed checklist is scored, not every phrase or aesthetics.', '식별 가능한 요청 물체를 셉니다. 잘린 몸체와 배경의 같은 종류도 포함하며 그림자·반사상은 별개로 세지 않습니다. 몸체의 주된 색과 이미지 좌표의 중심으로 좌우를 판단합니다. 개수나 종류를 확정할 수 없으면 판단 불확실을 선택하세요. 명시한 체크 항목만 채점하며 미적 품질이나 문장의 모든 표현을 채점하지 않습니다.')+'</p>'+a('../sources/case012-rubric.json.txt', t('Full frozen rubric', '전체 고정 규칙'))+'</details><p class="annotation-save-note">'+t('Keep an exported JSON file. Browser storage is only a convenience: other devices and cleared storage will not have your choices. Nothing is automatically uploaded to GitHub.', 'JSON 파일을 내려받아 보관하세요. 브라우저 저장은 보조 수단이며 다른 기기나 저장 기록 삭제 후에는 남지 않습니다. GitHub로 자동 업로드되지 않습니다.')+'</p><label for="evaluator">'+t('Your name or anonymous ID', '평가자 이름 또는 익명 ID')+'</label><input id="evaluator" autocomplete="off" maxlength="100"><p id="progress" role="status" aria-live="polite"></p><p id="position"></p><div class="case012-annotation-grid"><div><a id="original-image" target="_blank" rel="noopener"><img id="image" width="512" height="512" alt="'+t('Saved image to annotate', '직접 평가할 저장 이미지')+'"></a><p id="prompt-text"></p><details><summary>'+t('Original English input', '실제 영어 입력')+'</summary><p id="model-input"></p></details></div><div id="constraints"></div></div><div class="actions"><button id="prev" class="secondary">'+t('Previous', '이전')+'</button><button id="next">'+t('Next', '다음')+'</button><button id="unrated" class="secondary">'+t('Next unfinished image', '미평가로 이동')+'</button></div><div class="actions"><button id="export">'+t('Download my labels (JSON)', '내 체크 결과 다운로드 (JSON)')+'</button><label class="import-label">'+t('Restore labels from JSON', 'JSON으로 체크 복원')+'<input id="import" type="file" accept="application/json,.json"></label></div><p id="message" role="alert"></p><p>'+t('Uncertain counts as a failure in the primary score. A partial download retains unfinished constraints too. You may return later and import it.', '판단 불확실은 주 점수에서 미충족으로 계산합니다. 도중에 내려받아도 일부만 체크한 이미지까지 보존하며, 나중에 가져와 이어서 평가할 수 있습니다.')+'</p><noscript><p>'+t('JavaScript is required to record choices. The original images and rubric are available from the result page.', '체크 기록에는 JavaScript가 필요합니다. 원본 이미지와 평가 규칙은 결과 페이지에서 볼 수 있습니다.')+'</p></noscript>'+a('case012.html#results', t('Published AI results and study', '공개 AI 결과와 연구 설명'))+'</article>')


def image_viewer(root, data, lang):
    """Rebase the existing verified viewer without rewriting its judgments."""
    source = root/C12/f'publication/assessment_v1/viewer.{lang}.html'
    assets = {str((root/path).resolve()): name for name,path in data['images'].items()}
    # Show the original PNG in place of the review-only thumbnail; keep all IDs.
    assets.update({str((root/C12/'demo/thumbs'/(Path(name).stem+'.jpg')).resolve()): name for name in data['images']})
    assets.update({str((root/path).resolve()): 'sources/'+name for name,path in SOURCE_COPIES.items()})
    def replace(match):
        attr,url = match.groups(); parts = urlsplit(url)
        if not parts.path or parts.scheme: return match.group(0)
        target = (source.parent/parts.path).resolve()
        if target.name.startswith('viewer.') and target.suffix == '.html':
            language = target.name.split('.')[1]; dest = ('../'+language+'/' if language != lang else '')+'case012-images.html'
        elif target == (root/C12/'demo/style.css').resolve(): dest = '../assets/case012-review.css'
        elif target == (source.parent/'viewer.js').resolve(): dest = '../assets/case012-images.js'
        elif target.name in ('README.md','README.ko.md') and target.parent == (root/C12).resolve(): dest = 'case012.html'
        elif str(target) in assets: dest = '../'+assets[str(target)]
        else: raise ValueError('Unmapped Case012 viewer resource: '+url)
        return attr+'="'+escape(dest+('#'+parts.fragment if parts.fragment else ''), quote=True)+'"'
    return re.sub(r'(href|src)="([^"]+)"', replace, source.read_text())
