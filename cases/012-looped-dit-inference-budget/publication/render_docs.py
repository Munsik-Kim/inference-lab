"""Render beginner-facing documents from saved records, without model execution."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / 'publication'


def render():
    summary = json.loads((ROOT / 'analysis/summary.json').read_text())
    records = json.loads((ROOT / 'analysis/record_index.json').read_text())['rows']
    prompts = json.loads((ROOT / 'configs/prompts.json').read_text())['prompts']
    main = [r for r in records if r['job']['phase'] == 'main']
    lookup = {(r['job']['prompt_id'], r['job']['seed_label'], r['job']['setting']['id']): r for r in main}
    names = ['A_time', 'B_time', 'C_time']
    values = [summary['timing'][k] for k in names]
    table = '\n'.join(f'| {v["loops"]} | {v["steps"]} | {v["main_complete_median_seconds"]:.3f} |' for v in values)
    (PUB / 'figures').mkdir(exist_ok=True)
    from PIL import Image, ImageDraw, ImageFont
    seed = 72301
    fig = Image.new('RGB', (804, 316), '#f4f0e8')
    draw = ImageDraw.Draw(fig)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    draw.text((12, 8), 'Requested: 3 yellow cubes + 1 green sphere | same initial noise', font=font, fill='#251e17')
    members = []
    for j, k in enumerate(names):
        r = lookup['main-compound-2', seed, k]
        setting = r['job']['setting']
        draw.text((12 + j * 264, 35), f'Loops {setting["loops"]} / Steps {setting["steps"]}', font=font, fill='#66412b')
        with Image.open(ROOT / r['relative_image_path']) as im:
            fig.paste(im.resize((256, 256), Image.Resampling.LANCZOS), (12 + j * 264, 57))
        members.append({'job_id': r['job_id'], 'image_sha256': r['image_sha256'], 'prompt_id': 'main-compound-2', 'seed': seed, 'setting': k})
    fig.save(PUB / 'figures/compound-cubes.png')
    (PUB / 'example_identity.json').write_text(json.dumps({'status': 'POST_HOC_QUALITATIVE_NONBLIND_AI_INSPECTION', 'selection': 'Illustration chosen after inspecting the first frozen MAIN seed across all 16 prompts; not a scored benchmark or human annotation.', 'members': members, 'figure_sha256': hashlib.sha256((PUB / 'figures/compound-cubes.png').read_bytes()).hexdigest(), 'quality_labels_written': 0}, indent=2) + '\n')

    readme_ko = '''# Case 012 — 이미지 생성 시간, 어디에 계산을 더 쓸까?

[English](README.md) | 한국어

이미지 한 장을 만들 때, 잡음을 여러 번 고치는 **생성 단계**와 각 단계 안에서 모델을 다시 계산하는 **내부 반복**에 시간을 어떻게 나눠 쓸지 비교하는 도구를 구현했습니다. 같은 출발점에서 세 이미지를 생성하고, 걸린 시간과 개수·색·좌우 요구의 충족 여부를 함께 살펴볼 수 있습니다.

**동일 초기 잡음 · 설정별 시간·메모리 기록 · 조건을 가린 평가 화면**

[전체 이미지 비교](publication/IMAGES.ko.md) · [실행 코드](source/adapter.py) · [정식 보고서](REPORT.ko.md)

[Loop와 Step](#loops) · [현재 결과](#results) · [결과의 의미](#interpretation) · [직접 확인](#run) · [출처](#sources)

<a id="loops"></a>
## Loop와 Step은 무엇인가요?

- **Step(생성 단계)**: 잡음에서 시작한 이미지를 다음 상태로 갱신하는 횟수입니다.
- **Loop(내부 반복)**: 생성 단계 하나에서 같은 모델의 핵심 블록을 반복해서 사용하는 횟수입니다.

예를 들어 Loop 4 / Step 50은 이미지를 50단계로 갱신하며, 각 단계의 핵심 계산을 네 번 반복합니다. 여기서 반복 횟수와 단계 수를 함께 바꾸면 시간도 이미지도 달라집니다. B/32의 32는 이미지 patch 크기이고 loop 횟수가 아닙니다.

<a id="results"></a>
## 현재 무엇을 확인했나요?

16개 문장의 서로 다른 초기 잡음 4개씩, 같은 **64쌍**을 세 설정에서 생성했습니다. MAIN 192장과 실행 확인용 SMOKE 24장의 생성·측정을 마쳤습니다.

| 내부 반복 Loop | 생성 단계 Step | 요청 시간 중앙값(초) |
|---:|---:|---:|
__TABLE__

RTX 5080 · 512×512 · 같은 Looped-DiT B/32 저장본 · 설정별 64요청. 시간에는 문장 처리, T5 text encoder, 이미지 생성과 CPU 이미지 변환을 포함하며 모델 로딩·PNG 파일 쓰기는 제외합니다.

Loop 2 / Step 66이 가장 짧은 중앙 시간을 보였습니다. 실제 시간 맞추기의 ±5% 목표는 달성하지 못했으므로, 정확한 동일 시간 실험으로 표현하지 않습니다. 새 프로세스 반복에서는 Loop 1과 4의 시간 순위가 바뀌었습니다.

**전체 품질 평가는 주석 대기입니다.** 아직 조건 충족률이나 품질 기반 추천 설정을 계산하지 않았습니다. 이미지가 만들어진 것과 요구를 맞힌 것은 다른 결과입니다.

<a id="interpretation"></a>
## 결과를 어떻게 읽으면 좋을까요?

“노란 정육면체 세 개와 초록 구 한 개”를 요청한 같은 입력의 예시입니다.

![같은 초기 잡음에서 Loop 1/Step 89, Loop 2/Step 66, Loop 4/Step 50으로 생성한 세 이미지](publication/figures/compound-cubes.png)

시각 점검에서는 앞의 두 이미지에 정육면체 세 개, 마지막 이미지에 네 개가 보였습니다. 더 깊은 내부 반복과 더 적은 생성 단계의 조합이 이미 맞힌 개수 조건을 잃는 사례입니다. Loop와 Step을 동시에 바꿨으므로 Loop만의 원인으로 분리할 수는 없습니다.

이 예시는 코드 에이전트가 첫 고정 seed의 48장을 살펴본 뒤 선택한 **사후·비블라인드 AI 시각 점검**입니다. 공식 인간 주석이나 전체 정답률이 아닙니다. 네 개의 풍선을 요청했는데 세 설정 모두 다섯 개를 그린 사례도 있어, 계산을 달리 배분한다고 모든 오류가 해결되지는 않았습니다.

현재 관측은 “더 깊게 반복하면 언제나 더 정확하다”는 단순한 기대를 뒷받침하지 않습니다. 최종 비교는 실제 시간과 조건별 획득·손실을 함께 읽어야 합니다. [모든 64쌍](publication/IMAGES.ko.md)과 [세부 분석](REPORT.ko.md#s7)에서 다른 입력도 확인하세요.

<a id="run"></a>
## 직접 확인하기

GitHub에서는 [이미지 비교 문서](publication/IMAGES.ko.md)를 바로 읽을 수 있습니다. 저장소를 내려받으면 `demo/viewer.ko.html`을 브라우저로 열어 입력별 결과를 선택하거나 `demo/annotation.ko.html`에서 조건을 가리고 평가할 수 있습니다. HTML 링크를 GitHub에서 누르면 실행 화면 대신 소스가 표시됩니다.

모델 없이 저장된 기록을 검산하려면 Case 012 폴더에서 실행합니다.

```bash
python3 -m venv ../../.venv-case012-cpu
source ../../.venv-case012-cpu/bin/activate
python -m pip install -r requirements-cpu.txt
python analysis/audit.py
python publication/verify.py
```

모델을 다시 실행하지 않고 PNG·시간·입력 pairing·원형 보존을 확인합니다. [CPU 재현과 평가 방법](REPRODUCTION.md) · [모든 원본 PNG 목록](demo/gallery.ko.html) · [동결 설정](configs/main_settings.json)

<a id="sources"></a>
## 출처와 적용 범위

**사용 기술:** PyTorch · Transformers · NumPy · Pillow · HTML/CSS/JavaScript. Looped-DiT 모델은 OpenSenseNova의 구현이며, DIOVA는 동일 잡음 실행기·시간 예산 선택·평가·결과 탐색 도구를 구현했습니다. [출처와 기여](NOTICE.md)

같은 4-loop 학습 저장본을 Loop 1/2/4로 실행한 비교입니다. 별도로 학습한 세 모델의 비교나 새로운 이미지 생성 알고리즘을 제안하는 연구는 아닙니다. 이번 문항은 흰 배경의 개수·색·좌우 구성 16개이며 공식 GenEval 전체 점수와 구분합니다. [모델 식별](provenance/models.json) · [원형 기록과 공개본](publication/README.md)
'''.replace('__TABLE__', table)
    readme_en = '''# Case 012 — Allocating an Image-Generation Budget: Loops vs Steps

English | [한국어](README.ko.md)

Built a tool to compare how image-generation time is divided between **steps that update a noisy image** and **internal loops that repeat the model's core computation within each step**. It generates three images from the same starting noise and lets readers inspect time alongside count, color and left/right requirements.

**Identical initial noise · Per-setting time and memory records · Blinded annotation**

[Compare every image](publication/IMAGES.md) · [Adapter code](source/adapter.py) · [Full report](REPORT.md)

[Loops and steps](#loops) · [Current results](#results) · [Interpretation](#interpretation) · [Inspect directly](#run) · [Sources](#sources)

<a id="loops"></a>
## What are loops and steps?

- **Step:** one update of an image that starts as noise.
- **Loop:** another application of the same model's core blocks within a generation step.

Loops 4 / Steps 50 updates the image 50 times, repeating the core computation four times within each step. Changing both can change time and the final image. B/32 denotes patch size, not 32 loops.

<a id="results"></a>
## What has been measured?

The same **64 prompt–seed pairs**, from 16 prompts and four starting-noise seeds each, were generated at three settings. Generation and measurement are complete for 192 MAIN images and 24 separate SMOKE images.

| Internal loops | Generation steps | Median request time (seconds) |
|---:|---:|---:|
__TABLE__

RTX 5080 · 512×512 · one Looped-DiT B/32 checkpoint · 64 requests per setting. Time includes tokenization, the T5 text encoder, sampling and CPU image conversion; it excludes model loading and PNG writes.

Loops 2 / Steps 66 had the shortest median. The ±5% time-matching target was missed, so these are not exactly equal-time configurations. The time ranking of Loops 1 and 4 changed in fresh-process repeats.

**Quality annotations are pending.** Constraint pass rates and quality-ranked presets have not been calculated. A generated image is not automatically a correct image.

<a id="interpretation"></a>
## How should these results be read?

This example requests three yellow cubes and one green sphere from the same starting noise.

![Three images from identical initial noise: Loops 1/Steps 89, Loops 2/Steps 66, and Loops 4/Steps 50](publication/figures/compound-cubes.png)

Visual inspection shows three cubes in the first two images and four in the last. The deeper-loop/fewer-step configuration loses a count condition met by the other two. Since loops and steps change together, this does not isolate a causal effect of loops alone.

This illustration was selected after a code agent inspected 48 images from the first frozen seed. It is **post-hoc, non-blind AI visual inspection**, not human annotation or an overall accuracy score. A separate example requesting four balloons shows five at all three settings; reallocating computation did not resolve every observed error.

The inspected examples do not support a simple expectation that deeper loops always give more correct images. Compare measured time with constraint gains and losses. Inspect [all 64 pairs](publication/IMAGES.md) and the [detailed analysis](REPORT.md#s7).

<a id="run"></a>
## Inspect directly

The [image comparison document](publication/IMAGES.md) is readable directly on GitHub. After downloading the repository, open `demo/viewer.en.html` in a browser to select results or `demo/annotation.en.html` to annotate without setting labels. GitHub displays HTML source rather than running the viewer.

For model-free checks, run from the Case 012 directory:

```bash
python3 -m venv ../../.venv-case012-cpu
source ../../.venv-case012-cpu/bin/activate
python -m pip install -r requirements-cpu.txt
python analysis/audit.py
python publication/verify.py
```

These check saved PNGs, time, input pairing and original-file preservation without executing a model. [CPU reproduction and annotation](REPRODUCTION.md) · [Original PNG gallery](demo/gallery.en.html) · [Frozen settings](configs/main_settings.json)

<a id="sources"></a>
## Sources and scope

**Stack:** PyTorch · Transformers · NumPy · Pillow · HTML/CSS/JavaScript. OpenSenseNova provides Looped-DiT; DIOVA implements the paired-noise runner, measured-time selection, annotation and result inspection. [Attribution](NOTICE.md)

The same checkpoint trained at four loops runs at Loops 1/2/4. This is neither a comparison of three separately trained models nor a new image-generation algorithm. The 16 prompts cover count, color and left/right compositions on white backgrounds, separately from the full official GenEval benchmark. [Model identity](provenance/models.json) · [Original and public records](publication/README.md)
'''.replace('__TABLE__', table)
    (ROOT / 'README.ko.md').write_text(readme_ko)
    (ROOT / 'README.md').write_text(readme_en)

    for lang in ['ko', 'en']:
        ko = lang == 'ko'
        name = 'IMAGES.ko.md' if ko else 'IMAGES.md'
        opposite = 'IMAGES.md' if ko else 'IMAGES.ko.md'
        home = '../README.ko.md' if ko else '../README.md'
        text = ['# ' + ('모든 MAIN 이미지 비교' if ko else 'Every MAIN image comparison'), '', f'[{"English" if ko else "한국어"}]({opposite}) · [{"사례 소개" if ko else "Case overview"}]({home})', '',
            ('16문장 × 4초기 잡음 × 3설정의192장을 모두 표시합니다. 좋은 결과를 골라내지 않았습니다. 품질 주석은 미완료입니다. 아래 시간은 개별 요청 시간이고, 이미지는256px 썸네일입니다. 각 이미지 아래 원본 PNG 링크에서512px 원본을 확인할 수 있습니다.' if ko else 'All 192 MAIN images are shown: 16 prompts × four initial noises × three settings, without quality selection. Annotations are pending. Times are individual requests; these are 256px thumbnails. The link below each image opens its original 512px PNG.'), '']
        for p in prompts:
            if p['split'] != 'MAIN': continue
            text += ['## ' + p['prompt_id'], '', p['korean_display'] if ko else p['english'], '']
            for seed in [72301, 72302, 72303, 72304]:
                rr = [lookup[p['prompt_id'], seed, k] for k in names]
                text += [f'### Seed {seed}', '', '| L1 / S89 | L2 / S66 | L4 / S50 |', '|---|---|---|',
                         '| ' + ' | '.join(f'![{p["prompt_id"]}, L{r["job"]["setting"]["loops"]} S{r["job"]["setting"]["steps"]}](../demo/thumbs/{r["job_id"]}.jpg)' for r in rr) + ' |',
                         '| ' + ' | '.join(f'[{"원본 PNG" if ko else "Original PNG"}](../{r["relative_image_path"]}) · {r["complete_seconds"]:.3f}s · {"미평가" if ko else "Not annotated"}' for r in rr) + ' |', '']
        (PUB / name).write_text('\n'.join(text))

    additions = {
        'ko': '''
### 저장된 기록의 추가 분석 — 이번 공개 정리

같은273개 record JSON에서 MAIN192개와 반복18개를 다시 계산했습니다. 새 모델 실행과 품질 주석은0건이며, 이 분석은 **POST_HOC_SAME_RECORDED_SCALARS**입니다. [입력별 시간과 source hashes](publication/posthoc/analysis.json)

L2/S66의 MAIN 중앙 시간은 L1/S89보다3.28%, L4/S50보다7.93% 짧았습니다. 같은64쌍에서 각각45쌍·47쌍이 더 빨랐습니다. 한 MAIN process가 공유된 관측이므로 독립적인64개의 runtime 실험으로 확대하지 않습니다. 새 process에서 같은 두 입력을 반복했을 때 L2는 모두 가장 짧았지만 L1/L4 순위는 바뀌었습니다.

Sampler CUDA event가 전체 CPU wall 시간보다 긴 기록이192건 중45건 있었으며, 최대 차이는 약116ms였습니다. 서로 다른 clock 값을 빼서 T5·전송·PIL 비용을 계산할 수 없습니다. 원인은 아직 진단하지 않았고 원래 측정값은 보존했습니다. 전체 요청 시간의 기술통계와 반복값을 제시하되 미세한 속도 차이를 안정적인 우위로 확대하지 않습니다.

첫 고정 seed72301의 전체16문항·48장에 대한 비블라인드 AI 시각 점검에서는 풍선4개 요청이 세 설정 모두5개였고, cube3개 요청이 L1/2에서3개·L4에서4개로 보였습니다. 이 사례는 전체192장의 인간 품질 평가를 대체하지 않습니다. [사후 예시의 image identity](publication/example_identity.json) · [모든 이미지](publication/IMAGES.ko.md)

현재는 예산 배분에 따라 요구 조건을 얻거나 잃을 수 있는지 비교할 자료를 확보한 단계입니다. 요청된 개수·색·좌우를 각각 평가해야 그럴듯한 외관과 정확한 구성의 차이를 확인할 수 있습니다. 최종 품질 우위와 preset은 실제 blind 주석 후 원래 prompt-cluster 집계로 판단합니다.

''',
        'en': '''
### Additional analysis of saved records — publication preparation

The same 273 record JSON files support recalculation of 192 MAIN and 18 repeated-timing requests. New model executions and quality annotations are zero. This is **POST_HOC_SAME_RECORDED_SCALARS**. [Paired times and source hashes](publication/posthoc/analysis.json)

The L2/S66 MAIN median is 3.28% shorter than L1/S89 and 7.93% shorter than L4/S50. It is faster in 45 and 47 of the corresponding 64 pairs. MAIN shares one process; these are not 64 independent runtime experiments. L2 remains shortest in repeats of two fixed inputs across fresh processes, but the L1/L4 ranking changes.

In 45 of 192 MAIN records, the sampler CUDA event exceeds the surrounding CPU wall time, by up to about 116ms. Subtracting the two clocks cannot estimate T5, transfer or PIL cost. The cause remains undiagnosed and the original values are preserved. Complete-request summaries and repeats remain descriptive; small differences are not presented as a stable performance advantage.

Non-blind AI inspection of all 16 prompts at the first frozen seed72301 (48 images) finds five balloons where four were requested at every setting, and three requested cubes appearing as three at L1/2 but four at L4. These examples do not replace human quality annotations for all 192 images. [Post-hoc example identity](publication/example_identity.json) · [Every image](publication/IMAGES.md)

The present result is a recorded comparison for investigating which constraints different budget allocations gain or lose. Attractive appearance and correct requested composition must be assessed separately. Final quality rankings and presets await blind annotations and the original prompt-cluster aggregation.

'''}
    for lang, addition in additions.items():
        name = 'REPORT.ko.md' if lang == 'ko' else 'REPORT.md'
        original = (PUB / 'original_docs' / (name + '.txt')).read_text()
        counterpart = '[English](REPORT.md)' if lang == 'ko' else '[한국어](REPORT.ko.md)'
        original = original.replace('\n', '\n\n' + counterpart + '\n', 1)
        marker = '<a id="s8"></a>'
        assert original.count(marker) == 1
        (ROOT / name).write_text(original.replace(marker, addition + marker))
    reproduction = (PUB / 'original_docs/REPRODUCTION.md.txt').read_text()
    reproduction = reproduction.replace('python -m venv cpu-env', 'python3 -m venv ../../.venv-case012-cpu')
    reproduction = reproduction.replace('cpu-env/bin/python', '../../.venv-case012-cpu/bin/python')
    reproduction = reproduction.replace('../../.venv-case012-cpu/bin/python analysis/analyze.py\n', '')
    reproduction = reproduction.replace('../../.venv-case012-cpu/bin/python analysis/audit.py --output audit-receipt.json',
                                        '../../.venv-case012-cpu/bin/python analysis/audit.py\n../../.venv-case012-cpu/bin/python publication/verify.py')
    reproduction = reproduction.replace('../../.venv-case012-cpu/bin/python demo/build.py\n', '')
    reproduction = reproduction.replace('`analysis/analyze.py` rebuilds timing and pairing data.',
                                        'The commands above read saved records and print receipts without changing the study files. `analysis/analyze.py` rebuilds timing and pairing data; run it only in a disposable restored copy as shown below.')
    reproduction = reproduction.replace('../../.venv-case012-cpu/bin/python analysis/analyze.py --annotations case012_annotations.json',
                                        'python analysis/analyze.py --annotations /your/export/case012_annotations.json\npython demo/build.py')
    reproduction = reproduction.replace('```bash\npython analysis/analyze.py --annotations',
                                        'In a disposable restored copy with the CPU environment activated (section 5), apply the exported labels and rebuild the viewer. This changes that copy, not the published records:\n\n```bash\npython analysis/analyze.py --annotations')
    reproduction = reproduction.replace('-m pytest -q tests --ignore=tests/test_adapter_cpu.py',
                                        '-m pytest -q -p no:cacheprovider tests --ignore=tests/test_adapter_cpu.py')
    reproduction = reproduction.replace('Run from the Case 012 directory after extracting the complete review bundle:',
                                        'Run from the Case 012 directory. Keep the CPU environment outside the research subtree so the exact publication inventory contains only study files:')
    (ROOT / 'REPRODUCTION.md').write_text(reproduction + '''
## 5. Public GitHub edition and historical documents

The public edition adds beginner-facing bilingual README text, every MAIN image in GitHub-readable Markdown, and explicitly post-hoc analysis of saved records. No new generation or quality labels were added. `publication/original_docs/` preserves prior document bytes; `publication/original_inventory.json` maps every original case file to its original size and hash.

```bash
../../.venv-case012-cpu/bin/python publication/verify.py
../../.venv-case012-cpu/bin/python publication/restore_original.py --output /your/new-scratch/case012-original
```

The output must not already exist. The restored directory reproduces the original Case012 subtree, not the entire repository or its original review ZIP. Run `analysis/audit.py` and CPU tests from that restored directory to inspect the original code and saved measurements. Publication metadata records current scope separately from historical `LOCAL_REVIEW` fields.

To rebuild derived data or apply annotations, activate the same CPU environment, then change to the restored directory. Use your actual new scratch path in place of the example:

```bash
source ../../.venv-case012-cpu/bin/activate
cd /your/new-scratch/case012-original
python analysis/analyze.py
python demo/build.py
python analysis/audit.py
```

Keep generated receipts, environments and annotation exports outside the published study subtree. Its exact-inventory verifier checks the preserved public files, not a modified annotation workspace.

The paired image Markdown can be read on GitHub. The HTML viewer/annotation UI runs locally after cloning; no Pages deployment or browser-hosted model execution is implied. The post-hoc illustration is non-blind AI visual inspection, not human labels or a new primary score. The raw CPU wall and CUDA event values are both retained; do not subtract them to estimate preprocessing overhead.
''')
    print('Rendered bilingual introductions, all 64 pairs, and post-hoc report sections.')


if __name__ == '__main__':
    render()
