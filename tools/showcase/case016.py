"""Case016: source-verified native NVFP4 KV engineering records."""
from html import escape
import importlib.util
from common import read, require, sha

C16 = 'cases/016-kv-agent-bench'
TECH = C16 + '/followups/nvfp4-technical-r1'
TITLES = {'ko': 'RTX 5080에서 NVFP4 KV 캐시 검증', 'en': 'NVFP4 KV Cache on RTX 5080'}
SECTIONS = ('overview', 'kv-cache', 'storage', 'capacity', 'serving', 'numeric', 'limits', 'reproduce', 'references')
SOURCE_COPIES = {
    'case016-guide-ko.md.txt': C16 + '/README.ko.md', 'case016-guide-en.md.txt': C16 + '/README.md',
    'case016-report-ko.md.txt': C16 + '/REPORT.ko.md', 'case016-report-en.md.txt': C16 + '/REPORT.md',
    'case016-reproduce-ko.md.txt': C16 + '/REPRODUCE.ko.md', 'case016-reproduce-en.md.txt': C16 + '/REPRODUCE.md',
    'case016-notice.md.txt': C16 + '/NOTICE.md', 'case016-client.py.txt': TECH + '/src/nvfp4_technical/client.py',
    'case016-runner.py.txt': TECH + '/src/nvfp4_technical/runner.py', 'case016-worker.py.txt': TECH + '/src/nvfp4_technical/worker.py',
    'case016-audit.py.txt': TECH + '/src/nvfp4_technical/audit.py', 'case016-tests.py.txt': TECH + '/tests/test_contracts.py',
    'case016-protocol.json.txt': TECH + '/specs/protocol.json', 'case016-per-round.csv.txt': TECH + '/results/analysis/per-round.csv',
    'case016-capacity.csv.txt': TECH + '/results/analysis/capacity.csv', 'case016-fidelity.csv.txt': TECH + '/results/analysis/fidelity.csv',
}
FIGURES = {'assets/case016-' + kind + '-' + lang + '.svg': TECH + '/figures/' + kind + '-' + lang + '.svg' for kind in ('capacity', 'throughput', 'queue', 'fidelity') for lang in ('ko', 'en')}


def load_case016(root):
    case = root / C16
    if not (case / 'README.md').exists():
        return None
    for name in ('verify', 'derive'):
        spec = importlib.util.spec_from_file_location('case016_' + name, case / 'publication' / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if name == 'verify':
            module.verify(case)
        else:
            data = module.project(case)
    require(data == read(case / 'publication/summary/project.json'), 'Case016 display differs from raw records')
    data['display_source_files'] = {p: sha(root / p) for p in [*SOURCE_COPIES.values(), *FIGURES.values(), C16 + '/publication/summary/project.json', C16 + '/publication/PUBLIC_MANIFEST.json']}
    return data


def a(url, label):
    return '<a href="' + escape(url, quote=True) + '">' + escape(label) + '</a>'


def table(headers, rows, label):
    return '<div class="scroll" role="region" tabindex="0" aria-label="' + escape(label, quote=True) + '"><table><thead><tr>' + ''.join('<th scope="col">' + escape(str(x)) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def num(x, digits=3):
    return '—' if x is None else f'{x:.{digits}f}'


def home_case016(data, lang):
    txt = lambda k, e: k if lang == 'ko' else e
    primary = [r for r in data['summary'] if r['length'] == 16384 and r['concurrency'] in (2, 4)]
    changes = ' · '.join('16K/C' + str(r['concurrency']) + ' ' + num(r['ratio_median'], 2) + '×' for r in primary)
    return '<section class="panel kv-latest" id="kv-engineering"><p class="eyebrow">CASE 016 · NATIVE KV STORAGE</p><h2>' + a('case016.html#overview', TITLES[lang]) + '</h2><p>' + txt('모델이 읽은 내용의 중간 계산값을 작은 형식으로 저장하고, 같은 메모리에서 요청을 처리하는 도구를 만들었습니다. Qwen3-4B 가중치는 BF16으로 유지하고 KV 캐시 저장 형식만 비교합니다.', 'Built tools to verify compressed intermediate model memory and compare requests under the same memory budget. Qwen3-4B weights remain BF16; only KV storage changes.') + '</p><p><strong>' + txt('같은 약 4GiB pool에서 token 수용량 ', 'Token capacity at the same approximately4GiB pool: ') + num(data['capacity_ratio'], 2) + '×</strong> · ' + escape(changes) + '</p><p class="scope">' + txt('처리량 비율은 F4/B16의 3라운드 중앙값입니다. 8K·16K 합성 입력, 요청당 출력128토큰, 동시 요청1/2/4의 실측이며 일반 업무 성공률이나 전체 VRAM 감소율과 구분합니다.', 'Throughput ratios are three-round median F4/B16. Measurements use synthetic8K/16K inputs,128 output tokens/request and concurrency1/2/4; they are separate from business success or total VRAM reduction.') + '</p><div class="actions">' + a('case016.html#serving', txt('실제 처리량과 비용', 'Serving results and costs')) + ' · ' + a('case016.html#storage', txt('저장·읽기 검증', 'Storage/read checks')) + ' · ' + a('case016.html#reproduce', txt('CPU 검산', 'CPU replay')) + '</div></section>'


def render_case016(data, lang):
    txt = lambda k, e: k if lang == 'ko' else e
    labels = txt(['한눈에 보기', 'KV 캐시란?', '실제 압축 저장·읽기', '같은 메모리에서 수용량', '실제 서빙 비교', '수치·응답 확인', '실패와 적용 범위', '직접 확인·실행', '출처와 구현 기여'], ['Overview', 'What is a KV cache?', 'Actual compressed storage/read', 'Capacity at the same budget', 'Serving comparison', 'Numerical and response checks', 'Failures and scope', 'Check and reproduce', 'Sources and contributions'])
    out = '<header class="study-intro"><p class="eyebrow">DIOVA · CASE 016</p><h1>' + TITLES[lang] + '</h1><p class="lede">' + txt('같은 메모리로 더 많은 문맥을 수용할 수 있을까요?', 'How much context fits in the same memory budget?') + '</p><p>' + txt('RTX 5080 16GB에서 vLLM0.32.0과 FlashInfer의 실제 NVFP4 저장 경로를 연결하고, 압축된 캐시의 내용·수용량·요청 처리 비용을 검사했습니다.', 'On an RTX5080 16GB, we integrated vLLM0.32.0 and FlashInfer native NVFP4 KV storage and checked its contents, capacity and request-processing costs.') + '</p></header><div class="study-layout"><aside class="study-toc"><nav aria-label="' + txt('사례 목차', 'Case contents') + '"><ol>' + ''.join('<li>' + a('#' + i, label) + '</li>' for i, label in zip(SECTIONS, labels)) + '</ol></nav></aside><article class="study-article">'
    primary = [r for r in data['summary'] if r['length'] == 16384 and r['concurrency'] in (2,4)]
    highlight = '<p class="case016-highlights"><strong>' + txt('같은 약4GiB pool의 token 수용량 ', 'Token capacity at the same approximately4GiB pool: ') + num(data['capacity_ratio'],2) + '×</strong><br>' + ' · '.join('16K/C'+str(r['concurrency'])+' '+num(r['ratio_median'],2)+'×' for r in primary) + ' ' + txt('요청 처리량(F4/B16,3라운드 중앙값)', 'request throughput(F4/B16,three-round median)') + '</p><p class="scope">' + txt('8K/C2·16K/C1에서는 소폭 느렸습니다. 전체 VRAM 감소율·일반 업무 성공률과는 다른 측정입니다.', '8K/C2 and16K/C1 were slightly slower. These measurements are separate from total VRAM reduction or general workflow success.') + '</p>'
    out = out.replace('</header>', highlight+'</header>', 1)
    def section(i, content):
        return '<section class="study-section" id="' + SECTIONS[i] + '"><h2>' + labels[i] + '</h2>' + content + '</section>'
    def figure(kind, caption):
        return '<figure><img loading="lazy" src="../assets/case016-' + kind + '-' + lang + '.svg" alt="' + escape(caption, quote=True) + '"><figcaption>' + escape(caption) + '</figcaption></figure>'
    status_names = dict(zip(data['status'], txt(['실제 압축 캐시 저장·읽기', '같은 압축값 해석·유한한 계산', '실제 메모리·수용량 장부', '등록한 요청 처리 실측'], ['Native compressed-cache storage/read', 'Compressed-value interpretation and finite computation', 'Actual memory/capacity ledger', 'Registered serving measurement'])))
    out += section(0, '<p>' + txt('실제 packed NVFP4 KV 저장·읽기, 같은 pool의 용량 장부, 동시 요청 비교와 모델 없는 원기록 검산기를 구현했습니다.', 'Implemented actual packed NVFP4 KV path verification, a fixed-pool capacity ledger, concurrent workloads and a model-free raw-record auditor.') + '</p>' + table([txt('검증 항목', 'Check'), txt('상태', 'Status')], [[status_names[k], 'PASS' if v else 'PARTIAL / BLOCKED'] for k, v in data['status'].items()], 'Technical verification status') + '<p>' + escape(data['model']) + ' · RTX5080 · vLLM0.32.0 · ' + escape(data['mode']) + ' · BF16 weights · 36 attention layers.</p>')
    out += section(1, '<p>' + txt('모델은 글을 읽고 다음 말을 만들 때, 이미 읽은 내용의 중간 계산값을 잠시 저장합니다. 이 공간을 KV 캐시라고 합니다. BF16은 값 하나를16비트로 저장하는 기준선이고, NVFP4는4비트 값과 작은 scale을 함께 저장합니다. 모델 가중치 파일을4비트로 바꾼 실험은 아닙니다.', 'A model temporarily stores intermediate values for text it has already read while generating the next token. This is the KV cache. BF16 stores each value in16 bits; NVFP4 stores four-bit values plus small scales. Model weight files are not quantized in this experiment.') + '</p><ol class="case016-flow">' + ''.join('<li>' + escape(x) + '</li>' for x in txt(['같은 입력 토큰', 'BF16 또는 NVFP4 캐시에 저장', '실제 attention이 읽음', '동일 출력 길이와 요청 집합으로 비용 비교'], ['Same input tokens', 'BF16 or NVFP4 KV storage', 'Actual attention reads the cache', 'Compare a fixed output length and request set'])) + '</ol>')
    out += section(2, '<p>' + txt('36개 층의 실제 tensor와 공유 storage를 검사했습니다. F4는 uint8 packed data와 FP8 block scale을 사용하며, 짧은 profiler에서 NVFP4 writer와 압축 KV를 읽는 FlashInfer 경로를 확인했습니다. Prefill은 NVFP4 FA2, decode는 packed data와 block scale을 받는 XQA입니다. FA2 read는 BF16 저장 fallback을 뜻하지 않습니다.', 'Inspected actual tensors and shared storage across36 layers. F4 uses uint8 packed data and FP8 block scales. A short profile records the NVFP4 writer and compressed-KV read path:NVFP4 FA2 prefill and XQA decode receiving packed data/block scales. FA2 does not mean BF16 storage fallback.') + '</p><p>' + a('../sources/case016-worker.py.txt', txt('실제 tensor·유한성 검사 코드', 'Tensor/finite-check code')) + ' · ' + a('../sources/case016-protocol.json.txt', txt('사전 동결 조건', 'Frozen protocol')) + '</p>')
    out += section(3, table(['KV', 'B/token', txt('실제 pool bytes', 'Actual pool bytes'), 'Blocks ×16', 'Token slots'], [[r['kv'], r['bytes_per_token'], r['actual_pool_bytes'], r['blocks'], r['token_slots']] for r in data['capacity']], 'Actual KV storage') + figure('capacity', txt('같은 약4GiB pool의 수용량. FP8은 이전 P0 기록, B16/F4는 새 실측입니다.', 'Capacity at approximately4GiB. FP8 is historical P0; B16/F4 are newly measured.')) + '<p>' + txt('수용량 비율은 ', 'The token-capacity ratio is ') + num(data['capacity_ratio'], 4) + '×. ' + txt('한 요청의 최대 문맥은28672토큰입니다. 103552슬롯은 pool 전체 수용량이며100K 단일 문맥 지원이나 전체 VRAM 절감률이 아닙니다.', 'Maximum context/request is28672 tokens.103552 slots describe the whole pool, not a100K single context or total VRAM reduction.') + '</p>')
    rows = [[str(r['length'] // 1024) + 'K', 'C' + str(r['concurrency']), num(r['B16_requests_per_second']['median']), num(r['F4_requests_per_second']['median']), num(r['ratio_median'], 2) + '× [' + num(r['ratio_min'], 2) + ', ' + num(r['ratio_max'], 2) + ']', str(r['B16_success']) + '/24 · ' + str(r['F4_success']) + '/24'] for r in data['summary']]
    serving = '<p>' + txt('입력8K·16K × 동시 요청1/2/4 × 두 캐시 ×3라운드, 각 block은8요청입니다. 요청당 출력128토큰을 공식 ignore_eos 옵션으로 고정했습니다. 첫 요청 집합을 띄운 시점부터 마지막 응답의 수신·client 파싱 종료까지 실패·대기를 포함해 잽니다. 두 조건은 같은 pool·graph·scheduling을 사용합니다.', 'Input8K/16K × concurrency1/2/4 × two caches × three rounds; each block contains8 requests. Official ignore_eos fixes128 output tokens/request. Makespan runs from queue launch until the last response is received and parsed by the client, including failures and waiting. Pool, graph and scheduling are common.') + '</p>' + table(['Input', 'Clients', 'B16 req/s', 'F4 req/s', txt('F4/B16 중앙값 [범위]', 'F4/B16 median [range]'), txt('완료 B16 · F4', 'Completed B16 · F4')], rows, 'Serving throughput with all paired rounds') + figure('throughput', txt('막대는3라운드 중앙값, 점은 각 라운드입니다. 처리량은 높을수록 좋고 축은0부터 시작합니다.', 'Bars show three-round medians; dots show every round. Higher throughput is better; the axis starts at zero.')) + figure('queue', txt('엔진 대기시간과 선점 횟수. 전용 서버의 block 구간 지표이며 요청별 원인으로 환산하지 않습니다.', 'Engine queue time and preemptions. Dedicated-server interval metrics are not attributed to individual requests.'))
    serving += '<details><summary>' + txt('36개 block의 전체 값과 decode 관측', 'All36 blocks and decode observations') + '</summary>' + table(['R', 'KV', 'Input', 'C', 'OK/8', 'req/s', 'TTFT s', 'decode ms/token', 'queue s', 'preemptions', 'device MiB'], [[r['round'], r['kv'], r['length'], r['concurrency'], r['success'], num(r['requests_per_second']), num(r['median_client_ttft_seconds']), num(r['median_client_decode_ms_per_token']), num(r['request_queue_time_seconds_mean']), r['preemptions'], num(r['sampled_device_peak_MiB'], 0)] for r in data['per_round']], 'All technical serving blocks') + '<p>' + txt('TTFT는 실제 HTTP 전송부터 첫 token ID 포함 SSE까지입니다. decode는 첫·마지막 token-bearing event 시간차를 n−1로 나눈 client 평균이며, 개별 token ITL은 아닙니다. GPU 메모리는 배경 점유를 포함한 device 표본입니다.', 'TTFT is HTTP dispatch to the first SSE with token IDs. Decode is the first-to-last token-bearing event duration divided by n−1, a client average rather than individual-token ITL. Device memory samples include background use.') + '</p></details>'
    out += section(4, serving)
    fidelity = '<p>' + txt('동일 압축 캐시를 두 unpack/dequant 방식으로 읽은 차이와 BF16 원cache 대비 양자화 오차를 따로 검사했습니다. 실제 모델 attention 출력36층도 유한했습니다. 두 전체 decoder의 동등성 검사나 전체 KL/PPL은 아닙니다.', 'Compared two unpack/dequant interpretations of the same compressed cache separately from BF16-original quantization error. Actual model attention outputs were finite across36 layers. This is neither equivalence of two whole decoders nor full KL/PPL.') + '</p>' + figure('fidelity', txt('별도 greedy 응답 sanity. Copy canary의 정확 일치와 token sequence 차이를 별도로 기록합니다.', 'Separate greedy-response sanity. Exact copy-canary answers and token-sequence differences are recorded separately.'))
    exact = {kv: sum(r[kv + '_exact_answer'] is True for r in data['fidelity']) for kv in ('B16', 'F4')}
    equal = sum(r['token_sequences_equal'] for r in data['fidelity'])
    fidelity += '<p>' + txt('문자열 복사는 B16 ', 'Exact copied strings: B16 ') + str(exact['B16']) + '/8 · F4 ' + str(exact['F4']) + '/8. ' + txt('전체16개 입력 중 token sequence가 같은 경우는 ', 'Identical token sequences across all16 paired inputs: ') + str(equal) + '/16. ' + txt('정답이 정해지지 않은 일반 텍스트8개는 응답이 달랐으며, 그 차이를 품질 손실률로 채점하지 않았습니다.', 'Responses differ on the8 open-text inputs, which have no fixed answer; these differences are not scored as a quality-loss rate.') + '</p>'
    fidelity += '<details><summary>' + txt('별도16개 prompt의 두 응답과 최초 차이', 'Both responses and first divergence for16 separate prompts') + '</summary>'
    for r in data['fidelity']:
        fidelity += '<details><summary>' + escape(r['prompt_id']) + ' · ' + escape(r['kind']) + ' · divergence=' + str(r['first_token_divergence']) + '</summary><pre>' + escape('B16: ' + r['B16_text'] + '\nF4: ' + r['F4_text']) + '</pre></details>'
    out += section(5, fidelity + '</details>')
    agent = data['original_agent']
    limits = '<p>' + txt('이 결과는 한 장비·한 모델·합성 token workload에서 측정한 전체 구성의 비교입니다. 정밀도, 커널 선택, 캐시 용량과 queue가 함께 작용합니다. NVFP4 산술만의 인과 효과로 해석하지 않습니다.', 'This compares complete configurations on one device, model and synthetic token workload. Precision, kernel choice, capacity and queueing interact; the result is not the causal effect of quantization arithmetic alone.') + '</p><details><summary>' + txt('보존한 장문 agent 응용 탐색과 기준선 감사', 'Preserved long-context agent exploration and baseline audit') + '</summary><p>' + txt('이전 DEV96회에서 B16 ', 'The earlier96 DEV runs recorded B16 ') + str(agent['B16_verified']) + '/48 · F4 ' + str(agent['F4_verified']) + '/48. ' + txt('같은12개 과제를 메뉴·출력 한도로 반복한 수입니다. 기준선 gate는 NOT_MET이며 agent P2/P3는 실행하지 않았습니다. 원 점수 변경은0건입니다.', 'These repeat the same12 tasks across menus/output limits. The baseline gate is NOT_MET; agent P2/P3 were not run. Original scores changed in zero cases.') + '</p>' + a('../sources/case016-reproduce-' + lang + '.md.txt', txt('원형 복원과96개 CPU 재생 방법', 'Restore originals and replay all96 on CPU')) + '</details>'
    limits += '<p class="scope">' + txt('보존한 agent DEV: B16 ', 'Preserved agent DEV: B16 ') + str(agent['B16_verified']) + '/48 · F4 ' + str(agent['F4_verified']) + '/48 · baseline NOT_MET · agent P2/P3 NOT_RUN.</p>'
    out += section(6, limits)
    command = 'python3 -B cases/016-kv-agent-bench/publication/run_checks.py --output /tmp/case016-check-new --tests'
    out += section(7, '<p>' + txt('모델 없이 Python 표준 라이브러리로 원형 hash·SSE·용량·처리량을 재계산할 수 있습니다. 새 외부 출력 경로를 사용하세요.', 'Python standard-library checks recalculate original hashes, SSE, capacity and throughput without a model. Choose a new external output directory.') + '</p><pre>' + escape(command) + '</pre><div class="actions">' + a('../sources/case016-reproduce-' + lang + '.md.txt', txt('CPU/GPU 재현 안내', 'CPU/GPU reproduction')) + ' · ' + a('../sources/case016-runner.py.txt', 'Runner') + ' · ' + a('../sources/case016-audit.py.txt', 'Auditor') + ' · ' + a('../sources/case016-per-round.csv.txt', 'CSV') + '</div>')
    out += section(8, '<p>' + txt('NVFP4 알고리즘과 커널은 vLLM·FlashInfer가 제공합니다. 이 사례의 기여는 환경 통합, 실제 저장·읽기 경로 검사, 고정 pool benchmark, 원시 token/엔진 지표 기록과 별도 CPU 감사 도구입니다. Codex가 구현과 계산 검토를 보조했습니다.', 'vLLM and FlashInfer provide the NVFP4 algorithms and kernels. This case contributes integration, actual storage/read inspection, fixed-pool benchmarks, raw-token/engine recording and separate CPU audit tools. Codex assisted implementation and calculation review.') + '</p>' + a('../sources/case016-report-' + lang + '.md.txt', txt('정식 보고서', 'Formal report')) + ' · ' + a('../sources/case016-notice.md.txt', 'NOTICE') + ' · ' + a('../data/case016.json', txt('수치·출처 hash', 'Values and source hashes')))
    return out + '</article></div>'
