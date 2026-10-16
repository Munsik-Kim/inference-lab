"""Practical guide: a bounded projection of already-public, hash-pinned records.

Narrative lives in presentation/guides. Numbers are rendered from this projection,
shared by the home stories and guide pages. Never loads weights or runs a model.
"""
from __future__ import annotations

from html import escape
from math import isfinite
from pathlib import Path
import re
from statistics import mean

from common import checked_read, read, require, sha

C2 = 'cases/002-bf16-fp8-document-extraction'
C8 = 'cases/008-build-reconstruct-reload'
C9 = 'cases/009-q-serving-quality'
C13 = 'cases/013-dualquant-reference-quality'
FOLLOW = 'research/agent_tool_contract_repair_20261014'
REVISION = '09194c3f2e2c991feff5b6abab0cf8ab81a07052'
PAGES = ('llm-guide', 'qwen3-4b', 'measurements', 'compare')
COMPARE_EXAMPLES = {
    'examples/compare/baseline.jsonl': 'packages/diova-compare/examples/compare/baseline.jsonl',
    'examples/compare/candidate.jsonl': 'packages/diova-compare/examples/compare/candidate.jsonl',
}
TITLES = {
    'ko': ('16GB GPU에서 AI 모델 실행·검사하기', 'Qwen3-4B: 실행한 작업과 확인된 한계', '속도·메모리·정답 결과 읽는 법', '저장된 답변의 변화 찾기 · diova-compare'),
    'en': ('Run and check an AI model on a 16GB GPU', 'Qwen3-4B: work completed and known limits', 'Read speed, memory and answer results', 'Find changes in saved answers · diova-compare'),
}
# Only these previously published files can supply additional guide evidence.
# Existing Case008/009/013 extractors keep their original source checks as well.
SOURCE_PATHS = (
    C2+'/results/aggregate.json', C2+'/provenance/models.json', C2+'/README.md',
    C8+'/provenance/environment.json', C8+'/results/raw/Q/build.json',
    C8+'/REPORT.ko.md', C8+'/REPRODUCTION.md',
    C9+'/provenance/package_versions.json', C9+'/provenance/artifacts.json',
    C9+'/results/derived/serving_summary.json', C9+'/results/derived/quality_summary.json',
    C9+'/RUN_STATE.json', C9+'/configs/serving_protocol.json', C9+'/REPORT.ko.md', C9+'/REPRODUCTION.md',
    C13+'/provenance/observed_environment.json', C13+'/REPORT.ko.md',
    C13+'/summary/project.json',
    C13+'/records/attempts/suite_commands.json',
    C13+'/METHOD_CONTRACT.md', C13+'/PACKED_PROBE.md',
    FOLLOW+'/records/summary.json', FOLLOW+'/records/gpu_commands.json',
    FOLLOW+'/records/preregistration.json', FOLLOW+'/records/recommended_preset.json',
    FOLLOW+'/records/REPORT_KO.md', FOLLOW+'/README.ko.md',
)


def a(url: str, label: str) -> str:
    return '<a href="'+escape(url, quote=True)+'">'+escape(label)+'</a>'


def source(path: str, label: str) -> str:
    return a('https://github.com/Munsik-Kim/inference-lab/blob/'+REVISION+'/'+path, label)


def graph_memory_pairs(processes: list[dict]) -> dict:
    """Compare recorded whole-device peaks within each graph round, not by order."""
    rounds = {}
    for process in processes:
        match = re.fullmatch(r'round(\d+)-(BF16|W4)-graph', process['process_id'])
        if not match:
            continue
        round_id, arm = int(match[1]), match[2]
        peaks = rounds.setdefault(round_id, {})
        peak = process['sampled_device_peak_used_MiB']
        require(arm not in peaks, 'Duplicate graph memory arm')
        require(isfinite(peak) and peak > 0, 'Invalid graph memory peak')
        peaks[arm] = peak
    require(len(rounds) == 3, 'Expected three graph memory rounds')
    pairs = []
    for round_id, peaks in sorted(rounds.items()):
        require(set(peaks) == {'BF16', 'W4'}, 'Incomplete graph memory pair')
        saved = peaks['BF16'] - peaks['W4']
        pairs.append({'round': round_id, 'bf16_peak_mib': peaks['BF16'],
                      'w4_peak_mib': peaks['W4'], 'saved_mib': saved,
                      'reduction_fraction': saved / peaks['BF16']})
    return {'evidence_kind': 'CALCULATED_FROM_RECORDED_SAMPLES',
            'scope': 'whole-device per graph server, includes startup, warmup and desktop',
            'sample_interval_seconds': 1,
            'calculation': 'Pair by round; saved_MiB = BF16 - W4; reduction_fraction = saved_MiB / BF16; GiB = MiB / 1024',
            'pairs': pairs}


def load_practical(root: Path, data8: dict, data9: dict, data13: dict) -> dict:
    manifest = read(root/'presentation/practical_sources.json')
    require(manifest['evidence_revision'] == REVISION, 'Practical guide revision mismatch')
    require(set(manifest['files']) == set(SOURCE_PATHS), 'Unexpected practical guide source allowlist')
    for name in SOURCE_PATHS:
        path = root/name
        require(path.is_file() and not path.is_symlink() and sha(path) == manifest['files'][name],
                'Changed practical guide source: '+name)
    get = lambda name: checked_read(root, name, manifest)
    documents = get(C2+'/results/aggregate.json')
    models = get(C2+'/provenance/models.json')
    env8 = get(C8+'/provenance/environment.json')
    build8 = get(C8+'/results/raw/Q/build.json')
    env9 = get(C9+'/provenance/package_versions.json')
    env13 = get(C13+'/provenance/observed_environment.json')
    follow = get(FOLLOW+'/records/summary.json')
    commands = get(FOLLOW+'/records/gpu_commands.json')
    q = data8['tracks']['Q']
    require(q['model'] == data13['models']['4b']['id'] == models['bf16']['repo_id'], 'Guide model mismatch')
    require(q['model_revision'] == data13['models']['4b']['revision'] == models['bf16']['revision'], 'Guide model revision mismatch')
    require(all(r['exit_code'] == 0 for r in commands), 'Follow-up worker did not complete')
    require(follow['totals']['unique_previously_observed_tasks'] == 8, 'Unexpected DEV scope')
    sources = {**manifest['files'], **data8['sources'], **data9['sources'], **data13['sources']}
    return {
        'schema': 1, 'evidence_kind': 'REANALYSIS_OF_EXISTING_PUBLIC_RECORDS',
        'evidence_revision': REVISION, 'page_updated': '2026-10-15',
        'model': {'id': q['model'], 'revision': q['model_revision'],
                  'fp8_id': models['fp8']['repo_id'], 'fp8_revision': models['fp8']['revision']},
        'storage': {'weight_bytes': q['weight_bytes'], 'file_reduction_fraction': q['file_reduction_fraction'],
                    'allocator_peak_bytes': q['allocator_peak_bytes'],
                    'group_size': build8['quantization_config']['config_groups']['group_0']['weights']['group_size'],
                    'arms': q['arms']},
        'serving': {k: data9['serving'][k] for k in ('contrasts', 'cells', 'processes')},
        'device_memory': graph_memory_pairs(data9['serving']['processes']),
        'quality': data9['quality']['tasks'],
        'document_quality': {k: {'correct': v['all']['document_correct'], 'n': v['all']['n']}
                             for k, v in documents['quality'].items()},
        'reference': {'quality': [r for r in data13['quality'] if r['model'] == '4b'],
                      'packed': data13['packed'], 'long_not_run': data13['long_not_run']},
        'tool_repair': follow,
        'environment': {'build': env8['build']['effective_versions'], 'runtime': env8['runtime_versions'],
                        'case009': env9, 'reference': {'packages': env13['packages'], 'timestamp_utc': env13['timestamp_utc']},
                        'serving_frozen_utc': get(C9+'/configs/serving_protocol.json')['frozen_utc'],
                        'serving_completed_utc': get(C9+'/RUN_STATE.json')['updated_utc'],
                        'reference_worker_started_utc': next(r['started_utc'] for r in get(C13+'/records/attempts/suite_commands.json') if r['model']=='4b'),
                        'followup_started_utc': commands[0]['started_utc']},
        'sources': sources,
        'field_map': {
            'storage.weight_bytes': [C8+'/provenance/models.json', 'Q.files.*.bytes (safetensors sum)',
                                     C8+'/results/raw/Q/build.json', 'safetensors_bytes'],
            'serving': [C9+'/results/derived/serving_summary.json', 'contrasts / cells / processes'],
            'device_memory': [C9+'/results/derived/serving_summary.json',
                              'processes[process_id=roundN-{BF16,W4}-graph].sampled_device_peak_used_MiB',
                              'Calculated differences within each round; not per-request or per-concurrency peaks'],
            'quality': [C9+'/results/derived/quality_summary.json', 'tasks[].groups / overall_paired / process_status'],
            'document_quality': [C2+'/results/aggregate.json', 'quality.{bf16,fp8}.all.{document_correct,n}'],
            'reference': [C13+'/summary/project.json', 'quality[model=4b] / packed / long_not_run'],
            'tool_repair': [FOLLOW+'/records/summary.json', 'conditions[] / totals.unique_previously_observed_tasks'],
        },
    }


def table(headers: list[str], rows: list[list[str]], label: str) -> str:
    return ('<div class="scroll guide-table" role="region" tabindex="0" aria-label="'+escape(label, quote=True)+'">'
            '<table><caption>'+escape(label)+'</caption><thead><tr>'
            + ''.join('<th scope="col">'+escape(x)+'</th>' for x in headers)
            + '</tr></thead><tbody>'
            + ''.join('<tr>'+''.join(('<th scope="row">'+escape(str(v))+'</th>') if i == 0 else
                                    '<td>'+escape(str(v))+'</td>' for i, v in enumerate(row))+'</tr>' for row in rows)
            + '</tbody></table></div>')


def condition(data: dict, method: str, variant: str) -> dict:
    return next(r for r in data['tool_repair']['conditions'] if r['method'] == method and r['condition'] == variant)


def values(data: dict, lang: str) -> dict[str, str]:
    ko = lang == 'ko'; storage = data['storage']; s = data['serving']
    fmt_ratio = lambda r: f"{r['mean_ratio']:.3f} [{r['pointwise_95_interval'][0]:.3f}, {r['pointwise_95_interval'][1]:.3f}]"
    ratio = lambda length, concurrency: next(r['throughput_ratio']['mean_ratio'] for r in s['contrasts']
                                          if r['execution'] == 'graph' and r['input_tokens'] == length and r['client_concurrency'] == concurrency)
    tally = lambda method, variant: str(condition(data, method, variant)['successes'])+'/'+str(condition(data, method, variant)['tasks'])
    v = {'model_id': escape(data['model']['id']), 'model_revision': escape(data['model']['revision']),
         'fp8_id': escape(data['model']['fp8_id']), 'fp8_revision': escape(data['model']['fp8_revision']),
         'weight_files': ' → '.join(f"{storage['weight_bytes'][arm]/10**9:.3f} GB" for arm in ('Q-BF16', 'Q-W4')),
         'file_reduction': f"{100*storage['file_reduction_fraction']:.1f}%",
         'speed_small': f'{ratio(128,1):.3f}', 'speed_large': f'{ratio(1024,32):.3f}',
         'd16_before': tally('D16','BASELINE'), 'd16_after': tally('D16','BOTH'),
         'd4_before': tally('D4','BASELINE'), 'd4_after': tally('D4','BOTH'),
         'dev_tasks': str(data['tool_repair']['totals']['unique_previously_observed_tasks']),
         'episodes': str(data['tool_repair']['totals']['episodes']),
         'logical_conditions': str(data['tool_repair']['totals']['logical_conditions']),
         'd4_remaining': str(len(condition(data,'D4','BOTH')['failed_task_ids'])),
         'doc_bf16': f"{data['document_quality']['bf16']['correct']}/{data['document_quality']['bf16']['n']}",
         'doc_fp8': f"{data['document_quality']['fp8']['correct']}/{data['document_quality']['fp8']['n']}",
         'packed_error': f"{data['reference']['packed']['relative_rmse']:.5f}",
         'packed_threshold': f"{data['reference']['packed']['threshold']:.2f}",
         'updated': data['page_updated'], 'source_revision': REVISION,
         'followup_date': data['environment']['followup_started_utc'][:10]+' UTC',
         'reference_date': data['environment']['reference']['timestamp_utc'][:10]+' UTC',
         'serving_date': data['environment']['serving_frozen_utc'][:10]+' UTC',
         'reference_worker_date': data['environment']['reference_worker_started_utc'][:10]+' UTC',
    }
    v['storage_table'] = table(
        ['지표·단위' if ko else 'Metric / unit', 'BF16', 'GPTQ W4A16'],
        [[('학습된 숫자(가중치)의 저장 파일 · B' if ko else 'Saved learned numbers (weight files) · B'), *[f'{n:,}' for n in storage['weight_bytes'].values()]],
         [('PyTorch 계산용 메모리 최고값(Torch allocator peak) · GiB' if ko else 'Peak PyTorch calculation memory (Torch allocator peak) · GiB'), *[f'{n/2**30:.4f}' for n in storage['allocator_peak_bytes'].values()]]],
        'Case008 Q · 서로 다른 자원 지표' if ko else 'Case008 Q · separate resource measurements')
    v['speed_table'] = table(
        ['실행 방식 / 입력 조각(token) 수 / 동시 요청 상한' if ko else 'Execution method / input tokens / simultaneous request limit',
         '같은 시간에 생성한 출력량 W4/BF16 · 배 [95% 구간]' if ko else 'Output produced per second W4/BF16 · ratio [95% interval]'],
        [[f"{r['execution']} / {r['input_tokens']} / {r['client_concurrency']}", fmt_ratio(r['throughput_ratio'])] for r in s['contrasts']],
        'Case009 · 출력 256 token, 같은 조건의 원본·압축 서버를 묶은 3라운드(paired)' if ko else 'Case009 · 256 output tokens, three rounds pairing original and compressed servers')
    # Absolute request metrics at one stated condition; means of round medians.
    cells = {arm: [r for r in s['cells'] if r['arm'] == arm and r['execution'] == 'graph'
                   and r['input_tokens'] == 128 and r['client_concurrency'] == 1] for arm in ('BF16','W4')}
    v['latency_table'] = table(['지표·단위' if ko else 'Metric / unit', 'BF16', 'W4'],
        [[label, *[f'{mean(r[key] for r in cells[arm]):.3f}' for arm in ('BF16','W4')]] for key, label in
         [('ttft_p50_ms','첫 응답 TTFT · ms' if ko else 'Time to first response · ms'),
          ('tpot_p50_ms','이후 출력 토큰당 시간 TPOT · ms/token' if ko else 'Time per output token · ms/token'),
          ('latency_p50_ms','전체 요청 완료 시간 · ms' if ko else 'Whole request latency · ms'),
          ('output_tokens_per_second','출력 처리량 · token/s' if ko else 'Output throughput · token/s')]],
        'Case009 · graph, 입력 128 / 출력 256 / 동시 요청 상한 1, 라운드별 p50의 평균 (처리량은 라운드값 평균)' if ko
        else 'Case009 · graph, 128 input / 256 output / concurrency 1; mean of round medians (throughput: mean of round values)')
    pairs = data['device_memory']['pairs']
    span = lambda numbers, digits: f'{min(numbers):.{digits}f}–{max(numbers):.{digits}f}'
    v.update({
        'memory_bf16_gib': span([p['bf16_peak_mib']/1024 for p in pairs], 2)+' GiB',
        'memory_w4_gib': span([p['w4_peak_mib']/1024 for p in pairs], 2)+' GiB',
        'memory_saved_gib': span([p['saved_mib']/1024 for p in pairs], 2)+' GiB',
        'memory_saved_percent': span([100*p['reduction_fraction'] for p in pairs], 1)+'%',
        'memory_rounds': str(len(pairs)),
    })
    v['memory_table'] = table(['측정 대상' if ko else 'Measurement', 'BF16', 'GPTQ W4A16'],
        [[('장치 전체 최고 표본값의 범위' if ko else 'Range of whole-device sampled peaks'),
          v['memory_bf16_gib'], v['memory_w4_gib']]],
        'Case009 · graph 서버별 1초 간격(1 Hz) 측정, 시작·준비 실행(warmup)·바탕화면 포함' if ko else 'Case009 · samples once per second (1 Hz) per graph server, includes startup, warmup and desktop')
    v['memory_pairs_table'] = table(
        ['라운드' if ko else 'Round', 'BF16 · MiB', 'GPTQ W4A16 · MiB',
         '차이 · GiB' if ko else 'Difference · GiB', '감소율 · %' if ko else 'Reduction · %'],
        [[str(p['round']), f"{p['bf16_peak_mib']:,.0f}", f"{p['w4_peak_mib']:,.0f}",
          f"{p['saved_mib']/1024:.2f}", f"{100*p['reduction_fraction']:.1f}"] for p in pairs],
        'Case009 · 같은 라운드의 최고 표본값 비교, 차이·감소율은 계산값' if ko else 'Case009 · paired sampled peaks; differences and reductions are calculated')
    rows = []
    for task in data['quality']:
        if 'groups' not in task: continue
        status = task['process_status']
        state = ' / '.join(status[arm] for arm in ('BF16','W4'))
        if task['task'] == 'wikitext':
            arms = next(iter(task['groups'].values()))['arms']
            rows.append(['WikiText-2 · word PPL ↓', *[f"{arms[a]['word_perplexity']:.4f}" for a in ('BF16','W4')], state])
            continue
        groups = [('MMLU · acc',task['overall_paired'])] if task['task'] == 'mmlu' else list(task['groups'].items())
        for name, group in groups:
            rows.append([task['task']+' · '+name, *[f"{group[k]}/{group['n']} ({100*group[k]/group['n']:.2f}%)" for k in ('baseline_correct','candidate_correct')], state])
            if 'acc_norm' in group:
                r = group['acc_norm']; rows.append(['ARC · acc_norm', *[f"{r[k]}/{r['n']} ({100*r[k]/r['n']:.2f}%)" for k in ('baseline_correct','candidate_correct')],state])
    v['quality_table'] = table(['과제·지표' if ko else 'Task / metric', 'BF16', 'W4', '프로그램 정상 종료 여부 BF16 / W4' if ko else 'Program exit state BF16 / W4'], rows,
        'Case009 · 과제별 답변 평가, COMPLETE=정상 종료 / FAILED=계산 후 비정상 종료' if ko else 'Case009 · separate answer checks; COMPLETE=normal exit / FAILED=abnormal exit after calculation')
    v['reference_table'] = table(['조건' if ko else 'Condition', '다음 글 예측 점수(PPL) ↓' if ko else 'Next-text prediction score (PPL) ↓', '원본 확률과의 차이 KL(BF16 ∥ 후보) ↓' if ko else 'Difference from original probabilities KL(BF16 ∥ candidate) ↓', '원본 제곱합으로 나눈 가중치 제곱 오차(MSE) ↓' if ko else 'Weight squared error / original squared sum (MSE) ↓'],
        [[r['method'], f"{r['ppl']:.6f}", f"{r['forward_KL_BF16_to_candidate']:.6f}", f"{r['weight_normalized_mse']:.8f}"] for r in data['reference']['quality']],
        'Case013 · 4B 숫자를 BF16으로 복원한 품질 확인(reference), 같은 32개 2,048-token 구간' if ko else 'Case013 · 4B quality checks after BF16 restoration (reference), same 32 windows of 2,048 tokens')
    v['repair_table'] = table(['숫자 처리 조건' if ko else 'Number format condition', '원래 실행(BASELINE)' if ko else 'Original run (BASELINE)', '오류 안내(FEEDBACK)' if ko else 'Error guidance (FEEDBACK)', '등록 이름 제한(ENUM)' if ko else 'Registered names only (ENUM)', '두 방법 결합(BOTH)' if ko else 'Combined methods (BOTH)'],
        [[method,*[tally(method,variant) for variant in ('BASELINE','FEEDBACK','ENUM','BOTH')]] for method in ('D16','D4')],
        'Case013 후속 · 개발에 썼던 8개 문제(DEV)를 다시 실행한 실제 작업 성공' if ko else 'Case013 follow-up · actual task success on eight reused development tasks (DEV)')
    refs = {
        'source_storage': (C8+'/results/raw/Q/build.json','safetensors_bytes'),
        'source_bf16_files': (C8+'/provenance/models.json','Q.files.*.bytes'),
        'source_serving': (C9+'/results/derived/serving_summary.json','contrasts / cells / processes'),
        'source_quality': (C9+'/results/derived/quality_summary.json','tasks[].groups / process_status'),
        'source_document': (C2+'/results/aggregate.json','quality.*.all.document_correct'),
        'source_reference': (C13+'/summary/project.json','quality[model=4b] / packed'),
        'source_repair': (FOLLOW+'/records/summary.json','conditions[] / totals'),
        'source_environment8': (C8+'/provenance/environment.json','build.effective_versions / runtime_versions'),
        'source_environment9': (C9+'/provenance/package_versions.json','serving / evaluation'),
        'source_date9': (C9+'/configs/serving_protocol.json','frozen_utc'),
        'source_datefollow': (FOLLOW+'/records/gpu_commands.json','[0].started_utc'),
        'source_datereference': (C13+'/records/attempts/suite_commands.json','[model=4b].started_utc'),
    }
    for key,(path,field) in refs.items(): v[key] = source(path,path.split('/')[-1])+' · <code>'+escape(field)+'</code>'
    for key,path in {'report002':C2+'/README.md','report008':C8+'/REPORT.ko.md','reproduce008':C8+'/REPRODUCTION.md',
                     'report009':C9+'/REPORT.ko.md','reproduce009':C9+'/REPRODUCTION.md',
                     'report013':C13+'/REPORT.ko.md','reproduce013':C13+'/REPRODUCTION.ko.md',
                     'report_repair':FOLLOW+'/records/REPORT_KO.md','reproduce_repair':FOLLOW+'/README.ko.md'}.items():
        v[key] = 'https://github.com/Munsik-Kim/inference-lab/blob/'+REVISION+'/'+path
    v['source_index'] = '<ul class="source-list">'+''.join('<li>'+source(name,name)+' <code>'+digest+'</code></li>' for name,digest in sorted(data['sources'].items()))+'</ul>'
    return v


def render(root: Path, data: dict, lang: str, page: str) -> str:
    require(lang in TITLES and page in (*PAGES,'featured'), 'Unknown practical guide page')
    template = (root/f'presentation/guides/{lang}-{page}.html').read_text()
    replacements = values(data,lang)
    keys = set(re.findall(r'\{\{([a-z0-9_]+)\}\}',template))
    require(keys <= set(replacements), 'Unknown practical guide placeholder')
    return re.sub(r'\{\{([a-z0-9_]+)\}\}',lambda m: replacements[m[1]],template)
