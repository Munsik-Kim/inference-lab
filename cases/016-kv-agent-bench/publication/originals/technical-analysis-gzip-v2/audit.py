"""Stdlib-only independent raw-SSE, capacity and paired-block recalculation."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import pathlib
import statistics
from .common import load, save, sha_file, digest


def raw_request(path, prompts):
    """Does not trust saved success, token totals or the measurement parser."""
    row = load(path)
    prompt = prompts[row['prompt_id']]
    lines, tokens, text = [], [], ''
    prompt_seen, done, finish, usage = False, False, None, None
    token_times = []
    for e in row['sse']:
        line = e.get('line')
        if line is None:
            line = e['prefix'] + '[' + e['separator'].join(str(x) for x in prompts[e['prompt_ref']]['token_ids']) + ']' + e['suffix']
        lines.append(line)
        if not line.startswith('data:'):
            continue
        value = line[5:].strip()
        if value == '[DONE]':
            done = True
            continue
        event = json.loads(value)
        if event.get('usage'):
            usage = event['usage']
        for c in event.get('choices', []):
            assert c['index'] == 0, 'Unexpected choice index'
            if c.get('prompt_token_ids') is not None:
                assert c['prompt_token_ids'] == prompt['token_ids'], 'Actual prompt differs'
                prompt_seen = True
            ids = c.get('token_ids') or []
            tokens.extend(ids)
            if ids:
                token_times.append(e['elapsed'])
            text += c.get('text', '')
            if c.get('finish_reason') is not None:
                finish = c['finish_reason']
    assert hashlib.sha256(('\n'.join(lines) + '\n').encode()).hexdigest() == row['raw_sse_sha256'], 'Raw SSE changed'
    if 'output_token_ids' in row:
        assert tokens == row['output_token_ids'] and text == row['text'], 'Stored raw summary differs'
    valid = prompt_seen and done and usage is not None and usage['prompt_tokens'] == len(prompt['token_ids']) and usage['completion_tokens'] == len(tokens)
    if row['ignore_eos']:
        valid &= len(tokens) == row['output_limit'] and finish == 'length'
    if row['status'] in ('SUCCESS', 'RAW_OR_LENGTH_MISMATCH'):
        assert valid == (row['status'] == 'SUCCESS'), 'Recorded status differs from independent raw calculation'
    return {**{k: v for k, v in row.items() if k not in ('sse', 'text', 'output_token_ids', 'token_events')},
            'raw_success': valid, 'generated_tokens': len(tokens), 'tokens': tokens, 'text': text,
            'ttft': token_times[0] - row['http_dispatch'] if token_times else None,
            'decode_mean': (token_times[-1] - token_times[0]) / (len(tokens) - 1) if len(tokens) > 1 else None,
            'source_file': path.as_posix(), 'source_sha256': sha_file(path)}


def prometheus(text, name):
    rows = [line for line in text.splitlines() if not line.startswith('#') and line.split('{')[0].split(' ')[0] == name]
    return sum(float(line.split()[-1]) for line in rows) if rows else None


def counter_delta(before, after, name):
    a, b = prometheus(before, name), prometheus(after, name)
    if a is None or b is None:
        return None
    assert b >= a, 'Counter reset: ' + name
    return b - a


def csv_write(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(rows[0])
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def derive(root):
    prompts = {r['prompt_id']: r for r in load(root / 'data/prompts.json')}
    planned = load(root / 'data/planned_blocks.json')
    keys = [r['block_id'] for r in planned]
    assert len(keys) == len(set(keys)) == 36
    frozen = load(root / 'provenance/execution_freeze.json')
    for entry in frozen['files']:
        if 'path' in entry:
            assert sha_file(root / entry['path']) == entry['sha256'], 'Frozen execution input/source changed'
        else:
            assert sha_file(root.parent.parent / entry['original_path']) == entry['sha256'], 'Read-only original import changed'
    servers = [load(p) for p in (root / 'results/servers').glob('R*/cache_before.json')]
    signatures = {r['parameter_sample_sha256'] for r in servers}
    modes = {r['compilation_mode'] for r in servers}
    assert len(signatures) <= 1, 'Sampled model parameters differ between candidate servers'
    assert len(modes) <= 1, 'Different execution mode between timed servers'
    capacity = []
    for kv in ('B16', 'F4'):
        p = root / 'results/servers' / ('p0_' + kv) / 'cache_before.json'
        if not p.exists():
            continue
        cache = load(p)
        bpt = 2 * 36 * 8 * 128 * 2 if kv == 'B16' else 2 * 36 * 8 * (128 // 2 + 128 // 16)
        actual = cache['actual_unique_storage_bytes']
        block = cache['cache_config']['block_size']
        assert actual % (block * bpt) == 0, 'Pool not a whole number of blocks'
        blocks = actual // (block * bpt)
        assert cache['cache_config']['num_gpu_blocks'] in (None, blocks)
        capacity.append({'kv': kv, 'bytes_per_token': bpt, 'actual_pool_bytes': actual, 'block_size': block, 'blocks': blocks,
                         'token_slots': blocks * block, 'layers': len(cache['layers']), 'actual_cache_dtypes': sorted({r['dtype'] for r in cache['cache_tensors']}),
                         'source_file': str(p.relative_to(root)), 'source_sha256': sha_file(p)})
    blocks, request_rows = [], []
    for plan in planned:
        folder = root / 'results/blocks' / plan['block_id']
        if not (folder / 'block.json').exists():
            continue
        paths = sorted(folder.glob('request_*.json'))
        assert len(paths) == 8, 'Incomplete block explicitly requires missing ledger'
        raw = [raw_request(p, prompts) for p in paths]
        assert [r['prompt_id'] for r in raw] == plan['prompt_ids']
        for row in raw:
            for key in ('kv', 'round', 'length', 'concurrency', 'block_id'):
                assert row[key] == plan[key], 'Wrong block identity'
            row['source_file'] = str(pathlib.Path(row['source_file']).relative_to(root))
        saved = load(folder / 'block.json')
        makespan = max(r['terminal'] for r in raw)
        assert makespan == saved['makespan_seconds'] and makespan > 0
        before, after = (folder / 'metrics_before.txt').read_text(), (folder / 'metrics_after.txt').read_text()
        engine = {}
        for metric in ('request_queue_time_seconds', 'request_decode_time_seconds', 'time_to_first_token_seconds', 'request_prefill_time_seconds'):
            numerator = counter_delta(before, after, 'vllm:' + metric + '_sum')
            denominator = counter_delta(before, after, 'vllm:' + metric + '_count')
            engine[metric + '_mean'] = numerator / denominator if numerator is not None and denominator else None
        sample_file = folder / 'samples.json'
        samples = load(sample_file) if sample_file.exists() else json.loads(gzip.decompress(sample_file.with_suffix('.json.gz').read_bytes()))
        cache = load(folder / 'cache_after.json')
        device_used = []
        for sample in samples:
            values = sample.get('gpu', {}).get('csv', '').split(',')
            if len(values) >= 3:
                device_used.append(float(values[1]))
        valid = sum(r['raw_success'] for r in raw)
        ttfts = [r['ttft'] for r in raw if r['ttft'] is not None]
        decodes = [r['decode_mean'] for r in raw if r['decode_mean'] is not None]
        waiting = [prometheus(s.get('metrics', ''), 'vllm:num_requests_waiting') for s in samples]
        waiting = [v for v in waiting if v is not None]
        blocks.append({**{k: v for k, v in plan.items() if k != 'prompt_ids'}, 'requests': len(raw), 'success': valid,
                       'model_request_attempts': sum(r['model_request_attempts'] for r in raw), 'responses_received': sum(r['responses_received'] for r in raw),
                       'output_tokens': sum(r['generated_tokens'] for r in raw), 'makespan_seconds': makespan,
                       'requests_per_second': valid / makespan, 'output_tokens_per_second': sum(r['generated_tokens'] for r in raw) / makespan,
                       'median_client_ttft_seconds': statistics.median(ttfts) if ttfts else None,
                       'median_client_decode_ms_per_token': 1000 * statistics.median(decodes) if decodes else None,
                       'preemptions': counter_delta(before, after, 'vllm:num_preemptions_total'),
                       'max_waiting': max(waiting) if waiting else None,
                       'torch_peak_allocated': cache['peak_allocated'], 'torch_peak_reserved': cache['peak_reserved'],
                       'sampled_device_peak_MiB': max(device_used) if device_used else None,
                       **engine, 'explicit_recompute_count': None, 'recompute_reason': 'No per-block dedicated recompute counter observed; preemptions kept separate',
                       'source_file': str((folder / 'block.json').relative_to(root)), 'source_sha256': sha_file(folder / 'block.json')})
        request_rows.extend({k: v for k, v in r.items() if k not in ('tokens', 'text')} for r in raw)
    pairs = []
    for rnd in (1, 2, 3):
        for length in (8192, 16384):
            for concurrency in (1, 2, 4):
                rows = {r['kv']: r for r in blocks if (r['round'], r['length'], r['concurrency']) == (rnd, length, concurrency)}
                if set(rows) == {'B16', 'F4'}:
                    a, b = rows['B16'], rows['F4']
                    pairs.append({'round': rnd, 'length': length, 'concurrency': concurrency, 'B16_requests_per_second': a['requests_per_second'],
                                  'F4_requests_per_second': b['requests_per_second'], 'F4_over_B16': b['requests_per_second'] / a['requests_per_second'] if a['requests_per_second'] else None})
    summary = []
    for length in (8192, 16384):
        for concurrency in (1, 2, 4):
            pair = [p for p in pairs if p['length'] == length and p['concurrency'] == concurrency]
            ratios = [p['F4_over_B16'] for p in pair if p['F4_over_B16'] is not None]
            row = {'length': length, 'concurrency': concurrency, 'paired_rounds': len(pair), 'paired_ratios': ratios,
                   'ratio_median': statistics.median(ratios) if ratios else None, 'ratio_min': min(ratios) if ratios else None, 'ratio_max': max(ratios) if ratios else None}
            for kv in ('B16', 'F4'):
                group = [b for b in blocks if (b['kv'], b['length'], b['concurrency']) == (kv, length, concurrency)]
                for label in ('requests_per_second', 'median_client_decode_ms_per_token', 'median_client_ttft_seconds', 'sampled_device_peak_MiB', 'request_queue_time_seconds_mean'):
                    values = [r[label] for r in group if r[label] is not None]
                    row[kv + '_' + label] = {'median': statistics.median(values), 'min': min(values), 'max': max(values)} if values else None
                row[kv + '_success'] = sum(b['success'] for b in group)
                row[kv + '_preemptions'] = [b['preemptions'] for b in group]
            summary.append(row)
    sanity = []
    for length in (8192, 16384):
        for i in range(8):
            results = {}
            for kv in ('B16', 'F4'):
                path = root / f'results/sanity/{kv}_{length}/request_{i:02}.json'
                if path.exists():
                    results[kv] = raw_request(path, prompts)
            if set(results) != {'B16', 'F4'}:
                continue
            a, b = results['B16'], results['F4']
            expected = prompts[a['prompt_id']]['expected_answer']
            divergence = next((j for j, (x, y) in enumerate(zip(a['tokens'], b['tokens'])) if x != y), min(len(a['tokens']), len(b['tokens'])) if a['tokens'] != b['tokens'] else None)
            sanity.append({'prompt_id': a['prompt_id'], 'length': length, 'kind': prompts[a['prompt_id']]['kind'], 'expected': expected,
                           'B16_text': a['text'], 'F4_text': b['text'], 'B16_tokens': len(a['tokens']), 'F4_tokens': len(b['tokens']),
                           'B16_exact_answer': a['text'].strip() == expected if expected is not None else None,
                           'F4_exact_answer': b['text'].strip() == expected if expected is not None else None,
                           'B16_finish': a.get('finish_reason'), 'F4_finish': b.get('finish_reason'), 'first_token_divergence': divergence,
                           'token_sequences_equal': a['tokens'] == b['tokens']})
    historical_fp8 = []
    historical_file = root.parent.parent / 'results/p0_diagnostics_r2/F8_final_config_cache_before.json'
    if historical_file.exists():
        c = load(historical_file)
        actual = c['actual_unique_storage_bytes']
        bpt = 2 * 36 * 8 * 128
        historical_fp8.append({'kv': 'F8', 'evidence': 'HISTORICAL_P0', 'bytes_per_token': bpt, 'actual_pool_bytes': actual, 'token_slots': actual // bpt,
                               'source_file': '../../results/p0_diagnostics_r2/F8_final_config_cache_before.json', 'source_sha256': sha_file(historical_file)})
    counts = {}
    for phase, sub in [('p0_numeric', 'p0'), ('p0_graph', 'graph'), ('warmup', 'warmup'), ('sanity', 'sanity'), ('timed', 'blocks')]:
        rows = [raw_request(p, prompts) for p in sorted((root / 'results' / sub).glob('*/request_*.json'))]
        counts[phase] = {'records': len(rows), 'model_request_attempts': sum(r['model_request_attempts'] for r in rows), 'responses_received': sum(r['responses_received'] for r in rows), 'output_tokens': sum(r['generated_tokens'] for r in rows)}
    missing = [p for p in planned if p['block_id'] not in {b['block_id'] for b in blocks}]
    return {'schema': 'case016-technical-derived-v1', 'capacity': capacity, 'historical_fp8': historical_fp8, 'blocks': blocks, 'paired_rounds': pairs, 'summary': summary, 'sanity': sanity,
            'invariants': {'frozen_source_and_input_match': True, 'timed_servers': len(servers), 'parameter_sample_signatures': sorted(signatures), 'signature_scope': 'First/last8 values per named parameter, not full GPU weight hashing',
                           'actual_compilation_modes': sorted(modes), 'legacy_cudagraph_policy_note': 'Inherited CacheInspector.cudagraph_policy is a fixed eager-description string. Use actual compilation_mode and startup capture logs; raw legacy field is retained unchanged.'},
            'request_counts': counts, 'missing_blocks': missing, 'complete': len(blocks) == 36 and all(b['success'] == 8 for b in blocks),
            'model_revision': 'cdbee75f17c01a7cc42f958dc650907174af0554',
            'requests': request_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.output.exists():
        raise ValueError('Fresh output directory required')
    args.output.mkdir(parents=True)
    result = derive(root)
    for key, filename in [('blocks', 'per-round.csv'), ('paired_rounds', 'paired-rounds.csv'), ('capacity', 'capacity.csv'), ('historical_fp8', 'historical-fp8-capacity.csv'), ('sanity', 'fidelity.csv')]:
        csv_write(args.output / filename, result[key])
    summary = [{k: (json.dumps(v, separators=(',', ':')) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in result['summary']]
    csv_write(args.output / 'summary.csv', summary)
    save(args.output / 'derived.json', result)
    save(args.output / 'audit_receipt.json', {'status': 'PASS' if result['complete'] else 'PARTIAL', 'blocks': len(result['blocks']), 'timed_requests': len(result['requests']), 'raw_sse_and_prompt_ids_verified': True,
                                            'new_gpu_requests': 0, 'independent_from_saved_speed_and_success_summary': True, 'missing': result['missing_blocks'], 'request_counts': result['request_counts']})
    print(json.dumps({'complete': result['complete'], 'blocks': len(result['blocks']), 'counts': result['request_counts']}))


if __name__ == '__main__':
    main()
