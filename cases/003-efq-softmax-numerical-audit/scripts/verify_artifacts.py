"""Static cross-checks of the saved audit; no model, GPU or network calls."""
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import zipfile

CASE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((CASE / name).read_text())


def unit_key(row):
    return tuple(str(row[k]) for k in ('document_id', 'length', 'layer', 'head', 'block', 'method'))


def quantile(values, q):
    ordered = sorted(values)
    x = q * (len(ordered) - 1)
    lo = math.floor(x)
    hi = math.ceil(x)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (x - lo)


def close(a, b, tolerance=1e-12):
    assert math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance), (a, b)


def main():
    spec = read('configs/experiment_spec.json')
    expected_sha = (CASE / 'configs/experiment_spec.sha256').read_text().split()[0]
    assert sha(CASE / 'configs/experiment_spec.json') == expected_sha
    for name, expected in spec['frozen_files'].items():
        assert sha(CASE / name) == expected, name
    docs = read('inputs/manifest.json')['documents']
    assert Counter(d['split'] for d in docs) == {'dev': 8, 'eval': 16}
    assert len({d['document_id'] for d in docs}) == 24
    assert len({d['text_sha256'] for d in docs}) == 24
    assert len({tuple(d['token_ids_4096'][:512]) for d in docs}) == 24
    for d in docs:
        assert sha(CASE / 'inputs' / (d['document_id'] + '.txt')) == d['text_sha256']
        assert len(d['token_ids_4096']) == 4096
        for n in spec['plan']['lengths']:
            blob = json.dumps(d['token_ids_4096'][:n], separators=(',', ':')).encode()
            assert hashlib.sha256(blob).hexdigest() == d['prefix_sha256'][str(n)]
    doc_by_id = {d['document_id']: d for d in docs}
    for split, expected in [('dev', 72), ('eval', 144)]:
        traces = read(f'provenance/{split}_traces.json')['traces']
        assert len(traces) == expected
        assert len({(t['document_id'], t['length'], t['layer']) for t in traces}) == expected
        for t in traces:
            assert t['split'] == split
            assert t['heads'] == spec['plan']['heads'] and t['kv_heads'] == [0, 2, 5, 7]
            assert t['positions'] == spec['plan']['query_positions'][str(t['length'])]
            assert t['input_sha256'] == doc_by_id[t['document_id']]['prefix_sha256'][str(t['length'])]
            assert t['model_revision'] == spec['plan']['model_revision']
            close(t['scaling'], 1 / math.sqrt(128))
    for execution in read('provenance/evaluation_execution.json'):
        assert execution['exit_code'] == 0
        assert datetime.fromisoformat(execution['started_at_utc']) > datetime.fromisoformat(spec['frozen_at_utc'])
    checks = read('provenance/trace_validation.json')['checks']
    assert all(c['same_backend_pass'] and c['fp32_pass'] for c in checks)
    tests = read('provenance/numerical_tests.json')
    assert tests['success'] and tests['tests_run'] == 12 and tests['skipped'] == 0
    calibration = read('results/calibration.json')
    assert calibration['candidate_count'] == 49
    selected = min(calibration['candidates'], key=lambda c: c['relative_frobenius'])
    assert spec['calibrated_parameters'] == {k: selected[k] for k in ['tau', 'h']}
    close(calibration['selected_objective'], selected['relative_frobenius'])
    counts = {}
    all_units = {}
    for stage, expected in [('dev', 2592), ('synthetic-dev', 486), ('eval', 8640), ('synthetic-eval', 810)]:
        units = [json.loads(line) for line in (CASE / 'results' / f'{stage}_units.jsonl').read_text().splitlines()]
        assert len(units) == expected
        by_key = {unit_key(u): u for u in units}
        assert len(by_key) == expected
        row_counts = Counter()
        sums = defaultdict(lambda: [0., 0.])
        with gzip.open(CASE / 'results' / f'{stage}_rows.csv.gz', 'rt') as f:
            for row in csv.DictReader(f):
                key = unit_key(row)
                assert key in by_key
                row_counts[key] += 1
                assert row['valid'] == 'True' and row['failed_valid_row'] == 'False'
                assert row['near_zero_reference'] == 'False'
                assert float(row['denominator']) > 0 and math.isfinite(float(row['denominator']))
                assert float(row['js_nats']) >= 0 and math.isfinite(float(row['js_nats']))
                sums[key][0] += float(row['abs_error']) ** 2
                sums[key][1] += float(row['reference_norm']) ** 2
                if not stage.startswith('synthetic'):
                    assert int(row['query_position']) in spec['plan']['query_positions'][row['length']]
        assert len(row_counts) == expected and set(row_counts.values()) == {16}
        for key, u in by_key.items():
            assert u['failed_valid_rows'] == 0 and not u['near_zero_reference']
            close(u['absolute_error_frobenius'] ** 2, u['error_squared'])
            close(u['relative_error'], math.sqrt(u['error_squared'] / u['reference_squared']))
            # Row norms are rounded FP32 values; unit sums use FP64 aggregation.
            close(sums[key][0], u['error_squared'], 2e-6)
            close(sums[key][1], u['reference_squared'], 2e-6)
        run = read(f'results/{stage}_run.json')
        assert run['head_units'] == expected and run['row_records'] == 16 * expected
        assert run['valid_bad_rows'] == 0 and run['fully_masked_rows'] == 0
        assert run['probability_reconstruction_worst_relative'] < 5e-5
        counts[stage] = {'head_units': expected, 'rows': 16 * expected}
        all_units[stage] = units
    primary = [u for u in all_units['eval'] if u['block'] == 32]
    assert len(primary) == 5184
    primary_by_key = {unit_key(u): u for u in primary}
    for u in primary:
        if u['method'] == 'efq_calibrated':
            other = primary_by_key[unit_key(u)[:-1] + ('efq_mmlu',)]
            assert {k: v for k, v in u.items() if k != 'method'} == {k: v for k, v in other.items() if k != 'method'}
    aggregate = read('results/aggregate.json')
    readme = (CASE / 'README.md').read_text()
    for entry in aggregate['length']:
        values = [u['relative_error'] for u in primary if u['length'] == entry['length'] and u['method'] == entry['method']]
        assert len(values) == entry['n'] == 192
        for name, value in [('median', statistics.median(values)), ('p95', quantile(values, .95)), ('worst', max(values))]:
            close(entry[name], value)
        if entry['method'] not in ['dense_fp32', 'online_fp32']:
            assert f"{100*entry['median']:.2f}% / {100*entry['p95']:.2f}%" in readme
    for check in aggregate['screen']['checks']:
        def subset(method):
            return [u['relative_error'] for u in primary if u['method'] == method and u['length'] == check['length'] and ('layer' not in check or u['layer'] == check['layer'])]
        reducer = (lambda x: quantile(x, .95)) if check['scope'] == 'p95' else statistics.median
        base, calibrated = reducer(subset('nearest_efq_scale')), reducer(subset('efq_calibrated'))
        close(check['baseline'], base)
        close(check['calibrated'], calibrated)
        close(check['ratio'], calibrated / base)
        assert check['pass'] == (calibrated / base <= check['limit'])
    assert aggregate['screen']['passed'] == all(c['pass'] for c in aggregate['screen']['checks'])
    for p in CASE.rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p.relative_to(CASE)))
    linked_files = 0
    for p in CASE.rglob('*.md'):
        for link in re.findall(r'\]\(([^\s)]+)\)', p.read_text()):
            if link.startswith(('http:', 'https:', '#', 'mailto:')):
                continue
            assert (p.parent / link.split('#')[0]).exists(), (p.relative_to(CASE), link)
            linked_files += 1
    with zipfile.ZipFile(CASE / 'tests/fixtures/qwen_dev_small.npz') as z:
        assert z.testzip() is None
    report = {'scope': 'Static artifact validation only; no new GPU execution or external replication.',
              'status': 'PASS', 'spec_sha256': expected_sha,
              'frozen_files_verified': len(spec['frozen_files']), 'documents': {'dev': 8, 'eval': 16},
              'measurement_counts': counts, 'relative_markdown_links_checked': linked_files,
              'checks': ['Frozen hashes and pre-evaluation chronology', 'Document and tokenizer-prefix identity',
                         'Trace metadata and GQA head mapping', 'Numerical-test and native-check records',
                         'Candidate minimum and selected parameters', 'Row/unit correspondence and squared norms',
                         'Normalized JS nonnegativity and numerical failure counts',
                         'Primary medians, p95, worst and README table values', 'Screen recomputed from saved units',
                         'MMLU/calibrated exact unit equality', 'Python syntax, local links and fixture CRC']}
    (CASE / 'provenance/artifact_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
