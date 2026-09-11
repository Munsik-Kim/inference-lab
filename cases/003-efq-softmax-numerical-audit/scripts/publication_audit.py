"""Independent row-record reanalysis; no import of the original analysis code.

The original unit/aggregate files are read only after row-derived calculations,
to check them. New post-hoc outputs are written to an explicit output directory.
"""
import argparse
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

import numpy as np

CASE = Path(__file__).resolve().parents[1]
METHODS = ['nearest_ceil', 'nearest_efq_scale', 'efq_mmlu', 'efq_mean', 'efq_balance', 'efq_lut', 'efq_calibrated']
LABELS = {'nearest_ceil': 'Nearest, headroom scale', 'nearest_efq_scale': 'Nearest, EFQ scale',
          'efq_mmlu': 'EFQ-MMLU', 'efq_mean': 'EFQ-Mean', 'efq_balance': 'EFQ-Balance',
          'efq_lut': 'EFQ-LUT', 'efq_calibrated': 'EFQ calibrated', 'dense_fp32': 'FP32 dense', 'online_fp32': 'FP32 online'}
KEYS = ['document_id', 'length', 'layer', 'head', 'block', 'method']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(root, name):
    return json.loads((root / name).read_text())


def key(row):
    return (row['document_id'], int(row['length']), int(row['layer']), int(row['head']), int(row['block']), row['method'])


def quantile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary(values):
    return {'n': len(values), 'median': statistics.median(values), 'p95': quantile(values, .95), 'worst': max(values)}


def average_ranks(values):
    # Python sorted/group loops, separately from original NumPy rank routine.
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.] * len(values)
    a = 0
    while a < len(ordered):
        b = a + 1
        while b < len(ordered) and values[ordered[b]] == values[ordered[a]]:
            b += 1
        for i in ordered[a:b]:
            ranks[i] = (a + b - 1) / 2
        a = b
    return ranks


def spearman(pairs):
    x, y = zip(*pairs)
    return statistics.correlation(average_ranks(list(x)), average_ranks(list(y)))


def row_reconstruction(case, stage, plan):
    grouped = defaultdict(list)
    diagnostic_rows = []
    numerical = Counter()
    with gzip.open(case / 'results' / f'{stage}_rows.csv.gz', 'rt') as stream:
        for row in csv.DictReader(stream):
            k = key(row)
            grouped[k].append(row)
            numerical['rows'] += 1
            if row['valid'] != 'True':
                numerical['fully_masked'] += 1
                continue
            denominator = float(row['denominator']) if row['denominator'] else math.nan
            numerical['denominator_zero'] += denominator == 0
            numerical['denominator_nonfinite'] += not math.isfinite(denominator)
            numerical['recorded_bad'] += row['failed_valid_row'] == 'True'
            numerical['missing_or_nonfinite_error'] += not row['abs_error'] or not math.isfinite(float(row['abs_error']))
            numerical['near_zero_rows'] += row['near_zero_reference'] == 'True'
            assert row['js_nats'] and float(row['js_nats']) >= 0
            if stage in ['dev', 'eval']:
                assert int(row['query_position']) in plan['query_positions'][row['length']]
            if stage == 'eval' and k[4] == 32:
                diagnostic_rows.append(row)
    units = {}
    for k, rows in grouped.items():
        assert len(rows) == 16 and len({r['query_position'] for r in rows}) == 16, k
        if stage in ['dev', 'eval']:
            assert sorted(int(r['query_position']) for r in rows) == plan['query_positions'][str(k[1])]
        error_squared = math.fsum(float(r['abs_error']) ** 2 for r in rows)
        reference_squared = math.fsum(float(r['reference_norm']) ** 2 for r in rows)
        near = reference_squared <= (plan['metrics']['reference_norm_floor_rms'] ** 2) * len(rows) * 128
        units[k] = {**dict(zip(KEYS, k)), 'error_squared': error_squared, 'reference_squared': reference_squared,
                    'relative_error': None if near else math.sqrt(error_squared / reference_squared),
                    'near_zero_reference': near, 'language': rows[0]['language']}
    assert all(numerical[k] == 0 for k in ['fully_masked', 'denominator_zero', 'denominator_nonfinite', 'recorded_bad', 'missing_or_nonfinite_error', 'near_zero_rows']), dict(numerical)
    assert not any(u['near_zero_reference'] for u in units.values())
    return units, diagnostic_rows, dict(numerical)


def compute(case):
    post = read_json(case, 'configs/publication_audit_plan.json')
    spec = read_json(case, 'configs/experiment_spec.json')
    plan = spec['plan']
    spec_hash = digest(case / 'configs/experiment_spec.json')
    assert spec_hash == (case / 'configs/experiment_spec.sha256').read_text().split()[0]
    for name, expected in spec['frozen_files'].items():
        assert digest(case / name) == expected, name
    docs = read_json(case, 'inputs/manifest.json')['documents']
    ids = [d['document_id'] for d in docs]
    assert len(ids) == len(set(ids)) == 24
    assert len({d['text_sha256'] for d in docs}) == 24
    assert len({tuple(d['token_ids_4096'][:512]) for d in docs}) == 24
    assert len({d['topic'] for d in docs}) == 24
    languages = {split: dict(Counter(d['language'] for d in docs if d['split'] == split)) for split in ['dev', 'eval']}
    assert sum(languages['dev'].values()) == 8 and sum(languages['eval'].values()) == 16
    for d in docs:
        assert digest(case / 'inputs' / (d['document_id'] + '.txt')) == d['text_sha256']
        for n in plan['lengths']:
            encoded = json.dumps(d['token_ids_4096'][:n], separators=(',', ':')).encode()
            assert hashlib.sha256(encoded).hexdigest() == d['prefix_sha256'][str(n)]
    by_id = {d['document_id']: d for d in docs}
    for split, count in [('dev', 72), ('eval', 144)]:
        traces = read_json(case, f'provenance/{split}_traces.json')['traces']
        assert len(traces) == len({(t['document_id'], t['length'], t['layer']) for t in traces}) == count
        for t in traces:
            assert t['heads'] == [0, 5, 10, 15] and t['kv_heads'] == [0, 2, 5, 7]
            assert t['positions'] == plan['query_positions'][str(t['length'])]
            assert t['input_sha256'] == by_id[t['document_id']]['prefix_sha256'][str(t['length'])]
            assert t['model_revision'] == plan['model_revision']
            assert math.isclose(t['scaling'], 1 / math.sqrt(128), rel_tol=1e-15, abs_tol=0.)
    assert plan['tile'] == 128 and plan['primary_block'] == 32 and plan['sensitivity_blocks'] == [16, 64]
    for e in read_json(case, 'provenance/evaluation_execution.json'):
        assert e['exit_code'] == 0
        assert datetime.fromisoformat(e['started_at_utc']) > datetime.fromisoformat(spec['frozen_at_utc'])
    all_units, counts, rows = {}, {}, None
    sources = {}
    for stage, expected in [('dev', 2592), ('eval', 8640), ('synthetic-dev', 486), ('synthetic-eval', 810)]:
        units, diagnostic, numerical = row_reconstruction(case, stage, plan)
        assert len(units) == expected
        assert numerical['rows'] == expected * 16
        all_units[stage] = units
        counts[stage] = {'head_units': len(units), **numerical}
        sources[f'results/{stage}_rows.csv.gz'] = digest(case / 'results' / f'{stage}_rows.csv.gz')
        if stage == 'eval':
            rows = diagnostic
    primary = [u for u in all_units['eval'].values() if u['block'] == 32]
    for m in plan['methods']:
        for n in plan['lengths']:
            assert sum(u['method'] == m and u['length'] == n for u in primary) == 192
    assert all(k[4] == 32 or k[5] in plan['sensitivity_methods'] for k in all_units['eval'])
    length = [{ 'length': n, 'method': m, **summary([u['relative_error'] for u in primary if u['length'] == n and u['method'] == m]) }
              for n in plan['lengths'] for m in plan['methods']]
    index = {(e['length'], e['method']): e for e in length}
    decomposition = []
    for n in plan['lengths']:
        for contrast, numerator, denominator in post['contrasts']:
            a, b = index[n, numerator], index[n, denominator]
            decomposition.append({'length': n, 'contrast': contrast, 'numerator': numerator, 'denominator': denominator,
                                  **{f'{metric}_ratio': a[metric] / b[metric] for metric in ['median', 'p95']},
                                  **{f'{metric}_difference_pp': 100 * (a[metric] - b[metric]) for metric in ['median', 'p95']}})
    calibration = read_json(case, 'results/calibration.json')
    candidates = [(t, h) for t in plan['calibration']['tau'] for h in plan['calibration']['h']]
    assert len(candidates) == calibration['candidate_count'] == 49
    objectives = [math.sqrt(math.fsum(d['error_squared_by_candidate'][i] for d in calibration['per_document']) /
                            math.fsum(d['reference_squared'] for d in calibration['per_document'])) for i in range(49)]
    best = min(range(49), key=objectives.__getitem__)
    assert candidates[best] == (-2.9, 2.) and best == calibration['selected_index']
    assert {(c['tau'], c['h']) for c in calibration['candidates']} == set(candidates)
    assert [(d['document_id']) for d in calibration['per_document']] == [f'dev-{i:02d}' for i in range(8)]
    transfer = {}
    for split in ['dev', 'eval']:
        subset = [u for u in all_units[split].values() if u['method'] == 'efq_calibrated' and u['block'] == 32 and u['length'] == 2048 and u['layer'] == 13 and u['head'] in [0, 10]]
        transfer[split] = {'head_units': len(subset), 'relative_error': math.sqrt(math.fsum(u['error_squared'] for u in subset) / math.fsum(u['reference_squared'] for u in subset))}
    correlations = []
    for method in ['efq_mean', 'efq_calibrated']:
        for field in ['removed_reference_mass', 'entropy_nats', 'top_logit_gap', 'score_spread', 'max_row_max_jump']:
            pairs = [(float(r[field]), float(r['relative_error'])) for r in rows if r['method'] == method and r[field] and r['relative_error']]
            correlations.append({'method': method, 'feature': field, 'row_pairs': len(pairs), 'spearman': spearman(pairs)})
    rng = np.random.default_rng(post['bootstrap']['seed'])
    eval_ids = sorted(d['document_id'] for d in docs if d['split'] == 'eval')
    draws = rng.integers(0, len(eval_ids), size=(post['bootstrap']['replicates'], len(eval_ids)))
    intervals = []
    for n in plan['lengths']:
        arrays = {m: np.array([[all_units['eval'][(doc, n, layer, head, 32, m)]['relative_error'] for layer in plan['layers'] for head in plan['heads']] for doc in eval_ids]) for m in ['efq_calibrated', 'nearest_efq_scale', 'nearest_ceil']}
        medians = {m: np.median(a[draws].reshape(len(draws), -1), axis=1) for m, a in arrays.items()}
        for contrast, a, b in post['contrasts']:
            intervals.append({'length': n, 'contrast': contrast,
                              'median_difference_pp_ci95': np.quantile(100 * (medians[a] - medians[b]), [.025, .975]).tolist(),
                              'median_ratio_ci95': np.quantile(medians[a] / medians[b], [.025, .975]).tolist()})
    unitmap = all_units['eval']
    representatives = []
    for m in ['efq_mean', 'efq_calibrated']:
        u = max((u for u in primary if u['method'] == m), key=lambda u: u['relative_error'])
        other = unitmap[key(u)[:-1] + ('nearest_efq_scale',)]
        representatives.append({**u, 'nearest_same_scale_error': other['relative_error']})
    eligible = [u for u in primary if u['method'] == 'efq_calibrated' and unitmap[key(u)[:-1] + ('nearest_efq_scale',)]['relative_error'] > plan['screen']['baseline_relative_error_floor']]
    max_ratio_unit = max(eligible, key=lambda u: u['relative_error'] / unitmap[key(u)[:-1] + ('nearest_efq_scale',)]['relative_error'])
    max_ratio = max_ratio_unit['relative_error'] / unitmap[key(max_ratio_unit)[:-1] + ('nearest_efq_scale',)]['relative_error']
    groups = []
    for n in plan['lengths']:
        for layer in plan['layers']:
            values = {m: statistics.median(u['relative_error'] for u in primary if u['length'] == n and u['layer'] == layer and u['method'] == m) for m in ['efq_calibrated', 'nearest_efq_scale']}
            groups.append({'length': n, 'layer': layer, 'median_ratio': values['efq_calibrated'] / values['nearest_efq_scale']})
    screen_pass = all(e['median_ratio'] <= 1.1 and e['p95_ratio'] <= 1.25 for e in decomposition if e['contrast'] == 'mapping') and max(e['median_ratio'] for e in groups) <= 2.
    assert min(index[n, 'nearest_efq_scale']['median'] for n in plan['lengths']) > plan['screen']['baseline_relative_error_floor']
    flags = []
    rowmap = {(key(r), int(r['query_position'])): r for r in rows}
    rules = post['failure_diagnostics']
    for m in ['efq_mean', 'efq_calibrated']:
        bad = [r for r in rows if r['method'] == m and float(r['relative_error']) >= rules['row_error_threshold']]
        counter = Counter()
        for r in bad:
            assigned = []
            assigned.append(('removed_mass_association', float(r['removed_reference_mass']) >= rules['removed_mass_flag']))
            nvalid = int(r['query_position']) + 1
            assigned.append(('diffuse_high_entropy', nvalid > 1 and float(r['entropy_nats']) / math.log(nvalid) >= rules['normalized_entropy_flag']))
            assigned.append(('large_row_max_update', float(r['max_row_max_jump']) >= rules['row_max_jump_flag']))
            a, b = [float(rowmap[(key(r)[:-1] + (method,), int(r['query_position']))]['relative_error']) for method in ['nearest_efq_scale', 'nearest_ceil']]
            assigned.append(('scale_sensitive_nearest_control', b > 1e-5 and a / b >= rules['nearest_scale_error_ratio_flag'] and a - b >= rules['nearest_scale_error_difference_flag']))
            for label, value in assigned:
                counter[label] += value
            counter['none_of_these_flags'] += not any(value for _, value in assigned)
        flags.append({'method': m, 'high_error_rows': len(bad), 'all_primary_rows': 9216, **dict(counter), 'dominant_mechanism_unresolved': len(bad), 'nonzero_boundary_global_classification': 'Not available from aggregate row diagnostics.'})
    # Cross-check after all calculations; no reported aggregate drives output.
    discrepancies = []
    max_unit_delta = 0.
    for k, u in all_units['eval'].items():
        if k[4] == 32 and k[5] == 'efq_calibrated':
            assert u['relative_error'] == all_units['eval'][k[:-1] + ('efq_mmlu',)]['relative_error']
    for stage, units in all_units.items():
        reported = [json.loads(line) for line in (case / 'results' / f'{stage}_units.jsonl').read_text().splitlines()]
        assert len(reported) == len(units)
        for old in reported:
            difference = abs(units[key(old)]['relative_error'] - old['relative_error'])
            max_unit_delta = max(max_unit_delta, difference)
            if difference > post['unit_error_absolute_tolerance']:
                discrepancies.append({'location': f'{stage}:{key(old)}', 'difference': difference, 'kind': 'unit'})
    old_aggregate = read_json(case, 'results/aggregate.json')
    comparison_checks = []
    for old in old_aggregate['length']:
        new = index[old['length'], old['method']]
        for metric in ['median', 'p95', 'worst']:
            delta = new[metric] - old[metric]
            comparison_checks.append({'length': old['length'], 'method': old['method'], 'metric': metric, 'reported': old[metric], 'recomputed': new[metric], 'difference': delta, 'within_tolerance': abs(delta) <= post['unit_error_absolute_tolerance']})
            if abs(delta) > post['unit_error_absolute_tolerance']:
                discrepancies.append(comparison_checks[-1])
    for c in correlations:
        old = next(o for o in old_aggregate['exploratory_associations'] if o['method'] == c['method'] and o['feature'] == c['feature'])
        assert abs(old['spearman'] - c['spearman']) < 1e-12
    for new, old in zip(objectives, calibration['candidates']):
        assert abs(new - old['relative_frobenius']) < 1e-14
    assert abs(transfer['dev']['relative_error'] - objectives[best]) < post['unit_error_absolute_tolerance']
    native = read_json(case, 'provenance/trace_validation.json')['checks']
    return {'scope': 'Independent calculation path within the same Codex-assisted project; not a third-party or human replication.',
            'post_hoc_plan_sha256': digest(case / 'configs/publication_audit_plan.json'), 'original_spec_sha256': spec_hash,
            'input_hashes': sources, 'documents': {'dev': 8, 'eval': 16, 'languages': languages},
            'counts': counts, 'primary_length_summary': length, 'comparison_decomposition': decomposition,
            'calibration': {'candidate_count': 49, 'selected': dict(zip(['tau', 'h'], candidates[best])), 'selected_index': best,
                            'pooled_development_objective': objectives[best], 'matched_subset_transfer': transfer,
                            'plan_hash_difference_explained': calibration['plan_sha256'] != digest(case / 'configs/development_plan.json'),
                            'explanation': 'Calibration records the pre-JS-diagnostic-fix plan hash; development_corrections.json preserves it. Grid and output objective were unchanged.'},
            'online_fp32_max_error': max(u['relative_error'] for u in primary if u['method'] == 'online_fp32'),
            'native_bf16_validation': native, 'layer_length_ratios': groups, 'screen_pass': screen_pass,
            'correlations': correlations, 'post_hoc_clustered_uncertainty': intervals, 'failure_diagnostic_flags': flags,
            'worst_units': representatives, 'maximum_calibrated_same_scale_ratio': {'ratio': max_ratio, 'unit': max_ratio_unit},
            'cross_checks': {'all_numeric_mismatches_above_tolerance': discrepancies, 'max_unit_error_delta_from_fp32_row_rounding': max_unit_delta,
                             'reported_length_statistics': comparison_checks, 'interpretation': 'Every nonzero reporting difference is retained above; tiny differences reflect FP32 row-norm rounding.'}}


def tables(r):
    idx = {(x['length'], x['method']): x for x in r['primary_length_summary']}
    primary = '| Method | 512 tokens | 2048 tokens | 4096 tokens |\n|---|---:|---:|---:|\n'
    for m in METHODS:
        cells = [f"{100*idx[n,m]['median']:.2f}% / {100*idx[n,m]['p95']:.2f}%" for n in [512, 2048, 4096]]
        primary += '| ' + LABELS[m] + ' | ' + ' | '.join(cells) + ' |\n'
    contrasts = '| Contrast | Length | Median ratio | p95 ratio | Median difference | p95 difference |\n|---|---:|---:|---:|---:|---:|\n'
    for x in r['comparison_decomposition']:
        contrasts += f"| {x['contrast']} | {x['length']} | {x['median_ratio']:.3f} | {x['p95_ratio']:.3f} | {x['median_difference_pp']:+.3f} pp | {x['p95_difference_pp']:+.3f} pp |\n"
    cis = '| Contrast | Length | Median-difference 95% interval | Median-ratio 95% interval |\n|---|---:|---:|---:|\n'
    for x in r['post_hoc_clustered_uncertainty']:
        d, q = x['median_difference_pp_ci95'], x['median_ratio_ci95']
        cis += f"| {x['contrast']} | {x['length']} | [{d[0]:+.3f}, {d[1]:+.3f}] pp | [{q[0]:.3f}, {q[1]:.3f}] |\n"
    flags = '| Method | Rows with error >=20% | Removed mass >=10% | High entropy | Large max update | Scale-sensitive control | None of these flags |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for x in r['failure_diagnostic_flags']:
        flags += f"| {LABELS[x['method']]} | {x['high_error_rows']} | {x['removed_mass_association']} | {x['diffuse_high_entropy']} | {x['large_row_max_update']} | {x['scale_sensitive_nearest_control']} | {x['none_of_these_flags']} |\n"
    return primary, contrasts, cis, flags


def write_documents(case, output, r):
    primary, contrasts, cis, flags = tables(r)
    d, e = [100 * r['calibration']['matched_subset_transfer'][s]['relative_error'] for s in ['dev', 'eval']]
    maximum = max(x['median_ratio'] for x in r['layer_length_ratios'])
    failure = r['worst_units'][0]
    means = [x['median'] for x in r['primary_length_summary'] if x['method'] == 'efq_mean']
    full_ratios = [x['median_ratio'] for x in r['comparison_decomposition'] if x['contrast'] == 'full']
    first = (f'EFQ-Mean had {100*min(means):.2f}–{100*max(means):.2f}% median attention-output error on these Qwen3-0.6B traces. '
             'The calibrated/MMLU setting was close to same-scale nearest rounding in the aggregate, '
             'while headroom-scale nearest rounding had lower median and p95 error at every length. '
             'This was an FP32 numerical simulation; packed FP4 kernels, speedup and downstream quality were not measured.')
    readme = f'''# EFQ-Softmax on real Qwen attention traces

Can the published EFQ-Softmax operating points replace exp-then-nearest probability generation without materially increasing local attention-output error?

**{first}**

Values are **median / p95 relative output Frobenius error**, not task accuracy. Each length has 192 repeated head units (16 documents × 3 layers × 4 heads), with 16 sampled query rows per unit and all causal-valid keys per query. The independent document count is 16. Lengths share prefixes of the same documents.

{primary}
The development grid selected tau=-2.90, h=2.00, exactly the published EFQ-MMLU point. Both names remain in the table to separate the public setting from the development selection.

The prespecified screen uses **Nearest, EFQ scale** as its denominator. Under the same EFQ scale, the calibrated affine code generator stayed within the recorded aggregate error-ratio limits relative to exp-then-nearest rounding. That result does not compare EFQ with every MXFP4 implementation.

Against **Nearest, headroom scale**, calibrated EFQ's median error was {full_ratios[0]:.3f}×, {full_ratios[1]:.3f}× and {full_ratios[2]:.3f}× as large. The headroom reference uses an independently chosen scale rule; equivalence to the paper's MXFP4 baseline was not established.

A limited microkernel feasibility follow-up could test the cost of code generation while retaining both nearest controls. This audit does not recommend EFQ on fidelity alone. No kernel implementation or timing experiment was performed.

- [Analysis: mapping, scale, calibration and clustered uncertainty](ANALYSIS.md)
- [Methods, paper-to-code classification and reproduction commands](METHODS.md)
- [Independent row-derived calculations and all reporting differences](results/comparison_decomposition.json)
- [Raw per-query measurements](results/eval_rows.csv.gz) and [per-head records](results/eval_units.jsonl)
- [Fixed experiment specification](configs/experiment_spec.json) and [post-hoc audit plan](configs/publication_audit_plan.json)
- [Model](provenance/model.json), [environment](provenance/environment.json), [inputs](inputs/manifest.json), [small fixture](tests/fixtures/qwen_dev_small.npz)

The recorded setup was RTX 5080 SM120 / WSL2, Qwen3-0.6B BF16, Transformers 5.17.0 and PyTorch 2.13.0+cu130. Trace capture uses post-QK-normalization/post-RoPE Q/K with the corresponding V; QK, V, mask and FP32 accumulation are common to all methods. Tile size is 128 and the primary microscaling block is 32.

EFQ-Softmax and its published parameters are [Han et al.'s work](https://arxiv.org/html/2609.09721v1). This project contributes an independent implementation and numerical evidence. Codex assisted with synthetic inputs, code, execution, analysis and documentation. The independent audit refers to a separate calculation path within this project, not third-party replication. See [NOTICE](NOTICE.md).
'''
    analysis = f'''# Analysis

## Finding

{first}

## Experimental question

This audit tests the probability-generation path of [EFQ-Softmax v1](https://arxiv.org/html/2609.09721v1) under fixed Q/K/V and causal masking. Eight development and sixteen held-out documents use a shared self-authored synthetic English/Korean/code generator. It asks about sampled local attention fidelity within this prompt family and sampling design.

## Primary numerical results

Each cell is median / p95 relative output Frobenius error over 192 repeated head units. A unit pools its sixteen query outputs; it is not the mean of row-wise relative errors.

{primary}
The new calculation starts with each public row's absolute error and reference norm, sums their squares and divides the resulting Frobenius norms. Original unit and aggregate files serve only as cross-checks after calculation. The maximum unit discrepancy was {r['cross_checks']['max_unit_error_delta_from_fp32_row_rounding']:.3e}, attributable to rounding in the stored FP32 row norms. No discrepancy exceeded the fixed 2e-6 absolute-error tolerance; all two-decimal percentage table values agree. [All differences](results/comparison_decomposition.json) are retained, including those below tolerance.

FP32 online's maximum unit error was {r['online_fp32_max_error']:.6e}. The original 128-token trace check reproduced native BF16 SDPA exactly through the same backend; FP32 versus native BF16 differed by 0.142–0.186%. The latter is a precision/backend comparison, not EFQ error. Valid-row numerical failures and near-zero reference units were absent in the saved real and synthetic runs.

## Mapping effect, scale effect, full comparison

- **Mapping:** calibrated EFQ / nearest with the EFQ scale. Only the score-to-code mapping changes.
- **Scale:** nearest with the EFQ scale / nearest with the headroom scale. The nearest mapping is held fixed.
- **Full:** calibrated EFQ / nearest with the headroom scale. Both scale and code generation differ.

Ratios divide the indicated summary statistics; they are not medians of per-unit ratios. Differences are percentage points of relative output error (numerator minus denominator). They do not measure task-accuracy changes.

{contrasts}
![Comparison decomposition](results/comparison_decomposition.png)

The engineering screen's denominator is the **same-scale nearest reference**. Its length-wise median limits (1.10×), p95 limits (1.25×), valid-row numerical checks and layer×length limit (2×) all passed; the largest layer×length median ratio was {maximum:.3f}. This means the calibrated affine code generator stayed within the prespecified aggregate ratios under the same EFQ scale. Headroom nearest still gave lower median and p95 errors. The screen was recorded before evaluation; it was not externally preregistered, and this post-hoc analysis does not change it.

## Calibration transfer within the tested scope

The 7×7 Cartesian grid contains (-2.90, 2.00) because both values were listed before evaluation. Re-summing squared errors for all 49 candidates across the eight development documents selects that published MMLU point again. The objective is a concatenated-output Frobenius ratio, not a mean of unit ratios. The subset is length 2048, layer 13, heads 0/10 and the same sixteen query positions.

Using that subset definition, the objective changes from **{d:.3f}% development to {e:.3f}% evaluation**. The all-head median table covers three layers/four heads with a different aggregation and is not the same statistic. Calibration's older plan hash is explained by the documented development-only JS diagnostic fix; the grid and output objective did not change. The final specification predates evaluation extraction, and the selected parameters are fixed in all evaluation methods.

This checks transfer between document splits of the same model and synthetic generator. It says nothing about Qwen3-8B, other domains or downstream task accuracy. In v1 section IV-A, MMLU and Mean are selected for downstream MMLU and seven-task mean accuracy; Balance is selected on vision-language tasks. Higher local Frobenius error for Mean does not directly refute those task results. A follow-up question is why an operating point selected for downstream accuracy can differ from one selected for local attention-output fidelity.

## Representative counterexample

For `eval-13`, length 512, layer 27, head 0, EFQ-Mean's unit error was **{100*failure['relative_error']:.2f}%**, versus **{100*failure['nearest_same_scale_error']:.2f}%** for same-scale nearest. Query position **170 is zero-based** and uses keys 0–170. The recorded key 170 has reference probability 0.069700, EFQ-Mean probability 0.108802 and nearest probability 0.075533. Its scale exponent is -6 and residual is approximately -0.20839. EFQ's code-7 threshold is tau+6/h = -0.451304; nearest's 4/6 midpoint corresponds to log(5/6) = -0.182322. The residual lies between these thresholds, giving EFQ code 7 and nearest code 6.

Removed reference mass was only 0.35586%, and no finite row-max increase occurred after initialization. This is a **nonzero code-boundary counterexample that zero pruning alone cannot explain**. The normalized probabilities include historical rescaling. The public [key records](results/representative_details.json) and [case table](results/representative_cases.md) preserve the evidence; a small archived-score fixture and independent scalar check are described in Methods.

Calibrated EFQ's worst unit error was {100*r['worst_units'][1]['relative_error']:.2f}%; its largest unit-wise same-scale error ratio was {r['maximum_calibrated_same_scale_ratio']['ratio']:.2f}×. Grouped screen limits do not protect every head.

The following post-hoc flags describe rows with at least 20% error. High entropy means entropy/log(valid keys) >=0.8; a large max update is >=8 logits. A scale-sensitive control has nearest-EFQ error >=1.5× headroom-nearest error and at least 1 percentage point larger. The thresholds were fixed in the audit plan before this tally; they are not failure criteria from the paper.

{flags}
Flags overlap and rows are dependent. Removed mass, entropy and max updates do not establish a dominant cause. Global nonzero-boundary attribution needs per-key records that the row CSV does not contain. Dominant-mechanism classification remains unresolved rather than forcing each row into a causal label.

## Dependence structure and uncertainty

There are **16 independent document IDs, not 192 independent documents per length**. The 192 units repeat measurements across layers/heads, and lengths reuse document prefixes. English/Korean/code evaluation counts are 5/5/6; development counts are 3/3/2. All documents share generation rules, so even document-level intervals have limited population meaning.

The **post-hoc clustered uncertainty analysis** draws 16 document IDs with replacement, preserving all twelve layer/head units per document and pairing methods. The same 5,000 draws (seed 470011) are used at each length. Each draw recalculates method medians and then their difference/ratio. These percentile intervals supplement the original mean-document-difference bootstrap; they neither replace it nor alter the screen. The same-scale median intervals include both sides of zero difference (and ratio 1); this is not evidence of equivalence. The full-comparison intervals remain above ratio 1 in this sample.

{cis}
Row-level Spearman correlations for calibrated EFQ were +0.559 with removed mass, +0.511 with entropy and -0.405 with top-logit gap. They pool 9,216 dependent rows (8,640 for the gap, which is undefined for one-key rows). No independent-row p-value or causal interpretation is used. [Full counts and coefficients](results/comparison_decomposition.json) are retained.

## What this result establishes

The fixed v1 equations can be implemented and checked on real normalized/rotated Qwen3-0.6B attention inputs. Within this prompt family and sampling design, calibration finds a public operating point near the same-scale nearest reference in aggregate. The controlled comparisons separate that code-mapping observation from a larger scale-choice penalty, and the key-level counterexample exposes a nonzero boundary effect.

## What this result does not establish

This is FP32 simulation of decoded FP4 values on sampled traces. It does not validate packed FP4 tensor-core behavior, speedup, downstream quality, the authors' complete implementation or parameter transfer outside the tested scope. The paper leaves its precise hardware block layout and MXFP4 baseline details unresolved; both local baselines are explicit numerical references. No third-party or independent human reproduction is claimed.

## Decision for a possible microkernel follow-up

Consider only a limited feasibility experiment that measures code-generation cost at fixed operands and retains **both** nearest references. The same-scale screen motivates that narrow question; the present fidelity results alone favor headroom nearest. No microkernel was developed in this audit.

Recompute this document, README, JSON and figure from the row evidence with `python scripts/publication_audit.py --output-dir /path/to/new-derived-output`. Existing measured files are read-only. The generated bundle is post-hoc and is distinct from the frozen experiment.
'''
    (output / 'README.md').write_text(readme)
    (output / 'ANALYSIS.md').write_text(analysis)


def figure(output, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 140, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    names = {'mapping': 'Code mapping\nCalibrated / nearest EFQ scale', 'scale': 'Scale choice\nNearest EFQ / nearest headroom', 'full': 'Full comparison\nCalibrated / nearest headroom'}
    for ax, contrast in zip(axes, names):
        entries = [e for e in report['comparison_decomposition'] if e['contrast'] == contrast]
        xs = np.arange(3)
        ax.plot(xs, [e['median_ratio'] for e in entries], 'o-', label='Median ratio', color='#176a80')
        ax.plot(xs, [e['p95_ratio'] for e in entries], 's--', label='p95 ratio', color='#b05c31')
        ax.axhline(1., color='#555', linewidth=.8)
        ax.set_xticks(xs, ['512', '2048', '4096'])
        ax.set_xlabel('Input tokens')
        ax.set_title(names[contrast], fontsize=10)
        ax.set_ylim(.65, 1.8)
    axes[0].set_ylabel('Ratio of output-error summaries')
    axes[-1].legend(fontsize=9, loc='upper left')
    fig.suptitle('16 documents; 192 repeated head units per length; block 32')
    fig.tight_layout()
    fig.savefig(output / 'results/comparison_decomposition.png', metadata={'Software': 'CASE003 publication audit'})
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case-dir', type=Path, default=CASE)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    assert args.output_dir.resolve() != args.case_dir.resolve(), 'Use a separate output directory.'
    (args.output_dir / 'results').mkdir(parents=True, exist_ok=True)
    r = compute(args.case_dir)
    (args.output_dir / 'results/comparison_decomposition.json').write_text(json.dumps(r, indent=2, allow_nan=False) + '\n')
    write_documents(args.case_dir, args.output_dir, r)
    figure(args.output_dir, r)
    assert not r['cross_checks']['all_numeric_mismatches_above_tolerance'], 'Material reporting discrepancies found.'
    print(json.dumps({'counts': r['counts'], 'decomposition': r['comparison_decomposition'], 'max_row_rounding_delta': r['cross_checks']['max_unit_error_delta_from_fp32_row_rounding'], 'screen_pass': r['screen_pass']}, indent=2))


if __name__ == '__main__':
    main()
