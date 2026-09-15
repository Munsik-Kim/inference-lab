"""Post-hoc scalar analysis, separate from the frozen DEV decision.

Uses only the Python standard library. Threshold examples were selected after
results were known. They never select a finalist or rewrite original results.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

CONFIGS = ('B', 'A_PUBLIC', 'V1', 'V2', 'V4')
SCENARIOS = ((.01, .03, 1.50), (.01, .03, 1.25),
             (.02, .045, 1.50), (.02, .045, 1.25), (.03, .05, 1.50))


def quantile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def scalar_error(unit):
    numerator = math.fsum(x * x for x in unit['row_error_norms'])
    denominator = math.fsum(x * x for x in unit['row_reference_norms'])
    if denominator <= 0 or not math.isfinite(numerator + denominator):
        raise ValueError('Undefined scalar unit; no epsilon or dropped unit')
    return math.sqrt(numerator / denominator)


def paired_units(anchor, candidate):
    """Join unique (document, head) keys; never rely on array position."""
    def keyed(records):
        result = {}
        for document, head, error in records:
            key = (document, head)
            if key in result or not math.isfinite(error):
                raise ValueError('Duplicate or invalid paired unit')
            result[key] = error
        return result
    a, b = keyed(anchor), keyed(candidate)
    if not a or a.keys() != b.keys():
        raise ValueError('Missing paired unit')
    return [dict(document_id=d, head=h, anchor_error=a[d, h],
                 v4_error=b[d, h], error_reduction_pp=100 * (a[d, h] - b[d, h]))
            for d, h in sorted(a)]


def calculate(case):
    records, hashes = [], {}
    for round_id in range(2):
        path = case / f'results/dev/round-{round_id}.json'
        hashes[str(path.relative_to(case))] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = json.loads(path.read_text())
        assert raw['evidence_kind'] == 'gpu_measurement' and not raw['mock']
        assert raw['status'] == 'COMPLETED' and raw['round'] == round_id
        assert all(m['round'] == round_id and m['split'] == 'dev' for m in raw['measurements'])
        records.extend(raw['measurements'])
    index = {(m['document_id'], m['round'], m['config_id']): m for m in records}
    documents = sorted({m['document_id'] for m in records})
    assert len(documents) == 8 and len(index) == len(records) == 80
    assert {m['config_id'] for m in records} == set(CONFIGS)
    rows, units_by_config = [], {}
    for cid in CONFIGS:
        units, cosines, ratios = [], [], []
        for d in documents:
            primary = index[d, 0, cid]
            assert sorted(u['head'] for u in primary['units']) == list(range(16))
            for u in primary['units']:
                error = scalar_error(u)
                assert math.isclose(error, u['relative_output_error'], rel_tol=1e-12)
                units.append((d, u['head'], error))
                cosines.append(u['dot_sum'] / math.sqrt(u['actual_squared_sum'] * u['reference_squared_sum']))
            for r in range(2):
                base, other = index[d, r, 'B'], index[d, r, cid]
                assert base['input_sha256'] == other['input_sha256']
                assert base['full_output_nonfinite'] == other['full_output_nonfinite'] == 0
                a, b = base['wall_ms'], other['wall_ms']
                assert len(a) == len(b) == 5 and all(math.isfinite(t) and t > 0 for t in a + b)
                ratios.extend(x / y for x, y in zip(a, b))
        values = [u[2] for u in units]
        speedup = statistics.median(ratios)
        units_by_config[cid] = units
        rows.append(dict(config_id=cid, units=len(units), median_error=statistics.median(values),
                         p95_error=quantile(values, .95), speedup=speedup,
                         latency_reduction_percent=100 * (1 - 1 / speedup),
                         units_above_1pct=sum(x > .01 for x in values),
                         units_above_3pct=sum(x > .03 for x in values),
                         median_cosine=statistics.median(cosines)))
    pairs = paired_units(units_by_config['A_PUBLIC'], units_by_config['V4'])
    sensitivity = [dict(median_max=med, p95_max=tail, speedup_min=speed,
                        passing_new_configs=[r['config_id'] for r in rows
                                             if r['config_id'].startswith('V') and
                                             r['median_error'] <= med and r['p95_error'] <= tail and
                                             r['speedup'] >= speed]) for med, tail, speed in SCENARIOS]
    # This is an audit of the recorded stopped experiment, not a new selector.
    assert not sensitivity[0]['passing_new_configs'], 'Original-gate discrepancy'
    decision = json.loads((case / 'results/decision.json').read_text())
    assert decision['final_decision'] == 'STOP_DEV_SCREEN' and decision['finalist_id'] is None
    return dict(analysis_type='POST_HOC_DESCRIPTIVE; not a protocol revision or finalist selection',
                primary_numeric_round=0, independent_documents=8, units_per_config=128,
                rows=rows, sensitivity=sensitivity,
                v4_paired=dict(lower=sum(p['error_reduction_pp'] > 0 for p in pairs),
                               equal=sum(p['error_reduction_pp'] == 0 for p in pairs),
                               higher=sum(p['error_reduction_pp'] < 0 for p in pairs), total=len(pairs),
                               median_paired_error_reduction_pp=statistics.median(p['error_reduction_pp'] for p in pairs),
                               units=pairs),
                original_decision_unchanged='STOP_DEV_SCREEN', fresh_confirmation=False,
                v3_policy='Excluded validity failure; threshold changes never make it eligible',
                source_hashes=hashes,
                limits='128 units are repeated measurements from 8 document clusters; no binomial significance or downstream-quality inference')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.case.resolve()):
        parser.error('Write to a separate directory to preserve the publication evidence')
    result = calculate(args.case)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(original_decision=result['original_decision_unchanged'],
                          paired_units=result['v4_paired']['total'],
                          v4_lower=result['v4_paired']['lower']), indent=2))


if __name__ == '__main__':
    main()
