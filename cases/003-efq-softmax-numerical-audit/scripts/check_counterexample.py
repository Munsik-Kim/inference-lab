"""Scalar probability audit of one recorded failure, without model inference.

Optional --trace extracts a small score-only fixture from archived BF16 Q/K on
CPU. Normal use needs only that public fixture and the original key records.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

CASE = Path(__file__).resolve().parents[1]
VALUES = (0., .5, 1., 1.5, 2., 3., 4., 6.)


def probability(scores, affine):
    maximum = -math.inf
    weights, codes, exponents, residuals = [0.] * len(scores), [0] * len(scores), [0] * len(scores), [0.] * len(scores)
    maximum_jump = 0.
    for begin in range(0, len(scores), 128):
        end = min(begin + 128, len(scores))
        newmax = max(maximum, max(scores[begin:end]))
        if math.isfinite(maximum):
            maximum_jump = max(maximum_jump, newmax - maximum)
            weights = [w * math.exp(maximum - newmax) for w in weights]
        for b in range(begin, end, 32):
            stop = min(b + 32, end)
            local = max(scores[b:stop]) - newmax
            exponent = max(-127, min(127, math.floor((local + math.log(2 / 9)) / math.log(2))))
            scale = 2. ** exponent
            for i in range(b, stop):
                z = scores[i] - newmax - math.log(6 * scale)
                if affine:
                    code = max(0, min(7, math.floor((z + 3.06) * 2.3) + 1))
                else:
                    exact = math.exp(scores[i] - newmax) / scale
                    code = min(range(8), key=lambda c: (abs(exact - VALUES[c]), c % 2, c))
                weights[i] = VALUES[code] * scale
                codes[i], exponents[i], residuals[i] = code, exponent, z
        maximum = newmax
    total = math.fsum(weights)
    return [w / total for w in weights], codes, exponents, residuals, maximum_jump


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case-dir', type=Path, default=CASE)
    p.add_argument('--trace', type=Path)
    p.add_argument('--fixture', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    fixture = args.fixture or args.case_dir / 'tests/fixtures/efq_mean_counterexample.json'
    recorded = json.loads((args.case_dir / 'results/representative_details.json').read_text())[0]
    if args.trace:
        import torch
        assert not fixture.exists(), 'Preserve an existing fixture.'
        expected = next(t for t in json.loads((args.case_dir / 'provenance/eval_traces.json').read_text())['traces'] if t['document_id'] == 'eval-13' and t['length'] == 512 and t['layer'] == 27)
        trace_sha = hashlib.sha256(args.trace.read_bytes()).hexdigest()
        assert trace_sha == expected['trace_sha256']
        t = torch.load(args.trace, map_location='cpu', weights_only=True)
        meta = t['meta'];hi = meta['heads'].index(0);qi = meta['positions'].index(170)
        q = t['q'][hi, qi].float().tolist();keys = t['k'][hi, :171].float().tolist()
        scores = [math.fsum(a * b for a, b in zip(q, k)) * meta['scaling'] for k in keys]
        payload = {'scope': 'Post-hoc score-only fixture; one query. No Q/K/V tensors or model weights.',
                   'source_trace_sha256': trace_sha, 'model_revision': meta['model_revision'],
                   'document_id': 'eval-13', 'length': 512, 'layer': 27, 'head': 0,
                   'query_position_zero_based': 170, 'causal_keys': [0, 170],
                   'calculation': 'CPU math.fsum dot products from archived BF16 Q/K, then recorded attention scaling; FP64 diagnostic compared with original FP32 key records.',
                   'scores_fp64': scores}
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(json.dumps(payload, indent=2) + '\n')
    data = json.loads(fixture.read_text());scores = data['scores_fp64']
    assert data['query_position_zero_based'] == recorded['largest_row_query_position'] == 170 and len(scores) == 171
    w = [math.exp(s - max(scores)) for s in scores]
    reference = [v / math.fsum(w) for v in w]
    efq, ec, exponents, residuals, jump = probability(scores, True)
    nearest, nc, _, _, _ = probability(scores, False)
    discrepancies = []
    for old in recorded['top16_reference_keys']:
        i = old['key_position']
        assert abs(scores[i] - old['logit']) <= 2e-5
        assert ec[i] == old['method_code'] and nc[i] == old['nearest_code']
        for field, actual in [('reference_probability', reference[i]), ('method_effective_probability', efq[i]), ('nearest_same_scale_probability', nearest[i])]:
            delta = actual - old[field]
            assert abs(delta) <= 2e-6, (i, field, delta)
            discrepancies.append({'key': i, 'field': field, 'recorded_fp32': old[field], 'scalar_fp64': actual, 'difference': delta})
    mass = math.fsum(p for p, c in zip(reference, ec) if c == 0)
    assert abs(mass - recorded['row_diagnostics']['removed_reference_mass']) <= 2e-6
    assert jump == recorded['row_diagnostics']['max_row_max_jump'] == 0.
    assert ec[170] == 7 and nc[170] == 6 and exponents[170] == -6
    assert -3.06 + 6 / 2.3 <= residuals[170] < math.log(5 / 6)
    result = {'status': 'PASS', 'scope': 'Independent scalar reconstruction of one probability row; no model or full GPU reevaluation.',
              'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(), 'key_170': {
                  'reference_probability': reference[170], 'efq_mean_probability': efq[170],
                  'nearest_probability': nearest[170], 'efq_code': ec[170], 'nearest_code': nc[170],
                  'scale_exponent': exponents[170], 'residual': residuals[170],
                  'efq_code7_lower_boundary': -3.06 + 6 / 2.3, 'nearest_code7_lower_boundary': math.log(5 / 6)},
              'removed_reference_mass': mass, 'max_row_update_after_initialization': jump,
              'probability_absolute_tolerance': 2e-6, 'original_record_comparisons': discrepancies}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'original_record_comparisons'}, indent=2))


if __name__ == '__main__':
    main()
