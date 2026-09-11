"""Calibration, synthetic checks, specification freeze, and fixed evaluation."""
import argparse
import csv
import datetime
import gzip
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from numerics import dense, online, metrics
from prepare import CASE, sha, write_json
from extract_traces import verify_frozen


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_trace(path, heads=None):
    t = torch.load(path, map_location='cpu', weights_only=True)
    meta = t['meta'].copy()
    idx = list(range(len(meta['heads']))) if heads is None else [meta['heads'].index(h) for h in heads]
    q, k, v = [t[key][idx].float().cuda() for key in ['q', 'k', 'v']]
    s = q @ k.transpose(-1, -2) * meta['scaling']
    positions = torch.tensor(meta['positions'], device=s.device)
    s.masked_fill_(torch.arange(s.shape[-1], device=s.device)[None, None, :] > positions[None, :, None], -torch.inf)
    meta['heads'] = [meta['heads'][i] for i in idx]
    return s, v, meta


def calibration(plan, work):
    c = plan['calibration']
    candidates = list(itertools.product(c['tau'], c['h']))
    errors = np.zeros(len(candidates), dtype=np.float64)
    denominator = 0.
    per_document = []
    for i in range(c['documents']):
        path = work / 'traces/dev' / f"dev-{i:02d}-n{c['length']}-l{c['layer']}.pt"
        s, v, meta = load_trace(path, c['heads'])
        ref = dense(s, v)['out']
        count = s.shape[0]
        sb, vb = s.repeat(len(candidates), 1, 1), v.repeat(len(candidates), 1, 1)
        tau = torch.tensor([p[0] for p in candidates], device='cuda').repeat_interleave(count).reshape(-1, 1, 1)
        h = torch.tensor([p[1] for p in candidates], device='cuda').repeat_interleave(count).reshape(-1, 1, 1)
        result = online(sb, vb, 'efq_calibrated', plan['tile'], plan['primary_block'], tau, h, probabilities=False)
        diff = result['out'].reshape(len(candidates), *ref.shape) - ref
        ss = diff.double().square().sum((1, 2, 3)).cpu().numpy()
        refss = ref.double().square().sum().item()
        assert np.isfinite(ss).all() and refss > 0
        errors += ss
        denominator += refss
        per_document.append({'document_id': meta['document_id'], 'error_squared_by_candidate': ss.tolist(), 'reference_squared': refss})
        print('calibration document', i, flush=True)
    objective = np.sqrt(errors / denominator)
    best = int(objective.argmin())
    record = {'selected': {'tau': candidates[best][0], 'h': candidates[best][1]}, 'selected_index': best,
              'selected_objective': float(objective[best]), 'candidate_count': len(candidates),
              'plan_sha256': sha((CASE/'configs/development_plan.json').read_bytes()),
              'candidates': [{'tau': t, 'h': h, 'relative_frobenius': float(e)} for (t, h), e in zip(candidates, objective)],
              'per_document': per_document, 'selection_split': 'dev only; no refinement or evaluation selection'}
    write_json(CASE/'results/calibration.json', record)
    print(json.dumps(record['selected']), 'objective', record['selected_objective'], flush=True)


def synthetic_traces(plan, split):
    conf = plan['synthetic']
    base = conf['dev_seed'] if split == 'dev' else conf['eval_seed']
    for family_i, family in enumerate(conf['families']):
        for rep in range(conf['replicates_per_family']):
            for n in conf['lengths']:
                rng = np.random.default_rng(base + family_i * 100000 + rep * 10000 + n)
                r = conf['rows']
                s = rng.normal(size=(1, r, n)).astype('float32')
                if family == 'uniform': s *= 1e-4
                elif family == 'gaussian_low': s *= .1
                elif family == 'gaussian_medium': s *= 2
                elif family == 'gaussian_high': s *= 8
                elif family == 'single_peak':
                    s *= .1
                    for row in range(r): s[0, row, rng.integers(n)] = 30
                elif family == 'multiple_peaks':
                    s = s * .1 - 2
                    for row in range(r): s[0, row, rng.choice(n, 4, replace=False)] = [8., 7.9, 8.1, 8.05]
                elif family == 'heavy_tail': s = np.clip(rng.standard_t(2, size=s.shape) * 2, -100, 100).astype('float32')
                elif family == 'causal':
                    s *= 3
                    s[0, np.arange(n)[None, :] > np.array(plan['query_positions'][str(n)])[:, None]] = -np.inf
                elif family == 'max_jump':
                    s *= .1
                    s[..., :n//3] -= 10
                    s[..., 2*n//3:] += 80
                v = rng.normal(size=(1, n, 128)).astype('float32')
                meta = {'document_id': f'synthetic-{split}-{family}-{rep}', 'split': 'synthetic-' + split,
                        'language': 'synthetic', 'family': family, 'length': n, 'layer': -1, 'heads': [0],
                        'positions': plan['query_positions'][str(n)]}
                yield torch.tensor(s, device='cuda'), torch.tensor(v, device='cuda'), meta


def run_measurements(plan, stage, work):
    selected = json.loads((CASE/'results/calibration.json').read_text())['selected']
    if stage in ['eval', 'synthetic-eval']:
        verify_frozen()
    units_path = CASE/'results'/f'{stage}_units.jsonl'
    rows_path = CASE/'results'/f'{stage}_rows.csv.gz'
    assert not units_path.exists() and not rows_path.exists(), 'Preserve prior runs: output already exists.'
    if stage.startswith('synthetic'):
        stream = synthetic_traces(plan, stage.split('-')[1])
    else:
        stream = (load_trace(p) for p in sorted((work/'traces'/stage).glob('*.pt')))
    totals = {'head_units': 0, 'row_records': 0, 'valid_bad_rows': 0, 'fully_masked_rows': 0, 'probability_reconstruction_worst_relative': 0.}
    start = time.time()
    with units_path.open('w') as out, gzip.open(rows_path, 'wt', newline='') as rowout:
        writer = None
        for trace_i, (s, v, meta) in enumerate(stream):
            ref = dense(s, v)
            settings = [(m, plan['primary_block']) for m in plan['methods']]
            if stage in ['eval', 'synthetic-eval']:
                settings += [(m, b) for b in plan['sensitivity_blocks'] for m in plan['sensitivity_methods']]
            for method, block in settings:
                result = ref if method == 'dense_fp32' else online(s, v, method, plan['tile'], block, **selected)
                diag = {k: a.cpu().numpy() for k, a in metrics(s, v, ref, result).items()}
                valid = result['valid']
                delta = result['out'] - ref['out']
                ss = delta.double().square().sum((-1, -2)).cpu().numpy()
                rr = ref['out'].double().square().sum((-1, -2)).cpu().numpy()
                back = result['probs'] @ v
                backdiff = (back-result['out']).double().square().sum().sqrt().item()
                norm = result['out'].double().square().sum().sqrt().item()
                reconstruction_error = backdiff/norm if norm > 1e-12 else backdiff
                if math.isfinite(reconstruction_error):
                    totals['probability_reconstruction_worst_relative'] = max(totals['probability_reconstruction_worst_relative'], reconstruction_error)
                    assert reconstruction_error < 5e-5, 'Effective online probabilities do not reconstruct output.'
                bad = valid & ((result['denom'] <= 0) | ~torch.isfinite(result['denom']) |
                               ~torch.isfinite(result['out']).all(-1) | ~torch.isfinite(result['probs']).all(-1))
                bad_np = bad.cpu().numpy()
                valid_np = valid.cpu().numpy()
                for hi, head in enumerate(meta['heads']):
                    nv = int(valid_np[hi].sum())
                    near = math.sqrt(float(rr[hi])) <= plan['metrics']['reference_norm_floor_rms'] * math.sqrt(nv*v.shape[-1])
                    common = {'document_id': meta['document_id'], 'split': meta['split'], 'language': meta['language'],
                              'family': meta.get('family', ''), 'length': meta['length'], 'layer': meta['layer'],
                              'head': head, 'method': method, 'block': block}
                    unit = {**common, 'valid_rows': nv, 'fully_masked_rows': len(valid_np[hi])-nv,
                            'failed_valid_rows': int(bad_np[hi].sum()), 'near_zero_reference': near,
                            'relative_error': None if near else math.sqrt(float(ss[hi])/float(rr[hi])),
                            'absolute_error_frobenius': math.sqrt(float(ss[hi])), 'error_squared': float(ss[hi]),
                            'reference_squared': float(rr[hi]),
                            'underflow_blocks_trace_all_heads': result['underflow_blocks'].item(),
                            'zero_scales_trace_all_heads': result['zero_scales'].item()}
                    out.write(json.dumps(clean(unit), separators=(',', ':'), allow_nan=False)+'\n')
                    totals['head_units'] += 1
                    totals['valid_bad_rows'] += unit['failed_valid_rows']
                    totals['fully_masked_rows'] += unit['fully_masked_rows']
                    for ri, position in enumerate(meta['positions']):
                        row = {**common, 'query_position': position, **{k: clean(a[hi, ri]) for k, a in diag.items()},
                               'failed_valid_row': bool(bad_np[hi, ri])}
                        if method in ['dense_fp32', 'online_fp32']:
                            row['e2m1_zero_fraction'] = None
                            row['removed_reference_mass'] = None
                        if writer is None:
                            writer = csv.DictWriter(rowout, fieldnames=list(row)); writer.writeheader()
                        writer.writerow(row)
                        totals['row_records'] += 1
            if trace_i % 3 == 0:
                print(json.dumps({'stage': stage, 'traces_done': trace_i+1, 'elapsed_s': round(time.time()-start, 1)}), flush=True)
    totals['elapsed_s'] = time.time()-start
    totals['gpu'] = torch.cuda.get_device_name()
    write_json(CASE/'results'/f'{stage}_run.json', totals)


def freeze(plan):
    assert not (CASE/'configs/experiment_spec.json').exists()
    assert not (CASE/'provenance/eval_traces.json').exists()
    tests = json.loads((CASE/'provenance/numerical_tests.json').read_text())
    assert tests['success'] and tests['skipped'] == 0
    validation = json.loads((CASE/'provenance/trace_validation.json').read_text())
    assert all(x['same_backend_pass'] and x['fp32_pass'] for x in validation['checks'])
    for stage in ['dev', 'synthetic-dev']:
        run = json.loads((CASE/'results'/f'{stage}_run.json').read_text())
        assert run['valid_bad_rows'] == 0
    chosen = json.loads((CASE/'results/calibration.json').read_text())
    files = []
    for folder in ['scripts', 'tests', 'inputs', 'configs', 'provenance']:
        files += [p for p in (CASE/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [CASE/'results/calibration.json', CASE/'results/dev_run.json', CASE/'results/synthetic-dev_run.json']
    spec = {'experiment_id': 'case003-efq-v1-20260911', 'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'plan': plan, 'calibrated_parameters': chosen['selected'], 'frozen_files': {str(p.relative_to(CASE)): sha(p.read_bytes()) for p in sorted(files)}}
    write_json(CASE/'configs/experiment_spec.json', spec)
    digest = sha((CASE/'configs/experiment_spec.json').read_bytes())
    (CASE/'configs/experiment_spec.sha256').write_text(digest+'  experiment_spec.json\n')
    print('Frozen before evaluation:', digest, 'files:', len(files), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['calibrate', 'dev', 'eval', 'synthetic-dev', 'synthetic-eval', 'freeze'], required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads((CASE/'configs/development_plan.json').read_text())
    if args.stage == 'freeze':
        freeze(plan)
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    assert torch.cuda.is_available()
    if torch.cuda.mem_get_info()[0] < 2*2**30:
        raise RuntimeError('GPU_BUSY: less than 2 GiB free for numerical audit.')
    torch.cuda.set_per_process_memory_fraction(.45)
    with torch.inference_mode():
        if args.stage == 'calibrate': calibration(plan, args.work_dir)
        else: run_measurements(plan, args.stage, args.work_dir)


if __name__ == '__main__':
    main()
