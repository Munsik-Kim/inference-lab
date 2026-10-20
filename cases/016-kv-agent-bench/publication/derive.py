"""Verified display projection from raw technical records; no GPU import."""
import json
import pathlib
import sys
import hashlib
import zipfile

FOLLOWUP = 'followups/nvfp4-technical-r1'


def project(case):
    tech = case / FOLLOWUP
    sys.path.insert(0, str(tech / 'src'))
    from nvfp4_technical.audit import derive
    raw = derive(tech)
    read = lambda path: json.loads(path.read_text())
    frozen = read(tech / 'provenance/execution_freeze.json')
    configuration = read(tech / 'specs/final_configuration.json')
    finite = [read(tech / f'results/p0/{kv}/finite.json') for kv in ('B16', 'F4')]
    kernel = read(tech / 'results/kernel_check.json')
    cache = read(tech / 'results/servers/p0_F4/cache_before.json')
    profile = read(tech / 'results/p0/F4/profile.json')
    names = [r['kernel'] for r in profile['gpu_kernels']]
    path_valid = len(cache['nvfp4_views']) == 36 and all(r['is_kvcache_nvfp4'] and r['nvfp4_fa2'] for r in cache['layers']) and any('NVFP4QuantizeAppend' in n for n in names) and any('__nv_fp4x2_e2m1' in n for n in names) and any('kernel_mha(' in n for n in names)
    numeric = kernel['status'] == 'PASS' and all(f['all36_finite'] for f in finite)
    capacities = {r['kv']: r for r in raw['capacity']}
    with zipfile.ZipFile(case / 'followups/baseline-audit-r1/Case016_baseline_audit_v1.zip') as archive:
        old_audit = json.loads(archive.read('Case016_baseline_audit_v1/results/replay_r2/summary.json'))
    old_groups = old_audit['groups']
    old_agent = {'episodes': sum(g['n'] if 'n' in g else g.get('observed', 12) for g in old_groups), 'unique_dev_tasks': 12,
                 'B16_verified': sum(g['recomputed_verified'] for g in old_groups if g['kv'] == 'B16'),
                 'F4_verified': sum(g['recomputed_verified'] for g in old_groups if g['kv'] == 'F4'),
                 'per_kv_denominator': 48, 'baseline_gate': 'NOT_MET', 'P2_P3': 'NOT_RUN',
                 'audit_score_changes': len(old_audit['score_mismatches']), 'groups': old_groups}
    sources = {}
    for prefix in ('src', 'specs', 'data', 'results', 'provenance'):
        for path in sorted((tech / prefix).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                sources[str(path.relative_to(case))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'schema': 'case016-public-display-v1', 'stage': 'P1-T', 'run_id': 'nvfp4-technical-r1', 'evidence_kind': 'GPU_SERVING_MEASUREMENT_WITH_CPU_RAW_RECALCULATION',
            'model': 'Qwen/Qwen3-4B-Instruct-2507', 'model_revision': raw['model_revision'], 'weights': 'BF16', 'engine': 'vLLM0.32.0',
            'device': 'RTX5080 SM120 16GB', 'execution_boundary': 'Native NVFP4 KV, BF16 weights/query/output; FlashInfer prefill/decode. Not dense reload of KV.',
            'execution_source_sha256': frozen['execution_source_sha256'], 'git_base': frozen['git_base'], 'mode': configuration['mode'],
            'status': {'TECH_PATH_VALID': path_valid, 'TECH_NUMERIC_VALID': numeric, 'CAPACITY_VALID': len(capacities) == 2 and all(c['layers'] == 36 for c in capacities.values()), 'SERVING_MEASURED': raw['complete']},
            'capacity': raw['capacity'], 'capacity_ratio': capacities['F4']['token_slots'] / capacities['B16']['token_slots'],
            'bytes_per_token_ratio': capacities['B16']['bytes_per_token'] / capacities['F4']['bytes_per_token'],
            'summary': raw['summary'], 'per_round': raw['blocks'], 'paired_rounds': raw['paired_rounds'], 'fidelity': raw['sanity'],
            'counts': raw['request_counts'], 'worker': read(tech / 'results/worker_receipt.json'), 'missing_blocks': raw['missing_blocks'],
            'numeric_scope': {'same_compressed_dequant': kernel['rows'], 'all36_actual_attention_finite': numeric, 'whole_decoder_equivalence': 'NOT_TESTED', 'full_vocabulary_KL_PPL': 'NOT_TESTED'},
            'units': {'throughput': 'successful HTTP requests / closed8-request makespan seconds', 'requests_per_block': 8, 'rounds': 3, 'output_tokens_per_timed_request': 128,
                      'client_decode': '(last token-bearing event - first token-bearing event)/(observed token count-1); client average, not individual token ITL',
                      'queue': 'dedicated server histogram sum/count interval mean, not per-request attribution', 'device_memory': 'sampled total device MiB including background; not model-only memory'},
            'original_agent': old_agent,
            'scope': 'Synthetic token workload on one model/device/configuration. Capacity is not a100K single context; max context28672. No agent-quality gain or pure quantization-arithmetic causal effect claimed.',
            'sources': sources}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh output file required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(project(args.case), ensure_ascii=False, indent=2) + '\n')
