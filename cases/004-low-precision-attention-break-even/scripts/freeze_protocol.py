"""Freeze local choices from development only; refuse an existing protocol."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import CASE, fingerprint, shapes, sha, utc, write_json
from src.backends import SAGE_OPTIONS


def main():
    target = CASE / 'configs/experiment_spec.json'
    assert not target.exists(), 'Never replace a frozen protocol'
    assert not (CASE / 'provenance/confirmation_capture.json').exists()
    assert not list((CASE / 'results/confirmation').glob('*.json'))
    dev_path = CASE / 'results/dev/round-0.json'
    dev = json.loads(dev_path.read_text())
    gate = json.loads((CASE / 'provenance/backend_probe.json').read_text())
    manifest = json.loads((CASE / 'inputs/manifest.json').read_text())
    assert dev['stage'] == 'dev' and dev['status'] == 'COMPLETED'
    assert gate['status'] == 'PASS'
    assert dev['environment'] == gate['environment']
    selection, evidence = {}, {}
    for shape in shapes():
        sid = shape['id']
        candidate_costs = {}
        for backend in ('default', 'flash'):
            rows = [r for r in dev['measurements']
                    if r['shape']['id'] == sid and r['backend'] == backend]
            assert len(rows) == (8 if shape['family'] == 'qwen' else 1)
            assert all(r['status'] == 'OK' and r['profile']['fused_bf16']
                       and r['inputs_unchanged'] for r in rows)
            candidate_costs[backend] = float(np.median(
                [v for r in rows for v in r['wall_ms']]))
        selection[sid] = min(candidate_costs, key=candidate_costs.get)
        evidence[sid] = candidate_costs
    write_json(CASE / 'provenance/baseline_selection.json', dict(
        development_record_sha256=sha(dev_path),
        rule='Lowest median complete-adapter synchronized wall block cost across all development inputs for that shape. Exact tie chooses default. Both must be profiled fused paths.',
        development_median_wall_ms=evidence, selected=selection))
    frozen = {}
    for folder in ('src', 'scripts', 'tests', 'inputs', 'provenance'):
        for p in sorted((CASE / folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                frozen[str(p.relative_to(CASE))] = sha(p)
    frozen['results/dev/round-0.json'] = sha(dev_path)
    spec = dict(
        protocol_version=1, frozen_at_utc=utc(),
        registration='Local protocol fixed before confirmation; not external preregistration',
        environment=dev['environment'],
        environment_fingerprint=fingerprint(dev['environment']),
        model=dict(id=manifest['model_id'], revision=manifest['revision'], layer=13,
                   hq=16, hkv=8, head_dim=128, batch=1, dropout=0,
                   capture='Post-QK-normalization/post-RoPE, all query positions; native BF16 SDPA output'),
        backends=dict(baselines=['default', 'flash'], candidate='sage',
                      sage_commit='d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5',
                      api='sageattn_qk_int8_pv_fp8_cuda', options=SAGE_OPTIONS),
        gate0_status=gate['status'], shapes=shapes(), baseline_selection=selection,
        input_protocol=dict(manifest_sha256=sha(CASE/'inputs/manifest.json'),
                            development_documents=8, confirmation_documents=16,
                            confirmation_languages=dict(en=5, ko=5, code=6),
                            synthetic_distribution='Independent CPU FP32 standard Gaussian, then BF16 rounding',
                            development_seed_base=400000, confirmation_seed_base=410000,
                            synthetic_seed_rule='base + index in shapes; same input across process rounds',
                            dependence='Lengths are prefixes of one document; heads/queries are repeated units'),
        reference=dict(dtype='FP32 with TF32 disabled; FP64 NumPy small oracle',
                       positions='floor(i*(length-1)/31), i=0..31',
                       keys='Every causal-valid key for each sampled query; no KV expansion',
                       scale='1/sqrt(head_dim)', unit='document x length x layer x head, 32 sampled query outputs'),
        numerical_screen=dict(relative_error='Frobenius norm of difference / reference Frobenius norm',
                              median_max=.01, p95_max=.03, invalid_rows_max=0,
                              reference_rms_floor=1e-6, near_zero='Relative error null; NUMERICAL_REVIEW, no automatic approval',
                              primary_round=0, later_rounds='Repeatability only, not extra independent documents',
                              native_capture_max_relative_error=.01),
        timing=dict(process_rounds=5, counts=dict(warmup=20, blocks=20, calls=10),
                    primary='GPU-resident BF16 Q/K/V to BF16 O; complete public adapter; synchronized wall-clock blocks',
                    secondary='Separate CUDA-event blocks, after the corresponding wall block; end event synchronized',
                    distribution='p50/p95 of ten-call block-mean per-call costs, not service-request latency',
                    order='Seed 420000 + round*1000 + shape_index*31 + sum(document_id UTF-8 bytes); balanced rotations of shuffled backend names',
                    cache='Repeated-input warm-cache steady state; preprocessing repeated every call',
                    outside='Model load/tokenization/projections, CPU transfers, profiling, correctness/reference calculations',
                    cold='First complete call per backend/input retained separately; warmup excluded from steady-state',
                    kernel_only='Auxiliary only, after primary blocks; exact upstream internal call; require bitwise equality per input'),
        statistics=dict(timing_bootstrap_replicates=2000, document_bootstrap_replicates=2000,
                        timing_seed='470004 + shape_index', numerical_seed=480004,
                        speedup='Median paired BF16/Sage block-cost ratios; ratio of median costs also retained',
                        timing_resampling='Outer process-round cluster, inner paired block within each retained document; same documents retained at equal weight',
                        numerical_resampling='Document clusters; all heads retained together; percentile 95% intervals',
                        coverage='Shape-wise pointwise intervals; no multiple-comparison guarantee'),
        selection_rule=dict(minimum_median_speedup=1.10, minimum_ci_lower_strict=1.00,
                            require='Matching real Qwen geometry, local numerical PASS, confirmed interface and low-precision execution',
                            synthetic='SYNTHETIC_ONLY, including matching lengths: synthetic H8/H8 is not Qwen H16/H8',
                            unknown='BF16/UNKNOWN; no runtime correctness oracle or automatic deployment'),
        limits=dict(peak_allocated_bytes=13*2**30, other_gpu_jobs='Never terminate; retain background device telemetry',
                    oom='Record original error, terminate process; only smaller shapes may continue in a new process',
                    install_attempts_max=2, source_build_jobs_max=4, cuda_graph=False, torch_compile=False,
                    weights_download=False, confirmation_retuning=False),
        development_findings=dict(
            numerical='Sage development median Qwen errors already exceed the proposed 1% screen; threshold is retained.',
            causal_stress='Future V range changes can alter prefix approximation through global quantization statistics. Range-preserving future permutation leaves prefix unchanged. No prefix-invariance/decode guarantee.'),
        failure_policy='Preserve failures and missing points; no fake timings, silent retries, interpolation, threshold relaxation, or favorable-round selection',
        frozen_files=frozen)
    write_json(target, spec)
    (CASE/'configs/experiment_spec.sha256').write_text(sha(target)+'  experiment_spec.json\n')
    print(json.dumps(dict(spec_sha256=sha(target), frozen_files=len(frozen),
                          baseline_counts={name:list(selection.values()).count(name) for name in ['default','flash']})))


if __name__ == '__main__':
    main()
