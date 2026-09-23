"""Fresh-process checkpoint + byte-state resume on a separate fixed DEV input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_isolated import inspect_evaluation, load_frozen_evaluator


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--evaluation', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--child', action='store_true')
    ap.add_argument('--arm')
    args = ap.parse_args()
    inspection = inspect_evaluation(args.evaluation, args.checkpoint, require_complete=False)
    evaluation = load_frozen_evaluator(inspection)
    OnlineAdapter = evaluation.OnlineAdapter
    file_sha256 = evaluation.file_sha256
    torch.set_num_threads(2)
    model = evaluation.create_model(evaluation.load_upstream(args.upstream), checkpoint=args.checkpoint).eval()
    table = evaluation.token_table(model)
    config, data = evaluation.table_bytes(table)
    if (config != (args.evaluation/'token_coefficients.json').read_bytes()
            or data != (args.evaluation/'token_coefficients.bin').read_bytes()):
        raise ValueError('Canonical token coefficients differ from primary bytes')
    if args.child:
        folder = args.output/args.arm
        adapter = OnlineAdapter.from_shared((folder/'codec.json').read_bytes(), (folder/'basis.bin').read_bytes())
        tokens = np.load(folder/'suffix.npy', allow_pickle=False)
        state = adapter.from_bytes((folder/'prefix.bin').read_bytes(), batch_size=len(tokens))
        predictions, state, run = evaluation.sequence_call(model, table, tokens, adapter, initial=state, include_bos=False)
        if run['status'] != 'COMPLETE':
            raise ValueError('resume did not complete')
        np.save(folder/'resumed_predictions.npy', predictions, allow_pickle=False)
        (folder/'resumed.bin').write_bytes(state.payload.tobytes())
        return
    args.output.mkdir(parents=True, exist_ok=False)
    batch = evaluation.frozen_sequences('S3', 'DEV', 8, 64, 2003)
    checks = []
    for name in ['NATIVE_FP32', 'UNIFORM_8', 'STOCHASTIC_4', 'FULL_RESIDUAL_4_4', 'LOWRANK_4_8_R2']:
        folder = args.output/name
        folder.mkdir()
        config = (args.evaluation/'codecs'/f'{name}.json').read_bytes()
        basis_path = args.evaluation/'codecs'/f'{name}.basis.bin'
        basis = basis_path.read_bytes() if basis_path.exists() else b''
        adapter = OnlineAdapter.from_shared(config, basis)
        seeds = evaluation.stream_seeds(2003, 8)
        whole_pred, whole, whole_run = evaluation.sequence_call(model, table, batch.tokens, adapter, seeds)
        _, prefix, prefix_run = evaluation.sequence_call(model, table, batch.tokens[:, :17], adapter, seeds)
        if whole_run['status'] != 'COMPLETE' or prefix_run['status'] != 'COMPLETE':
            raise ValueError('parent sequence did not complete')
        (folder/'codec.json').write_bytes(config)
        (folder/'basis.bin').write_bytes(basis)
        (folder/'prefix.bin').write_bytes(prefix.payload.tobytes())
        np.save(folder/'suffix.npy', batch.tokens[:, 17:], allow_pickle=False)
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--child', '--upstream', str(args.upstream),
                   '--checkpoint', str(args.checkpoint), '--evaluation', str(args.evaluation),
                   '--output', str(args.output), '--arm', name]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
        (folder/'child-private.log').write_text(completed.stdout+completed.stderr)
        if completed.returncode:
            raise RuntimeError(f'fresh process failed for {name}: return code{completed.returncode}')
        actual = np.load(folder/'resumed_predictions.npy', allow_pickle=False)
        payload_matches = (folder/'resumed.bin').read_bytes() == whole.payload.tobytes()
        prediction_matches = np.array_equal(actual, whole_pred[:, 18:])
        if not payload_matches or not prediction_matches:
            raise AssertionError(f'restart mismatch {name}')
        checks.append(dict(arm=name, new_process_exit_code=completed.returncode,
                           prefix_tokens=17, suffix_tokens=47, sequences=8, bos_not_repeated=True,
                           exact_final_payload=payload_matches, exact_suffix_predictions=prediction_matches,
                           final_payload_sha256=hashlib.sha256(whole.payload.tobytes()).hexdigest(),
                           stored_prefix_bytes=prefix.payload.nbytes,
                           prefix_payload_sha256=file_sha256(folder/'prefix.bin'),
                           suffix_input_file_sha256=file_sha256(folder/'suffix.npy'),
                           codec_config_sha256=file_sha256(folder/'codec.json'),
                           codec_basis_sha256=file_sha256(folder/'basis.bin')))
    receipt = dict(scope='DEV only; fresh checkpoint reload + packed cache, no native floating cache restored',
                   checkpoint_sha256=file_sha256(args.checkpoint), checks=checks, passed=True,
                   runner_sha256=file_sha256(__file__), provenance_helper_sha256=file_sha256(ROOT/'scripts/benchmark_isolated.py'),
                   primary_manifest_sha256=inspection['manifest_sha256'],
                   source_sha256=inspection['manifest']['source_sha256'],
                   runtime_sha256=inspection['manifest']['runtime_sha256'],
                   protocol_sha256=inspection['manifest']['protocol_sha256'],
                   calibration_sha256=inspection['manifest']['calibration_sha256'],
                   token_table_config_sha256=inspection['manifest']['token_table_config_sha256'],
                   token_table_data_sha256=inspection['manifest']['token_table_data_sha256'],
                   codec_sha256=inspection['codecs'], input_identity=evaluation.input_identity(batch),
                   source_policy='parent and every fresh child import the verified primary frozen-source snapshot',
                   cpu_threads=2, test_generated=False)
    (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
