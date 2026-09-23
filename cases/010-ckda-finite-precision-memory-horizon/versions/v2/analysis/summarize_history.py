"""Model-free audit of retained historical predictions and byte-hash receipts.

No Torch/model imports, checkpoint loading, inference, or unit-test counting.
The four boundary appendices extend existing cases; they add no memory trials.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAIN_ARMS = ('NATIVE_FP32', 'UNIFORM_8', 'UNIFORM_5', 'LOWRANK_4_8_R2')
EXTRA_ARMS = ('STOCHASTIC_8', 'FULL_RESIDUAL_4_4', 'LOWRANK_4_8_R1')
FAILURES = ((0, 'UNIFORM_2', 77, 838), (0, 'UNIFORM_3', 8, 1741),
            (1, 'UNIFORM_2', 31, 623), (2, 'UNIFORM_2', 5, 867))
MAIN_COHORT = 'HISTORICAL_FIRST16_TEST3001'
EXTRA_COHORT = 'EXTRA_FIRST4_DEV2001_T64'
FAILURE_COHORT = 'ORIGINAL_INVALID_STREAM_B1_TEST3001'
CODES = {0: 'ACTIVE', 1: 'RECONSTRUCTION_NONFINITE', 2: 'TRANSITION_NONFINITE',
         3: 'STATE_RANGE', 4: 'RESIDUAL_NONFINITE', 5: 'RESIDUAL_RANGE', 6: 'POST_WRITE_NONFINITE'}
PREDICTIONS = ('v1_full', 'v1_split', 'v2_full', 'v2_split', 'v2_fresh')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON number: '+value)
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=invalid)


def first_difference(reference, candidate):
    require(reference.shape == candidate.shape, 'Prediction shape mismatch')
    positions = np.argwhere(reference != candidate)
    if not len(positions):
        return None
    write = int(positions[:, 1].min())
    row = int(positions[positions[:, 1] == write, 0].min())
    return dict(kind='prediction', row=row, write_index=write,
                group_token_index=None if write == 0 else write,
                reference=int(reference[row, write]), candidate=int(candidate[row, write]))


class Audit:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files = {}

    def path(self, relative):
        rel = PurePosixPath(relative)
        require(isinstance(relative, str) and not rel.is_absolute() and '\\' not in relative and
                '..' not in rel.parts and rel.as_posix() == relative, 'Unsafe artifact path')
        path = self.root / relative
        require(path.resolve().is_relative_to(self.root) and path.is_file() and not path.is_symlink(),
                'Missing or unsafe artifact: '+relative)
        return path

    def track(self, relative, expected=None):
        path = self.path(relative)
        raw = path.read_bytes()
        digest = sha(raw)
        require(expected is None or expected == digest, 'Artifact hash mismatch: '+relative)
        self.files[relative] = dict(sha256=digest, bytes=len(raw))
        return path

    def read(self, relative, expected=None):
        return json_read(self.track(relative, expected))

    def sources(self, mapping):
        require(isinstance(mapping, dict) and bool(mapping), 'Missing runtime source identities')
        for name, digest in mapping.items():
            self.track(name, digest)

    def protocol(self):
        freeze = self.read('provenance/protocol_freeze.json')
        self.protocol_hash = freeze['protocol_sha256']
        self.design = self.read('protocol_v2.json', self.protocol_hash)
        for name, digest in freeze['frozen_files'].items():
            self.track(name, digest)
        require(self.design['historical']['arms'] == list(MAIN_ARMS) and
                self.design['historical']['extras']['arms'] == list(EXTRA_ARMS), 'Historical design changed')

    def main_cell(self, relative, digest, seed, arm, cohort):
        result = self.read(relative, digest)
        identity = result['identity']
        require((identity['model_seed'], identity['arm'], identity['cohort']) == (seed, arm, cohort), 'Historical cell identity mismatch')
        require(identity['protocol_sha256'] == self.protocol_hash and
                identity['checkpoint_sha256'] == self.design['checkpoint_sha256'][str(seed)], 'Protocol/checkpoint mismatch')
        require(result['status'] == 'PASS' and result['diagnostic_only'] and result['fresh_primary_sample_count'] == 0 and
                result['reviewer_probe_available'] is False and result['reviewer_probe_executed'] is False and
                result['same_batch_shape'] and result['fresh_processes'] == 1 and result['fresh_process_version'] == 'v2' and
                result['serialized_at_all_boundaries'] == ['v1', 'v2'] and result['legacy_error'] is None,
                'Historical execution scope/completion mismatch')
        n, t = (16, 2048) if cohort == MAIN_COHORT else ((4, 64) if cohort == EXTRA_COHORT else (1, 2048))
        require((result['N'], result['T']) == (n, t), 'Historical sample dimensions changed')
        ids = result['sample_ids']
        require(len(ids) == n and len(set(ids)) == n and
                sha(json.dumps(ids, sort_keys=True, separators=(',', ':')).encode()) == identity['sample_ids_sha256'], 'Sample identity mismatch')
        self.sources(result['source_sha256'])
        artifact = result['predictions_artifact']
        require(artifact['file'] == 'predictions.npz', 'Unexpected prediction artifact path')
        prediction_file = (PurePosixPath(relative).parent / artifact['file']).as_posix()
        with np.load(self.track(prediction_file, artifact['sha256']), allow_pickle=False) as data:
            require(set(data.files) == set(PREDICTIONS) | {'gold'}, 'Missing or extra historical prediction arrays')
            arrays = {name: data[name].copy() for name in data.files}
        gold = arrays.pop('gold')
        require(gold.shape == (n, t) and gold.dtype == np.uint8 and np.all(gold <= 5) and
                sha(gold.tobytes()) == identity['gold_sha256'], 'Historical gold identity mismatch')
        for key, pred in arrays.items():
            require(pred.shape == (n, t+1) and pred.dtype == np.int8 and np.all((pred >= -1) & (pred <= 5)),
                    'Invalid prediction array')
            require(sha(pred.tobytes()) == result['predictions_sha256'][key], 'Decoded prediction bytes differ from receipt')
        labels = [('v1_full_vs_split', 'v1_full', 'v1_split', False),
                  ('v1_vs_v2_full_body', 'v1_full', 'v2_full', True),
                  ('v2_full_vs_split', 'v2_full', 'v2_split', False),
                  ('v2_full_vs_fresh', 'v2_full', 'v2_fresh', False)]
        differences = {}
        for label, left, right, body_only in labels:
            comparison = result['comparisons'][label]
            difference = first_difference(arrays[left], arrays[right])
            require(comparison['available'] and comparison['predictions_match'] == (difference is None) and
                    comparison['first_prediction_difference'] == difference, 'Reported first prediction difference is wrong')
            payload = comparison['payload']
            require(payload['match'] == (payload['reference_sha256'] == payload['candidate_sha256']), 'Payload hash equality flag is wrong')
            require((payload['first_difference'] is None) == payload['match'], 'Payload first-difference scope is inconsistent')
            if not body_only:
                require(payload['reference_sha256'] == result['final_payload_sha256'][left] and
                        payload['candidate_sha256'] == result['final_payload_sha256'][right], 'Final payload hash references disagree')
            differences[label] = difference
        require(differences['v2_full_vs_split'] is None and differences['v2_full_vs_fresh'] is None and
                result['comparisons']['v2_full_vs_split']['payload']['match'] and
                result['comparisons']['v2_full_vs_fresh']['payload']['match'] and
                result['full_split_match'] and result['fresh_serialized_boundary_trace_match'], 'v2 restart parity failure')
        full = arrays['v2_full']
        codes, firsts = result['final_terminal_codes'], result['first_terminal_write']
        require(len(codes) == n and len(firsts) == n, 'Terminal metadata dimensions')
        terminal = np.zeros(full.shape, bool)
        active_attempts = 0
        for row, (code, first) in enumerate(zip(codes, firsts)):
            require(type(code) is int and code in CODES and (first is None) == (code == 0), 'Terminal code/first-write mismatch')
            if first is None:
                active_attempts += t+1
            else:
                require(type(first) is int and 0 <= first <= t, 'Invalid terminal write')
                terminal[row, first:] = True
                active_attempts += first+1
        require(np.all(full[terminal] == -1), 'Terminal prediction revival')
        invalid_active = (full == -1) & ~terminal
        expected_counts = dict(active_update_attempts=active_attempts, terminal_noop_steps=n*(t+1)-active_attempts,
                               invalid_readout_counts=invalid_active.sum(axis=1).tolist())
        execution = result['execution']['v2_full']
        require(execution['completed_writes'] == t+1 and execution['terminal_codes'] == codes and
                execution['first_terminal_write'] == firsts, 'Execution terminal/cursor metadata mismatch')
        require(all(execution[key] == value for key, value in expected_counts.items()), 'A/B/C counter mismatch')
        require(result['execution']['v2_split']['counts'] == expected_counts, 'Split lost consumed-step/readout counts')
        expected_boundaries = [value for value in (1, 17, 63, 256, 1024, 2048) if value <= t]
        if t not in expected_boundaries: expected_boundaries.append(t)
        require(result['boundaries'] == expected_boundaries, 'Historical chunk cuts changed')
        trace = result['execution']['v2_split']['trace']
        require([item['group_stop'] for item in trace] == expected_boundaries, 'Missing historical split trace')
        for item in trace:
            require(item['prediction_sha256'] == sha(full[:, :item['group_stop']+1].tobytes()), 'Split prefix prediction hash mismatch')
        require(trace[-1]['payload_sha256'] == result['final_payload_sha256']['v2_full'], 'Split final trace hash mismatch')
        row = dict(model_seed=seed, arm=arm, cohort=cohort, N=n, T=t,
            source_sequence_id=ids[0] if cohort == FAILURE_COHORT else None,
            original_row=None, original_failure_write=None, actual_terminal_write=None, terminal_code=None, terminal_name=None,
            v1_v2_full_predictions_equal=differences['v1_vs_v2_full_body'] is None,
            v1_v2_final_body_hashes_equal=result['comparisons']['v1_vs_v2_full_body']['payload']['match'],
            v1_full_split_predictions_equal=differences['v1_full_vs_split'] is None,
            v1_first_split_difference_write=None if differences['v1_full_vs_split'] is None else differences['v1_full_vs_split']['write_index'],
            v1_first_split_difference_reference=None if differences['v1_full_vs_split'] is None else differences['v1_full_vs_split']['reference'],
            v1_first_split_difference_resumed=None if differences['v1_full_vs_split'] is None else differences['v1_full_vs_split']['candidate'],
            v2_full_split_predictions_equal=True, v2_full_fresh_predictions_equal=True,
            v2_full_split_payload_hashes_equal=True, v2_full_fresh_payload_hashes_equal=True,
            post_failure_appendix_pass=None, post_failure_body_hash_frozen=None, post_failure_cursor_advances=None,
            fresh_resume_immediately_after_failure=None,
            A_finite_wrong_scored_tokens=int(((full[:, 1:] >= 0) & (full[:, 1:] != gold)).sum()),
            B_invalid_active_readouts_including_BOS=int(invalid_active.sum()),
            BOS_invalid_predictions=int((full[:, 0] == -1).sum()), C_terminal_sequences=sum(code != 0 for code in codes),
            active_update_attempts=active_attempts, terminal_noop_steps=expected_counts['terminal_noop_steps'],
            receipt=relative, receipt_sha256=digest, boundary_receipt=None, boundary_receipt_sha256=None)
        if cohort == FAILURE_COHORT:
            selected = next(item for item in FAILURES if item[:2] == (seed, arm))
            _, _, original_row, write = selected
            require(ids == [f'S3:TEST:3001:{original_row:06d}'] and firsts == [write] and codes == [2] and
                    result['extra']['original_row'] == original_row and result['extra']['original_first_invalid_write'] == write and
                    result['original_invalid_case_replay']['v1_B1_first_invalid_write'] == write and
                    result['original_invalid_case_replay']['same_write_as_original'], 'Selected historical failure did not reproduce')
            row.update(original_row=original_row, original_failure_write=write, actual_terminal_write=write,
                       terminal_code=2, terminal_name=CODES[2])
        else:
            require(all(code == 0 for code in codes) and row['v1_v2_full_predictions_equal'] and
                    row['v1_v2_final_body_hashes_equal'], 'Healthy reference parity mismatch')
        return result, arrays, row, differences

    def boundaries(self, cells):
        relative = 'provenance/historical_failure_boundaries_v1.json'
        expected = self.track('provenance/historical_failure_boundaries_v1.sha256').read_text().strip()
        appendix = self.read(relative, expected)
        require(appendix['new_memory_trials'] == 0 and appendix['protocol_sha256'] == self.protocol_hash,
                'Appendix study scope changed')
        self.sources(appendix['source_sha256'])
        expected_keys = {(seed, arm) for seed, arm, _, _ in FAILURES}
        require({(item['seed'], item['arm']) for item in appendix['cases']} == expected_keys and len(appendix['cases']) == 4,
                'Appendix fixed case inventory mismatch')
        index = self.read('results/historical-failure-boundaries/index.json')
        require(index['status'] == 'COMPLETE' and index['appendix_sha256'] == expected and len(index['results']) == 4,
                'Incomplete appendix index')
        self.sources(index['source_sha256'])
        seen = set()
        for item in index['results']:
            path = 'results/historical-failure-boundaries/' + item['file']
            result = self.read(path, item['sha256'])
            key = (result['seed'], result['arm'])
            require(key in expected_keys and key not in seen, 'Duplicate/unplanned boundary case')
            seen.add(key)
            baseline, arrays, row, _ = cells[(key[0], key[1], FAILURE_COHORT)]
            frozen = next(entry for entry in appendix['cases'] if (entry['seed'], entry['arm']) == key)
            require(frozen['receipt'] == row['receipt'] and frozen['receipt_sha256'] == row['receipt_sha256'] and
                    result['original_receipt'] == row['receipt'] and result['original_receipt_sha256'] == row['receipt_sha256'],
                    'Boundary check is not linked to its exact original case')
            self.track(frozen['receipt'], frozen['receipt_sha256'])
            self.track(frozen['predictions'], frozen['predictions_sha256'])
            self.sources(result['source_sha256'])
            first = row['actual_terminal_write']
            cuts = [first-1, first, first+1, 2048]
            require(result['status'] == 'PASS' and result['new_memory_trials'] == 0 and result['N'] == 1 and result['T'] == 2048 and
                    result['appendix_sha256'] == expected and result['first_terminal_write'] == first and
                    result['group_token_cuts'] == cuts and frozen['group_token_cuts'] == cuts and
                    result['fresh_process_resume_cut'] == first and result['fresh_processes'] == 1 and
                    result['reviewer_probe_executed'] is False and result['first_prediction_difference'] is None,
                    'Boundary execution scope mismatch')
            artifact = result['predictions_artifact']
            require(artifact['file'] == 'predictions.npz', 'Unexpected boundary prediction path')
            raw_path = (PurePosixPath(path).parent / artifact['file']).as_posix()
            with np.load(self.track(raw_path, artifact['sha256']), allow_pickle=False) as data:
                require(set(data.files) == {'parent', 'fresh'}, 'Boundary prediction array inventory mismatch')
                for name in data.files:
                    require(data[name].dtype == np.int8 and np.array_equal(data[name], arrays['v2_full']),
                            'Boundary restart predictions differ from retained full run')
            trace = result['trace']
            require([entry['group_stop'] for entry in trace] == cuts and result['fresh_trace'] == trace[2:], 'Missing or changed boundary trace')
            frozen_bodies = {entry['body_sha256'] for entry in trace}
            require(len(frozen_bodies) == 1 and next(iter(frozen_bodies)) == baseline['comparisons']['v1_vs_v2_full_body']['payload']['candidate_sha256'],
                    'Terminal body/RNG hash was not frozen at last committed body')
            for entry in trace:
                stop = entry['group_stop']
                require(entry['cursor'] == stop+1 and entry['terminal_code'] == (0 if stop < first else 2) and
                        entry['first_terminal_write'] == (None if stop < first else first), 'Terminal flag/write/cursor trace invalid')
                require(entry['prediction_sha256'] == sha(arrays['v2_full'][:, :stop+1].tobytes()), 'Boundary prefix prediction hash mismatch')
            require(trace[-1]['final_payload_sha256'] == baseline['final_payload_sha256']['v2_full'] and
                    result['counts'] == baseline['execution']['v2_split']['counts'], 'Boundary final hash or counters changed')
            require(len(result['comparisons']) == 6 and all(value is True for value in result['comparisons'].values()),
                    'Boundary reported comparison is inconsistent')
            row.update(post_failure_appendix_pass=True, post_failure_body_hash_frozen=True,
                post_failure_cursor_advances=True, fresh_resume_immediately_after_failure=True,
                boundary_receipt=path, boundary_receipt_sha256=item['sha256'])
        require(seen == expected_keys, 'Missing boundary case')
        return dict(file=relative, sha256=expected, checks=4, new_memory_trials=0)

    def synthetic(self):
        path = 'results/v1_failure_probe_from_request.json'
        result = self.read(path)
        self.track('source/probe_v1_failure.py')
        full = np.asarray(result['uninterrupted_predictions'])
        suffix = np.asarray(result['resumed_suffix_predictions'])
        require(full.shape == (2, 4) and suffix.shape == (2, 3), 'Synthetic probe dimensions changed')
        require(result['kind'] == 'SYNTHETIC_CONTRACT_REPRODUCTION_FROM_REQUEST' and result['reviewer_probe_available'] is False,
                'Synthetic probe misidentified as reviewer execution')
        differs = not np.array_equal(full[:, 1:], suffix)
        require(differs and result['v1_failure_reproduced'] and result['suffix_matches'] is False and
                np.array_equal(full[1, 1:], suffix[1]) and result['healthy_row_predictions_match'], 'Synthetic probe receipt mismatch')
        return dict(kind=result['kind'], source_receipt=path, source_receipt_sha256=self.files[path]['sha256'],
            reviewer_probe_available=False, reviewer_probe_executed=False, model_inference=False,
            v1_failure_reproduced=True, healthy_row_predictions_match=True,
            scope='Request-described injected numerical fault; separate from the 19 actual model replay cells and four continuation checks.',
            first_suffix_difference=first_difference(full[:, 1:], suffix),
            position_convention='Synthetic arrays have no BOS; first_suffix_difference.write_index is local suffix index.')


def summarize(root):
    audit = Audit(root)
    audit.protocol()
    cells = {}
    for seed in range(3):
        path = f'results/historical/seed{seed}/index.json'
        index = audit.read(path)
        require(index['status'] == 'COMPLETE' and index['model_seed'] == seed and not index['technical_dev_smoke'] and
                index['protocol_sha256'] == audit.protocol_hash and index['checkpoint_sha256'] == audit.design['checkpoint_sha256'][str(seed)],
                'Historical index scope/completion mismatch')
        audit.sources(index['source_sha256'])
        expected = {(name, MAIN_COHORT) for name in MAIN_ARMS}
        if seed == 0: expected |= {(name, EXTRA_COHORT) for name in EXTRA_ARMS}
        expected |= {(arm, FAILURE_COHORT) for model_seed, arm, _, _ in FAILURES if model_seed == seed}
        require(len(index['results']) == len(expected), 'Wrong historical cell count')
        seen = set()
        for entry in index['results']:
            key = entry['arm'], entry['cohort']
            require(key in expected and key not in seen and entry['status'] == 'PASS', 'Duplicate or unplanned historical cell')
            seen.add(key)
            relative = f'results/historical/seed{seed}/' + entry['file']
            cells[(seed, *key)] = audit.main_cell(relative, entry['sha256'], seed, *key)
        require(seen == expected, 'Missing historical cell')
    appendix = audit.boundaries(cells)
    probe = audit.synthetic()
    rows = [value[2] for _, value in sorted(cells.items())]
    failures = [row for row in rows if row['cohort'] == FAILURE_COHORT]
    healthy = [row for row in rows if row['cohort'] != FAILURE_COHORT]
    differences = [dict(model_seed=key[0], arm=key[1], cohort=key[2], comparisons=value[3]) for key, value in sorted(cells.items())]
    summary = dict(schema='case010-historical-model-free-summary-v1', status='PASS', protocol_sha256=audit.protocol_hash,
        model_free=True, checkpoint_bytes_loaded=False, model_inference=False, new_memory_trials=0,
        counts=dict(historical_cells=19, main_first16_cells=12, fixed_DEV_extra_cells=3,
            numerically_healthy_cells=15, selected_actual_failure_cases=4, technical_failure_boundary_checks=4,
            v2_full_split_fresh_prediction_matches=sum(row['v2_full_fresh_predictions_equal'] and row['v2_full_split_predictions_equal'] for row in rows),
            v2_full_split_fresh_final_payload_hash_matches=sum(row['v2_full_fresh_payload_hashes_equal'] and row['v2_full_split_payload_hashes_equal'] for row in rows),
            v1_v2_uninterrupted_prediction_matches=sum(row['v1_v2_full_predictions_equal'] for row in rows),
            healthy_v1_v2_body_hash_matches=sum(row['v1_v2_final_body_hashes_equal'] for row in healthy),
            v1_uninterrupted_split_prediction_mismatches=sum(not row['v1_full_split_predictions_equal'] for row in rows),
            exact_original_failure_write_reproductions=sum(row['original_failure_write'] == row['actual_terminal_write'] for row in failures)),
        count_scope='Each of the 19 fixed arm/cohort/checkpoint cells is counted once. Four boundary checks refer to four existing failure cases. Full/split/fresh executions, appendices, synthetic probes and unit-test reruns are not additional statistical trials.',
        verification_scope=dict(prediction_arrays='Decoded public NPZ arrays, dtype/range/shapes, raw-array hashes, all prefix hashes, first prediction differences and terminal predictions recomputed independently.',
            payloads='Consistency of retained final/body hash references and terminal cursor/code traces verified; raw runtime bodies intentionally private and not reread by this model-free audit.',
            sources='Frozen protocol files, runtime source identities, indices, receipt references and prediction artifact hashes checked against actual public bytes.',
            unit_test_counts='Not counted here; use the separate final test receipt.'),
        prediction_contract=dict(A='Finite wrong label is a scored error and does not terminate the state.',
            B='Nonfinite readout of a finite ACTIVE state yields INVALID (-1), commits that finite state, and continues.',
            C='Numeric state failure freezes the last committed body/RNG, stores the first terminal write, and makes every later prediction INVALID while cursor advances.',
            BOS='Write0 consumes BOS and invokes active-state readout; it is excluded from scored token errors. A terminal BOS failure would remain terminal at write1. Resumed calls never prepend BOS.',
            read_boundary='Every write; terminal rows produce INVALID rather than classify zero scratch. Per-cell A/B/C counts use only the uninterrupted v2 trajectory.'),
        interpretation=['All 19 uninterrupted v1/v2 prediction sequences match; 15 numerically healthy cases also retain equal state-body hashes.',
            'Three U2 legacy replays differ after restarting at group-token1024; the earliest split difference is write1025.',
            'The U3 failure at write1741 lies after the last interior original cut1024. Its original split equality alone does not test a restart after failure; the separately frozen appendix does.',
            'All four selected actual failures reproduce at their original recorded positions. Their boundary checks preserve terminal body/RNG hashes and advancing cursors through fresh-process continuation.',
            'These are historical implementation diagnostics, not independent memory-capacity confirmation or numerical-failure incidence estimates.'],
        failure_cases=failures, healthy_cells=healthy, first_prediction_differences=differences,
        boundary_appendix=appendix, request_based_synthetic_probe=probe, source_artifacts=audit.files)
    return summary, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'Summary destination must be new; never overwrite raw or derived results')
    summary, rows = summarize(args.case_root)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'comparisons.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary['comparisons_csv_sha256'] = sha((args.output/'comparisons.csv').read_bytes())
    summary['analysis_source_sha256'] = sha(Path(__file__).read_bytes())
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(status=summary['status'], counts=summary['counts']), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
