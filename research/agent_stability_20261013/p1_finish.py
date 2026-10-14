"""Deterministic common-prefix registration and P1/P2 gate readback (CPU)."""
import argparse
from pathlib import Path
from contracts import (atomic, append, read, rows, token_hash, common_length,
                       tool_blocks, first_patch_correct, select_projection, worker_dir, METHODS, TASK_IDS)


def common(result):
    result = Path(result)
    natural = {method: {r['task_id']: r for r in rows(worker_dir(result, 'first', method) / 'first_responses.jsonl')}
               for method in METHODS}
    inputs = {r['task_id']: r for r in read(result / 'input_manifest.json')['natural']}
    probes = []
    for key in TASK_IDS:
        for baseline, candidate in (('BF16', 'D16'), ('BF16', 'D4'), ('RTN', 'D16'), ('RTN', 'D4')):
            a, b = natural[baseline][key], natural[candidate][key]
            if a['input_token_sha256'] != b['input_token_sha256'] or a['input_token_sha256'] != inputs[key]['input_hash']:
                raise ValueError('natural prompt mismatch')
            ids_a, ids_b = a['output_token_ids'], b['output_token_ids']
            n = common_length(ids_a, ids_b)
            value = {'task_id': key, 'baseline': baseline, 'candidate': candidate,
                     'common_response_token_count': n, 'baseline_response_tokens': len(ids_a),
                     'candidate_response_tokens': len(ids_b)}
            if n == len(ids_a) == len(ids_b):
                value['status'] = 'NO_RESPONSE_TOKEN_DIVERGENCE'
            elif n == min(len(ids_a), len(ids_b)):
                value['status'] = 'ONE_RESPONSE_ENDED_BEFORE_DIFFERENT_TOKEN'
            else:
                prefix = inputs[key]['input_ids'] + ids_a[:n]
                value.update(status='DIVERGENCE', prefix_ids=prefix, prefix_hash=token_hash(prefix),
                             baseline_next_id=ids_a[n], candidate_next_id=ids_b[n],
                             role='same-prefix teacher forcing; not a causal attribution')
            probes.append(value)
    atomic(result / 'common_prefix_input_manifest.json', probes, fresh=True)
    print({'status': 'FROZEN_BEFORE_COMMON_PREFIX_FORWARDS', 'pairs': len(probes),
           'divergent_pairs': sum(p['status'] == 'DIVERGENCE' for p in probes)})


def finish(result):
    result = Path(result)
    runtime = read(result / 'runtime_manifest.json')
    research = Path(runtime['original_result_root'])
    tasks = {r['id']: r for r in read(research / 'inputs/agent_tasks.json')}
    natural = {}
    counts = {}
    for method in METHODS:
        first, common_dir = worker_dir(result, 'first', method), worker_dir(result, 'common', method)
        if read(first / 'head_parity.json')['status'] != 'PASS' or read(common_dir / 'head_parity.json')['status'] != 'PASS':
            raise ValueError('head parity not passed')
        natural[method] = {r['task_id']: r for r in rows(first / 'first_responses.jsonl')}
        controlled, recovery = rows(first / 'controlled_name_scores.jsonl'), rows(first / 'error_recovery.jsonl')
        if set(natural[method]) != set(TASK_IDS) or len(controlled) != 4 or len(recovery) != 2:
            raise ValueError('P1 mandatory measurement incomplete')
        counts[method] = {'natural': 4, 'controlled_name_probes': 4, 'error_recovery': 2}
        for filename, record_list in [('p1_first_responses.jsonl', list(natural[method].values())),
                                      ('p1_controlled_name_scores.jsonl', controlled),
                                      ('p1_error_recovery.jsonl', recovery),
                                      ('p1_common_prefix_scores.jsonl', rows(common_dir / 'common_prefix_scores.jsonl'))]:
            for row in record_list:
                append(result / filename, row)
    gated = []
    for key in ('AGENT-6', 'AGENT-7'):
        baseline_correct = all(first_patch_correct(natural[m][key], tasks[key]) for m in ('BF16', 'RTN'))
        calls = tool_blocks(natural['D16'][key]['raw_text'])
        unknown_first = bool(calls and calls[0].get('name') not in
                             ('read_document', 'read_record', 'query_table', 'calculate', 'apply_fixture_patch', 'validate_fixture_test'))
        if baseline_correct and unknown_first:
            gated.append(key)
    gate = {'P1': 'COMPLETED' if gated else 'NOT_REPRODUCED', 'P2_initial_gate': bool(gated),
            'gated_task_ids': gated, 'same_input_artifact_dtype_head_contract': 'PASS', 'counts': counts,
            'scope': 'first tool action, not complete episode success; selected historical DEV'}
    atomic(result / 'p1_gate.json', gate, fresh=True)
    if gated:
        first = worker_dir(result, 'first', 'D16')
        errors = {key: read(first / ('projection_errors_' + key + '.json')) for key in ('AGENT-6', 'AGENT-7')}
        selection = select_projection(errors, read(Path(runtime['artifact_root']) / 'D16/manifest.json'))
        atomic(result / 'p2_selection.json', selection, fresh=True)
    print(gate)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--phase', choices=('common', 'finish'), required=True)
    a = p.parse_args()
    (common if a.phase == 'common' else finish)(a.result)
