"""Conditional request accounting, without creating model responses."""
from .io import read
from .upstream import ROOT, digest


def limits(eligible_failures):
    f = eligible_failures
    if type(f) is not int or not 0 <= f <= 24:
        raise ValueError('F must be an integer in [0,24]')
    return {'F': f, 'first_diagnostic_responses': 48 + 2 * f,
        'later_continuation_requests_upper_bound': 240 + 3 * f,
        'total_new_model_requests_upper_bound': 288 + 5 * f,
        'total_new_tokens_upper_bound': (288 + 5 * f) * 192}


def dry_run():
    protocol = read(ROOT / 'specs/protocol.json')
    checkpoints = read(ROOT / 'data/checkpoints.json')
    cells = []
    for checkpoint in checkpoints:
        for method in protocol['methods']:
            for arm in ['C0', 'G', 'N', 'E']:
                cells.append({'checkpoint_id': checkpoint['checkpoint_id'], 'method': method, 'condition': arm,
                    'status': 'NOT_RUN', 'success': None,
                    'trigger': 'same NEW method/checkpoint C0 malformed-only and remaining cumulative budget' if arm in ['N', 'E'] else 'future explicit approval',
                    'prefix_calls_charged_not_regenerated': 2, 'prefix_tool_attempts_charged': 2})
    return {'status': 'NOT_RUN_GPU_GATE_CLOSED', 'protocol_sha256': digest(protocol),
        'selected_fixtures': 4, 'api_checkpoints': 8, 'C0_first_requests': 24, 'G_first_requests': 24,
        'historical_F8_resource_scenario_not_a_prediction': limits(8), 'absolute_upper_bound': limits(24),
        'not_independent_episodes': True, 'prefix_regeneration': False,
        'future_total_worker_wall_budget': 'NOT_AUTHORIZED_OR_SET_IN_THIS_CPU_STAGE',
        'local_inputs_required': ['existing local snapshot/config/tokenizer/template at registered revision',
            'RTN and D16 each252 pinned projection files and manifests; no regeneration',
            'original_result_root and approved original adapter arithmetic',
            'compatible existing GPU Python/environment; no install or upgrade',
            'future reviewed model/grammar hook and measured new-C0 output-token reproduction'],
        'cells': cells, 'model_responses_generated': 0}

