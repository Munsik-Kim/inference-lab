"""Raw -> CPU endpoint replay -> required publication tables and decision."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

P2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P2 / 'execution'))
from audit import audit, pair
from state import read, write
from diagnostics import diagnose


def export_csv(path, rows):
    if not rows:
        return
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def derive(results, output):
    results = Path(results); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    summary = audit(results, output / 'raw_replay')
    groups = diagnose(results, output / 'diagnostics')
    export_csv(output / 'method_api_arm.csv', summary['table'])
    export_csv(output / 'paired_outcomes.csv', summary['paired'])
    export_csv(output / 'C0_reproduction.csv', read(output / 'raw_replay/reproduction.json'))
    records = [read(f) for f in sorted((results / 'cells').glob('*.json'))]
    requests = [json.loads(line) for line in (results / 'requests.jsonl').read_text().splitlines() if line]
    costs = []
    for method in ('BF16', 'RTN', 'D16'):
        for arm in ('C0', 'G', 'N', 'E'):
            rows = [r for r in requests if r['method'] == method and r['arm'] == arm]
            costs.append({'method': method, 'arm': arm, 'new_unique_requests': len(rows),
                'focal': sum(r['request_role'] == 'FOCAL' for r in rows),
                'continuation': sum(r['request_role'] == 'CONTINUATION' for r in rows),
                'generated_tokens': sum(r['output_tokens'] for r in rows),
                'generation_seconds_sum': sum(r['seconds'] for r in rows)})
    export_csv(output / 'request_costs.csv', costs)
    compact = []
    for method in ('BF16', 'RTN', 'D16'):
        item = {'method': method}
        for arm in ('C0', 'G', 'N', 'E'):
            rows = [r for r in records if r['method'] == method and r['arm'] == arm and r['observation_status'] == 'OBSERVED']
            item[arm] = {'success': sum(r['VERIFIED_WORKFLOW_SUCCESS'] for r in rows), 'denominator': len(rows),
                         'status': 'OBSERVED' if rows else 'NOT_APPLICABLE' if all(t['not_applicable_cells'] == t['planned_checkpoints'] for t in summary['table'] if t['method'] == method and t['arm'] == arm) else 'NOT_RUN'}
        compact.append(item)
    group = {(r['method'], r['arm']): r for r in groups}
    d = next(r for r in compact if r['method'] == 'D16')
    quarter = d['G']
    if summary['C0_reproduction']['matched'] != 24 or summary['status'] != 'COMPLETED':
        verdict = 'JUDGEMENT_HELD'
    elif quarter['success'] > d['C0']['success']:
        verdict = 'RECOVERY_OBSERVED'
    elif group[('D16', 'G')]['correct_write_cells'] > group[('D16', 'C0')]['correct_write_cells']:
        verdict = 'PARTIAL_RECOVERY'
    else:
        verdict = 'NO_WORKFLOW_RECOVERY_OBSERVED'
    ne_pairs = next(r for r in summary['paired'] if r['method'] == 'D16' and r['left'] == 'N')
    decision = {'status': summary['status'], 'verdict': verdict, 'C0_reproduction': summary['C0_reproduction'],
        'method_summary': compact, 'F_actual': summary['F_actual'],
        'N_correct_write_cells': group[('D16', 'N')]['correct_write_cells'],
        'N_state_only_success': group[('D16', 'N')]['state_only_success'],
        'E_vs_N_paired': ne_pairs,
        'N_remaining_stages': group[('D16', 'N')]['terminal_failure_stages'],
        'E_remaining_stages': group[('D16', 'E')]['terminal_failure_stages'],
        'registered_primary_endpoint': 'VERIFIED_WORKFLOW_SUCCESS',
        'E_better_than_N_in_verified_success': d['E']['success'] > d['N']['success'],
        'equivalence_claim': False, 'independent_selected_fixtures': 4, 'API_checkpoints': 8,
        'fresh_heldout_or_full_end_to_end_agent_evaluation': False,
        'historical_A1_failures_preserved': True, 'packed_serving_benefit': 'NOT_EVALUATED',
        'execution_source_commit': read(results / 'identity.json')['execution_source_commit']}
    write(output / 'decision.json', decision)
    write(output / 'display.json', {'summary': compact, 'table': summary['table'], 'paired': summary['paired'],
                                  'costs': summary['costs'], 'costs_by_arm': costs, 'error_groups': groups})
    return decision


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--results', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(derive(args.results, args.output)))
