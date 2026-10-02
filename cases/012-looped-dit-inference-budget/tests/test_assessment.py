"""Supplemental AI scoring contracts, separate from synthetic or human labels."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('supplement_audit', ROOT / 'publication/assessment_v1/audit.py')
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)
spec2 = importlib.util.spec_from_file_location('supplement_analyze', ROOT / 'publication/assessment_v1/analyze.py')
analysis_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(analysis_module)


def synthetic():
    records, items, rows = [], [], []
    for seed in (72301, 72302, 72303, 72304):
        for setting in ('A_time', 'B_time', 'C_time'):
            iid = f'synthetic_test-{seed}-{setting}'
            records.append({'job_id': iid, 'image_sha256': 'fixture', 'status': 'SUCCESS',
                            'job': {'prompt_id': 'synthetic-prompt', 'seed_label': seed, 'setting': {'id': setting}}})
            items.append({'image_id': iid, 'image_sha256': 'fixture', 'category': 'synthetic_test', 'constraints': [{'id': 'c1'}, {'id': 'c2'}]})
            rows.append({'image_id': iid, 'image_sha256': 'fixture', 'values': {'c1': 'satisfied', 'c2': 'satisfied'}})
    return records, {'rows': rows}, items


def test_completed_real_assessment_independently_recalculates():
    result = audit_module.audit(ROOT)
    assert result['scored_main_images'] == 192
    assert result['human_raters'] == 0
    assert sum(result['paired'].values()) == result['prompt_seed_pairs'] == 64


def test_uncertain_fails_primary_but_has_separate_sensitivity():
    records, ann, items = synthetic()
    ann['rows'][0]['values']['c1'] = 'uncertain'
    out = audit_module.recompute(records, ann, items)
    a = out['settings']['A_time']
    assert a['all_constraint_pass_rate'] == .75
    assert a['all_uncertain_as_pass_rate'] == 1
    assert a['mean_constraint_fraction'] == .875
    assert out['uncertain_constraint_count'] == 1


def test_primary_is_c_minus_a_not_reverse_or_b():
    records, ann, items = synthetic()
    for row in ann['rows']:
        if row['image_id'].endswith('A_time'):
            row['values']['c1'] = 'not_satisfied'
    out = audit_module.recompute(records, ann, items)
    assert out['primary']['estimate'] == 1
    assert out['paired']['C_only'] == 4


def test_row_and_metric_key_permutations_do_not_change_scores():
    records, ann, items = synthetic()
    expected = audit_module.recompute(records, ann, items)
    ann['rows'].reverse()
    for row in ann['rows']:
        row['values'] = dict(reversed(list(row['values'].items())))
    assert audit_module.recompute(list(reversed(records)), ann, list(reversed(items))) == expected


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'unknown', 'hash', 'constraint', 'value', 'failed'])
def test_invalid_label_or_source_contract_rejected(fault):
    records, ann, items = synthetic()
    if fault == 'missing':
        ann['rows'].pop()
    elif fault == 'duplicate':
        ann['rows'].append(copy.deepcopy(ann['rows'][0]))
    elif fault == 'unknown':
        ann['rows'][0]['image_id'] = 'not-in-cohort'
    elif fault == 'hash':
        ann['rows'][0]['image_sha256'] = 'different'
    elif fault == 'constraint':
        ann['rows'][0]['values'].pop('c2')
    elif fault == 'value':
        ann['rows'][0]['values']['c1'] = 'made-up-label'
    else:
        records[0]['status'] = 'FAILED'
    with pytest.raises(ValueError):
        audit_module.recompute(records, ann, items)


def test_mutating_completed_labels_requires_new_version(tmp_path):
    target = tmp_path / 'publication/assessment_v1'
    target.mkdir(parents=True)
    for name in ('protocol.json', 'annotations.json'):
        (target / name).write_bytes((ROOT / 'publication/assessment_v1' / name).read_bytes())
    ann = json.loads((target / 'annotations.json').read_text())
    ann['rows'][0]['values']['c1'] = 'not_satisfied'
    (target / 'annotations.json').write_text(json.dumps(ann))
    with pytest.raises(ValueError, match='new version'):
        analysis_module.calculate(tmp_path)


def test_historical_quality_stays_pending_while_ai_supplement_complete():
    original = json.loads((ROOT / 'analysis/summary.json').read_text())
    current = json.loads((ROOT / 'publication/assessment_v1/summary.json').read_text())
    assert original['quality']['primary'] is None
    assert original['quality']['status'] == 'ANNOTATION_PENDING'
    assert current['status'] == 'MODEL_ASSISTED_EVALUATION_COMPLETE'
    assert current['human_evaluation'] == 'NOT_RUN'
    assert current['evaluator']['inter_rater_agreement'] == 'NOT_MEASURED'
