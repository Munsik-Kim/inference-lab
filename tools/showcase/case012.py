"""Verified archive registration only; Case012 has no deployed Pages view yet."""
from pathlib import Path
from common import read, require, sha

C12 = 'cases/012-looped-dit-inference-budget'


def load_case012(root: Path):
    path = root / C12 / 'analysis/summary.json'
    if not path.exists():
        return None
    data = read(path)
    require(data['schema'] == 'case012-summary-v1', 'Case012 summary schema')
    require(data['quality']['status'] == 'ANNOTATION_PENDING' and data['quality']['primary'] is None,
            'Case012 archive requires the recorded pending-quality scope')
    require(data['generation']['main_success'] == data['generation']['main_expected'] == 192,
            'Case012 recorded generation coverage')
    for setting in ('A_time', 'B_time', 'C_time'):
        require(data['timing'][setting]['images'] == 64, 'Case012 per-setting denominator')
    for key, path in [('protocol', 'configs/protocol.json'), ('prompts', 'configs/prompts.json'), ('rubric', 'configs/rubric.json'), ('models', 'provenance/models.json')]:
        require(sha(root / C12 / path) == data['source_hashes'][key], 'Case012 archive source identity')
    return data
