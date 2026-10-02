"""Original-file preservation, publication identity and saved-record contracts."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from restore_original import original_bytes
from assessment_v1.audit import audit as audit_assessment

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root = Path(root).resolve()
    if hashlib.sha256((root / 'publication/original_inventory.json').read_bytes()).hexdigest() != '3cf346b6ce9303a7100e39ca48d67dd3388cf1d38a9e83e0dd7375d51a110278':
        raise ValueError('Frozen original inventory identity changed')
    original = json.loads((root / 'publication/original_inventory.json').read_text())
    count = 0
    for name, meta in original['files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe original path')
        blob = original_bytes(root, name)
        if len(blob) != meta['bytes'] or hashlib.sha256(blob).hexdigest() != meta['sha256']:
            raise ValueError('Original identity mismatch: ' + name)
        count += 1
    checksum = root / 'publication/PUBLICATION_SHA256SUMS'
    listed = set()
    for line in checksum.read_text().splitlines():
        h, name = line.split('  ', 1)
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or name in listed:
            raise ValueError('Unsafe or duplicate publication path')
        listed.add(name)
        if (root / p).is_symlink() or not (root / p).resolve().is_relative_to(root):
            raise ValueError('Unsafe publication source: ' + name)
        if hashlib.sha256((root / p).read_bytes()).hexdigest() != h:
            raise ValueError('Publication checksum mismatch: ' + name)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and not {'__pycache__', '.pytest_cache'} & set(p.parts)}
    if listed != actual - {'publication/PUBLICATION_SHA256SUMS'}:
        raise ValueError('Publication inventory mismatch')
    summary = json.loads((root / 'analysis/summary.json').read_text())
    posthoc = json.loads((root / 'publication/posthoc/analysis.json').read_text())
    rows = []
    for name in original['files']:
        if name.startswith('results/attempts/') and name.endswith('/record.json'):
            rows.append(json.loads((root / name).read_text()))
    main = [r for r in rows if r['job']['phase'] == 'main']
    if len(main) != 192 or any(r['status'] != 'SUCCESS' for r in main):
        raise ValueError('MAIN completion mismatch')
    if summary['quality']['primary'] is not None or summary['quality']['status'] != 'ANNOTATION_PENDING':
        raise ValueError('Unannotated quality contract changed')
    for setting in ['A_time', 'B_time', 'C_time']:
        rr = [r for r in main if r['job']['setting']['id'] == setting]
        med = statistics.median(r['complete_seconds'] for r in rr)
        if len(rr) != 64 or med != summary['timing'][setting]['main_complete_median_seconds'] or med != posthoc['settings'][setting]['complete_median_s']:
            raise ValueError('Source timing mismatch: ' + setting)
    neg = sum(r['complete_seconds'] * 1000 < r['sampler_gpu_ms'] for r in main)
    if neg != 45:
        raise ValueError('Clock diagnostic mismatch')
    example = json.loads((root / 'publication/example_identity.json').read_text())
    if example['status'] != 'POST_HOC_QUALITATIVE_NONBLIND_AI_INSPECTION' or example['quality_labels_written'] != 0:
        raise ValueError('Example evaluation scope changed')
    for name in ['README.md', 'README.ko.md']:
        text = (root / name).read_text()
        for value in ['192', '64', '48', '4.534', '4.385', '4.762', 'publication/IMAGES', 'REPORT', 'source/adapter.py']:
            if value not in text:
                raise ValueError('Missing introduction evidence/link: ' + name + '/' + value)
        if 'id="interpretation"' not in text or 'id="run"' not in text:
            raise ValueError('Missing reader route')
    for name in ['IMAGES.md', 'IMAGES.ko.md']:
        text = (root / 'publication' / name).read_text()
        if text.count('### Seed ') != 64 or text.count('../demo/thumbs/') != 192:
            raise ValueError('Incomplete image comparison: ' + name)
    assessment = audit_assessment(root)
    status = json.loads((root / 'publication/status.json').read_text())
    if status['quality_score'] != 'publication/assessment_v1/summary.json' or status['new_model_assisted_images'] != 192 or status['new_human_annotations'] != 0:
        raise ValueError('Publication evaluation scope mismatch')
    for name in ['README.md', 'README.ko.md']:
        text = (root / name).read_text()
        for value in ['51/64', '53/64', 'assessment_v1', 'AI']:
            if value not in text:
                raise ValueError('Missing assessment route: ' + name)
    return {'status': 'PASS', 'original_files': count, 'publication_files': len(listed), 'original_archive_sha256': original['source_archive_sha256'], 'main_images': 192, 'historical_quality': 'ANNOTATION_PENDING', 'current_quality': assessment['assessment'], 'human_raters': 0, 'clock_discrepancies': neg, 'new_generations': 0}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    result = verify(a.root)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))
