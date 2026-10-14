"""Create a fresh redacted compact export; preserve original local evidence."""
import argparse
import json
from pathlib import Path
import re
import shutil
from contracts import sha


def export(result, output):
    result, output = Path(result), Path(output)
    if output.exists():
        raise FileExistsError('fresh public output required')
    output.mkdir(parents=True)
    selected = ['identity.json', 'preregistration.json', 'input_manifest.json', 'runtime_manifest.json',
                'common_prefix_input_manifest.json', 'p1_gate.json', 'p2_selection.json', 'owned_inputs.json',
                'p1_first_responses.jsonl', 'p1_controlled_name_scores.jsonl', 'p1_error_recovery.jsonl',
                'p1_common_prefix_scores.jsonl', 'gpu_commands.json', 'independent_scalar_audit.json',
                'decision.json', 'verification.json', 'preservation.json', 'adapter_validation.json',
                'adapter_commands.json', 'adapter_validation_final.json', 'report_ko.md', 'README_REPRODUCE.md']
    for folder in ('workers', 'p2_compare_inputs', 'p2_compare_reports', 'adapter_export', 'adapter_reports'):
        selected += [str(p.relative_to(result)) for p in (result / folder).rglob('*') if p.is_file()]
    selected += ['site_regression_v2/' + name for name in ('checks.json','docs.json','site.json')]
    mappings = []; private = re.compile('/'+'home'+r'/[^/\s]+/|/mnt/[a-z]/|[A-Za-z]:\\Users\\|gh[pousr]_[A-Za-z0-9]{20,}')
    runtime = json.loads((result / 'runtime_manifest.json').read_text())
    replacements = [(str(result), 'LOCAL_FOLLOWUP_RESULT'),
                    (str(result.parents[1]), 'LOCAL_RESEARCH'),
                    (str(Path(__file__).resolve().parents[2]), 'LOCAL_REPO'),
                    (runtime['model']['path'], 'LOCAL_MODEL_SNAPSHOT')]
    for command in json.loads((result / 'adapter_commands.json').read_text()):
        cpu_root = str(Path(command['command'][0]).parents[1])
        if (cpu_root, 'LOCAL_CPU_CLIENT') not in replacements:
            replacements.append((cpu_root, 'LOCAL_CPU_CLIENT'))
    for name in sorted(set(selected)):
        source = result / name
        if not source.is_file() or source.is_symlink() or source.suffix not in ('.json','.jsonl','.csv','.md'):
            raise ValueError('missing/unsafe compact source: ' + name)
        text = source.read_text()
        if name == 'README_REPRODUCE.md':
            text = '# 재계산과 재실행\n\n[상위 안내](../README_REPRODUCE.md)에 실제 CPU 명령과 외부 모델/환경이 필요한 GPU 요건을 구분했습니다. 이 records는 경로를 치환한 공개 준비이며 원 로컬 명령은 external receipt에 보존합니다.\n'
        for old, new in replacements:
            text = text.replace(old, new)
        if private.search(text):
            raise ValueError('private path/credential remains: ' + name)
        target = output / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        mappings.append({'path': name, 'original_sha256': sha(source), 'public_sha256': sha(target),
                         'bytes': target.stat().st_size,
                         'operation': 'PUBLIC_DOCUMENTATION_ENTRY' if name == 'README_REPRODUCE.md' else
                                      'BYTE_IDENTICAL' if source.read_bytes() == target.read_bytes() else 'LOCAL_PATH_REDACTION'})
    (output / 'export_map.json').write_text(json.dumps({'files': mappings,
        'rules': 'Data: explicit local path substitutions only. Local reproduction guide gets a relative public entry. No metrics/parser/output correction. Original local evidence retained.',
        'excluded': ['model weights','INT4 payloads','full logits','activation arrays','environment','credentials'],
        'external_hashes': 'Original private runtime/artifact identities do not imply public weight bytes'}, indent=2)+'\n')
    print({'files': len(mappings), 'bytes': sum(r['bytes'] for r in mappings)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();export(a.result,a.output)
