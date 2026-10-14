"""Review-only export, manifest and ZIP; excludes models and private paths."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(result, output):
    if output.exists():
        raise FileExistsError('preserve review export; choose fresh output')
    output.mkdir(parents=True)
    identity = json.loads((result / 'identity.json').read_text())
    runtime = json.loads((result / 'runtime_manifest.json').read_text())
    replacements = [(runtime['model']['path'], 'EXTERNAL_MODEL_SNAPSHOT'),
                    (identity['result'], 'LOCAL_RESULT'), (identity['repo'], 'LOCAL_REPO'),
                    (identity['research'], 'LOCAL_RESEARCH')]
    replacements.sort(key=lambda x: -len(x[0]))
    mapping = []

    def export(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = source.read_bytes(); content = raw.decode('utf-8')
        for a, b in replacements:
            content = content.replace(a, b)
        if ('/home' + '/mskim') in content or ('/mnt' + '/c/Users' + '/user') in content:
            raise ValueError('unmapped private path: ' + str(source))
        destination.write_text(content)
        kind = 'CODE/' if source.parent == Path(__file__).parent else 'EXTERNAL_ORIGINAL_INPUTS/'
        mapping.append({'source_relative_path': str(source.relative_to(result)) if source.is_relative_to(result) else kind + source.name,
                        'source_sha256': hashlib.sha256(raw).hexdigest(), 'export_path': str(destination.relative_to(output)),
                        'export_sha256': sha(destination), 'path_redacted': raw != destination.read_bytes()})

    for name in ('preregistration.json', 'runtime_manifest.json', 'input_manifest.json', 'identity.json',
                 'protected_before.json', 'gpu_commands.json', 'cpu_verification.json', 'cpu_tests_attempt03.log',
                 'summary.json', 'summary.csv', 'verification.json', 'decision.json', 'REPORT_KO.md', 'REPORT.md',
                 'README.ko.md', 'README.md', 'failure_examples.json', 'recommended_preset.json',
                 'original_case013_public_verification.json', 'validation_attempts.json', 'compare_verification.json'):
        source = result / name
        if not source.exists():
            raise FileNotFoundError('required reviewed artifact: ' + name)
        export(source, output / 'records' / name)
    for source in sorted((result / 'workers').rglob('*')):
        if source.is_file() and source.suffix in ('.json', '.jsonl'):
            export(source, output / 'records' / source.relative_to(result))
    for source in sorted((result / 'attempts').glob('*.log')):
        export(source, output / 'records' / source.relative_to(result))
    for directory in ('compare_inputs', 'compare_reports'):
        for source in sorted((result / directory).rglob('*')):
            if source.is_file():
                export(source, output / 'records' / source.relative_to(result))
    export(Path(identity['repo']) / 'LICENSE', output / 'LICENSE')
    (output / 'SOURCE_NOTICE.md').write_text('''# Sources and reuse\n\nThis follow-up adds a tool-name enum decoder, registry-aware feedback, a bounded runner and CPU event replay. It reuses DIOVA's existing independent INT4 reference and synthetic local-tool harness as external dependencies. DualQuant is the upstream work of Mahsa Salmani, Sayeh Sharify and Hesham Mostafa; Qwen supplies the model/tokenizer; PyTorch and Transformers supply inference. The vendor tree, private block-format library and model/INT4 weights are not redistributed.\n\nSee the preserved [Case 013 notice](https://github.com/Munsik-Kim/inference-lab/blob/27294fc5cfde575cbed638c17b9a547e126a2f01/cases/013-dualquant-reference-quality/NOTICE.md) for original attribution and external license boundaries. The repository's existing Apache-2.0 LICENSE is included unchanged; third-party rights are not relicensed. Synthetic task fixtures are experiment-owned. Codex assisted implementation and reporting.\n''')
    for name in ('agent_tasks.json', 'tool_schema.json'):
        export(Path(runtime['original_result_root']) / 'inputs' / name, output / 'records/inputs' / name)
    for source in sorted(Path(__file__).parent.iterdir()):
        if source.is_file() and source.suffix in ('.py', '.md'):
            export(source, output / 'code' / source.name)
    (output / 'README.md').write_text('''# Local review: Case 013 tool-contract repair\n\nStart with [Korean report](records/REPORT_KO.md) or [English report](records/REPORT.md).\n\nModel-free replay after extraction, from this directory:\n\n```bash\npython3 -B code/analyze.py --result records --inputs records/inputs --output /tmp/tool-repair-review-audit-new --skip-external-preservation\npython3 -B -m unittest discover -s code -p 'test_*.py' -v\n```\n\nThe skip flag does not certify external models/cases; the original preservation receipt records the local check. Models, quantized weights, environments and full logits are excluded. Runtime hashes before redaction are external identities; exported bytes have a separate manifest. See the code guide for actual GPU prerequisites. This is a local review bundle, not a model download or remote publication.\n''')
    (output / 'EXPORT_MAP.json').write_text(json.dumps({'rule': 'replace private absolute paths only; measurements/answers unchanged', 'files': mapping}, indent=2) + '\n')
    files = [{'path': str(p.relative_to(output)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(output.rglob('*')) if p.is_file()]
    (output / 'MANIFEST.json').write_text(json.dumps({'schema_version': 1, 'files': files}, indent=2) + '\n')
    archive = output.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError('archive already exists')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                z.write(p, str(p.relative_to(output)))
    metadata = {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive),
                'files': len(files) + 1, 'weights_included': False, 'remote_publication': False}
    result.joinpath('archive_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); main(a.result, a.output)
