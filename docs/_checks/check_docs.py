"""Read-only checks for this bounded bilingual Markdown set, not a general parser.

Supported: ATX headings, fenced blocks, single-line inline links/images without
whitespace/parentheses in destinations, explicit <a id="..."></a>, claim comments.
Reference-style links and other HTML are rejected rather than silently passed.
Source links may target preserved case Markdown outside this authored subset.
Translation meaning requires a separate editorial review.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlsplit

try:
    from markdown_it import MarkdownIt
except ImportError:
    MarkdownIt = None

GUIDES = ('CASEBOOK.md', 'GETTING_STARTED.md', 'PORTFOLIO.md', 'START_HERE.md', 'GLOSSARY.md')
PAGES = ('README.md', 'README.ko.md', *(f'docs/{lang}/{name}' for lang in ('en', 'ko') for name in GUIDES))
CONCEPT_FIGURE = 'docs/assets/inference-reading-path.svg'
PUBLIC_FILES = (*PAGES, CONCEPT_FIGURE, 'docs/_meta/claims.json', 'docs/_meta/MAINTENANCE.md',
                'docs/_checks/check_docs.py', 'docs/_checks/test_check_docs.py')
LINK = re.compile(r'!?\[[^\]\n]*\]\(([^\s()]+)\)')
CLAIMS = re.compile(r'<!-- claims: ([a-zA-Z0-9_\- ]+) -->')
FEEDBACK_DOCS = (
    'docs/feedback_actions.md', 'docs/interview-notes.ko.md', 'docs/phase2a-design.md',
    'docs/related-work/README.md', 'docs/related-work/README.ko.md',
    'packages/diova-compare/README.md', 'packages/diova-compare/README.ko.md',
    *(f'cases/009-q-serving-quality/{name}' for name in
      ('README.md','README.ko.md','REPORT.md','REPORT.ko.md','METHODS.md','NOTICE.md','REPRODUCTION.md')),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prose(text: str) -> str:
    return re.sub(r'^```[^\n]*\n.*?^```\s*$', '', text, flags=re.M | re.S)


def anchors(text: str) -> set[str]:
    body = prose(text)
    result = set(re.findall(r'<a id="([\w-]+)"></a>', body))
    seen: dict[str, int] = {}
    for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', body, re.M):
        slug = re.sub(r'[^\w\- ]', '', heading.strip().lower()).replace(' ', '-')
        count = seen.get(slug, 0)
        result.add(slug if not count else f'{slug}-{count}')
        seen[slug] = count + 1
    return result


def check_links(root: Path, pages: list[str], errors: list[str]) -> int:
    count = 0
    for name in pages:
        path = root / name
        if not path.is_file():
            errors.append('Missing page: ' + name)
            continue
        text = prose(path.read_text())
        if re.search(r'\[[^\]\n]+\]\s*\[|^\s*\[[^\]\n]+\]:', text, re.M):
            errors.append('Unsupported reference-link syntax: ' + name)
        clean = re.sub(r'<!--.*?-->|<a id="[\w-]+"></a>', '', text, flags=re.S)
        if re.search(r'<\s*/?[A-Za-z][^>]*>', clean):
            errors.append('Unsupported HTML: ' + name)
        links = list(LINK.finditer(text))
        if text.count('](') != len(links):
            errors.append('Unsupported/malformed inline link: ' + name)
        targets = [m.group(1) for m in links]
        if MarkdownIt is not None:
            parsed = []
            for token in MarkdownIt('commonmark').parse(text):
                for child in token.children or []:
                    if child.type == 'link_open':
                        parsed.append(child.attrGet('href'))
                    elif child.type == 'image':
                        parsed.append(child.attrGet('src'))
            if parsed != targets:
                errors.append('Inline syntax differs from installed Markdown parser: ' + name)
            targets = parsed
        for target in targets:
            url = urlsplit(target)
            if url.scheme:
                if url.scheme not in ('https', 'http'):
                    errors.append('Unsupported URL scheme: ' + name)
                continue
            if target.startswith(('/', '//')) or '\\' in target:
                errors.append('Non-relative local destination: ' + name)
                continue
            dest = (path.parent / unquote(url.path)).resolve() if url.path else path
            count += 1
            if not dest.is_relative_to(root.resolve()) or not dest.exists():
                errors.append(f'Broken local link: {name} -> {target}')
            elif url.fragment:
                if dest.suffix != '.md' or unquote(url.fragment) not in anchors(dest.read_text()):
                    errors.append(f'Missing heading/explicit anchor: {name} -> {target}')
    return count


def check_pairs(root: Path, errors: list[str]) -> None:
    pairs = [('README.md', 'README.ko.md', 'README.ko.md', 'README.md')]
    pairs += [(f'docs/en/{n}', f'docs/ko/{n}', f'../ko/{n}', f'../en/{n}') for n in GUIDES]
    for en, ko, to_ko, to_en in pairs:
        if not (root/en).is_file() or not (root/ko).is_file():
            errors.append('Missing language counterpart: ' + en + ' / ' + ko)
            continue
        en_text, ko_text = (root/en).read_text(), (root/ko).read_text()
        if f']({to_ko})' not in en_text or f']({to_en})' not in ko_text:
            errors.append('Wrong language-switch destination: ' + en)
        if en.startswith('docs/') and ('](../../README.md)' not in en_text or '](../../README.ko.md)' not in ko_text):
            errors.append('Missing same-language home: ' + en)
        ids = lambda t: {v for b in CLAIMS.findall(t) for v in b.split()}
        if ids(en_text) != ids(ko_text):
            errors.append('Unequal language claim coverage: ' + en)
    for lang in ('en', 'ko'):
        path = root / f'docs/{lang}/CASEBOOK.md'
        if path.exists():
            for n in range(1, 8):
                if f'case-{n:03d}' not in anchors(path.read_text()):
                    errors.append(f'Missing case anchor: {lang}/{n}')


def pointer(obj, path: str):
    if not path.startswith('/'):
        raise ValueError('JSON pointer must start with /')
    for token in path[1:].split('/'):
        key = token.replace('~1', '/').replace('~0', '~')
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj


def check_beginner_routes(root: Path, errors: list[str]) -> None:
    """Check navigation/structure; reading difficulty and meaning need human review."""
    glossary_ids = []
    for lang, home in (('en', 'README.md'), ('ko', 'README.ko.md')):
        for name, prefix in ((home, f'docs/{lang}/'), (f'docs/{lang}/START_HERE.md', '')):
            path = root/name
            if not path.is_file():
                errors.append('Missing beginner entry: ' + name)
                continue
            targets = {m.group(1) for m in LINK.finditer(prose(path.read_text()))}
            for n in range(1, 8):
                if prefix + f'CASEBOOK.md#case-{n:03d}' not in targets:
                    errors.append(f'Missing beginner case route: {name}/{n}')
            if prefix + 'GLOSSARY.md' not in targets:
                errors.append('Missing glossary route: ' + name)
            if name == home and prefix + 'START_HERE.md' not in targets:
                errors.append('Missing introduction route: ' + name)
        path = root/f'docs/{lang}/GLOSSARY.md'
        if path.is_file():
            ids = re.findall(r'<a id="([\w-]+)"></a>', path.read_text())
            glossary_ids.append(set(ids))
            if not 15 <= len(ids) <= 20 or len(set(ids)) != len(ids):
                errors.append('Invalid glossary entry count/duplicate anchor: ' + lang)
    if len(glossary_ids) == 2 and glossary_ids[0] != glossary_ids[1]:
        errors.append('Unequal glossary concept anchors')


def check_concept_figure(root: Path, errors: list[str]) -> None:
    """A small declarative SVG; no scripts, events or embedded external content."""
    path = root/CONCEPT_FIGURE
    if not path.is_file() or path.is_symlink():
        errors.append('Missing/unsupported concept figure')
        return
    try:
        text = path.read_text()
        if '<!DOCTYPE' in text.upper() or '<!ENTITY' in text.upper():
            raise ValueError('Document entities are unsupported')
        tree = ET.fromstring(text)
        allowed = {'svg', 'title', 'desc', 'g', 'rect', 'path', 'text'}
        for node in tree.iter():
            if node.tag.removeprefix('{http://www.w3.org/2000/svg}') not in allowed:
                raise ValueError('Unsupported SVG element')
            for key, value in node.attrib.items():
                if key.lower().startswith('on') or key.endswith('href') or 'url(' in value.lower():
                    raise ValueError('Active/external SVG attribute')
        if tree.tag != '{http://www.w3.org/2000/svg}svg' or not tree.findtext('{http://www.w3.org/2000/svg}title') or not tree.findtext('{http://www.w3.org/2000/svg}desc'):
            raise ValueError('Missing accessible SVG title/description')
    except (ET.ParseError, ValueError) as exc:
        errors.append('Invalid concept figure: ' + str(exc))


def check_copy_hygiene(root: Path, names: tuple[str, ...], errors: list[str]) -> None:
    """Flag copy patterns for review, not proof that all sensitive data is absent."""
    blocked = re.compile(r'\bTODO\b|\bTBD\b|\bPLACEHOLDER\b|example\.(?:com|org)|turn\d+(?:search|view|fetch)\d+|filecite|/home/[^/\s]+/|/mnt/|C:\\\\Users|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}')
    for name in names:
        if (root/name).is_file() and blocked.search((root/name).read_text()):
            errors.append('Private identifier/credential/unfinished citation pattern: ' + name)


def check_claims(root: Path, mapping: dict, pages: list[str], errors: list[str]) -> int:
    claims = mapping.get('claims', [])
    ids = [c['claim_id'] for c in claims]
    if len(ids) != len(set(ids)):
        errors.append('Duplicate claim ID mapping')
    used = {}
    for name in pages:
        if (root/name).exists():
            used[name] = {v for block in CLAIMS.findall((root/name).read_text()) for v in block.split()}
            for cid in used[name] - set(ids):
                errors.append('Unknown claim ID: ' + cid)
    for c in claims:
        cid = c['claim_id']
        source = (root/c['source_path']).resolve()
        if not source.is_relative_to(root.resolve()) or not source.is_file():
            errors.append('Missing source: ' + cid)
            continue
        if sha(source) != c['source_sha256']:
            errors.append('Changed source hash: ' + cid)
        archive_source = any(
            c['source_revision'] == 'archive:'+entry['archive_sha256']
            and c['source_path'].startswith(entry['case_path']+'/')
            for entry in mapping.get('additional_publications', []))
        if not archive_source and (c['source_revision'] != mapping['source_revision'] or not re.fullmatch('[0-9a-f]{40}', c['source_revision'])):
            errors.append('Invalid source revision: ' + cid)
        for key in ('claim_en', 'claim_ko', 'scope', 'limits'):
            if not c.get(key):
                errors.append('Missing claim meaning/scope: ' + cid + '/' + key)
        loc = c['source_locator']
        try:
            if loc['kind'] == 'text':
                if not loc['contains'] or any(t not in source.read_text() for t in loc['contains']):
                    errors.append('Missing source text locator: ' + cid)
            elif loc['kind'] == 'json':
                obj = json.loads(source.read_text())
                if not loc['checks'] or any(pointer(obj, v['pointer']) != v['equals'] for v in loc['checks']):
                    errors.append('Source numeric/value mismatch: ' + cid)
            else:
                errors.append('Unsupported source locator: ' + cid)
        except (KeyError, IndexError, ValueError, TypeError):
            errors.append('Invalid source locator: ' + cid)
        locations = c['docs_locations']
        if len(locations) != len(set(locations)):
            errors.append('Duplicate document mapping: ' + cid)
        actual = {n for n, values in used.items() if cid in values}
        if not actual or set(locations) != actual:
            errors.append('Claim/document mapping mismatch: ' + cid)
        for d in c.get('display_checks', []):
            f = root/d['path']
            if d['path'] not in locations or not f.is_file() or any(t not in f.read_text() for t in d['contains']):
                errors.append('Document numeric/identifier display mismatch: ' + cid)
    return len(claims)


def check_protection(root: Path, base: str, errors: list[str], inventory: Path | None, additional: set[str] | None = None) -> int:
    if inventory:
        data = json.loads(inventory.read_text())
        if data['base_revision'] != base:
            errors.append('Protection inventory revision mismatch')
        for name, expected in data['files'].items():
            f = root/name
            if not f.is_file() or sha(f) != expected:
                errors.append('Protected file changed: ' + name)
        check_protected_additions(root, set(data['files']) | (additional or set()), errors)
        return len(data['files'])
    result = subprocess.run(['git', 'ls-tree', '-r', '-z', base], cwd=root, capture_output=True, check=True)
    count = 0
    known = set()
    for entry in result.stdout.split(b'\0'):
        if not entry:
            continue
        info, rawname = entry.split(b'\t', 1)
        mode, kind, expected = info.decode().split()
        name = rawname.decode()
        known.add(name)
        if name in PUBLIC_FILES:
            continue
        count += 1
        f = root/name
        if kind != 'blob' or not f.is_file() or f.is_symlink():
            errors.append('Protected entry missing/unsupported: ' + name)
            continue
        data = f.read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != expected:
            errors.append('Protected Git blob changed: ' + name)
    check_protected_additions(root, known | (additional or set()), errors)
    return count


def check_protected_additions(root: Path, known: set[str], errors: list[str]) -> None:
    for prefix in ('cases', 'downloads', 'notes'):
        for path in (root/prefix).rglob('*'):
            if path.is_file() or path.is_symlink():
                name = path.relative_to(root).as_posix()
                if name not in known:
                    errors.append('Unexpected file in protected tree: ' + name)


def check_packages(root: Path, mapping: dict, errors: list[str]) -> list[str]:
    """Check both linked publication archives against existing metadata."""
    downloads = []
    for cid in ('c006-package', 'c007-package'):
        matches = [c for c in mapping['claims'] if c['claim_id'] == cid]
        if len(matches) != 1:
            errors.append('Missing/duplicate package mapping: ' + cid)
            continue
        metadata_path = (root/matches[0]['source_path']).resolve()
        if not metadata_path.is_relative_to(root.resolve()) or not metadata_path.is_file():
            errors.append('Missing/unsafe package metadata: ' + cid)
            continue
        meta = json.loads(metadata_path.read_text())
        filename = meta['filename']
        if not isinstance(filename, str) or Path(filename).name != filename or '\\' in filename:
            errors.append('Unsafe package filename: ' + cid)
            continue
        archive = metadata_path.parent/filename
        if archive.is_symlink() or not archive.is_file() or sha(archive) != meta['sha256'] or archive.stat().st_size != meta['bytes']:
            errors.append('Default download identity mismatch: ' + cid)
        for lang in ('en', 'ko'):
            guide = (root/f'docs/{lang}/GETTING_STARTED.md').read_text()
            if filename not in guide or metadata_path.name not in guide:
                errors.append('Current archive/metadata link missing: ' + cid + '/' + lang)
        downloads.append(filename)
    return downloads


def check_additional_publications(root: Path, mapping: dict, errors: list[str]) -> set[str]:
    """Authorize only exact files from an independently verified new publication."""
    known = set()
    for entry in mapping.get('additional_publications', []):
        case = entry.get('case_path')
        # This extension is specifically the reviewed Case008, not a directory wildcard.
        if case != 'cases/008-build-reconstruct-reload' or entry.get('archive_sha256') != '49cc2640b63d7662cf3d40c168eaf036505bbf5f7f5f73de8de48099d452daca':
            errors.append('Unknown additional publication'); continue
        try:
            with tempfile.TemporaryDirectory() as temp:
                proc = subprocess.run([sys.executable, '-B', str(root/case/'publication/verify_publication.py'),
                    '--root', str(root), '--output', str(Path(temp)/'check.json')], capture_output=True, text=True)
                if proc.returncode:
                    errors.append('Additional publication verification failed'); continue
            if entry['manifest'] != case+'/PUBLICATION_SHA256SUMS':
                raise ValueError('Unexpected manifest')
            for line in (root/entry['manifest']).read_text().splitlines():
                digest, name = line.split('  ', 1)
                if name in known or sha(root/name) != digest:
                    raise ValueError('Duplicate/hash mismatch')
                known.add(name)
            known.add(entry['manifest'])
            expected = 'downloads/case008_build_reconstruct_reload_reviewed_publication_v2.json'
            if entry['download_metadata'] != expected:
                raise ValueError('Unexpected download metadata')
            meta=json.loads((root/expected).read_text());name=meta['filename']
            if name!='case008_build_reconstruct_reload_reviewed_publication_v2.zip':
                raise ValueError('Unexpected download filename')
            zip_path=root/'downloads'/name
            if zip_path.is_symlink() or sha(zip_path)!=meta['sha256'] or zip_path.stat().st_size!=meta['bytes']:
                raise ValueError('Download hash/size mismatch')
            if sha(root/entry['manifest'])!=meta['publication_manifest_sha256']:
                raise ValueError('Download manifest mismatch')
            known.update([expected,'downloads/'+name])
        except (OSError, KeyError, ValueError) as exc:
            errors.append('Invalid additional publication: '+str(exc))
    return known


def check_new_study(root: Path, errors: list[str]) -> set[str]:
    """An exact new-case inventory; no exception within historical Cases001–008."""
    base = root/'cases/009-q-serving-quality'
    if not base.exists():
        return set()
    try:
        protocol = base/'configs/serving_protocol.json'
        if sha(protocol) != 'df84d13a33639f10102c372dc0ea55b7520fe068e030cc6818735a9df1e3f13f':
            raise ValueError('Frozen serving protocol changed')
        for name,digest in {
            'quality_protocol.json':'bd2a8ae107e9abaf070e7bb99f30e059e06b141fd19daaeab165f6d2bdbb4052',
            'quality_input_freeze.json':'f1145adb18a8e71c49a15096d8db07aa33f91930913a6c71538bda80a8c30cc1',
            'quality_analysis_plan.json':'9be69986602eebe1ccbe4428dff683d12e22536da1dba09d921a1f535aa99650',
        }.items():
            if sha(base/'configs'/name) != digest:
                raise ValueError('Frozen quality source changed: '+name)
        for name,digest in {
            'serving.py':'2d63b6a070fc08aeb148873173b5a2627955696fa284c94137a3f75cdeab44fd',
            'freeze_quality_inputs.py':'d4d12e5fc04a19203293e82dde75d5d7c3a010bafe8600d32bf91d08d22ae152',
            'quality.py':'31fe5772051f7fbde1cc985c2b5072ab661890e8c0f850eefce4ab27e293fc71',
        }.items():
            if sha(base/'scripts'/name) != digest:
                raise ValueError('Frozen execution source changed: '+name)
        publication = base/'publication/verify_publication.py'
        if publication.exists():
            import importlib.util
            spec=importlib.util.spec_from_file_location('case009_publication_docs',publication)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            module.verify(root)
            known={p.relative_to(root).as_posix() for p in base.rglob('*') if p.is_file()}
            for stem in ['case009_serving_quality_reviewed_publication_v1','diova_compare-0.1.1']:
                meta_name=stem+'.json' if stem.startswith('case009') else stem+'.metadata.json'
                meta=json.loads((root/'downloads'/meta_name).read_text());p=root/'downloads'/meta['filename']
                if p.is_symlink() or sha(p)!=meta['sha256'] or p.stat().st_size!=meta['bytes']:
                    raise ValueError('New publication download hash/size mismatch')
                known.update(['downloads/'+meta_name,'downloads/'+meta['filename']])
            return known
        expected = {}
        for line in (base/'SHA256SUMS').read_text().splitlines():
            digest, name = line.split('  ', 1)
            path = base/name
            if name in expected or name == 'SHA256SUMS' or Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
                raise ValueError('Unsafe/duplicate new-study inventory')
            if sha(path) != digest:
                raise ValueError('New-study checksum mismatch: '+name)
            expected[name] = digest
        actual = {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file() or p.is_symlink()}
        if actual != set(expected) | {'SHA256SUMS'}:
            raise ValueError('New-study exact inventory mismatch')
        return {'cases/009-q-serving-quality/'+name for name in actual}
    except (OSError, ValueError) as exc:
        errors.append('New study verification failed: '+str(exc))
        return set()


def check_feedback_docs(root: Path, errors: list[str]) -> int:
    links = check_links(root, list(FEEDBACK_DOCS), errors)
    check_copy_hygiene(root, FEEDBACK_DOCS, errors)
    for folder,stem in [('docs/related-work','README'),('packages/diova-compare','README'),
                        ('cases/009-q-serving-quality','README'),('cases/009-q-serving-quality','REPORT')]:
        en,ko=root/folder/(stem+'.md'),root/folder/(stem+'.ko.md')
        if not en.is_file() or not ko.is_file():
            errors.append('Missing feedback language counterpart: '+folder+'/'+stem)
        elif f']({stem}.ko.md)' not in en.read_text() or f']({stem}.md)' not in ko.read_text():
            errors.append('Wrong feedback language link: '+folder+'/'+stem)
    return links


def audit(root: Path, inventory: Path | None = None) -> dict:
    root = root.resolve()
    errors: list[str] = []
    mapping = json.loads((root/'docs/_meta/claims.json').read_text())
    links = check_links(root, [*PAGES, 'docs/_meta/MAINTENANCE.md'], errors)
    feedback_links = check_feedback_docs(root, errors) if (root/'cases/009-q-serving-quality').exists() else 0
    check_pairs(root, errors)
    check_beginner_routes(root, errors)
    check_concept_figure(root, errors)
    count = check_claims(root, mapping, list(PAGES), errors)
    additional = check_additional_publications(root, mapping, errors)
    additional |= check_new_study(root, errors)
    protected = check_protection(root, mapping.get('protection_revision', mapping['source_revision']), errors, inventory, additional)
    check_copy_hygiene(root, (*PAGES, CONCEPT_FIGURE, 'docs/_meta/claims.json', 'docs/_meta/MAINTENANCE.md'), errors)
    downloads = check_packages(root, mapping, errors)
    return {'status': 'PASS' if not errors else 'FAIL', 'errors': errors,
            'markdown_parser': 'markdown-it-py' if MarkdownIt is not None else 'documented limited-syntax stdlib fallback',
            'authored_pages': len(PAGES), 'local_links_checked': links, 'claims_checked': count,
            'feedback_local_links_checked': feedback_links,
            'protected_files_checked': protected, 'source_revision': mapping['source_revision'],
            'downloads': downloads, 'scope': 'Supported Markdown syntax, path/anchor existence, paired beginner routes and glossary IDs, declarative SVG, claim-ID coverage, source hashes/locators, selected literal displays, existing archive identity and protected bytes. No experimental result recalculation.',
            'not_proven': ['Semantic equivalence of translations (requires editorial review)',
                           'Every numeric token or claim entailment', 'External URL availability',
                           'Browser layout or real iPad behavior', 'Scientific/GPU reproduction']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--inventory', type=Path, help='Optional private starting SHA256 inventory; otherwise compare protected Git blobs at protection_revision; only declared guide files are editable')
    parser.add_argument('--output', type=Path, required=True, help='New report outside the repository')
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.repo.resolve()):
        parser.error('Choose a new output outside the repository; never overwrite source')
    result = audit(args.repo, args.inventory)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
