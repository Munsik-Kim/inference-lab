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
from pathlib import Path
from urllib.parse import unquote, urlsplit

try:
    from markdown_it import MarkdownIt
except ImportError:
    MarkdownIt = None

GUIDES = ('CASEBOOK.md', 'GETTING_STARTED.md', 'PORTFOLIO.md')
PAGES = ('README.md', 'README.ko.md', *(f'docs/{lang}/{name}' for lang in ('en', 'ko') for name in GUIDES))
PUBLIC_FILES = (*PAGES, 'docs/_meta/claims.json', 'docs/_meta/MAINTENANCE.md',
                'docs/_checks/check_docs.py', 'docs/_checks/test_check_docs.py')
LINK = re.compile(r'!?\[[^\]\n]*\]\(([^\s()]+)\)')
CLAIMS = re.compile(r'<!-- claims: ([a-zA-Z0-9_\- ]+) -->')


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
        if c['source_revision'] != mapping['source_revision'] or not re.fullmatch('[0-9a-f]{40}', c['source_revision']):
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


def check_protection(root: Path, base: str, errors: list[str], inventory: Path | None) -> int:
    if inventory:
        data = json.loads(inventory.read_text())
        if data['base_revision'] != base:
            errors.append('Protection inventory revision mismatch')
        for name, expected in data['files'].items():
            f = root/name
            if not f.is_file() or sha(f) != expected:
                errors.append('Protected file changed: ' + name)
        check_protected_additions(root, set(data['files']), errors)
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
    check_protected_additions(root, known, errors)
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


def audit(root: Path, inventory: Path | None = None) -> dict:
    root = root.resolve()
    errors: list[str] = []
    mapping = json.loads((root/'docs/_meta/claims.json').read_text())
    links = check_links(root, [*PAGES, 'docs/_meta/MAINTENANCE.md'], errors)
    check_pairs(root, errors)
    count = check_claims(root, mapping, list(PAGES), errors)
    protected = check_protection(root, mapping.get('protection_revision', mapping['source_revision']), errors, inventory)
    blocked = re.compile(r'\bTODO\b|\bTBD\b|\bPLACEHOLDER\b|example\.(?:com|org)|turn\d+(?:search|view|fetch)\d+|filecite|/home/[^/\s]+/|/mnt/|C:\\\\Users|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}')
    for name in (*PAGES, 'docs/_meta/claims.json', 'docs/_meta/MAINTENANCE.md'):
        if (root/name).is_file() and blocked.search((root/name).read_text()):
            errors.append('Private identifier/credential/unfinished citation pattern: ' + name)
    downloads = check_packages(root, mapping, errors)
    return {'status': 'PASS' if not errors else 'FAIL', 'errors': errors,
            'markdown_parser': 'markdown-it-py' if MarkdownIt is not None else 'documented limited-syntax stdlib fallback',
            'authored_pages': len(PAGES), 'local_links_checked': links, 'claims_checked': count,
            'protected_files_checked': protected, 'source_revision': mapping['source_revision'],
            'downloads': downloads, 'scope': 'Supported Markdown syntax, path/anchor existence, language navigation/claim-ID coverage, source hashes/locators, selected literal numeric displays, existing archive identity and protected bytes.',
            'not_proven': ['Semantic equivalence of translations (requires editorial review)',
                           'Every numeric token or claim entailment', 'External URL availability',
                           'Browser layout or real iPad behavior', 'Scientific/GPU reproduction']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--inventory', type=Path, help='Optional private starting SHA256 inventory; otherwise compare protected Git blobs at protection_revision; authored guides are editable')
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
