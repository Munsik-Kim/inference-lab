"""Small successful/failing fixtures for the documentation checker, no ML imports."""
import copy
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from check_docs import anchors, check_claims, check_links, check_pairs, check_packages, check_protected_additions, PAGES


class DocumentChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in PAGES:
            p = self.root/name
            p.parent.mkdir(parents=True, exist_ok=True)
            if name == 'README.md': nav = '[KO](README.ko.md)'
            elif name == 'README.ko.md': nav = '[EN](README.md)'
            elif '/en/' in name: nav = f'[KO](../ko/{p.name}) [Home](../../README.md)'
            else: nav = f'[EN](../en/{p.name}) [홈](../../README.ko.md)'
            case_anchors = '\n'.join(f'<a id="case-{n:03d}"></a>' for n in range(1, 8)) if p.name == 'CASEBOOK.md' else ''
            p.write_text('# Fixture\n'+nav+'\n'+case_anchors+'\n<!-- claims: measured -->\nCount: 7\n')
        src=self.root/'source.json'
        src.write_text('{"count":7}\n')
        self.mapping={'source_revision':'a'*40,'claims':[{
            'claim_id':'measured','claim_en':'Seven recorded fixtures.','claim_ko':'기록된 fixture 일곱 개.',
            'scope':'Unit test only','limits':'Not GPU evidence','source_revision':'a'*40,
            'source_path':'source.json','source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
            'source_locator':{'kind':'json','checks':[{'pointer':'/count','equals':7}]},
            'docs_locations':list(PAGES),'display_checks':[{'path':'README.md','contains':['Count: 7']}]}]}

    def claims_errors(self):
        errors=[]
        check_claims(self.root,self.mapping,list(PAGES),errors)
        return errors

    def test_valid_pages_sources_and_pairs(self):
        errors=[]
        check_links(self.root,list(PAGES),errors)
        check_pairs(self.root,errors)
        self.assertEqual(errors+self.claims_errors(),[])

    def test_documented_fallback_without_optional_parser(self):
        errors=[]
        with patch('check_docs.MarkdownIt', None):
            check_links(self.root,list(PAGES),errors)
        self.assertEqual(errors,[])

    def test_addition_in_protected_tree_is_rejected(self):
        folder=self.root/'cases/unchanged';folder.mkdir(parents=True)
        (folder/'extra.md').write_text('Unapproved fixture addition')
        errors=[];check_protected_additions(self.root,set(),errors)
        self.assertTrue(any('Unexpected file in protected tree' in e for e in errors))

    def test_broken_local_link(self):
        with (self.root/'README.md').open('a') as f:f.write('[Broken](absent.md)\n')
        errors=[];check_links(self.root,list(PAGES),errors)
        self.assertTrue(any('Broken local link' in e for e in errors))

    def test_missing_language_counterpart(self):
        (self.root/'docs/ko/CASEBOOK.md').unlink()
        errors=[];check_pairs(self.root,errors)
        self.assertTrue(any('Missing language counterpart' in e for e in errors))

    def test_missing_seventh_case_anchor(self):
        p = self.root/'docs/ko/CASEBOOK.md'
        p.write_text(p.read_text().replace('<a id="case-007"></a>', ''))
        errors = []
        check_pairs(self.root, errors)
        self.assertIn('Missing case anchor: ko/7', errors)

    def package_fixture(self):
        folder = self.root/'downloads'
        folder.mkdir()
        claims = []
        for number in ('006', '007'):
            archive = folder/f'case{number}.zip'
            archive.write_bytes(b'Unit fixture only; no measured results')
            meta = folder/f'case{number}.json'
            meta.write_text(json.dumps({'filename': archive.name, 'bytes': archive.stat().st_size,
                                       'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}))
            claims.append({'claim_id': f'c{number}-package', 'source_path': str(meta.relative_to(self.root))})
            for lang in ('en', 'ko'):
                with (self.root/f'docs/{lang}/GETTING_STARTED.md').open('a') as stream:
                    stream.write(f'\n{archive.name} {meta.name}\n')
        return {'claims': claims}

    def test_both_package_identities(self):
        mapping = self.package_fixture()
        errors = []
        self.assertEqual(check_packages(self.root, mapping, errors), ['case006.zip', 'case007.zip'])
        self.assertEqual(errors, [])

    def test_changed_case007_archive_is_rejected(self):
        mapping = self.package_fixture()
        (self.root/'downloads/case007.zip').write_bytes(b'Changed fixture')
        errors = []
        check_packages(self.root, mapping, errors)
        self.assertIn('Default download identity mismatch: c007-package', errors)

    def test_unsafe_package_name_is_rejected(self):
        mapping = self.package_fixture()
        p = self.root/'downloads/case007.json'
        meta = json.loads(p.read_text())
        meta['filename'] = '../case007.zip'
        p.write_text(json.dumps(meta))
        errors = []
        check_packages(self.root, mapping, errors)
        self.assertIn('Unsafe package filename: c007-package', errors)

    def test_wrong_counterpart_target(self):
        p=self.root/'docs/en/CASEBOOK.md';p.write_text(p.read_text().replace('../ko/CASEBOOK.md','../ko/PORTFOLIO.md'))
        errors=[];check_pairs(self.root,errors)
        self.assertTrue(any('Wrong language-switch' in e for e in errors))

    def test_missing_source(self):
        (self.root/'source.json').unlink()
        self.assertTrue(any('Missing source' in e for e in self.claims_errors()))

    def test_changed_source_hash(self):
        (self.root/'source.json').write_text('{"count":8}\n')
        self.assertTrue(any('Changed source hash' in e for e in self.claims_errors()))

    def test_wrong_claim_id(self):
        p=self.root/'README.md';p.write_text(p.read_text().replace('claims: measured','claims: invented'))
        self.assertTrue(any('Unknown claim ID' in e for e in self.claims_errors()))

    def test_duplicate_claim_mapping(self):
        self.mapping['claims'].append(copy.deepcopy(self.mapping['claims'][0]))
        self.assertIn('Duplicate claim ID mapping',self.claims_errors())

    def test_duplicate_document_mapping(self):
        self.mapping['claims'][0]['docs_locations'].append('README.md')
        self.assertTrue(any('Duplicate document mapping' in e for e in self.claims_errors()))

    def test_changed_numeric_display(self):
        p=self.root/'README.md';p.write_text(p.read_text().replace('Count: 7','Count: 8'))
        self.assertTrue(any('Document numeric' in e for e in self.claims_errors()))

    def test_wrong_json_pointer(self):
        self.mapping['claims'][0]['source_locator']['checks'][0]['pointer']='/missing'
        self.assertTrue(any('Invalid source locator' in e for e in self.claims_errors()))

    def test_heading_anchor_and_missing_anchor(self):
        self.assertEqual(anchors('# Repeat\n## Repeat\n<a id="manual"></a>\n'),{'repeat','repeat-1','manual'})
        with (self.root/'README.md').open('a') as f:f.write('[Bad](README.ko.md#missing)\n')
        errors=[];check_links(self.root,list(PAGES),errors)
        self.assertTrue(any('Missing heading' in e for e in errors))

    def test_unsupported_markdown_fails(self):
        with (self.root/'README.md').open('a') as f:f.write('[Unsupported][ref]\n[ref]: README.ko.md\n')
        errors=[];check_links(self.root,list(PAGES),errors)
        self.assertTrue(any('Unsupported reference' in e for e in errors))


if __name__ == '__main__':
    unittest.main()
