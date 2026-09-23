"""Small successful/failing fixtures for the documentation checker, no ML imports."""
import copy
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from check_docs import (anchors, check_claims, check_links, check_pairs, check_packages,
                        check_protected_additions, check_protection,
                        check_beginner_routes, check_concept_figure, check_copy_hygiene,
                        check_unified_case010, CONCEPT_FIGURE, PAGES)


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

    def test_incomplete_unified_case_is_not_added_to_protection_allowlist(self):
        case = self.root/'cases/010-ckda-finite-precision-memory-horizon'
        case.mkdir(parents=True)
        (case/'README.md').write_text('Synthetic incomplete integration fixture')
        errors = []
        self.assertEqual(check_unified_case010(self.root, errors), set())
        self.assertTrue(any('Case010 integration verification failed' in e for e in errors))

    def test_unified_archive_claim_cannot_authorize_unrelated_source(self):
        self.mapping['unified_snapshots'] = [{
            'case_path':'cases/010-ckda-finite-precision-memory-horizon',
            'version':'v2',
            'archive_sha256':'efe3ab2586a3cec757429386a7f874b594a9893fc771c804f5871dfaa61ed9cc'}]
        self.mapping['claims'][0]['source_revision'] = 'archive:' + self.mapping['unified_snapshots'][0]['archive_sha256']
        self.assertTrue(any('Invalid source revision' in e for e in self.claims_errors()))

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

    def beginner_fixture(self):
        for lang, home in (('en', 'README.md'), ('ko', 'README.ko.md')):
            for name, prefix in ((home, f'docs/{lang}/'), (f'docs/{lang}/START_HERE.md', '')):
                with (self.root/name).open('a') as stream:
                    stream.write('\n'.join(f'[Case {n}]({prefix}CASEBOOK.md#case-{n:03d})' for n in range(1, 8)))
                    stream.write(f'\n[Terms]({prefix}GLOSSARY.md) [Start]({prefix}START_HERE.md)\n')
            with (self.root/f'docs/{lang}/GLOSSARY.md').open('a') as stream:
                stream.write('\n'.join(f'<a id="term-{n}"></a>' for n in range(20)))

    def test_beginner_routes_and_glossary_pairs(self):
        self.beginner_fixture()
        errors=[];check_beginner_routes(self.root, errors)
        self.assertEqual(errors, [])

    def test_missing_beginner_case_route(self):
        self.beginner_fixture()
        p=self.root/'docs/en/START_HERE.md'
        p.write_text(p.read_text().replace('CASEBOOK.md#case-007', 'CASEBOOK.md'))
        errors=[];check_beginner_routes(self.root, errors)
        self.assertIn('Missing beginner case route: docs/en/START_HERE.md/7', errors)

    def test_missing_new_language_pages(self):
        for name in ('START_HERE.md', 'GLOSSARY.md'):
            with self.subTest(name=name):
                p=self.root/f'docs/ko/{name}';old=p.read_text();p.unlink()
                errors=[];check_pairs(self.root, errors)
                self.assertTrue(any('Missing language counterpart' in e for e in errors))
                p.write_text(old)

    def test_glossary_concept_mismatch(self):
        self.beginner_fixture()
        p=self.root/'docs/ko/GLOSSARY.md';p.write_text(p.read_text().replace('term-19', 'different'))
        errors=[];check_beginner_routes(self.root, errors)
        self.assertIn('Unequal glossary concept anchors', errors)

    def test_broken_image_link(self):
        with (self.root/'README.md').open('a') as stream:stream.write('\n![Concept](absent.svg)\n')
        errors=[];check_links(self.root,['README.md'],errors)
        self.assertTrue(any('Broken local link' in e for e in errors))

    def svg_fixture(self, content=''):
        p=self.root/CONCEPT_FIGURE;p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('<svg xmlns="http://www.w3.org/2000/svg"><title>Route</title><desc>Concept only</desc>'+content+'</svg>')
        return p

    def test_declarative_svg(self):
        self.svg_fixture('<text x="5" y="10">RUN</text>')
        errors=[];check_concept_figure(self.root,errors)
        self.assertEqual(errors, [])

    def test_active_svg_is_rejected(self):
        for content in ('<script>ignored()</script>', '<text onclick="ignored()">X</text>', '<image href="https://invalid.invalid/a"/>'):
            with self.subTest(content=content):
                self.svg_fixture(content)
                errors=[];check_concept_figure(self.root, errors)
                self.assertTrue(any('Invalid concept figure' in e for e in errors))

    def test_svg_needs_accessible_description(self):
        p=self.svg_fixture();p.write_text(p.read_text().replace('<desc>Concept only</desc>', ''))
        errors=[];check_concept_figure(self.root,errors)
        self.assertTrue(any('Invalid concept figure' in e for e in errors))

    def test_scientific_change_not_hidden_by_editable_guides(self):
        p=self.root/'cases/frozen/result.json';p.parent.mkdir(parents=True);p.write_text('original')
        inv=self.root/'inventory.json'
        inv.write_text(json.dumps({'base_revision':'a'*40,'files':{'cases/frozen/result.json':hashlib.sha256(p.read_bytes()).hexdigest()}}))
        errors=[];self.assertEqual(check_protection(self.root,'a'*40,errors,inv),1);self.assertEqual(errors,[])
        p.write_text('changed');errors=[];check_protection(self.root,'a'*40,errors,inv)
        self.assertIn('Protected file changed: cases/frozen/result.json',errors)

    def test_copy_hygiene_rejects_private_path_and_unfinished_copy(self):
        for value in ('/home/fixture/private.json', 'ghp_'+'x'*25, 'turn42search8', 'TODO'):
            with self.subTest(value=value):
                (self.root/'README.md').write_text(value)
                errors=[];check_copy_hygiene(self.root,('README.md',),errors)
                self.assertTrue(any('Private identifier/credential/unfinished citation' in e for e in errors))

    def test_copy_hygiene_accepts_normal_introduction(self):
        (self.root/'README.md').write_text('Compare recorded answers. 결과를 비교합니다.')
        errors=[];check_copy_hygiene(self.root,('README.md',),errors)
        self.assertEqual(errors, [])


class ProtectionScope(unittest.TestCase):
    def test_guides_are_editable_but_science_is_protected(self):
        from check_docs import PUBLIC_FILES
        self.assertIn('README.ko.md', PUBLIC_FILES)
        self.assertIn('docs/en/PORTFOLIO.md', PUBLIC_FILES)
        self.assertNotIn('cases/006-attention-decision-stability/README.md', PUBLIC_FILES)
        self.assertNotIn('LICENSE', PUBLIC_FILES)

    def test_changed_protected_bytes_are_detected(self):
        from check_docs import check_protection
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'cases').mkdir();path=root/'cases/frozen.json';path.write_text('old')
            inventory=root/'inventory.json';inventory.write_text(json.dumps({'base_revision':'a'*40,'files':{'cases/frozen.json':hashlib.sha256(b'old').hexdigest()}}))
            path.write_text('changed');errors=[];check_protection(root,'a'*40,errors,inventory)
            self.assertIn('Protected file changed: cases/frozen.json',errors)


class NewStudyBoundaryTests(unittest.TestCase):
    def test_new_inventory_cannot_approve_changed_frozen_protocol(self):
        from check_docs import check_new_study
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);case=root/'cases/009-q-serving-quality';(case/'configs').mkdir(parents=True)
            p=case/'configs/serving_protocol.json';p.write_text('{"changed": true}\n')
            (case/'SHA256SUMS').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  configs/serving_protocol.json\n')
            errors=[];self.assertEqual(check_new_study(root,errors),set())
            self.assertTrue(any('Frozen serving protocol changed' in x for x in errors))

    def test_new_docs_reject_missing_counterpart_and_broken_link(self):
        from check_docs import check_feedback_docs
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'docs/related-work/README.md';p.parent.mkdir(parents=True)
            p.write_text('[Missing](absent.md)')
            errors=[];check_feedback_docs(root,errors)
            self.assertTrue(any('Broken local link' in x for x in errors))
            self.assertTrue(any('Missing feedback language counterpart' in x for x in errors))


if __name__ == '__main__':
    unittest.main()
